// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file contains routines for generating summaries of the key findings from a report,
// based on the results of the more detailed topic and subtopic summaries

import { SummaryStats, TopicStats } from "../../stats/summary_stats";
import { SummaryContent, Summary } from "../../types";
import { RecursiveSummary } from "./recursive_summarization";
import {
  getAbstractPrompt,
  decimalToPercent,
  filterSummaryContent,
  retryCall,
} from "../../sensemaker_utils";

function oneShotInstructions(topicNames: string[]) {
  return (
    `Your job is to compose a summary of the key findings from a public discussion, based on already composed summaries corresponding to topics and subtopics identified in said discussion. ` +
    `These topic and subtopic summaries are based on comments and voting patterns that participants submitted as part of the discussion. ` +
    `You should format the results as a markdown list, to be included near the top of the final report, which shall include the complete topic and subtopic summaries. ` +
    `Do not pretend that you hold any of these opinions. You are not a participant in this discussion. ` +
    `Do not include specific numbers about how many comments were included in each topic or subtopic, as these will be included later in the final report output. ` +
    `You also do not need to recap the context of the conversation, as this will have already been stated earlier in the report. ` +
    `Where possible, prefer describing the results in terms of the "statements" submitted or the overall "conversation", rather than in terms of the participants' perspectives (Note: "comments" and "statements" are the same thing, but for the sake of this portion of the summary, only use the term "statements"). ` +
    `Remember: this is just one component of a larger report, and you should compose this so that it will flow naturally in the context of the rest of the report. ` +
    `Be clear and concise in your writing, and do not use the passive voice, or ambiguous pronouns.` +
    `\n\n` +
    `The structure of the list you output should be in terms of the topic names, in the order that follows. ` +
    `Each list item should start in bold with topic name name (including percentage, exactly as listed below), then a colon, and then a short one or two sentence summary for the corresponding topic.` +
    `The complete response should be only the markdown list, and no other text. ` +
    `For example, a list item might look like this:\n` +
    `<output_format format="markdown">* **Topic Name (45%):**  Topic summary.</output_format>\n` +
    `Here are the topics:
    ${topicNames.map((s) => "* " + s).join("\n")}`
  );
}

function perTopicInstructions(topicName: string) {
  return (
    `Your job is to compose a summary of the key findings from a public discussion, based on already composed summaries corresponding to topics and subtopics identified in said discussion. ` +
    `These topic and subtopic summaries are based on comments and voting patterns that participants submitted as part of the discussion. ` +
    `This summary will be formatted as a markdown list, to be included near the top of the final report, which shall include the complete topic and subtopic summaries. ` +
    `Do not pretend that you hold any of these opinions. You are not a participant in this discussion. ` +
    `Where possible, prefer descriging the results in terms of the "statements" submitted or the overall "conversation", rather than in terms of the participants' perspectives (Note: "comments" and "statements" are the same thing, but for the sake of this portion of the summary, only use the term "statements"). ` +
    `Do not include specific numbers about how many comments were included in each topic or subtopic, as these will be included later in the final report output. ` +
    `You also do not need to recap the context of the conversation, as this will have already been stated earlier in the report. ` +
    `Remember: this is just one component of a larger report, and you should compose this so that it will flow naturally in the context of the rest of the report. ` +
    `Be clear and concise in your writing, and do not use the passive voice, or ambiguous pronouns.` +
    `\n\n` +
    `Other topics will come later, but for now, your job is to compose a very short one or two sentence summary of the following topic: ${topicName}. ` +
    `This summary will be put together into a list with other such summaries later.`
  );
}

/**
 * The interface is the input structure for the OverviewSummary class, and controls
 * which specific method is used to generate this part of the summary.
 */
export interface OverviewInput {
  summaryStats: SummaryStats;
  topicsSummary: SummaryContent;
  method?: "one-shot" | "per-topic";
}

/**
 * Generates a summary of the key findings in the conversation, in terms of the top-level
 * topics.
 */
export class OverviewSummary extends RecursiveSummary<OverviewInput> {
  async getSummary(): Promise<SummaryContent> {
    const method = this.input.method || "one-shot";
    const result = await (method == "one-shot" ? this.oneShotSummary() : this.perTopicSummary());

    const preamble =
      `Below is a high level overview of the topics discussed in the conversation, as well as the percentage of statements categorized under each topic. ` +
      `Note that the percentages may add up to greater than 100% when statements fall under more than one topic.\n\n`;
    return { title: "## Overview", text: preamble + result };
  }

  /**
   * Produces a summary of the key findings within the conversation, based on the
   * results of the topicsSummary.
   * @returns A promise of the resulting summary string
   */
  async oneShotSummary(): Promise<string> {
    const topicNames = this.topicNames();
    const prompt = getAbstractPrompt(
      oneShotInstructions(topicNames),
      [filterSectionsForOverview(this.input.topicsSummary)],
      (summary: SummaryContent) =>
        `<topicsSummary>\n` +
        `${new Summary([summary], []).getText("XML")}\n` +
        `  </topicsSummary>`,
      this.additionalContext
    );
    return await retryCall(
      async function (model, prompt) {
        console.log(`Generating OVERVIEW SUMMARY in one shot`);
        let result = await model.generateText(prompt);
        result = removeEmptyLines(result);
        if (!result) {
          throw new Error(`Overview summary failed to conform to markdown list format.`);
        } else {
          return result;
        }
      },
      (result) => isMdListValid(result, topicNames),
      3,
      "Overview summary failed to conform to markdown list format, or did not include all topic descriptions exactly as intended.",
      undefined,
      [this.model, prompt],
      []
    );
  }

  /**
   * Generates a summary one topic at a time, and then programatically concatenates them.
   * @returns A promise of the resulting summary string
   */
  async perTopicSummary(): Promise<string> {
    let text = "";
    for (const topicStats of this.input.summaryStats.getStatsByTopic()) {
      text += `* __${this.getTopicNameAndCommentPercentage(topicStats)}__: `;
      const prompt = getAbstractPrompt(
        perTopicInstructions(topicStats.name),
        [filterSectionsForOverview(this.input.topicsSummary)],
        (summary: SummaryContent) =>
          `<topicsSummary>\n` +
          `${new Summary([summary], []).getText("XML")}\n` +
          `  </topicsSummary>`,
        this.additionalContext
      );
      console.log(`Generating OVERVIEW SUMMARY for topic: "${topicStats.name}"`);
      text += (await this.model.generateText(prompt)).trim() + "\n";
    }
    return text;
  }

  /**
   * @returns Topic names with the percentage of comments classified thereunder in parentheses
   */
  private topicNames() {
    const summaryStats = this.input.summaryStats;
    return summaryStats.getStatsByTopic().map((topicStats: TopicStats) => {
      return this.getTopicNameAndCommentPercentage(topicStats);
    });
  }

  private getTopicNameAndCommentPercentage(topicStats: TopicStats): string {
    const totalCommentCount = this.input.summaryStats.commentCount;
    const percentage = decimalToPercent(topicStats.commentCount / totalCommentCount, 0);
    return `${topicStats.name} (${percentage})`;
  }
}

/**
 * This function removes all of the common ground and differences of opinion components
 * from the input topicSummary object, leaving the original unmodified.
 * @param topicSummary The result of the TopicsSummary component
 * @returns the resulting summary, as a new data structure
 */
function filterSectionsForOverview(topicSummary: SummaryContent): SummaryContent {
  return filterSummaryContent(
    topicSummary,
    (subtopicSummary: SummaryContent) =>
      !subtopicSummary.title?.includes("Common ground") &&
      !subtopicSummary.title?.includes("Differences of opinion")
  );
}

/**
 * Remove all empty lines from the input string, useful when a model response formats
 * list items with empty lines between them (as though they are paragraphs, each containing
 * a single list item).
 * @param mdList A string, presumably representing a markdown list
 * @returns The input string, with all empty lines removed
 */
export function removeEmptyLines(mdList: string): string {
  return mdList.replace(/\s*[\r\n]+\s*/g, "\n").trim();
}

/**
 * This function processes the input markdown list string, ensuring that it matches
 * the expected format, normalizing it with `removeEmptyLines`, and ensuring that each
 * lines matches the expected format (* **bold topic**: summary...)
 */
export function isMdListValid(mdList: string, topicNames: string[]): boolean {
  const lines = mdList.split("\n");
  for (const [index, line] of lines.entries()) {
    // Check to make sure that every line matches the expected format
    // Valid examples:
    // * **Topic Name:** A summary.
    // *   **Topic Name with extra spaces in front:** A summary.
    // * __Topic Name:__ A summary.
    // - **Topic Name**:  A summary.
    // - __Topic Name__:  A summary.
    if (!line.match(/^[\*\-]\s+\*\*.*:?\*\*:?\s/) && !line.match(/^[\*\-]\s+\_\_.*:?\_\_:?\s/)) {
      console.log("Line does not match expected format:", line);
      return false;
    }
    // Check to make sure that every single topicName in topicNames is in the list, and in the right order
    if (!line.includes(topicNames[index])) {
      console.log(`Topic "${topicNames[index]}" not found at line:\n`, line);
      return false;
    }
  }
  return true;
}
