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

// Import localization system
import { getReportSectionTitle, getReportContent } from "../../../templates/l10n";
import { getOverviewOneShotPrompt, getOverviewPerTopicPrompt } from "../../../templates/l10n/prompts";
import { SupportedLanguage } from "../../../templates/l10n/languages";

function oneShotInstructions(topicNames: string[], output_lang: string) {
  return getOverviewOneShotPrompt(output_lang as SupportedLanguage, topicNames);
}

function perTopicInstructions(topicName: string, output_lang: string) {
  return getOverviewPerTopicPrompt(output_lang as SupportedLanguage, topicName);
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
    // Debug: 檢查 output_lang 值
    console.log(`[DEBUG] OverviewSummary.getSummary() output_lang: ${this.output_lang}`);
    
    const method = this.input.method || "one-shot";
    const result = await (method == "one-shot" ? this.oneShotSummary() : this.perTopicSummary());

    // Get localized title and preamble from localization system
    const title = getReportSectionTitle("overview", this.output_lang);
    const preamble = getReportContent("overview", "preamble", this.output_lang);
    
    // Debug: 檢查本地化函式的調用參數和結果
    console.log(`[DEBUG] OverviewSummary.getSummary() calling getReportSectionTitle with: section="overview", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] OverviewSummary.getSummary() calling getReportContent with: section="overview", content="preamble", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] OverviewSummary.getSummary() title result: "${title}"`);
    console.log(`[DEBUG] OverviewSummary.getSummary() preamble result: "${preamble}"`);
    
    return { title, text: preamble + result };
  }

  /**
   * Produces a summary of the key findings within the conversation, based on the
   * results of the topicsSummary.
   * @returns A promise of the resulting summary string
   */
  async oneShotSummary(): Promise<string> {
    const topicNames = this.topicNames();
    const output_lang = this.output_lang;
    
    // Debug: 檢查 oneShotSummary 中的 output_lang 值
    console.log(`[DEBUG] OverviewSummary.oneShotSummary() output_lang: ${output_lang}`);
    
    const prompt = getAbstractPrompt(
      oneShotInstructions(topicNames, output_lang),
      [filterSectionsForOverview(this.input.topicsSummary)],
      (summary: SummaryContent) =>
        `<topicsSummary>\n` +
        `${new Summary([summary], []).getText("XML")}\n` +
        `  </topicsSummary>`,
      this.additionalContext,
      this.output_lang  // ← 加入 output_lang 參數
    );
    
    // Debug: 檢查 getAbstractPrompt 的調用參數
    console.log(`[DEBUG] OverviewSummary.oneShotSummary() calling getAbstractPrompt with: output_lang="${this.output_lang}"`);
    
    return await retryCall(
      async function (model, prompt, output_lang) {
        console.log(`Generating OVERVIEW SUMMARY in one shot`);
        console.log(`[DEBUG] retryCall function received output_lang: ${output_lang}`);
        let result = await model.generateText(prompt, output_lang);
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
      [this.model, prompt, output_lang],  // ← 加入 output_lang
      []
    );
  }

  /**
   * Generates a summary one topic at a time, and then programatically concatenates them.
   * @returns A promise of the resulting summary string
   */
  async perTopicSummary(): Promise<string> {
    // Debug: 檢查 perTopicSummary 中的 output_lang 值
    console.log(`[DEBUG] OverviewSummary.perTopicSummary() output_lang: ${this.output_lang}`);
    
    let text = "";
    for (const topicStats of this.input.summaryStats.getStatsByTopic()) {
      text += `* __${this.getTopicNameAndCommentPercentage(topicStats)}__: `;
      const prompt = getAbstractPrompt(
        perTopicInstructions(topicStats.name, this.output_lang),
        [filterSectionsForOverview(this.input.topicsSummary)],
        (summary: SummaryContent) =>
          `<topicsSummary>\n` +
          `${new Summary([summary], []).getText("XML")}\n` +
          `  </topicsSummary>`,
        this.additionalContext,
        this.output_lang  // ← 加入 output_lang 參數
      );
      
      // Debug: 檢查 getAbstractPrompt 的調用參數
      console.log(`[DEBUG] OverviewSummary.perTopicSummary() calling getAbstractPrompt with: output_lang="${this.output_lang}"`);
      
      console.log(`Generating OVERVIEW SUMMARY for topic: "${topicStats.name}"`);
      console.log(`[DEBUG] Calling model.generateText with output_lang: ${this.output_lang}`);
      text += (await this.model.generateText(prompt, this.output_lang)).trim() + "\n";
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
    // 使用更寬鬆的檢查，處理可能的格式變化
    const expectedTopicName = topicNames[index];
    const normalizedExpected = normalizeTopicName(expectedTopicName);
    const normalizedLine = normalizeTopicName(line);
    
    if (!normalizedLine.includes(normalizedExpected)) {
      console.log(`Topic "${expectedTopicName}" not found at line:\n`, line);
      console.log(`Normalized expected: "${normalizedExpected}"`);
      console.log(`Normalized line: "${normalizedLine}"`);
      return false;
    }
  }
  return true;
}

/**
 * 標準化主題名稱，移除可能影響匹配的字符
 */
function normalizeTopicName(topicName: string): string {
  return topicName
    .toLowerCase()
    .replace(/["""]/g, '')           // 移除各種引號
    .replace(/['']/g, '')            // 移除各種單引號
    .replace(/[()]/g, '')            // 移除括號
    .replace(/\s+/g, ' ')            // 標準化空白字符
    .trim();
}
