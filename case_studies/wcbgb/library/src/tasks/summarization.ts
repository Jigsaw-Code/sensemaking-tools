// Copyright 2024 Google LLC
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

// Functions for different ways to summarize Comment and Vote data.

import { Model } from "../models/model";
import { Comment, SummarizationType, Summary, SummaryContent } from "../types";
import { IntroSummary } from "./summarization_subtasks/intro";
import { OverviewSummary } from "./summarization_subtasks/overview";
import { GroupsSummary } from "./summarization_subtasks/groups";
import { GroupedSummaryStats } from "../stats/group_informed";
import { MajoritySummaryStats } from "../stats/majority_vote";
import { SummaryStats, TopicStats } from "../stats/summary_stats";
import { TopSubtopicsSummary } from "./summarization_subtasks/top_subtopics";
import { AllTopicsSummary } from "./summarization_subtasks/topics";

/**
 * Summarizes comments based on the specified summarization type.
 *
 * @param model The language model to use for summarization.
 * @param comments An array of `Comment` objects containing the comments to summarize.
 * @param summarizationType The type of summarization to perform (e.g., GROUP_INFORMED_CONSENSUS).
 * @param additionalContext Optional additional instructions to guide the summarization process. These instructions will be included verbatim in the prompt sent to the LLM.
 * @returns A Promise that resolves to the generated summary string.
 * @throws {TypeError} If an unknown `summarizationType` is provided.
 */
export async function summarizeByType(
  model: Model,
  comments: Comment[],
  summarizationType: SummarizationType,
  additionalContext?: string
): Promise<Summary> {
  let summaryStats: SummaryStats;
  if (summarizationType === SummarizationType.GROUP_INFORMED_CONSENSUS) {
    summaryStats = new GroupedSummaryStats(comments);
  } else if (summarizationType === SummarizationType.AGGREGATE_VOTE) {
    summaryStats = new MajoritySummaryStats(comments);
  } else {
    throw new TypeError("Unknown Summarization Type.");
  }
  return new MultiStepSummary(summaryStats, model, additionalContext).getSummary();
}

/**
 *
 */
export class MultiStepSummary {
  private summaryStats: SummaryStats;
  private model: Model;
  // TODO: Figure out how we handle additional instructions with this structure.
  private additionalContext?: string;

  constructor(summaryStats: SummaryStats, model: Model, additionalContext?: string) {
    this.summaryStats = summaryStats;
    this.model = model;
    this.additionalContext = additionalContext;
  }

  async getSummary(): Promise<Summary> {
    const topicsSummary = await new AllTopicsSummary(
      this.summaryStats,
      this.model,
      this.additionalContext
    ).getSummary();
    const summarySections: SummaryContent[] = [];
    summarySections.push(
      await new IntroSummary(this.summaryStats, this.model, this.additionalContext).getSummary()
    );
    summarySections.push(
      await new OverviewSummary(
        { summaryStats: this.summaryStats, topicsSummary: topicsSummary, method: "one-shot" },
        this.model,
        this.additionalContext
      ).getSummary()
    );
    summarySections.push(
      await new TopSubtopicsSummary(
        this.summaryStats,
        this.model,
        this.additionalContext
      ).getSummary()
    );
    if (this.summaryStats.groupBasedSummarization) {
      summarySections.push(
        await new GroupsSummary(
          this.summaryStats as GroupedSummaryStats,
          this.model,
          this.additionalContext
        ).getSummary()
      );
    }
    summarySections.push(topicsSummary);
    return new Summary(summarySections, this.summaryStats.comments);
  }
}

/**
 * Quantifies topic names by adding the number of associated comments in parentheses.
 *
 * @param topics An array of `TopicStats` objects.
 * @returns A map where keys are quantified topic names and values are arrays of quantified subtopic names.
 *
 * @example
 * Example input:
 * [
 *   {
 *     name: 'Topic A',
 *     commentCount: 5,
 *     subtopicStats: [
 *       { name: 'Subtopic 1', commentCount: 2 },
 *       { name: 'Subtopic 2', commentCount: 3 }
 *     ]
 *   }
 * ]
 *
 * Expected output:
 * {
 *   'Topic A (5 comments)': [
 *     'Subtopic 1 (2 comments)',
 *     'Subtopic 2 (3 comments)'
 *   ]
 * }
 */
export function _quantifyTopicNames(topics: TopicStats[]): { [key: string]: string[] } {
  const result: { [key: string]: string[] } = {};

  for (const topic of topics) {
    const topicName = `${topic.name} (${topic.commentCount} comments)`;

    if (topic.subtopicStats) {
      result[topicName] = topic.subtopicStats.map(
        (subtopic) => `${subtopic.name} (${subtopic.commentCount} comments)`
      );
    }
  }

  return result;
}
