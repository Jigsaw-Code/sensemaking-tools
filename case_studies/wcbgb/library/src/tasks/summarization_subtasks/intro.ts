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

// Functions for different ways to summarize Comment and Vote data.

import { SummaryStats, TopicStats } from "../../stats/summary_stats";
import { SummaryContent } from "../../types";
import { RecursiveSummary } from "./recursive_summarization";

export class IntroSummary extends RecursiveSummary<SummaryStats> {
  getSummary(): Promise<SummaryContent> {
    let text = `This report summarizes the results of public input, encompassing:\n`;
    const commentCountFormatted = this.input.commentCount.toLocaleString();
    text += ` * __${commentCountFormatted} statements__\n`;
    const voteCountFormatted = this.input.voteCount.toLocaleString();
    text += ` * __${voteCountFormatted} votes__\n`;
    const statsByTopic = this.input.getStatsByTopic();
    text += ` * ${statsByTopic.length} topics\n`;
    const subtopicCount = statsByTopic
      .map((topic: TopicStats) => {
        return topic.subtopicStats ? topic.subtopicStats.length : 0;
      })
      .reduce((a, b) => a + b, 0);
    text += ` * ${subtopicCount} subtopics\n\n`;
    // TODO: Add how many themes there are when it's available.
    text += "All voters were anonymous.";

    return Promise.resolve({ title: "## Introduction", text: text });
  }
}
