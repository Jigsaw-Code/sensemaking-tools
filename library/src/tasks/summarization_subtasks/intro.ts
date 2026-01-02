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

import { RecursiveSummary } from "./recursive_summarization";
import { SummaryStats } from "../../stats/summary_stats";
import { SummaryContent } from "../../types";

// Import localization system
import { 
  getReportSectionTitle, 
  getReportContent 
} from "../../../templates/l10n";

export class IntroSummary extends RecursiveSummary<SummaryStats> {
  getSummary(): Promise<SummaryContent> {
    // Get localized title and text from localization system
    const title = getReportSectionTitle("introduction", this.output_lang);
    const text = getReportContent("introduction", "text", this.output_lang);
    const statementsLabel = getReportContent("introduction", "statements", this.output_lang);
    const votesLabel = getReportContent("introduction", "votes", this.output_lang);
    const topicsLabel = getReportContent("introduction", "topics", this.output_lang);
    const subtopicsLabel = getReportContent("introduction", "subtopics", this.output_lang);
    const anonymousText = getReportContent("introduction", "anonymous", this.output_lang);
    
    // Build the content with dynamic values
    const content = `${text}\n` +
      ` * __${this.input.commentCount.toLocaleString()} ${statementsLabel}__\n` +
      ` * __${this.input.voteCount.toLocaleString()} ${votesLabel}__\n` +
      ` * ${this.input.getStatsByTopic().length} ${topicsLabel}\n` +
      ` * ${this.getSubtopicCount()} ${subtopicsLabel}\n\n` +
      `${anonymousText}`;
    
    return Promise.resolve({ title, text: content });
  }
  
  private getSubtopicCount(): number {
    const statsByTopic = this.input.getStatsByTopic();
    return statsByTopic.map(topic => topic.subtopicStats?.length || 0).reduce((a, b) => a + b, 0);
  }
}
