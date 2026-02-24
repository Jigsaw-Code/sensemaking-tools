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

// A summary of the top subtopics.

import { SummaryStats, TopicStats } from "../../stats/summary_stats";
import { Comment, SummaryContent } from "../../types";
import { RecursiveSummary } from "./recursive_summarization";
import { getPrompt } from "../../sensemaker_utils";

export class TopSubtopicsSummary extends RecursiveSummary<SummaryStats> {
  async getSummary(): Promise<SummaryContent> {
    const allSubtopics = getFlattenedSubtopics(this.input.getStatsByTopic());
    const topSubtopics = getTopSubtopics(allSubtopics);

    const subtopicSummaryContents: SummaryContent[] = [];
    for (let i = 0; i < topSubtopics.length; ++i) {
      subtopicSummaryContents.push(await this.getSubtopicSummary(topSubtopics[i], i));
    }
    return Promise.resolve({
      title: `## Top ${topSubtopics.length} Most Discussed Subtopics`,
      text: `${allSubtopics.length} subtopics of discussion emerged. These ${topSubtopics.length} subtopics had the most statements submitted.`,
      subContents: subtopicSummaryContents,
    });
  }

  async getSubtopicSummary(st: TopicStats, index: number): Promise<SummaryContent> {
    const subtopicComments = st.summaryStats.comments;
    console.log(`Generating PROMINENT THEMES for top 5 subtopics: "${st.name}"`);
    const text = await this.model.generateText(
      getPrompt(
        `Please generate a concise bulleted list identifying up to 5 prominent themes across all statements. Each theme should be less than 10 words long.  Do not use bold text. Do not preface the bulleted list with any text. These statements are all about ${st.name}`,
        subtopicComments.map((comment: Comment): string => comment.text),
        this.additionalContext
      )
    );
    const themesSummary = { title: "Prominent themes were:", text: text };
    return Promise.resolve({
      title: `### ${index + 1}. ${st.name} (${st.commentCount} statements)`,
      text: "",
      subContents: [themesSummary],
    });
  }
}

function getTopSubtopics(allSubtopics: TopicStats[], max = 5) {
  // Sort all subtopics by comment count, desc
  allSubtopics.sort((a, b) => b.commentCount - a.commentCount);

  // Get top subtopics, skipping other
  const topSubtopics = [];
  for (const st of allSubtopics) {
    if (st.name == "Other") {
      continue;
    }
    topSubtopics.push(st);
    if (topSubtopics.length >= max) {
      break;
    }
  }
  return topSubtopics;
}

// Returns all subtopics in a flat array.
function getFlattenedSubtopics(allTopicStats: TopicStats[]): TopicStats[] {
  const allSubtopics = [];
  for (const t of allTopicStats) {
    if (t.subtopicStats) {
      for (const st of t.subtopicStats) {
        allSubtopics.push(st);
      }
    }
  }
  return allSubtopics;
}
