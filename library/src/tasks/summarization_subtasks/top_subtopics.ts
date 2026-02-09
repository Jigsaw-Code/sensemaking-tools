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

// Import localization system
import { 
  getReportSectionTitle, 
  getReportContent, 
  getSubsectionTitle,
  getTopicName,
  type SupportedLanguage
} from "../../../templates/l10n";

// Import localized prompts
import { getTopSubtopicsThemesPrompt, getTopSubtopicsTitleTemplate } from "../../../templates/l10n/prompts";

export class TopSubtopicsSummary extends RecursiveSummary<SummaryStats> {
  async getSummary(): Promise<SummaryContent> {
    // Debug: 檢查 output_lang 值
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() output_lang: ${this.output_lang}`);
    
    const allSubtopics = getFlattenedSubtopics(this.input.getStatsByTopic());
    const topSubtopics = getTopSubtopics(allSubtopics, 5, this.output_lang);
    
    // Debug: 檢查 getTopSubtopics 的調用參數
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() calling getTopSubtopics with: max=5, output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() allSubtopics count: ${allSubtopics.length}, topSubtopics count: ${topSubtopics.length}`);

    const subtopicSummaryContents: SummaryContent[] = [];
    for (let i = 0; i < topSubtopics.length; ++i) {
      subtopicSummaryContents.push(await this.getSubtopicSummary(topSubtopics[i], i));
    }
    
    // Get localized title and text from localization system
    const title = getReportSectionTitle("topSubtopics", this.output_lang, topSubtopics.length);
    const text = getReportContent("topSubtopics", "text", this.output_lang, {
      totalCount: allSubtopics.length,
      topCount: topSubtopics.length
    });
    
    // Debug: 檢查本地化函式的調用參數和結果
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() calling getReportSectionTitle with: section="topSubtopics", output_lang="${this.output_lang}", count=${topSubtopics.length}`);
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() calling getReportContent with: section="topSubtopics", content="text", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() title result: "${title}"`);
    console.log(`[DEBUG] TopSubtopicsSummary.getSummary() text result: "${text}"`);
    
    return Promise.resolve({
      title: title,
      text: text,
      subContents: subtopicSummaryContents,
    });
  }

  async getSubtopicSummary(st: TopicStats, index: number): Promise<SummaryContent> {
    // Debug: 檢查 getSubtopicSummary 中的 output_lang 值
    console.log(`[DEBUG] TopSubtopicsSummary.getSubtopicSummary() output_lang: ${this.output_lang}`);
    
    const subtopicComments = st.summaryStats.comments;
    console.log(`Generating PROMINENT THEMES for top 5 subtopics: "${st.name}"`);
    console.log(`[DEBUG] Calling model.generateText with output_lang: ${this.output_lang}`);
    
    const text = await this.model.generateText(
      getPrompt(
        getTopSubtopicsThemesPrompt(this.output_lang, st.name),
        subtopicComments.map((comment: Comment): string => comment.text),
        this.additionalContext,
        this.output_lang
      ),
      this.output_lang
    );
    
    // Get localized themes title from localization system
    const themesTitle = getSubsectionTitle("prominentThemes", this.output_lang);
    
    // Debug: 檢查本地化函式的調用參數和結果
    console.log(`[DEBUG] TopSubtopicsSummary.getSubtopicSummary() calling getSubsectionTitle with: section="prominentThemes", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] TopSubtopicsSummary.getSubtopicSummary() themesTitle result: "${themesTitle}"`);
    
    const themesSummary = { title: themesTitle, text: text };
    return Promise.resolve({
      title: getTopSubtopicsTitleTemplate(this.output_lang, index, st.name, st.commentCount),
      text: "",
      subContents: [themesSummary],
    });
  }
}

function getTopSubtopics(allSubtopics: TopicStats[], max = 5, output_lang: SupportedLanguage = "en") {
  // Debug: 檢查 getTopSubtopics 函數接收到的 output_lang 參數
  console.log(`[DEBUG] getTopSubtopics() received output_lang: ${output_lang}`);
  
  // Sort all subtopics by comment count, desc
  allSubtopics.sort((a, b) => b.commentCount - a.commentCount);

  // Get top subtopics, skipping other
  const topSubtopics = [];
  for (const st of allSubtopics) {
    if (st.name == getTopicName("other", output_lang)) {
      // Debug: 檢查 getTopicName 的調用
      console.log(`[DEBUG] getTopSubtopics() calling getTopicName with: topic="other", output_lang="${output_lang}"`);
      console.log(`[DEBUG] getTopSubtopics() getTopicName result: "${getTopicName("other", output_lang)}"`);
      console.log(`[DEBUG] getTopSubtopics() skipping subtopic: "${st.name}"`);
      continue;
    }
    topSubtopics.push(st);
    if (topSubtopics.length >= max) {
      break;
    }
  }
  
  // Debug: 檢查最終結果
  console.log(`[DEBUG] getTopSubtopics() returning ${topSubtopics.length} subtopics`);
  
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
