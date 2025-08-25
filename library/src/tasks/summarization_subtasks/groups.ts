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
import { GroupedSummaryStats, GroupStats } from "../../stats/group_informed";
import { SummaryContent, Comment } from "../../types";
import { getPrompt } from "../../sensemaker_utils";

// Import localization system
import { 
  getReportSectionTitle, 
  getReportContent 
} from "../../../templates/l10n";

/**
 * Format a list of strings to be a human readable list ending with "and"
 * @param items the strings to concatenate
 * @returns a string with the format "<item>, <item>, and <item>"
 */
function formatStringList(items: string[]): string {
  if (items.length === 0) {
    return "";
  }

  if (items.length === 1) {
    return items[0];
  }

  if (items.length === 2) {
    return `${items[0]} and ${items[1]}`;
  }

  const lastItem = items.pop(); // Remove the last item
  return `${items.join(", ")} and ${lastItem}`;
}

export class GroupsSummary extends RecursiveSummary<GroupedSummaryStats> {
  async getSummary(): Promise<SummaryContent> {
    const groupStats = this.input.getStatsByGroup();
    const groupCount = groupStats.length;
    const groupNamesWithQuotes = groupStats.map((stat: GroupStats) => { return `"${stat.name}"`; });
    const groupNames = groupStats.map((stat: GroupStats) => { return stat.name; });
    
    // Get localized title and text from localization system
    const title = getReportSectionTitle("opinionGroups", this.output_lang);
    const text = getReportContent("opinionGroups", "text", this.output_lang, {
      groupCount,
      groupNames: formatStringList(groupNamesWithQuotes)
    });
    
    const content: SummaryContent = { title: title, text: text, subContents: await this.getGroupDescriptions(groupNames), };
    return content;
  }

  async getGroupDescriptions(groupNames: string[]): Promise<SummaryContent[]> {
    const groupDescriptions: SummaryContent[] = [];
    for (const groupName of groupNames) {
      const groupStats = this.input.getStatsByGroup().find((stat: GroupStats) => stat.name === groupName);
      if (groupStats) {
        const groupDescription = await this.getGroupDescription(groupStats);
        groupDescriptions.push(groupDescription);
      }
    }
    return groupDescriptions;
  }

  async getGroupDescription(groupStats: GroupStats): Promise<SummaryContent> {
    // Get representative comments for this group
    const groupComments = this.input.getGroupRepresentativeComments(groupStats.name);
    const prompt = getPrompt(
      `Please write a concise summary of the key viewpoints and perspectives of the group "${groupStats.name}". This summary should be based on the statements submitted by members of this group and should reflect their common viewpoints and concerns. The summary should be at least one sentence and at most three sentences long. Do not pretend that you hold any of these opinions. You are not a participant in this discussion.`,
      groupComments.map((comment: Comment) => comment.text),
      this.additionalContext,
      this.output_lang
    );
    const groupDescription = await this.model.generateText(prompt, this.output_lang);
    return { title: `### ${groupStats.name}`, text: groupDescription };
  }
}
