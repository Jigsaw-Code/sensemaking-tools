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
import { getMaxGroupAgreeProbDifference, getMinAgreeProb } from "../../stats/stats_util";
import {
  getPrompt,
  getAbstractPrompt,
  commentTableMarkdown,
  ColumnDefinition,
  executeConcurrently,
} from "../../sensemaker_utils";
import { Comment, SummaryContent, isCommentType } from "../../types";
import { Model } from "../../models/model";
import { SummaryStats, TopicStats } from "../../stats/summary_stats";
import { RelativeContext } from "./relative_context";
// Import localization system
import { 
  type SupportedLanguage,
  getReportSectionTitle, 
  getReportContent, 
  getSubsectionTitle,
  getTopicSummaryText,
  getPluralForm,
  localizeTopicName,
  getStatisticsMessage
} from "../../../templates/l10n";
// Import multi-language prompts
import { 
  getThemesPrompt,
  getDifferencesOfOpinionInstructions,
  getDifferencesOfOpinionSingleCommentInstructions,
  getCommonGroundInstructions,
  getCommonGroundSingleCommentInstructions,
  getRecursiveTopicSummaryInstructions
} from "../../../templates/l10n/prompts";

// Note: These constants are now replaced by multi-language versions from prompts.ts
// const COMMON_INSTRUCTIONS = ...
// const GROUP_SPECIFIC_INSTRUCTIONS = ...

// Note: These functions are now replaced by multi-language versions from prompts.ts
// function getCommonGroundInstructions(containsGroups: boolean): string { ... }
// function getCommonGroundSingleCommentInstructions(containsGroups: boolean): string { ... }

// TODO: Test whether conditionally including group specific text in this prompt improves
// performance.
// Note: This constant is now replaced by getDifferencesOfOpinionInstructions() function
// const DIFFERENCES_OF_OPINION_INSTRUCTIONS =
//   `You are going to be presented with several comments from a discussion on which there were differing opinions, ` +
//   `as well as a summary of points of common ground from this discussion. Your job is summarize the ideas ` +
//   `contained in the comments, keeping in mind the points of common ground as backgrounnd in describing ` +
//   `the differences of opinion. Do not pretend that you hold any of these opinions. You are not a ` +
//   `participant in this discussion. Write a concise summary of these comments that is at least ` +
//   `one sentence and at most five sentences long. Refer to the people who made these comments as ` +
//   `participants, not commenters.  Do not talk about how strongly they disagree with these ` +
//   `comments. Use complete sentences. ${COMMON_INSTRUCTIONS}

// Do not assume that these comments were written by different participants. These comments could be from ` +
//   `the same participant, so do not say some participants prosed one things while other ` +
//   `participants proposed another.  Do not say "Some participants proposed X while others Y".  ` +
//   `Instead say "One statement proposed X while another Y"

// Where the difference of opinion comments refer to topics that are also covered in the common ground ` +
//   `summary, your output should begin in some variant of the form "While there was broad support for ..., ` +
//   `opinions differed with respect to ...". When this is not the case, you can beging simple as ` +
//   `"There was disagreement ..." or something similar to contextualize that the comments you are ` +
//   `summarizing had mixed support.`;



// Note: This function is now replaced by getRecursiveTopicSummaryInstructions() function from prompts.ts
// function getRecursiveTopicSummaryInstructions(topicStat: TopicStats): string { ... }

/**
 * This RecursiveSummary subclass constructs a top level "Topics" summary section,
 * calling out to the separate TopicSummary and SubtopicSummary classes to generate
 * content for individual subsections corresponding to specific topics and subtopics.
 */
export class AllTopicsSummary extends RecursiveSummary<SummaryStats> {
  async getSummary(): Promise<SummaryContent> {
    // Debug: 檢查 output_lang 值
    console.log(`[DEBUG] AllTopicsSummary.output_lang: ${this.output_lang}`);
    
    // First construct the introductory description for the entire section
    const topicStats: TopicStats[] = this.input.getStatsByTopic();
    const nTopics: number = topicStats.length;
    const nSubtopics: number = topicStats
      .map((t) => t.subtopicStats?.length || 0)
      .reduce((n, m) => n + m, 0);
    const hasSubtopics: boolean = nSubtopics > 0;
    const subtopicsCountText: string = hasSubtopics ? getReportContent("subtopics", "text", this.output_lang, { count: nSubtopics }) : "";
    
    // Debug: 檢查 getReportContent 的調用參數
    if (hasSubtopics) {
      console.log(`[DEBUG] AllTopicsSummary.getSummary() calling getReportContent with: section="subtopics", content="text", output_lang="${this.output_lang}", count=${nSubtopics}`);
      console.log(`[DEBUG] AllTopicsSummary.getSummary() subtopicsCountText result: "${subtopicsCountText}"`);
    }
    
    const usesGroups = topicStats.some((t) => t.summaryStats.groupBasedSummarization);
    
    // Get localized title and overview text from localization system
    const title = getReportSectionTitle("topics", this.output_lang);
    const overviewText = getReportContent("topics", "overview", this.output_lang, {
      topicCount: nTopics,
      subtopicsText: subtopicsCountText,
      groupsText: usesGroups ? " between the opinion groups described above," : "",
      groupsBetweenText: usesGroups ? "between the groups " : ""
    });
    
    // Debug: 檢查本地化函式的調用參數
    console.log(`[DEBUG] AllTopicsSummary.getSummary() calling getReportSectionTitle with: section="topics", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] AllTopicsSummary.getSummary() calling getReportContent with: section="topics", content="overview", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] AllTopicsSummary.getSummary() title result: "${title}"`);
    console.log(`[DEBUG] AllTopicsSummary.getSummary() overviewText result: "${overviewText}"`);

    // Now construct the individual Topic summaries
    const relativeContext = new RelativeContext(topicStats, this.output_lang);
    const topicSummaries: (() => Promise<SummaryContent>)[] = topicStats.map(
      (topicStat) =>
        // Create a callback function for each summary and add it to the list, preparing them for parallel execution.
        () =>
          new TopicSummary(
            topicStat,
            this.model,
            relativeContext,
            this.additionalContext,
            this.output_lang
          ).getSummary()
    );
    return {
      title: title,
      text: overviewText,
      subContents: await executeConcurrently(topicSummaries),
    };
  }
}

/**
 * This RecursiveSummary subclass generates summaries for individual topics.
 */
export class TopicSummary extends RecursiveSummary<SummaryStats> {
  // TopicSummary also needs to know about the topic, like name and subtopics
  topicStat: TopicStats;
  relativeContext: RelativeContext;

  // This override is necessary to pass through a TopicStat object, rather than a SummaryStats object
  constructor(
    topicStat: TopicStats,
    model: Model,
    relativeContext: RelativeContext,
    additionalContext?: string,
    output_lang?: SupportedLanguage
  ) {
    super(topicStat.summaryStats, model, additionalContext, output_lang);
    this.topicStat = topicStat;
    this.relativeContext = relativeContext;
    
    // Debug: 檢查建構函數中的 output_lang 值
    // console.log(`[DEBUG] TopicSummary constructor output_lang: ${this.output_lang}`);
  }

  async getSummary(): Promise<SummaryContent> {
    // Debug: 檢查 getSummary 中的 output_lang 值
    // console.log(`[DEBUG] TopicSummary.getSummary() output_lang: ${this.output_lang}`);
    
    const nSubtopics: number = this.topicStat.subtopicStats?.length || 0;
    if (nSubtopics == 0) {
      return this.getCommentSummary();
    } else {
      return this.getAllSubTopicSummaries();
    }
  }

  /**
   * Returns the section title for this topics summary section of the final report
   */
  getSectionTitle(): string {
    // Debug: 檢查 localizeTopicName 的調用參數
    console.log(`[DEBUG] TopicSummary.getSectionTitle() calling localizeTopicName with: topicName="${this.topicStat.name}", output_lang="${this.output_lang}"`);
    
    return `### ${localizeTopicName(this.topicStat.name, this.output_lang)} (${this.topicStat.commentCount} ${getStatisticsMessage("statements", this.output_lang, {})})`;
  }

  /**
   * When subtopics are present, compiles the individual summaries for those subtopics
   * @returns a promise of the summary string
   */
  async getAllSubTopicSummaries(): Promise<SummaryContent> {
    // Debug: 檢查 getAllSubTopicSummaries 中的 output_lang 值
    console.log(`[DEBUG] TopicSummary.getAllSubTopicSummaries() output_lang: ${this.output_lang}`);
    
    // Create subtopic summaries for all subtopics with > 1 statement.
    const subtopicSummaries: (() => Promise<SummaryContent>)[] = (
      this.topicStat.subtopicStats || []
    )
      .filter((subtopicStat) => subtopicStat.commentCount > 1)
      .map(
        // Create a callback function for each summary and add it to the list, preparing them for parallel execution.
        (subtopicStat) => () =>
          new SubtopicSummary(
            subtopicStat,
            this.model,
            this.relativeContext,
            this.additionalContext,
            this.output_lang
          ).getSummary()
      );

    const subtopicSummaryContents = await executeConcurrently(subtopicSummaries);

    const nSubtopics: number = subtopicSummaries.length;
    let topicSummary = "";
    if (nSubtopics > 0) {
      // Get localized topic summary text from localization system
      topicSummary = getTopicSummaryText("topicSummary", this.output_lang, {
        subtopicCount: nSubtopics,
        subtopicPlural: getPluralForm(nSubtopics, this.output_lang),
        statementCount: this.topicStat.commentCount,
        statementPlural: getPluralForm(this.topicStat.commentCount, this.output_lang)
      });
      
      // Debug: 檢查本地化函式的調用參數
      console.log(`[DEBUG] TopicSummary.getAllSubTopicSummaries() calling getTopicSummaryText with: content="topicSummary", output_lang="${this.output_lang}"`);
      console.log(`[DEBUG] TopicSummary.getAllSubTopicSummaries() calling getPluralForm with: count=${nSubtopics}, output_lang="${this.output_lang}"`);
      console.log(`[DEBUG] TopicSummary.getAllSubTopicSummaries() topicSummary result: "${topicSummary}"`);
      
      const subtopicSummaryPrompt = getAbstractPrompt(
        getRecursiveTopicSummaryInstructions(this.output_lang, this.topicStat.name),
        subtopicSummaryContents,
        (summary: SummaryContent) =>
          `<subtopicSummary>\n` +
          `    <title>${summary.title}</title>\n` +
          `    <text>\n${summary.subContents?.map((s) => s.title + s.text).join("\n\n")}\n` +
          `    </text>\n  </subtopicSummary>`,
        this.additionalContext,
        this.output_lang
      );
      console.log(`Generating TOPIC SUMMARY for: "${this.topicStat.name}"`);
      console.log(`[DEBUG] Calling model.generateText with output_lang: ${this.output_lang}`);
      subtopicSummaryContents.unshift({
        type: "TopicSummary",
        text: await this.model.generateText(subtopicSummaryPrompt, this.output_lang),
      });
    }

    return {
      title: this.getSectionTitle(),
      text: topicSummary,
      subContents: subtopicSummaryContents,
    };
  }

  /**
   * Summarizes the comments associated with the given topic
   * @returns a promise of the summary string
   */
  async getCommentSummary(): Promise<SummaryContent> {
    const relativeAgreement = this.relativeContext.getRelativeAgreement(
      this.topicStat.summaryStats
    );
    
    // Get localized agreement description from localization system
    const agreementDescription = getTopicSummaryText("relativeAgreement", this.output_lang, {
      level: relativeAgreement
    });
    
    // Debug: 檢查本地化函式的調用參數
    console.log(`[DEBUG] TopicSummary.getCommentSummary() calling getTopicSummaryText with: content="relativeAgreement", output_lang="${this.output_lang}", level="${relativeAgreement}"`);
    console.log(`[DEBUG] TopicSummary.getCommentSummary() agreementDescription result: "${agreementDescription}"`);
    
    const subContents = [await this.getThemesSummary()];
    // check env variable to decide whether to compute common ground and difference of opinion summaries
    // 智能環境變量讀取，支持 Node.js 和 Cloudflare Workers
    function getEnvVar(key: string, defaultValue: string): string {
      // 檢查是否在 Node.js 環境中
      if (typeof process !== 'undefined' && process.env && process.versions && process.versions.node) {
        return process.env[key] || defaultValue;
      }
      
      // 檢查是否在 Cloudflare Workers 環境中
      if (typeof globalThis !== 'undefined') {
        return (globalThis as unknown as Record<string, string>)[key] || defaultValue;
      }
      
      return defaultValue;
    }
    
    const skipCommonGround = getEnvVar("SKIP_COMMON_GROUND_AND_DIFFERENCES_OF_OPINION", "false");
    if (skipCommonGround !== "true") {
      const commonGroundSummary = await this.getCommonGroundSummary(this.topicStat.name);
      const differencesOfOpinionSummary = await this.getDifferencesOfOpinionSummary(
        commonGroundSummary,
        this.topicStat.name
      );
      subContents.push(commonGroundSummary, differencesOfOpinionSummary);
    }

    const debugMode = getEnvVar("DEBUG_MODE", "false");
    if (debugMode === "true") {
      // Based on the common ground and differences of opinion comments,
      // TODO: Should also include common ground disagree comments (aka what everyone agrees they
      // don't like)
      const commonGroundComments = this.input.getCommonGroundAgreeComments();
      const differencesComments = this.input.getDifferenceOfOpinionComments();

      // Figure out what comments aren't currently being summarized
      const allSummarizedCommentIds = new Set([
        ...commonGroundComments.map((c) => c.id),
        ...differencesComments.map((c) => c.id),
      ]);
      const otherComments = this.topicStat.summaryStats.comments.filter(
        (comment) => !allSummarizedCommentIds.has(comment.id)
      );

      const otherCommentsTable = commentTableMarkdown(otherComments, [
        { columnName: "minAgreeProb", getValue: getMinAgreeProb } as ColumnDefinition,
        {
          columnName: "maxAgreeDiff",
          getValue: getMaxGroupAgreeProbDifference,
        } as ColumnDefinition,
      ]);

      const otherCommentsSummary = {
        title: getSubsectionTitle("otherStatements", this.output_lang, otherComments.length),
        text: otherCommentsTable,
      };
      subContents.push(otherCommentsSummary);
    }

    return {
      title: this.getSectionTitle(),
      text: agreementDescription,
      subContents: subContents,
    };
  }

  /**
   * Summarizes the themes that recur across all comments
   * @returns a single sentence describing the themes, without citations.
   */
  async getThemesSummary(): Promise<SummaryContent> {
    // Debug: 檢查 getThemesSummary 中的 output_lang 值
    console.log(`[DEBUG] TopicSummary.getThemesSummary() output_lang: ${this.output_lang}`);
    
    const allComments = this.input.comments;
    // TODO: add some edge case handling in case there is only 1 comment, etc
    console.log(`Generating PROMINENT THEMES for subtopic: "${this.topicStat.name}"`);
    console.log(`[DEBUG] Calling model.generateText with output_lang: ${this.output_lang}`);
    const text = await this.model.generateText(
      getPrompt(
        getThemesPrompt(this.output_lang, this.topicStat.name),
        allComments.map((comment: Comment): string => comment.text),
        this.additionalContext,
        this.output_lang
      ),
      this.output_lang
    );
    
    // Get localized themes title from localization system
    const title = getSubsectionTitle("prominentThemes", this.output_lang);
    
    // Debug: 檢查本地化函式的調用參數
    console.log(`[DEBUG] TopicSummary.getThemesSummary() calling getSubsectionTitle with: section="prominentThemes", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] TopicSummary.getThemesSummary() title result: "${title}"`);
    
    return { title, text };
  }

  /**
   * Summarizes the comments on which there was the strongest agreement.
   * @returns a short paragraph describing the similarities, including comment citations.
   */
  async getCommonGroundSummary(topic: string): Promise<SummaryContent> {
    // Debug: 檢查 getCommonGroundSummary 中的 output_lang 值
    console.log(`[DEBUG] TopicSummary.getCommonGroundSummary() output_lang: ${this.output_lang}`);
    
    // TODO: Should also include common ground disagree comments (aka what everyone agrees they
    // don't like)
    const commonGroundComments = this.input.getCommonGroundAgreeComments();
    const nComments = commonGroundComments.length;
    let text = "";
    if (nComments === 0) {
      text = this.input.getCommonGroundNoCommentsMessage();
    } else {
      console.log(`Generating COMMON GROUND for "${topic}"`);
      console.log(`[DEBUG] Calling model.generateText with output_lang: ${this.output_lang}`);
      const summary = this.model.generateText(
        getPrompt(
          nComments === 1
            ? getCommonGroundSingleCommentInstructions(this.output_lang, this.input.groupBasedSummarization)
            : getCommonGroundInstructions(this.output_lang, this.input.groupBasedSummarization),
          commonGroundComments.map((comment: Comment): string => comment.text),
          this.additionalContext,
          this.output_lang
        ),
        this.output_lang
      );
      text = await summary;
    }
    
    // Get localized common ground title from localization system
    const title = this.input.groupBasedSummarization
      ? getSubsectionTitle("commonGroundBetweenGroups", this.output_lang)
      : getSubsectionTitle("commonGround", this.output_lang);
    
    // Debug: 檢查本地化函式的調用參數
    console.log(`[DEBUG] TopicSummary.getCommonGroundSummary() calling getSubsectionTitle with: section="${this.input.groupBasedSummarization ? 'commonGroundBetweenGroups' : 'commonGround'}", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] TopicSummary.getCommonGroundSummary() title result: "${title}"`);
    
    return {
      title,
      text: text,
      citations: commonGroundComments.map((comment) => comment.id),
    };
  }

  /**
   * Summarizes the comments on which there was the strongest disagreement.
   * @returns a short paragraph describing the differences, including comment citations.
   */
  async getDifferencesOfOpinionSummary(
    commonGroundSummary: SummaryContent,
    topic: string
  ): Promise<SummaryContent> {
    // Debug: 檢查 getDifferencesOfOpinionSummary 中的 output_lang 值
    console.log(`[DEBUG] TopicSummary.getDifferencesOfOpinionSummary() output_lang: ${this.output_lang}`);
    
    const topDisagreeCommentsAcrossGroups = this.input.getDifferenceOfOpinionComments();
    const nComments = topDisagreeCommentsAcrossGroups.length;
    let text = "";
    if (nComments === 0) {
      text = this.input.getDifferencesOfOpinionNoCommentsMessage();
    } else {
      const prompt = getAbstractPrompt(
        nComments === 1
          ? getDifferencesOfOpinionSingleCommentInstructions(this.output_lang, this.input.groupBasedSummarization)
          : getDifferencesOfOpinionInstructions(this.output_lang),
        [commonGroundSummary].concat(topDisagreeCommentsAcrossGroups),
        formatDifferenceOfOpinionData,
        this.additionalContext,
        this.output_lang
      );
      console.log(`Generating DIFFERENCES OF OPINION for "${topic}"`);
      console.log(`[DEBUG] Calling model.generateText with output_lang: ${this.output_lang}`);
      const summary = this.model.generateText(prompt, this.output_lang);
      text = await summary;
    }
    
    // Get localized differences of opinion title from localization system
    const title = getSubsectionTitle("differencesOfOpinion", this.output_lang);
    
    // Debug: 檢查本地化函式的調用參數
    console.log(`[DEBUG] TopicSummary.getDifferencesOfOpinionSummary() calling getSubsectionTitle with: section="differencesOfOpinion", output_lang="${this.output_lang}"`);
    console.log(`[DEBUG] TopicSummary.getDifferencesOfOpinionSummary() title result: "${title}"`);
    
    const resp = {
      title,
      text: text,
      citations: topDisagreeCommentsAcrossGroups.map((comment) => comment.id),
    };

    // Since common ground is part of the summary, include its citations for evaluation
    if (commonGroundSummary.citations) {
      resp.citations = resp.citations.concat(commonGroundSummary.citations);
    }
    return resp;
  }
}

/**
 * This TopicSummary subclass contains overrides for subtopics. At present, this is just an
 * override for the section title, but may evolve to different on other functionality.
 */
export class SubtopicSummary extends TopicSummary {
  override getSectionTitle(): string {
    // Debug: 檢查 SubtopicSummary 中的 output_lang 值
    console.log(`[DEBUG] SubtopicSummary.getSectionTitle() output_lang: ${this.output_lang}`);
    
    return `#### ${this.topicStat.name} (${this.topicStat.commentCount} ${getStatisticsMessage("statements", this.output_lang, {})})`;
  }
}

function formatDifferenceOfOpinionData(datum: SummaryContent | Comment) {
  // Warning: `Comment` and `SummaryContent` types are very similar, and comments actually pass
  // the `isSummaryContent` typecheck function. We are checking for isCommentType
  // first because comments _must_ have `id` fields, so the code below works.
  // However, if for some reason `SummaryContent` ended up getting an `id` field, this would no
  // longer work. There does not seem to be a simple way around this though because of the
  // differences between types and interfaces in typescript.
  // TODO: Add some testing of this in case there's ever a regression, or write with a more
  // custom prompt construction function.
  if (isCommentType(datum)) {
    return `<comment>${datum.text}</comment>`;
  } else {
    return (
      `<commonGroundSummary>\n` +
      `    <text>\n${datum.text}` +
      `    </text>\n  </commonGroundSummary>`
    );
  }
}
