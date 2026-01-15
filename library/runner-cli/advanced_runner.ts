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

// Advanced data output mode.
//
// There are 3 outputs:
// - a topic stats JSON that includes the topics found, their size, their votes, their engagement
//        and alignment, and their subtopics
// - a comments JSON that includes the comment id, text, votes, pass rate, agree rate, disagree
//        rate,  whether the comment is high alignment, low alignment, uncertain, and whether the
//        comment is filtered out.
// - the summary object as a JSON which includes the section titles, text, and cited comment ids.
//
// The input CSV is expected to have the following columns: comment-id, comment_text, and topics.
// Vote data should also be included, for data without group information the columns should be:
// agrees, disagrees, and optionally passes. For data with group information the columns should be:
// {group name}-agree-count, {group name}-disagree-count, and optionally {group name}-pass-count
// for each group.
//
// Sample Usage:
// time npx ts-node ./runner-cli/advanced_runner.ts --outputBasename final-copy \
//   --vertexProject "{CLOUD_PROJECT_ID}" \
//   --inputFile "./data.csv"

import { Command } from "commander";
import { writeFileSync } from "fs";
import { concatTopics, getCommentsFromCsv, getSummary } from "./runner_utils";
import { MajoritySummaryStats } from "../src/stats/majority_vote";
import { TopicStats } from "../src/stats/summary_stats";
import { RelativeContext } from "../src/tasks/summarization_subtasks/relative_context";
import { Comment, CommentWithVoteInfo, VoteInfo } from "../src/types";
import { getTotalAgreeRate, getTotalDisagreeRate, getTotalPassRate } from "../src/stats/stats_util";
import { SupportedLanguage } from "../templates/l10n/languages";

interface MinimalTopicStat {
  name: string;
  commentCount: number;
  voteCount: number;
  subtopicStats?: MinimalTopicStat[];
  relativeEngagement: string;
  relativeAlignment: string;
}

interface CommentWithScores {
  id: string;
  text: string;
  votes?: VoteInfo;
  topics?: string;

  agreeRate?: number;
  disagreeRate?: number;
  passRate?: number;

  isHighAlignment?: boolean;
  highAlignmentScore?: number;

  isLowAlignment?: boolean;
  lowAlignmentScore?: number;

  isHighUncertainty?: boolean;
  highUncertaintyScore?: number;

  isFilteredOut?: boolean;
}

function createMinimalStats(
  stats: TopicStats[],
  output_lang: SupportedLanguage = "en",
  relativeContext: RelativeContext | null = null
): MinimalTopicStat[] {
  if (!relativeContext) relativeContext = new RelativeContext(stats, output_lang);
  return stats.map((stat): MinimalTopicStat => {
    const minimalStat: MinimalTopicStat = {
      name: stat.name,
      commentCount: stat.commentCount,
      voteCount: stat.summaryStats.voteCount,
      relativeAlignment: relativeContext.getRelativeAgreement(stat.summaryStats),
      relativeEngagement: relativeContext.getRelativeEngagement(stat.summaryStats),
      // Recursively process subtopics if they exist
      subtopicStats: stat.subtopicStats
        ? createMinimalStats(stat.subtopicStats, output_lang, relativeContext)
        : undefined,
    };
    return minimalStat;
  });
}

function getCommentsWithScores(
  comments: Comment[],
  stats: MajoritySummaryStats
): CommentWithScores[] {
  const highAlignmentCommentIDs = stats
    .getCommonGroundComments(Number.MAX_VALUE)
    .map((comment) => comment.id);
  const lowAlignmentCommentIDs = stats
    .getDifferenceOfOpinionComments(Number.MAX_VALUE)
    .map((comment) => comment.id);
  const highUncertaintyCommentIDs = stats
    .getUncertainComments(Number.MAX_VALUE)
    .map((comment) => comment.id);
  const filteredCommentIds = stats.filteredComments.map((comment) => comment.id);
  return comments.map((comment) => {
    const commentWithScores: CommentWithScores = {
      id: comment.id,
      text: comment.text,
      votes: comment.voteInfo,
      topics: concatTopics(comment),
    };

    if (comment.voteInfo) {
      const commentWithVoteInfo = comment as CommentWithVoteInfo;
      commentWithScores.passRate = getTotalPassRate(comment.voteInfo, stats.asProbabilityEstimate);
      commentWithScores.agreeRate = getTotalAgreeRate(
        comment.voteInfo,
        stats.includePasses,
        stats.asProbabilityEstimate
      );
      commentWithScores.disagreeRate = getTotalDisagreeRate(
        comment.voteInfo,
        stats.includePasses,
        stats.asProbabilityEstimate
      );
      commentWithScores.isHighAlignment = highAlignmentCommentIDs.includes(comment.id);
      commentWithScores.highAlignmentScore = stats.getCommonGroundScore(commentWithVoteInfo);

      commentWithScores.isLowAlignment = lowAlignmentCommentIDs.includes(comment.id);
      commentWithScores.lowAlignmentScore = stats.getDifferenceOfOpinionScore(commentWithVoteInfo);

      commentWithScores.isHighUncertainty = highUncertaintyCommentIDs.includes(comment.id);
      commentWithScores.highUncertaintyScore = stats.getUncertainScore(commentWithVoteInfo);

      commentWithScores.isFilteredOut = !filteredCommentIds.includes(comment.id);
    }
    return commentWithScores;
  });
}

async function main(): Promise<void> {
  // Parse command line arguments.
  const program = new Command();
  program
    .option(
      "-o, --outputBasename <file>",
      "The output basename, this will be prepended to 'summary.html' and 'summaryClaimsAndComments.csv'."
    )
    .option("-i, --inputFile <file>", "The input file name.")
    .option(
      "-a, --additionalContext <context>",
      "A short description of the conversation to add context."
    )
    .option("-v, --vertexProject <project>", "The Vertex Project name.")
    .option("-l, --outputLang <language>", "The output language (en, zh-TW, zh-CN, fr, es, ja).", "en");
  program.parse(process.argv);
  const options = program.opts();

  const comments = await getCommentsFromCsv(options.inputFile);
  const stats = new MajoritySummaryStats(comments);
  if (stats.getStatsByTopic().length === 0) {
    throw Error(
      "Expected input comments to have topics. Please categorize them using the " +
        "categorization_runner.ts"
    );
  }

  // Modify the SummaryStats output to drop comment info and add RelativeContext.
  const minimalTopicStats = createMinimalStats(stats.getStatsByTopic(), options.outputLang);
  writeFileSync(
    options.outputBasename + "-topic-stats.json",
    JSON.stringify(minimalTopicStats, null, 2)
  );

  const commentsWithScores = getCommentsWithScores(comments, stats);
  writeFileSync(
    options.outputBasename + "-comments-with-scores.json",
    JSON.stringify(commentsWithScores, null, 2)
  );

  const summary = await getSummary(
    options.vertexProject,
    comments,
    undefined,
    options.additionalContext,
    options.outputLang
  );
  writeFileSync(options.outputBasename + "-summary.json", JSON.stringify(summary, null, 2));
}

main();
