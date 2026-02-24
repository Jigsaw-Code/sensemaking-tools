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

import { getStandardDeviation } from "../../stats/stats_util";
import { SummaryStats, TopicStats } from "../../stats/summary_stats";

/**
 * Holds information for the relative agreement and engagement across all pieces of the summary.
 */
export class RelativeContext {
  averageHighAgreeRate: number;
  highAgreeStdDeviation: number;

  maxCommentCount: number;
  maxVoteCount: number;
  engagementStdDeviation: number;
  averageEngagement: number;

  constructor(topicStats: TopicStats[]) {
    const subtopicStats = topicStats.flatMap((t) => t.subtopicStats || []);
    const highAgreementRatePerSubtopic = subtopicStats.map((subtopicStats) =>
      this.getHighAgreementRate(subtopicStats.summaryStats)
    );
    this.averageHighAgreeRate =
      highAgreementRatePerSubtopic.reduce((sum, num) => sum + num, 0) /
      highAgreementRatePerSubtopic.length;
    this.highAgreeStdDeviation = getStandardDeviation(highAgreementRatePerSubtopic);

    this.maxCommentCount = subtopicStats
      .map((subtopicStats) => subtopicStats.summaryStats.commentCount)
      .reduce((a, b) => Math.max(a, b), 0);
    this.maxVoteCount = subtopicStats
      .map((subtopicStats) => subtopicStats.summaryStats.voteCount)
      .reduce((a, b) => Math.max(a, b), 0);
    const engagementBySubtopic = subtopicStats.map((subtopicStats) =>
      this.getEngagementNumber(subtopicStats.summaryStats)
    );
    this.engagementStdDeviation = getStandardDeviation(engagementBySubtopic);
    this.averageEngagement =
      engagementBySubtopic.reduce((sum, num) => sum + num, 0) / engagementBySubtopic.length;
  }

  /**
   * Get the rate of all comments being considered high agreement (both all agree and all disagree)
   * @param summaryStats the subset of comments to consider
   * @returns the count of all potential high agreement comments.
   */
  private getHighAgreementRate(summaryStats: SummaryStats): number {
    // Allow all the comments to be chosen if they match the requirements.
    const maxLength = summaryStats.comments.length;
    const highAgreeConsensusCount = summaryStats.getCommonGroundComments(maxLength).length;
    const highDisagreeConsensusCount =
      summaryStats.getCommonGroundDisagreeComments(maxLength).length;
    return (highAgreeConsensusCount + highDisagreeConsensusCount) / summaryStats.commentCount;
  }

  getRelativeEngagement(summaryStats: SummaryStats): string {
    const engagmenet = this.getEngagementNumber(summaryStats);
    if (engagmenet < this.averageEngagement - this.engagementStdDeviation) {
      return "low engagement";
    }
    if (engagmenet < this.averageEngagement) {
      return "moderately low engagement";
    }
    if (engagmenet < this.averageEngagement + this.engagementStdDeviation) {
      return "moderately high engagement";
    } else {
      return "high engagement";
    }
  }

  /**
   * Gets an engagement number that weighs votes and comment counts equally.
   *
   * This is done by normalizing the vote count to be in the range 0-1 and the comment count to be
   * in the range 0-1. Then these numbers are added together to get a score from 0-2 with 2 being
   * the max value.
   *
   * @param summaryStats the comments and votes to consider for engagement
   * @returns the engagement number from 0-2 for the comments.
   */
  private getEngagementNumber(summaryStats: SummaryStats): number {
    return (
      summaryStats.commentCount / this.maxCommentCount + summaryStats.voteCount / this.maxVoteCount
    );
  }

  getRelativeAgreement(summaryStats: SummaryStats): string {
    const highAgreementRate = this.getHighAgreementRate(summaryStats);
    if (highAgreementRate < this.averageHighAgreeRate - this.highAgreeStdDeviation) {
      return "low alignment";
    }
    if (highAgreementRate < this.averageHighAgreeRate) {
      return "moderately low alignment";
    }
    if (highAgreementRate < this.averageHighAgreeRate + this.highAgreeStdDeviation) {
      return "moderately high alignment";
    } else {
      return "high alignment";
    }
  }
}
