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

import { decimalToPercent } from "../sensemaker_utils";
import { Comment, CommentWithVoteInfo } from "../types";
import { getTotalAgreeRate, getTotalDisagreeRate, getTotalPassRate } from "./stats_util";
import { SummaryStats } from "./summary_stats";

// Stats basis for the summary that is based on majority vote algorithms. Does not use groups.
export class MajoritySummaryStats extends SummaryStats {
  // Must be above this threshold to be considered high agreement.
  minCommonGroundProb = 0.7;
  // Agreement and Disagreement must be between these values to be difference of opinion.
  minDifferenceProb = 0.4;
  maxDifferenceProb = 0.6;

  // Whether to include pass votes in agree and disagree rate calculations.
  includePasses = false;

  groupBasedSummarization = false;
  // This outlier protection isn't needed since we already filter our comments without many votes.
  asProbabilityEstimate = false;

  // Buffer between uncertainty comments and high/low alignment comments.
  uncertaintyBuffer = 0.05;

  /**
   * An override of the SummaryStats static factory method,
   * to allow for MajoritySummaryStats specific initialization.
   */
  static override create(comments: Comment[]): MajoritySummaryStats {
    return new MajoritySummaryStats(comments);
  }

  /**
   * Returns the top k comments according to the given metric.
   */
  override topK(
    sortBy: (comment: CommentWithVoteInfo) => number,
    k: number = this.maxSampleSize,
    filterFn: (comment: CommentWithVoteInfo) => boolean = () => true
  ): Comment[] {
    return this.filteredComments
      .filter(filterFn)
      .sort((a, b) => sortBy(b) - sortBy(a))
      .slice(0, k);
  }

  /** Returns a score indicating how well a comment represents when everyone agrees. */
  getCommonGroundAgreeScore(comment: CommentWithVoteInfo): number {
    return getTotalAgreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate);
  }

  /** Returns a score indicating how well a comment represents the common ground. */
  getCommonGroundScore(comment: CommentWithVoteInfo): number {
    return Math.max(
      this.getCommonGroundAgreeScore(comment),
      getTotalDisagreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate)
    );
  }

  meetsCommonGroundAgreeThreshold(comment: CommentWithVoteInfo): boolean {
    return (
      getTotalAgreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) >=
        this.minCommonGroundProb &&
      getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate) <=
        this.minUncertaintyProb - this.uncertaintyBuffer
    );
  }

  /**
   * Gets the topK agreed upon comments based on highest % of agree votes.
   *
   * @param k the number of comments to get
   * @returns the top agreed on comments
   */
  getCommonGroundAgreeComments(k: number = this.maxSampleSize) {
    return this.topK(
      (comment) => this.getCommonGroundAgreeScore(comment),
      k,
      // Before getting the top agreed comments, enforce a minimum level of agreement
      (comment: CommentWithVoteInfo) => this.meetsCommonGroundAgreeThreshold(comment)
    );
  }

  /**
   * Gets the topK common ground comments where either everyone agrees or everyone disagrees.
   *
   * @param k the number of comments to get
   * @returns the top common ground comments
   */
  getCommonGroundComments(k: number = this.maxSampleSize) {
    return this.topK(
      (comment) => this.getCommonGroundScore(comment),
      k,
      // Before getting the top agreed comments, enforce a minimum level of agreement
      (comment: CommentWithVoteInfo) =>
        this.meetsCommonGroundAgreeThreshold(comment) ||
        this.meetsCommonGroundDisagreeThreshold(comment)
    );
  }

  getCommonGroundNoCommentsMessage(): string {
    return (
      `No statements met the thresholds necessary to be considered as a point of common ` +
      `ground (at least ${this.minVoteCount} votes, and at least ` +
      `${decimalToPercent(this.minCommonGroundProb)} agreement).`
    );
  }

  /** Returns a score indicating how well a comment represents an uncertain viewpoint based on pass
   *  votes */
  getUncertainScore(comment: CommentWithVoteInfo): number {
    return getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate);
  }

  /**
   * Gets the topK uncertain comments based on pass votes.
   *
   * @param k the number of comments to get
   * @returns the top uncertain comments
   */
  getUncertainComments(k: number = this.maxSampleSize) {
    return this.topK(
      (comment) => this.getUncertainScore(comment),
      k,
      // Before getting the top comments, enforce a minimum level of uncertainty
      (comment: CommentWithVoteInfo) =>
        getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate) >= this.minUncertaintyProb
    );
  }

  meetsCommonGroundDisagreeThreshold(comment: CommentWithVoteInfo): boolean {
    return (
      getTotalDisagreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) >=
        this.minCommonGroundProb &&
      getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate) <=
        this.minUncertaintyProb - this.uncertaintyBuffer
    );
  }

  /**
   * Gets the topK disagreed upon comments across.
   *
   * @param k dfaults to this.maxSampleSize
   * @returns the top disagreed on comments
   */
  getCommonGroundDisagreeComments(k: number = this.maxSampleSize) {
    return this.topK(
      (comment) =>
        getTotalDisagreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate),
      k,
      // Before using Group Informed Consensus a minimum bar of agreement between groups is enforced
      (comment: CommentWithVoteInfo) => this.meetsCommonGroundDisagreeThreshold(comment)
    );
  }

  /** Returns a score indicating how well a comment represents a difference of opinions. This
   * score prioritizes comments where the agreement rate and disagreement rate are
   * both high, and the pass rate is low.*/
  getDifferenceOfOpinionScore(comment: CommentWithVoteInfo): number {
    return (
      1 -
      Math.abs(
        getTotalAgreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) -
          getTotalDisagreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate)
      ) -
      getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate)
    );
  }

  /**
   * Gets the topK agreed upon comments based on highest % of agree votes.
   *
   * @param k the number of comments to get
   * @returns the top differences of opinion comments
   */
  getDifferenceOfOpinionComments(k: number = this.maxSampleSize) {
    return this.topK(
      // Rank comments with the same agree and disagree rates the most highly and prefer when these
      // values are higher. So the best score would be when both the agree rate and the disagree
      // rate are 0.5.
      (comment) => this.getDifferenceOfOpinionScore(comment),
      k,
      // Before getting the top differences comments, enforce a minimum level of difference of
      // opinion.
      (comment: CommentWithVoteInfo) =>
        getTotalAgreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) >=
          this.minDifferenceProb &&
        getTotalAgreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) <=
          this.maxDifferenceProb &&
        getTotalDisagreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) >=
          this.minDifferenceProb &&
        getTotalDisagreeRate(comment.voteInfo, this.includePasses, this.asProbabilityEstimate) <=
          this.maxDifferenceProb &&
        getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate) <=
          this.minUncertaintyProb - this.uncertaintyBuffer
    );
  }

  getDifferencesOfOpinionNoCommentsMessage(): string {
    const minThreshold = decimalToPercent(this.minDifferenceProb);
    const maxThreshold = decimalToPercent(this.maxDifferenceProb);
    return (
      `No statements met the thresholds necessary to be considered as a significant ` +
      `difference of opinion (at least ${this.minVoteCount} votes, and both an agreement rate ` +
      `and disagree rate between ${minThreshold}% and ${maxThreshold}%).`
    );
  }
}
