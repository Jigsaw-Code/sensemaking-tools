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

import { groupCommentsBySubtopic } from "../sensemaker_utils";
import { Comment, CommentWithVoteInfo, isCommentWithVoteInfoType } from "../types";
import { getCommentVoteCount, getTotalPassRate } from "./stats_util";

function get75thPercentile(arr: number[]): number {
  const sortedArr = [...arr].sort((a, b) => a - b);
  const index = (sortedArr.length - 1) * 0.75;

  if (Math.floor(index) === index) {
    return sortedArr[index];
  }

  const lowerIndex = Math.floor(index);
  const upperIndex = lowerIndex + 1;
  return (sortedArr[lowerIndex] + sortedArr[upperIndex]) / 2;
}

// Base class for statistical basis for summaries

/**
 * This class is the input interface for the RecursiveSummary abstraction, and
 * therefore the vessel through which all data is ultimately communicated to
 * the individual summarization routines.
 */
export abstract class SummaryStats {
  comments: Comment[];
  // Comments with at least minVoteCount votes.
  filteredComments: CommentWithVoteInfo[];
  minCommonGroundProb = 0.6;
  minAgreeProbDifference = 0.3;
  // Must be above this threshold to be considered an uncertain comment. This can be overriden in
  // the constructor if the particular conversation has relatively high passes.
  minUncertaintyProb: number = 0.2;
  asProbabilityEstimate = false;

  maxSampleSize = 12;
  public minVoteCount = 20;
  // Whether group data is used as part of the summary.
  groupBasedSummarization: boolean = true;

  constructor(comments: Comment[]) {
    this.comments = comments;
    this.filteredComments = comments.filter(isCommentWithVoteInfoType).filter((comment) => {
      return getCommentVoteCount(comment, true) >= this.minVoteCount;
    });
    const topQuartilePassRate = get75thPercentile(
      this.filteredComments.map((comment) =>
        getTotalPassRate(comment.voteInfo, this.asProbabilityEstimate)
      )
    );
    // Uncertain comments must have at least a certain minimum pass rate.
    this.minUncertaintyProb = Math.max(topQuartilePassRate, this.minUncertaintyProb);
  }

  /**
   * A static factory method that creates a new instance of SummaryStats
   * or a subclass. This is meant to be overriden by subclasses.
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  static create(comments: Comment[]): SummaryStats {
    throw new Error("Cannot instantiate abstract class SummaryStats");
  }

  /**
   * Get the top common ground comments that everyone either agrees on or disagrees on.
   * @param k the number of comments to return
   */
  abstract getCommonGroundComments(k?: number): Comment[];

  /** Returns a score indicating how well a comment represents the common ground. */
  abstract getCommonGroundScore(comment: Comment): number;

  /**
   * Get the top common ground comments that everyone agrees on.
   * @param k the number of comments to return
   */
  abstract getCommonGroundAgreeComments(k?: number): Comment[];

  /**
   * Returns an error message explaining why no common ground comments were found. The
   * requirements for inclusion and thresholds are typically mentioned.
   */
  abstract getCommonGroundNoCommentsMessage(): string;

  /** Get the top common ground comments that everyone disagrees on.
   * @param k the number of comments to return
   */
  abstract getCommonGroundDisagreeComments(k?: number): Comment[];

  /**
   * Based on how the implementing class defines it, get the top disagreed on comments.
   * @param k the number of comments to return.
   */
  abstract getDifferenceOfOpinionComments(k?: number): Comment[];

  /** Returns a score indicating how well a comment represents a difference of opinions. */
  abstract getDifferenceOfOpinionScore(comment: Comment): number;

  /**
   * Gets the topK uncertain comments.
   * @param k the number of comments to get
   */
  abstract getUncertainComments(k?: number): Comment[];

  /** Returns a score indicating how well a comment represents an uncertain viewpoint */
  abstract getUncertainScore(comment: Comment): number;

  /**
   * Returns an error message explaining why no differences of opinion comments were found. The
   * requirements for inclusion and thresholds are typically mentioned.
   */
  abstract getDifferencesOfOpinionNoCommentsMessage(): string;

  // The total number of votes across the entire set of input comments
  get voteCount(): number {
    return this.comments.reduce((sum: number, comment: Comment) => {
      return sum + getCommentVoteCount(comment, true);
    }, 0);
  }

  // The total number of comments in the set of input comments
  get commentCount(): number {
    return this.comments.length;
  }

  get containsSubtopics(): boolean {
    for (const comment of this.comments) {
      if (comment.topics) {
        for (const topic of comment.topics) {
          // Check if the topic matches the 'NestedTopic' type
          if ("subtopics" in topic && Array.isArray(topic.subtopics)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Returns the top k comments according to the given metric.
   */
  topK(
    sortBy: (comment: Comment) => number,
    k: number = this.maxSampleSize,
    filterFn: (comment: Comment) => boolean = () => true
  ): Comment[] {
    return this.comments
      .filter(filterFn)
      .sort((a, b) => sortBy(b) - sortBy(a))
      .slice(0, k);
  }

  /**
   * Sorts topics and their subtopics based on comment count, with
   * "Other" topics and subtopics going last in sortByDescendingCount order.
   * @param topicStats what to sort
   * @param sortByDescendingCount whether to sort by comment count sortByDescendingCount or ascending
   * @returns the topics and subtopics sorted by comment count
   */
  private sortTopicStats(
    topicStats: TopicStats[],
    sortByDescendingCount: boolean = true
  ): TopicStats[] {
    topicStats.sort((a, b) => {
      if (a.name === "Other") return sortByDescendingCount ? 1 : -1;
      if (b.name === "Other") return sortByDescendingCount ? -1 : 1;
      return sortByDescendingCount
        ? b.commentCount - a.commentCount
        : a.commentCount - b.commentCount;
    });

    topicStats.forEach((topic) => {
      if (topic.subtopicStats) {
        topic.subtopicStats.sort((a, b) => {
          if (a.name === "Other") return sortByDescendingCount ? 1 : -1;
          if (b.name === "Other") return sortByDescendingCount ? -1 : 1;
          return sortByDescendingCount
            ? b.commentCount - a.commentCount
            : a.commentCount - b.commentCount;
        });
      }
    });

    return topicStats;
  }

  /**
   * Gets a sorted list of stats for each topic and subtopic.
   *
   * @returns A list of TopicStats objects sorted by comment count with "Other" topics last.
   */
  getStatsByTopic(): TopicStats[] {
    const commentsByTopic = groupCommentsBySubtopic(this.comments);
    const topicStats: TopicStats[] = [];

    for (const topicName in commentsByTopic) {
      const subtopics = commentsByTopic[topicName];
      const subtopicStats: TopicStats[] = [];
      const topicComments = new Set<Comment>();

      for (const subtopicName in subtopics) {
        // get corresonding comments, and update counts
        const comments = new Set<Comment>(Object.values(subtopics[subtopicName]));
        const commentCount = comments.size;
        // aggregate comment objects
        comments.forEach((comment) => topicComments.add(comment));
        subtopicStats.push({
          name: subtopicName,
          commentCount,
          summaryStats: (this.constructor as typeof SummaryStats).create([...comments]),
        });
      }

      topicStats.push({
        name: topicName,
        commentCount: topicComments.size,
        subtopicStats: subtopicStats,
        summaryStats: (this.constructor as typeof SummaryStats).create([...topicComments]),
      });
    }

    return this.sortTopicStats(topicStats);
  }
}

/**
 * Represents statistics about a topic and its subtopics.
 */
export interface TopicStats {
  name: string;
  commentCount: number;
  subtopicStats?: TopicStats[];
  // The stats for the subset of comments.
  summaryStats: SummaryStats;
}
