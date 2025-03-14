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
import { Comment, CommentWithVoteTallies, isCommentWithVoteTalliesType } from "../types";
import { getCommentVoteCount } from "./stats_util";

// Base class for statistical basis for summaries

/**
 * This class is the input interface for the RecursiveSummary abstraction, and
 * therefore the vessel through which all data is ultimately communicated to
 * the individual summarization routines.
 */
export abstract class SummaryStats {
  comments: Comment[];
  // Comments with at least minVoteCount votes.
  filteredComments: CommentWithVoteTallies[];
  minCommonGroundProb = 0.6;
  minAgreeProbDifference = 0.3;
  maxSampleSize = 12;
  public minVoteCount = 20;
  // Whether group data is used as part of the summary.
  groupBasedSummarization: boolean = true;

  constructor(comments: Comment[]) {
    this.comments = comments;
    this.filteredComments = comments.filter(isCommentWithVoteTalliesType).filter((comment) => {
      return getCommentVoteCount(comment) >= this.minVoteCount;
    });
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
   * Based on how the implementing class defines it, get the top agreed on comments.
   * @param k the number of comments to return
   */
  abstract getCommonGroundComments(k?: number): Comment[];

  /**
   * Returns an error message explaining why no common ground comments were found. The
   * requirements for inclusion and thresholds are typically mentioned.
   */
  abstract getCommonGroundNoCommentsMessage(): string;

  /**
   * Based on how the implementing class defines it, get the top disagreed on comments.
   * @param k the number of comments to return.
   */
  abstract getDifferenceOfOpinionComments(k?: number): Comment[];

  /**
   * Returns an error message explaining why no differences of opinion comments were found. The
   * requirements for inclusion and thresholds are typically mentioned.
   */
  abstract getDifferencesOfOpinionNoCommentsMessage(): string;

  // The total number of votes across the entire set of input comments
  get voteCount(): number {
    return this.comments.reduce((sum: number, comment: Comment) => {
      return sum + getCommentVoteCount(comment);
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
   * Allow comments to only be used in one subtopic. They are kept in the smallest subtopic only.
   * @param topicStats the stats by topic to filter repeat comments out of
   * @returns the TopicStats with each comment only appearing once.
   */
  private useCommentOnlyOnce(topicStats: TopicStats[]): TopicStats[] {
    let alreadyUsedComments = new Set<string>();
    // Consider the subtopics with the least comments first so they will keep the comment and all
    // future duplicates will be removed.
    const reversedTopicStats = this.sortTopicStats(topicStats, false);

    const filteredTopicStats: TopicStats[] = [];
    for (const topic of reversedTopicStats) {
      const newTopicStats: TopicStats = {
        name: topic.name,
        commentCount: topic.commentCount,
        summaryStats: topic.summaryStats,
        subtopicStats: [],
      };
      if (topic.subtopicStats) {
        for (const subtopic of topic.subtopicStats) {
          // Ignore repeats in the "Other":"Other" category, these are a special case and should
          // eventually be handled separately.
          if (topic.name === "Other" && subtopic.name === "Other") {
            newTopicStats.subtopicStats!.push(subtopic);
            continue;
          }
          // Remove comments that have already been used previously and regenerate the
          // SummmaryStats object.
          const unusedComments = subtopic.summaryStats.comments.filter((comment) => {
            return !alreadyUsedComments.has(comment.id);
          });
          newTopicStats.subtopicStats!.push({
            name: subtopic.name,
            commentCount: unusedComments.length,
            summaryStats: (this.constructor as typeof SummaryStats).create(unusedComments),
          });
          alreadyUsedComments = new Set<string>([
            ...alreadyUsedComments,
            ...unusedComments.map((comment) => comment.id),
          ]);
        }
      }
      filteredTopicStats.push(newTopicStats);
    }
    return this.sortTopicStats(filteredTopicStats);
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
   * @param forceOneSubtopicEach whether to force comments with multiple topic/subtopics to only
   * have one. This is done by keeping only the subtopic with the least comments that's not in
   * the Other:Other category.
   *
   * @returns A list of TopicStats objects sorted by comment count with "Other" topics last.
   */
  getStatsByTopic(forceOneSubtopicEach: boolean = false): TopicStats[] {
    const commentsByTopic = groupCommentsBySubtopic(this.comments);
    const topicStats: TopicStats[] = [];

    for (const topicName in commentsByTopic) {
      const subtopics = commentsByTopic[topicName];
      const subtopicStats: TopicStats[] = [];
      let totalTopicComments: number = 0;
      const topicComments: Comment[] = [];

      for (const subtopicName in subtopics) {
        // get corresonding comments, and update counts
        const comments: Comment[] = Object.values(subtopics[subtopicName]);
        const commentCount = comments.length;
        totalTopicComments += commentCount;
        // aggregate comment objects
        topicComments.push(...comments);
        subtopicStats.push({
          name: subtopicName,
          commentCount,
          summaryStats: (this.constructor as typeof SummaryStats).create(comments),
        });
      }

      topicStats.push({
        name: topicName,
        commentCount: totalTopicComments,
        subtopicStats: subtopicStats,
        summaryStats: (this.constructor as typeof SummaryStats).create(topicComments),
      });
    }

    if (forceOneSubtopicEach) {
      return this.useCommentOnlyOnce(topicStats);
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
