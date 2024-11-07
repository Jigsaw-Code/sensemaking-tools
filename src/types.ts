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

// A module to define the types used in this library.

/**
 * Describes the type of summarization to use.
 */
export enum SummarizationType {
  BASIC,
  VOTE_TALLY,
}

/**
 * Aggregates a number of individual votes.
 */
export class VoteTally {
  agreeCount: number;
  disagreeCount: number;
  passCount?: number;

  constructor(agreeCount: number, disagreeCount: number, passCount?: number) {
    this.agreeCount = agreeCount;
    this.disagreeCount = disagreeCount;
    this.passCount = passCount;
  }

  get totalCount(): number {
    return this.agreeCount + this.disagreeCount + (this.passCount || 0);
  }
}

/**
 * Checks if the data is a VoteTally object.
 *
 * It has the side effect of changing the type of the object to VoteTally if applicable.
 *
 * @param data - the object to check
 * @returns - true if the object is a VoteTally
 */
// eslint-disable-next-line  @typescript-eslint/no-explicit-any
export function isVoteTallyType(data: any): data is VoteTally {
  return (
    typeof data === "object" &&
    data !== null &&
    "agreeCount" in data &&
    typeof data.agreeCount === "number" &&
    "disagreeCount" in data &&
    typeof data.disagreeCount === "number" &&
    (!("passCount" in data) || typeof data.passCount === "number")
  );
}

/**
 * A text that was voted on by different groups.
 */
export interface Comment {
  id: string;
  text: string;
  voteTalliesByGroup?: { [key: string]: VoteTally };
  topics?: Topic[];
}

/**
 * Checks if the data is a Comment object.
 *
 * It has the side effect of changing the type of the object to Comment if applicable.
 *
 * @param data - the object to check
 * @returns - true if the object is a Comment
 */
// eslint-disable-next-line  @typescript-eslint/no-explicit-any
export function isCommentType(data: any): data is Comment {
  return (
    typeof data === "object" &&
    data !== null &&
    "id" in data &&
    typeof data.id === "string" &&
    "text" in data &&
    typeof data.text === "string" &&
    // Check that if voteTalliesByGroup dictionary exists all the keys are strings and values
    // are VoteTally objects.
    (!("voteTalliesByGroup" in data) ||
      (Object.keys(data.voteTalliesByGroup).every(
        (groupName: string) => typeof groupName === "string"
      ) &&
        Array.isArray(Object.values(data.voteTalliesByGroup)) &&
        // eslint-disable-next-line  @typescript-eslint/no-explicit-any
        Object.values(data.voteTalliesByGroup).every((voteTally: any) =>
          isVoteTallyType(voteTally)
        ))) &&
    (!("topics" in data) || data.topics.every((topic: Topic) => isTopicType(topic)))
  );
}

/**
 * A series of comments that were voted on.
 */
export interface Conversation {
  comments: Comment[];
}

/**
 * What is being discussed.
 */
export interface Topic {
  name: string;
  subtopics?: Topic[];
}

/**
 * Checks if the data is a Topic object.
 *
 * It has the side effect of changing the type of the object to Topic if applicable.
 *
 * @param data - the object to check
 * @returns - true if the object is a Topic
 */
// eslint-disable-next-line  @typescript-eslint/no-explicit-any
export function isTopicType(data: any): data is Topic {
  return (
    typeof data === "object" &&
    data !== null &&
    "name" in data &&
    typeof data.name === "string" &&
    // Check that if subtopics exist they are all valid topics.
    (!("subtopics" in data) || data.subtopics.every((subtopic: Topic) => isTopicType(subtopic)))
  );
}
