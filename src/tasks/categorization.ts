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

import { Comment, Topic } from "../types";
import { MAX_RETRIES, RETRY_DELAY_MS } from "../models/vertex_model";
import { Model } from "../models/model";
import { getPrompt } from "../sensemaker_utils";

/**
 * @fileoverview Helper functions for performing comments categorization.
 */

/**
 * Makes API call to generate JSON and retries with any comments that were not properly categorized.
 * @param instructions Instructions for the LLM on how to categorize the comments.
 * @param inputComments The comments to categorize.
 * @param includeSubtopics Whether to include subtopics in the categorization.
 * @param topics The topics and subtopics provided to the LLM for categorization.
 * @param additionalInstructions - extra context to be included to the LLM prompt
 * @returns The categorized comments.
 */
export async function categorizeWithRetry(
  model: Model,
  instructions: string,
  inputComments: Comment[],
  includeSubtopics: boolean,
  topics: Topic[],
  additionalInstructions?: string
): Promise<Comment[]> {
  // a holder for uncategorized comments: first - input comments, later - any failed ones that need to be retried
  let uncategorized: Comment[] = [...inputComments];
  const categorized: Comment[] = [];
  // Lookup map to get comments by ID (a LLM returns IDs only, this is used to pull the text to build a proper Comment)
  const inputCommentsLookup = new Map<string, Comment>(
    inputComments.map((comment) => [comment.id, comment])
  );

  for (let attempts = 1; attempts <= MAX_RETRIES; attempts++) {
    // convert JSON to string representation that will be sent to the model
    const uncategorizedCommentsForModel: string[] = uncategorized.map((comment) =>
      JSON.stringify({ id: comment.id, text: comment.text })
    );

    const newCategorized: Comment[] = await model.generateComments(
      getPrompt(instructions, uncategorizedCommentsForModel, additionalInstructions),
      includeSubtopics
    );
    // Add missing 'text' properties to the result using the lookup map, so we can cast to Comment type that requires text.
    const newCategorizedComments: Comment[] = addMissingTextToCategorizedComments(
      newCategorized,
      inputCommentsLookup
    );

    // perform validation, update categorized, reset uncategorized
    uncategorized = processCategorizedComments(
      newCategorizedComments,
      inputComments,
      uncategorized,
      includeSubtopics,
      topics,
      categorized
    );

    if (uncategorized.length === 0) {
      break; // All comments categorized successfully
    }

    if (attempts < MAX_RETRIES) {
      console.warn(
        `Expected all ${uncategorizedCommentsForModel.length} comments to be categorized, but ${uncategorized.length} are not categorized properly. Retrying in ${RETRY_DELAY_MS / 1000} seconds...`
      );
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    } else {
      assignDefaultCategory(uncategorized, categorized);
    }
  }

  return categorized;
}

export function topicCategorizationPrompt(topics: Topic[]): string {
  return `
For each of the following comments, identify the most relevant topic from the list below.

Input Topics:
${JSON.stringify(topics)}

Important Considerations:
- Ensure the assigned topic accurately reflects the meaning of the comment.
- A comment can be assigned to multiple topics if relevant.
- Prioritize using the existing topics whenever possible.
- All comments must be assigned at least one existing topic.
- If no existing topic fits a comment well, assign it to the "Other" topic.
- Do not create any new topics that are not listed in the Input Topics.
`;
}

export function subtopicCategorizationPrompt(topics: Topic[]): string {
  return `
For each of the following comments, identify the most relevant topic and subtopic from the list below.

Input Topics and Subtopics (JSON formatted):
${JSON.stringify(topics)}

Important Considerations:
- Ensure the assignment of comments to subtopics is accurate and reflects the meaning of the comment.
- If comments relate to multiple topics, they should be added to each of the corresponding topics and their relevant subtopics.
- Prioritize assigning comments to existing subtopics whenever possible.
- All comments must be assigned to at least one existing topic and subtopic.
- If none of the provided topic–subtopic pairs accurately fit the comment, assign it to the 'Other' topic and its 'Other' subtopic.
- Do not create any new topics that are not listed in the Input Topics and Subtopics.
- Do not create any new subtopics that are not listed in the Input Topics and Subtopics.
`;
}

/**
 * Generates a prompt for an LLM to categorize comments based on a predefined set of topics (and subtopics).
 *
 * @param topics The user provided topics (and subtopic).
 * @param includeSubtopics Whether to include subtopics in the categorization.
 * @returns The generated prompt string, including instructions, output format, and considerations for categorization.
 */
export function generateCategorizationPrompt(topics: Topic[], includeSubtopics: boolean): string {
  return includeSubtopics
    ? subtopicCategorizationPrompt(topics)
    : topicCategorizationPrompt(topics);
}

/**
 * Validates categorized comments, checking for:
 *  - Extra comments (not present in the original input)
 *  - Empty topics or subtopics
 *  - Invalid topic or subtopic names
 * @param categorizedComments The categorized comments to validate.
 * @param inputComments The original input comments.
 * @param includeSubtopics Whether to include subtopics in the categorization.
 * @param topics The topics and subtopics provided to the LLM for categorization.
 * @returns An object containing:
 *  - `validCategorizedComments`: An array of validated categorized comments.
 *  - `commentsWithInvalidTopics`: An array of comments that failed validation.
 */
export function validateCategorizedComments(
  categorizedComments: Comment[],
  inputComments: Comment[],
  includeSubtopics: boolean,
  topics: Topic[]
): {
  commentsPassedValidation: Comment[];
  commentsWithInvalidTopics: Comment[];
} {
  const commentsPassedValidation: Comment[] = [];
  const commentsWithInvalidTopics: Comment[] = [];
  // put all input comment ids together for output ids validation
  const inputCommentIds: string[] = inputComments.map((comment) => comment.id);
  // topic -> subtopics lookup for naming validation
  const topicLookup: Record<string, string[]> = createTopicLookup(topics);

  categorizedComments.forEach((comment) => {
    if (isExtraComment(comment, inputCommentIds)) {
      return; // Skip to the next comment
    }

    if (hasEmptyTopicsOrSubtopics(comment, includeSubtopics)) {
      commentsWithInvalidTopics.push(comment);
      return; // Skip to the next comment
    }

    if (hasInvalidTopicNames(comment, includeSubtopics, topicLookup)) {
      commentsWithInvalidTopics.push(comment);
      return; // Skip to the next comment
    }

    // If all checks pass, add the comment to the valid ones
    commentsPassedValidation.push(comment);
  });

  return { commentsPassedValidation, commentsWithInvalidTopics };
}

/**
 * Creates a lookup table (dictionary) from an array of input Topic objects.
 * This table maps topic names to arrays of their corresponding subtopic names.
 *
 * @param inputTopics The array of Topic objects to create the lookup table from.
 * @returns A dictionary where keys are topic names (strings) and values are arrays of subtopic names (strings).
 *   If a topic has no subtopics, an empty array is used as the value to avoid dealing with undefined values.
 */
function createTopicLookup(inputTopics: Topic[]): Record<string, string[]> {
  const lookup: Record<string, string[]> = {};
  for (const topic of inputTopics) {
    lookup[topic.name] = topic.subtopics ? topic.subtopics.map((subtopic) => subtopic.name) : [];
  }
  return lookup;
}

/**
 * Checks if a comment is an extra comment (not present in the original input).
 * @param comment The categorized comment to check.
 * @param inputCommentIds An array of IDs of the original input comments.
 * @returns True if the comment is extra, false otherwise.
 */
function isExtraComment(comment: Comment, inputCommentIds: string[]): boolean {
  if (!inputCommentIds.includes(comment.id)) {
    console.warn(`Extra comment in model's response: ${JSON.stringify(comment)}`);
    return true;
  }
  return false;
}

/**
 * Checks if a comment has empty topics or subtopics.
 * @param comment The categorized comment to check.
 * @param includeSubtopics Whether to include subtopics in the categorization.
 * @returns True if the comment has empty topics or subtopics, false otherwise.
 */
function hasEmptyTopicsOrSubtopics(comment: Comment, includeSubtopics: boolean): boolean {
  if (!comment.topics || comment.topics.length === 0) {
    console.warn(`Comment with empty topics: ${JSON.stringify(comment)}`);
    return true;
  }
  if (
    includeSubtopics &&
    comment.topics.some((topic) => !topic.subtopics || topic.subtopics.length === 0)
  ) {
    console.warn(`Comment with empty subtopics: ${JSON.stringify(comment)}`);
    return true;
  }
  return false;
}

/**
 * Checks if a categorized comment has topic or subtopic names different from the provided ones to the LLM.
 * @param comment The categorized comment to check.
 * @param includeSubtopics Whether to include subtopics in the categorization.
 * @param inputTopics The lookup table mapping the input topic names to arrays of their subtopic names.
 * @returns True if the comment has invalid topic or subtopic names, false otherwise.
 */
function hasInvalidTopicNames(
  comment: Comment,
  includeSubtopics: boolean,
  inputTopics: Record<string, string[]>
): boolean {
  // TODO: Currently comment.topics can be undefined, so we need this. Remove it once we have a new type that has topics required.
  const topicsToCheck = comment.topics || [];

  // We use `some` here to return as soon as we find an invalid topic (or subtopic).
  return topicsToCheck.some((topic) => {
    if (topic.name === "Other") {
      return false; // "Other" topic can have any subtopic names - we can skip checking them.
    }

    const isValidTopic = topic.name in inputTopics;
    if (!isValidTopic) {
      console.warn(
        `Comment has an invalid topic: ${topic.name}, comment: ${JSON.stringify(comment)}`
      );
      return true; // Invalid topic found, stop checking and return `hasInvalidTopicNames` true for this comment.
    }

    if (includeSubtopics && topic.subtopics) {
      const areAllSubtopicsValid = areSubtopicsValid(topic.subtopics, inputTopics[topic.name]);
      if (!areAllSubtopicsValid) {
        console.warn(
          `Comment has invalid subtopics under topic: ${topic.name}, comment: ${JSON.stringify(comment)}`
        );
        return true; // Invalid subtopics found, stop checking and return `hasInvalidTopicNames` true for this comment.
      }
    }

    // The current topic (and all its subtopics) is valid, go to the next one.
    return false;
  });
}

/**
 * Checks if an array of subtopics is valid against a list of valid subtopic names.
 * A subtopic is considered valid if its name is present in the input subtopics or if it's named "Other".
 *
 * @param subtopicsToCheck An array of subtopic objects, each having a 'name' property.
 * @param inputSubtopics An array of input subtopic names.
 * @returns True if all subtopics are valid, false otherwise.
 */
function areSubtopicsValid(
  subtopicsToCheck: { name: string }[],
  inputSubtopics: string[]
): boolean {
  return subtopicsToCheck.every(
    (subtopic) => inputSubtopics.includes(subtopic.name) || subtopic.name === "Other"
  );
}

/**
 * Finds comments that are missing from the categorized output.
 * @param categorizedComments The categorized comments received from the model.
 * @param uncategorized The current set of uncategorized comments to check if any are missing in the model response.
 * @returns An array of comments that were present in the input, but not in categorized.
 */
export function findMissingComments(
  categorizedComments: Comment[],
  uncategorized: Comment[]
): Comment[] {
  const categorizedCommentIds: string[] = categorizedComments.map((comment) => comment.id);
  const missingComments = uncategorized.filter(
    (uncategorizedComment) => !categorizedCommentIds.includes(uncategorizedComment.id)
  );

  if (missingComments.length > 0) {
    console.warn(`Missing comments in model's response: ${JSON.stringify(missingComments)}`);
  }
  return missingComments;
}

/**
 * Adds the missing 'text' property to categorized comments using a lookup map.
 * This is necessary because the LLM returns only IDs, and we need the original text.
 *
 * @param categorizedComments The categorized comments received from the model (missing the 'text' property).
 * @param inputCommentsLookup A map to look up the original input comments by their ID.
 * @returns An array of Comment objects with the 'text' property added.
 */
export function addMissingTextToCategorizedComments(
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  categorizedComments: any[],
  inputCommentsLookup: Map<string, Comment>
): Comment[] {
  // eslint-disable-next-line  @typescript-eslint/no-explicit-any
  categorizedComments.forEach((categorizedComment: any) => {
    const inputComment = inputCommentsLookup.get(categorizedComment.id);
    if (inputComment) {
      categorizedComment.text = inputComment.text;
    } else {
      // This may happen if model returned a comment id that was not in the input. We filter such comments on the later steps.
      console.warn(`Could not find input comment for ID ${categorizedComment.id}`);
    }
  });
  return categorizedComments as Comment[];
}

/**
 * Parses a JSON string representing topics into an array of Topic objects.
 *
 * @param topicsJsonString The JSON string to parse.
 * @returns An array of Topic objects.
 * @throws An error if the input is not a valid JSON string representing an array of Topic objects.
 */
export function parseTopicsJson(topicsJsonString: string): Topic[] {
  try {
    return JSON.parse(topicsJsonString);
  } catch (error) {
    throw new Error(
      `Invalid topics JSON string. Please provide a valid JSON array of Topic objects,
      JSON parse error: ${error}`
    );
  }
}

// TODO: Reconsider how this feature should be used, it is currently unused, but there has been
// desire in the past to group by Topics instead of Comments (like to assist with deduplication).
/**
 * Groups categorized comments by topic and subtopic.
 *
 * @param categorizedComments An array of categorized comments.
 * @returns A JSON string representing the comments grouped by topic and subtopic.
 *
 * Example:
 * {
 *   "Topic 1": {
 *     "Subtopic 2": {
 *       "id 1": "comment 1",
 *       "id 2": "comment 2"
 *     }
 *   }
 * }
 */
export function groupCommentsByTopic(categorized: Comment[]): string {
  const commentsByTopics: {
    [topicName: string]: {
      [subtopicName: string]: { [commentId: string]: string };
    };
  } = {};
  for (const comment of categorized) {
    if (!comment.topics || comment.topics.length === 0) {
      throw new Error(`Comment with ID ${comment.id} has no topics assigned.`);
    }
    for (const topic of comment.topics) {
      if (!commentsByTopics[topic.name]) {
        commentsByTopics[topic.name] = {}; // init new topic name
      }
      for (const subtopic of topic.subtopics || []) {
        if (!commentsByTopics[topic.name][subtopic.name]) {
          commentsByTopics[topic.name][subtopic.name] = {}; // init new subtopic name
        }
        commentsByTopics[topic.name][subtopic.name][comment.id] = comment.text;
      }
    }
  }
  return JSON.stringify(commentsByTopics, null, 2);
}

/**
 * Processes the categorized comments, validating them and updating the categorized and uncategorized arrays.
 *
 * @param newCategorizedComments The newly categorized comments from the LLM.
 * @param inputComments The original input comments.
 * @param uncategorized The current set of uncategorized comments to check if any are missing in the model response.
 * @param includeSubtopics Whether to include subtopics in the categorization.
 * @param topics The topics and subtopics provided to the LLM for categorization.
 * @param categorized The array of already categorized comments.
 * @returns The updated array of uncategorized comments.
 */
function processCategorizedComments(
  newCategorizedComments: Comment[],
  inputComments: Comment[],
  uncategorized: Comment[],
  includeSubtopics: boolean,
  topics: Topic[],
  categorized: Comment[]
): Comment[] {
  // Check for comments that were never in the input, have no topics, or non-matching topic names.
  const { commentsPassedValidation, commentsWithInvalidTopics } = validateCategorizedComments(
    newCategorizedComments,
    inputComments,
    includeSubtopics,
    topics
  );
  categorized.push(...commentsPassedValidation);
  // Check for comments completely missing in the model's response
  const missingComments: Comment[] = findMissingComments(newCategorizedComments, uncategorized);
  // Combine all invalid comments for retry
  return [...missingComments, ...commentsWithInvalidTopics];
}

/**
 * Assigns the default "Other" topic and "Uncategorized" subtopic to comments that failed categorization.
 *
 * @param uncategorized The array of comments that failed categorization.
 * @param categorized The array of successfully categorized comments.
 */
function assignDefaultCategory(uncategorized: Comment[], categorized: Comment[]) {
  console.warn(
    `Failed to categorize ${uncategorized.length} comments after maximum number of retries. Assigning "Other" topic and "Uncategorized" subtopic to failed comments.`
  );
  console.warn("Uncategorized comments:", JSON.stringify(uncategorized));
  uncategorized.forEach((comment) => {
    comment.topics = [
      {
        name: "Other",
        subtopics: [{ name: "Uncategorized" }],
      },
    ];
    categorized.push(comment);
  });
}
