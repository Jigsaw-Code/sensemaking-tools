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

import { CommentRecord, Comment, Topic, FlatTopic, TopicCategorizedComment } from "../types";
import { Model } from "../models/model";
import { executeConcurrently, getPrompt, hydrateCommentRecord } from "../sensemaker_utils";
import { TSchema, Type } from "@sinclair/typebox";
import { learnOneLevelOfTopics } from "./topic_modeling";
import { MAX_RETRIES, RETRY_DELAY_MS } from "../models/model_util";
import { SupportedLanguage } from "../../templates/l10n";

/**
 * @fileoverview Helper functions for performing comments categorization.
 */

/**
 * Makes API call to generate JSON and retries with any comments that were not properly categorized.
 * @param instructions Instructions for the LLM on how to categorize the comments.
 * @param inputComments The comments to categorize.
 * @param topics The topics and subtopics provided to the LLM for categorization.
 * @param additionalContext - extra context to be included to the LLM prompt
 * @returns The categorized comments.
 */
export async function categorizeWithRetry(
  model: Model,
  instructions: string,
  inputComments: Comment[],
  topics: Topic[],
  additionalContext?: string,
  output_lang: SupportedLanguage = "en"
): Promise<CommentRecord[]> {
  // a holder for uncategorized comments: first - input comments, later - any failed ones that need to be retried
  let uncategorized: Comment[] = [...inputComments];
  let categorized: CommentRecord[] = [];

  for (let attempts = 1; attempts <= MAX_RETRIES; attempts++) {
    // convert JSON to string representation that will be sent to the model
    const uncategorizedCommentsForModel: string[] = uncategorized.map((comment) =>
      JSON.stringify({ id: comment.id, text: comment.text })
    );
    const outputSchema: TSchema = Type.Array(TopicCategorizedComment);
    let newCategorized: CommentRecord[];
    
    try {
      const rawResponse = await model.generateData(
        getPrompt(instructions, uncategorizedCommentsForModel, additionalContext, output_lang),
        outputSchema,
        output_lang
      );

      
      // 確保回應是一個數組
      if (!Array.isArray(rawResponse) && !Array.isArray((rawResponse as { items: CommentRecord[] }).items)) {
        console.error('LLM response is not an array:', typeof rawResponse, rawResponse);
        throw new Error('LLM response format error: expected array of comments');
      }
      
      newCategorized = Array.isArray(rawResponse) ? rawResponse : (rawResponse as { items: CommentRecord[] }).items;
      console.log(`LLM returned ${newCategorized.length} categorized comments`);
    } catch (error) {
      console.error('Error in LLM categorization:', error);
      throw error;
    }

    const newProcessedComments = processCategorizedComments(
      newCategorized,
      inputComments,
      uncategorized,
      topics
    );
    categorized = categorized.concat(newProcessedComments.commentRecords);
    uncategorized = newProcessedComments.uncategorizedComments;

    if (uncategorized.length === 0) {
      break; // All comments categorized successfully
    }

    if (attempts < MAX_RETRIES) {
      console.warn(
        `Expected all ${uncategorizedCommentsForModel.length} comments to be categorized, but ${uncategorized.length} are not categorized properly. Retrying in ${RETRY_DELAY_MS / 1000} seconds...`
      );
      await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
    } else {
      categorized = categorized.concat(assignDefaultCategory(uncategorized));
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
- A comment can be assigned to multiple topics if necessary but prefer to assign only one topic 
- Prioritize using the existing topics whenever possible.
- All comments must be assigned at least one existing topic.
- If no existing topic fits a comment well, assign it to the "Other" topic.
- Do not create any new topics that are not listed in the Input Topics.
- When generating the JSON output, minimize the size of the response. For example, prefer this compact format: {"id": "5258", "topics": [{"name": "Arts, Culture, And Recreation"}]} instead of adding unnecessary whitespace or newlines.
`;
}

/**
 * Validates categorized comments, checking for:
 *  - Extra comments (not present in the original input)
 *  - Empty topics or subtopics
 *  - Invalid topic or subtopic names
 * @param commentRecords The categorized comments to validate.
 * @param inputComments The original input comments.
 * @param topics The topics and subtopics provided to the LLM for categorization.
 * @returns An object containing:
 *  - `validCommentRecords`: comments that passed validation.
 *  - `commentsWithInvalidTopics`: comments that failed validation.
 */
export function validateCommentRecords(
  commentRecords: CommentRecord[],
  inputComments: Comment[],
  topics: Topic[]
): {
  commentsPassedValidation: CommentRecord[];
  commentsWithInvalidTopics: CommentRecord[];
} {
  const commentsPassedValidation: CommentRecord[] = [];
  const commentsWithInvalidTopics: CommentRecord[] = [];
  
  // 防護措施：確保 commentRecords 是一個數組
  if (!Array.isArray(commentRecords)) {
    console.error('validateCommentRecords: commentRecords is not an array:', typeof commentRecords, commentRecords);
    throw new Error('Invalid input: commentRecords must be an array');
  }
  
  // 防護措施：確保 inputComments 是一個數組
  if (!Array.isArray(inputComments)) {
    console.error('validateCommentRecords: inputComments is not an array:', typeof inputComments, inputComments);
    throw new Error('Invalid input: inputComments must be an array');
  }
  
  // 防護措施：確保 topics 是一個數組
  if (!Array.isArray(topics)) {
    console.error('validateCommentRecords: topics is not an array:', typeof topics, topics);
    throw new Error('Invalid input: topics must be an array');
  }
  
  // put all input comment ids together for output ids validation
  const inputCommentIds = new Set<string>(inputComments.map((comment) => comment.id));
  // topic -> subtopics lookup for naming validation
  const topicLookup: Record<string, string[]> = createTopicLookup(topics);

  commentRecords.forEach((comment) => {
    if (isExtraComment(comment, inputCommentIds)) {
      return; // Skip to the next comment
    }

    if (hasEmptyTopicsOrSubtopics(comment)) {
      commentsWithInvalidTopics.push(comment);
      return; // Skip to the next comment
    }

    if (hasInvalidTopicNames(comment, topicLookup)) {
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
    if ("subtopics" in topic) {
      lookup[topic.name] = topic.subtopics.map((subtopic) => subtopic.name);
    } else {
      lookup[topic.name] = [];
    }
  }
  return lookup;
}

/**
 * Checks if a comment is an extra comment (not present in the original input).
 * @param comment The categorized comment to check.
 * @param inputCommentIds An array of IDs of the original input comments.
 * @returns True if the comment is extra, false otherwise.
 */
function isExtraComment(comment: Comment | CommentRecord, inputCommentIds: Set<string>): boolean {
  if (!inputCommentIds.has(comment.id)) {
    console.warn(`Extra comment in model's response: ${JSON.stringify(comment)}`);
    return true;
  }
  return false;
}

/**
 * Checks if a comment has empty topics or subtopics.
 * @param comment The categorized comment to check.
 * @returns True if the comment has empty topics or subtopics, false otherwise.
 */
function hasEmptyTopicsOrSubtopics(comment: CommentRecord): boolean {
  // 檢查 topics 欄位是否存在
  if (!comment.topics || comment.topics.length === 0) {
    console.warn(`Comment with missing or empty topics: ${JSON.stringify(comment)}`);
    return true;
  }
  if (
    comment.topics.some(
      (topic: Topic) => "subtopics" in topic && (!topic.subtopics || topic.subtopics.length === 0)
    )
  ) {
    console.warn(`Comment with empty subtopics: ${JSON.stringify(comment)}`);
    return true;
  }
  return false;
}

/**
 * Checks if a categorized comment has topic or subtopic names different from the provided ones to the LLM.
 * @param comment The categorized comment to check.
 * @param inputTopics The lookup table mapping the input topic names to arrays of their subtopic names.
 * @returns True if the comment has invalid topic or subtopic names, false otherwise.
 */
function hasInvalidTopicNames(
  comment: CommentRecord,
  inputTopics: Record<string, string[]>
): boolean {
  // 檢查 topics 欄位是否存在
  if (!comment.topics || comment.topics.length === 0) {
    console.warn(`Comment with missing or empty topics: ${JSON.stringify(comment)}`);
    return true;
  }
  
  // We use `some` here to return as soon as we find an invalid topic (or subtopic).
  return comment.topics.some((topic: Topic) => {
    const isValidTopic = topic.name in inputTopics;
    if (!isValidTopic && topic.name !== "Other") {
      console.warn(
        `Comment has an invalid topic: ${topic.name}, comment: ${JSON.stringify(comment)}`
      );
      return true; // Invalid topic found, stop checking and return `hasInvalidTopicNames` true for this comment.
    }

    if ("subtopics" in topic) {
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
 * @param commentRecords The categorized comments received from the model.
 * @param uncategorized The current set of uncategorized comments to check if any are missing in the model response.
 * @returns An array of comments that were present in the input, but not in categorized.
 */
export function findMissingComments(
  commentRecords: CommentRecord[],
  uncategorized: Comment[]
): Comment[] {
  const commentRecordIds: string[] = commentRecords.map((comment) => comment.id);
  const missingComments = uncategorized.filter(
    (uncommentRecord) => !commentRecordIds.includes(uncommentRecord.id)
  );

  if (missingComments.length > 0) {
    console.warn(`Missing comments in model's response: ${JSON.stringify(missingComments)}`);
  }
  return missingComments;
}

/**
 * Processes the categorized comments, validating them and updating the categorized and uncategorized arrays.
 *
 * @param commentRecords The newly categorized comments from the LLM.
 * @param inputComments The original input comments.
 * @param uncategorized The current set of uncategorized comments to check if any are missing in the model response.
 * @param topics The topics and subtopics provided to the LLM for categorization.
 * @returns The successfully categorized comments and the unsuccessfully categorized comments with
 * the topics removed.
 */
function processCategorizedComments(
  commentRecords: CommentRecord[],
  inputComments: Comment[],
  uncategorized: Comment[],
  topics: Topic[]
): {
  commentRecords: CommentRecord[];
  uncategorizedComments: Comment[];
} {
  // Check for comments that were never in the input, have no topics, or non-matching topic names.
  const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
    commentRecords,
    inputComments,
    topics
  );

  // Check for comments completely missing in the model's response
  const missingComments: Comment[] = findMissingComments(commentRecords, uncategorized);
  // Remove invalid topics from comments to prepare for retry.
  let invalidComments = hydrateCommentRecord(commentsWithInvalidTopics, inputComments);
  invalidComments = invalidComments.map((comment: Comment): Comment => {
    comment.topics = undefined;
    return comment;
  });
  // Combine all invalid comments for retry
  return {
    commentRecords: commentsPassedValidation,
    uncategorizedComments: [...missingComments, ...invalidComments],
  };
}

/**
 * Assigns the default "Other" topic and optionally "Uncategorized" subtopic to comments that
 * failed categorization.
 *
 * @param uncategorized The array of comments that failed categorization.
 * @returns the uncategorized comments now categorized into a "Other" category.
 */
function assignDefaultCategory(uncategorized: Comment[]): CommentRecord[] {
  console.warn(
    `Failed to categorize ${uncategorized.length} comments after maximum number of retries. Assigning "Other" topic and "Uncategorized" subtopic to failed comments.`
  );
  console.warn("Uncategorized comments:", JSON.stringify(uncategorized));
  return uncategorized.map((comment: Comment): CommentRecord => {
    return {
      ...comment,
      topics: [{ name: "Other" } as FlatTopic],
    };
  });
}

export function getTopicDepthFromTopics(topics: Topic[], currentDepth: number = 1): number {
  if (!topics || topics.length === 0) {
    return currentDepth - 1; // avoid infinite recursion for empty topics
  }
  return topics.every((topic) => {
    return "subtopics" in topic && topic.subtopics.length > 0;
  })
    ? getTopicDepthFromTopics(
        topics.map((topic) => ("subtopics" in topic ? topic.subtopics : [])).flat(),
        currentDepth + 1
      )
    : currentDepth;
}

/**
 * Get the minimum topic depth across all comments.
 */
function getTopicDepth(comments: Comment[]): number {
  return comments
    .map((comment) => {
      return comment.topics ? getTopicDepthFromTopics(comment.topics, 1) : 0;
    })
    .reduce((minDepth, depth) => Math.min(minDepth, depth), Number.MAX_VALUE);
}

// Return a flat list of topics representing all the topics at a depth
function getTopicsAtDepth(topics: Topic[], depth: number): Topic[] {
  if (depth === 1) {
    return topics;
  } else if (depth >= 2) {
    return getTopicsAtDepth(
      topics
        .map((topic) => {
          return "subtopics" in topic ? topic.subtopics : [];
        })
        .flat(),
      depth - 1
    );
  } else {
    throw Error("Invalid depth value provided, depth: " + depth);
  }
}

function getCommentsWithTopic(comments: Comment[], topicName: string) {
  return comments.filter(
    (comment) => comment.topics && comment.topics.map((topic) => topic.name).includes(topicName)
  );
}

/**
 * Gets the comment texts and ids with a topic at a given level.
 *
 * Note the comment topics are from the given depth level and have been modified.
 *
 * @param comments the categorized comments to search
 * @param topicName the name of the topic to match
 * @param depth the depth to search at
 * @returns the comments with the given topicName at the given depth
 */
function getCommentTextsWithTopicsAtDepth(
  comments: Comment[],
  topicName: string,
  depth: number = 1
): Comment[] {
  if (depth === 1) {
    return getCommentsWithTopic(comments, topicName);
  } else if (depth >= 2) {
    return getCommentTextsWithTopicsAtDepth(
      comments
        .filter((comment) => {
          return comment.topics !== undefined;
        })
        .map((comment) => {
          return {
            id: comment.id,
            text: comment.text,
            topics: comment
              .topics!.map((topic) => ("subtopics" in topic ? topic.subtopics : []))
              .flat(),
          };
        }),
      topicName,
      depth - 1
    );
  } else {
    throw Error("Invalid depth value provided, depth: " + depth);
  }
}

/**
 * Add subtopics to an existing topic
 * @param topic the topic to add the subtopics to
 * @param parentSubtopic the topic that is the parent of the new subtopics
 * @param newSubtopics the new subtopics to add to topic
 * @returns the topic with the new subtopics added at the right level
 */
function addNewLevelToTopic(topic: Topic, parentSubtopic: Topic, newSubtopics: Topic[]): Topic {
  if ("subtopics" in topic) {
    if (!("subtopics" in parentSubtopic)) {
      throw Error("Expected parent topic to have subtopics");
    }
    for (let i = 0; i < topic.subtopics.length; i++) {
      if (topic.subtopics[i].name === parentSubtopic.name) {
        topic.subtopics[i] = addNewLevelToTopic(
          topic.subtopics[i],
          parentSubtopic.subtopics[0],
          newSubtopics
        );
      }
    }
    return topic;
  } else {
    return { name: topic.name, subtopics: newSubtopics };
  }
}

/**
 * Combine full comments with newly categorized comments with one extra level of categorization
 * @param comments the existing comments to merge into
 * @param categorizedComments a subset of comments that have been newly categorized. The
 * categorization is always topics only but should be merged at the topicDepth level
 * @param topic the parent topic to the topics associated with categorizedComments
 * @param topicDepth the depth of the newly categorizedComments on the comment object
 * @returns all the comments with the one new level of categorization added
 */
function mergeCommentTopics(
  comments: Comment[],
  categorizedComments: Comment[],
  topic: Topic,
  topicDepth: number
): Comment[] {
  const commentIdsInTopic = getCommentTextsWithTopicsAtDepth(comments, topic.name, topicDepth).map(
    (comment) => comment.id
  );

  for (const commentId of commentIdsInTopic) {
    const matchingCategorized = categorizedComments.find(
      (categorized) => categorized.id === commentId
    );
    if (!matchingCategorized || !matchingCategorized.topics) {
      continue;
    }
    // Iterate through comments using indices so that the value can be changed.
    for (let i = 0; i < comments.length; i++) {
      const currentComment = comments[i];
      if (currentComment.id !== commentId || currentComment.topics === undefined) {
        continue;
      }

      // Merge in matchingCategorized either as a new subtopic or a new subsubtopic.
      for (let j = 0; j < currentComment.topics.length; j++) {
        const existingTopic = currentComment.topics[j];
        if (existingTopic.name === topic.name) {
          currentComment.topics[j] = addNewLevelToTopic(
            existingTopic,
            topic,
            matchingCategorized.topics
          );
        } else if ("subtopics" in existingTopic) {
          for (let k = 0; k < existingTopic.subtopics.length; k++) {
            const existingSubtopic = existingTopic.subtopics[k];
            if (existingSubtopic.name === topic.name) {
              if ("subtopics" in currentComment.topics[j]) {
                currentComment.topics[j] = {
                  name: existingTopic.name,
                  subtopics: [
                    ...existingTopic.subtopics.slice(0, k),
                    addNewLevelToTopic(existingSubtopic, topic, matchingCategorized.topics),
                    ...existingTopic.subtopics.slice(k + 1),
                  ],
                };
              }
            }
          }
        }
      }
    }
  }
  return comments;
}

/**
 * Merge an existing topic with new subtopics into a list of all topics.
 * @param topics the existing topics to merge into
 * @param topicAndNewSubtopics the existing topic (must match a topic name from topics) with the
 * new subtopics to add to it
 * @returns the list of existing topics with the new subtopics added to the appropriate topic
 */
function mergeTopics(topics: Topic[], topicAndNewSubtopics: Topic): Topic[] {
  if (!("subtopics" in topicAndNewSubtopics)) {
    return topics;
  }
  for (let i = 0; i < topics.length; i++) {
    if (topics[i].name === topicAndNewSubtopics.name) {
      topics[i] = { name: topics[i].name, subtopics: topicAndNewSubtopics.subtopics };
      return topics;
    }
  }
  return topics;
}

/**
 * Categorize comments one level at a time.
 *
 * For comments without topics, first all the topics are learned, then the comments are
 * categorized into the topics, then for each topic the subset of relevant comments are selected
 * and this is repeated recursively.
 *
 * @param comments the comments to categorize to the given depthLevel
 * @param topicDepth the depth of categorization and topic learning, 1 is topic only; 2 is topics
 * and subtopics; 3 is topics, subtopics, and subsubtopics
 * @param model the model to use for topic learning and categorization
 * @param topics a given set of topics to categorize the comments into
 * @param additionalContext information to give the model
 * @returns the comments categorized to the level specified by topicDepth
 */
export async function categorizeCommentsRecursive(
  comments: Comment[],
  topicDepth: 1 | 2 | 3,
  model: Model,
  topics?: Topic[],
  additionalContext?: string,
  output_lang: SupportedLanguage = "en"
): Promise<Comment[]> {
  // The exit condition - if the requested topic depth matches the current depth of topics on the
  // comments then exit.
  const currentTopicDepth = getTopicDepth(comments);
  console.log("Identifying topics and categorizing statements at depth=", currentTopicDepth);
  if (currentTopicDepth >= topicDepth) {
    return comments;
  }

  if (!topics) {
    topics = await learnOneLevelOfTopics(comments, model, undefined, undefined, additionalContext, output_lang);
    comments = await oneLevelCategorization(comments, model, topics, additionalContext, output_lang);
    // Sometimes comments are categorized into an "Other" topic if no given topics are a good fit.
    // This needs included in the list of topics so these are processed downstream.
    topics.push({ name: "Other" });
    return categorizeCommentsRecursive(comments, topicDepth, model, topics, additionalContext, output_lang);
  }

  if (topics && currentTopicDepth === 0) {
    comments = await oneLevelCategorization(comments, model, topics, additionalContext, output_lang);
    // Sometimes comments are categorized into an "Other" topic if no given topics are a good fit.
    // This needs included in the list of topics so these are processed downstream.
    topics.push({ name: "Other" });
    return categorizeCommentsRecursive(comments, topicDepth, model, topics, additionalContext, output_lang);
  }

  let index = 0;
  const parentTopics = getTopicsAtDepth(topics, currentTopicDepth);
  for (let topic of parentTopics) {
    console.log(
      "Categorizing statements into subtopics under: ",
      topic.name,
      ` (${++index}/${parentTopics.length} topics)`
    );
    const commentsInTopic = structuredClone(
      getCommentTextsWithTopicsAtDepth(comments, topic.name, currentTopicDepth)
    );
    if (commentsInTopic.length === 0) {
      continue;
    }
    if (!("subtopics" in topic)) {
      // The subtopics are added to the existing topic, so a list of length one is returned.
      const newTopics = await learnOneLevelOfTopics(commentsInTopic, model, topic, parentTopics, additionalContext, output_lang);
      
      // Check if we got any topics back
      if (newTopics.length === 0) {
        console.log(`No subtopics learned for topic "${topic.name}", skipping subtopic categorization`);
        continue;
      }
      
      const newTopicAndSubtopics = newTopics[0];
      if (!newTopicAndSubtopics || !("subtopics" in newTopicAndSubtopics)) {
        throw Error("Badly formed LLM response - expected 'subtopics' to be in topics ");
      }
      topic = { name: topic.name, subtopics: newTopicAndSubtopics.subtopics };
    }

    // Use the subtopics as high-level topics and merge them in later.
    const categorizedComments = await oneLevelCategorization(
      commentsInTopic,
      model,
      topic.subtopics,
      additionalContext,
      output_lang
    );
    comments = mergeCommentTopics(comments, categorizedComments, topic, currentTopicDepth);
    // Sometimes comments are categorized into an "Other" subtopic if no given subtopics are a good fit.
    // This needs included in the list of subtopics so these are processed downstream.
    const topicWithNewSubtopics = topic;
    topicWithNewSubtopics.subtopics.push({ name: "Other" });
    topics = mergeTopics(topics, topicWithNewSubtopics);
  }
  return categorizeCommentsRecursive(comments, topicDepth, model, topics, additionalContext, output_lang);
}

export async function oneLevelCategorization(
  comments: Comment[],
  model: Model,
  topics: Topic[],
  additionalContext?: string,
  output_lang: SupportedLanguage = "en"
): Promise<Comment[]> {
  const instructions = topicCategorizationPrompt(topics);
  // TODO: Consider the effects of smaller batch sizes. 1 comment per batch was much faster, but
  // the distribution was significantly different from what we're currently seeing. More testing
  // is needed to determine the ideal size and distribution.
  const batchesToCategorize: (() => Promise<CommentRecord[]>)[] = []; // callbacks
  for (let i = 0; i < comments.length; i += model.categorizationBatchSize) {
    const uncategorizedBatch = comments.slice(i, i + model.categorizationBatchSize);

    // Create a callback function for each batch and add it to the list, preparing them for parallel execution.
    batchesToCategorize.push(() =>
      categorizeWithRetry(model, instructions, uncategorizedBatch, topics, additionalContext, output_lang)
    );
  }

  // 恢復並行處理：每個請求現在都有獨立的 buffer 上下文，不會互相干擾
  const totalBatches = Math.ceil(comments.length / model.categorizationBatchSize);
  console.log(
    `Categorizing ${comments.length} statements in batches (${totalBatches} batches of ${model.categorizationBatchSize} statements) - using parallel processing`
  );
  
  // 使用 executeConcurrently 進行並行處理，每個批次都有獨立的 buffer 上下文
  const CategorizedBatches: CommentRecord[][] = await executeConcurrently(batchesToCategorize);
  console.log(`✅ All ${totalBatches} batches completed successfully in parallel`);

  // flatten categorized batches
  const categorized: CommentRecord[] = [];
  CategorizedBatches.forEach((batch) => categorized.push(...batch));

  const categorizedComments = hydrateCommentRecord(categorized, comments);
  return categorizedComments;
}
