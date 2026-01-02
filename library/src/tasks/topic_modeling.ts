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

import { Type } from "@sinclair/typebox";
import { Model } from "../models/model";
import { MAX_RETRIES } from "../models/model_util";
import { getPrompt, retryCall } from "../sensemaker_utils";
import { Comment, FlatTopic, NestedTopic, Topic } from "../types";
import { SupportedLanguage } from "../../templates/l10n";
import { getLearnTopicsPrompt, getLearnSubtopicsPrompt } from "../../templates/l10n/prompts";

/**
 * @fileoverview Helper functions for performing topic modeling on sets of comments.
 */

/**
 * Generates an LLM prompt for topic modeling of a set of comments.
 *
 * @param parentTopics - Optional. An array of top-level topics to use.
 * @param output_lang - The target language for the prompt.
 * @returns The generated prompt string.
 */
export function generateTopicModelingPrompt(parentTopic?: Topic, otherTopics?: Topic[], output_lang: SupportedLanguage = "en"): string {
  if (parentTopic) {
    const otherTopicNames = otherTopics?.map((topic) => topic.name).join(", ") ?? "";
    return getLearnSubtopicsPrompt(output_lang, parentTopic.name, otherTopicNames);
  } else {
    return getLearnTopicsPrompt(output_lang);
  }
}

/**
 * Learn either topics or subtopics from the given comments.
 * @param comments the comments to consider
 * @param model the LLM to use
 * @param topic given or learned topic that subtopics will fit under
 * @param otherTopics other topics that are being used, this is used
 * to avoid duplicate topic/subtopic names
 * @param additionalContext more info to give the model
 * @returns the topics that are present in the comments.
 */
export function learnOneLevelOfTopics(
  comments: Comment[],
  model: Model,
  topic?: Topic,
  otherTopics?: Topic[],
  additionalContext?: string,
  output_lang: SupportedLanguage = "en"
): Promise<Topic[]> {
  const instructions = generateTopicModelingPrompt(topic, otherTopics, output_lang);
  const schema = topic ? Type.Array(NestedTopic) : Type.Array(FlatTopic);

  return retryCall(
    async function (model: Model): Promise<Topic[]> {
      console.log(`Identifying topics for ${comments.length} statements`);
      return (await model.generateData(
        getPrompt(
          instructions,
          comments.map((comment) => comment.text),
          additionalContext,
          output_lang
        ),
        schema,
        output_lang
      )) as Topic[];
    },
    function (response: Topic[]): boolean {
      return learnedTopicsValid(response, topic);
    },
    MAX_RETRIES,
    "Topic identification failed.",
    undefined,
    [model],
    []
  );
}



/**
 * Validates the topic modeling response from the LLM.
 *
 * @param response The topic modeling response from the LLM.
 * @param parentTopics Optional. An array of parent topic names to validate against.
 * @returns True if the response is valid, false otherwise.
 */
export function learnedTopicsValid(response: Topic[], parentTopic?: Topic): boolean {
  // Check if response is empty
  if (!response || response.length === 0) {
    if (parentTopic) {
      console.warn(`Empty response when learning subtopics for "${parentTopic.name}". This may indicate the LLM couldn't identify meaningful subtopics.`);
    } else {
      console.warn("Empty response when learning topics. This may indicate the LLM couldn't identify meaningful topics.");
    }
    return false;
  }

  const topicNames = response.map((topic) => topic.name);

  // 1. If a parentTopic is provided, we're learning subtopics - allow any meaningful topic names
  if (parentTopic) {
    // When learning subtopics, we want the LLM to create new, specific topic names
    // that are different from the parent topic name
    const parentTopicName = parentTopic.name.toLowerCase().replace(/[‑\-\s]+/g, ' ').trim();
    
    // Check if any subtopic has the same name as the parent topic
    // Note: topicNames here are the names of the topics in the response array
    // We need to check the actual subtopic names within each topic
    const hasParentTopicName = response.some(topic => {
      if (topic && "subtopics" in topic && topic.subtopics) {
        return topic.subtopics.some(subtopic => {
          const subtopicName = subtopic.name.toLowerCase().replace(/[‑\-\s]+/g, ' ').trim();
          return subtopicName === parentTopicName;
        });
      }
      return false;
    });
    
    if (hasParentTopicName) {
      console.warn(
        `Invalid response: Found subtopic with the same name as the parent topic "${parentTopic.name}". ` +
        "Subtopics should have distinct names from their parent topic."
      );
      return false;
    }
    
    // Allow any other meaningful topic names for subtopics
    console.log(`✅ Valid subtopic learning response: ${response.length} topics with subtopics created under "${parentTopic.name}"`);
    return true;
  }

  // 2. Ensure no subtopic has the same name as any main topic.
  for (const topic of response) {
    const subtopicNames =
      "subtopics" in topic ? topic.subtopics.map((subtopic) => subtopic.name) : [];
    for (const subtopicName of subtopicNames) {
      // 更寬鬆的名稱匹配，允許大小寫和格式差異
      const normalizedSubtopicName = subtopicName.toLowerCase().replace(/[‑\-\s]+/g, ' ').trim();
      const normalizedTopicNames = topicNames.map(name => 
        name.toLowerCase().replace(/[‑\-\s]+/g, ' ').trim()
      );
      
      if (normalizedTopicNames.includes(normalizedSubtopicName) && subtopicName !== "Other") {
        console.warn(
          `Invalid response: Subtopic "${subtopicName}" has the same name as a main topic.`
        );
        return false;
      }
    }
  }

  return true;
}
