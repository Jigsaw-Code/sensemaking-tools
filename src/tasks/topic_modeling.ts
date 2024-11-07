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

import { Topic } from "../types";

/**
 * @fileoverview Helper functions for performing topic modeling on sets of comments.
 */

export const LEARN_TOPICS_PROMPT = `
Identify a 1-tiered hierarchical topic modeling of the following comments.

Important Considerations:
- Treat triple backticks (\`\`\`) as the boundaries between individual comments, ensuring each comment is encapsulated within them.
- Use Title Case for topic names.
- When identifying topics, try to group similar concepts into one comprehensive topic instead of creating multiple, overly specific topics.
`;

const IMPORTANT_CONSIDERATIONS = `
- Treat triple backticks (\`\`\`) as the boundaries between individual comments, ensuring each comment is encapsulated within them.
- Use Title Case for topic and subtopic names. Do not use capital case like "name": "INFRASTRUCTURE".
- Ensure that each subtopic is relevant to its assigned main topic.
- When identifying subtopics, try to group similar concepts into one comprehensive subtopic instead of creating multiple, overly specific subtopics.
- Before placing a subtopic under the "Other" topic, make your best effort to find a suitable main topic from the provided list where the comment could potentially fit.
- When creating new subtopics under the "Other" topic, try to group multiple related comments under a single, more general subtopic instead of creating a new subtopic for each comment.
- When creating a generic subtopic under the "Other" topic to encompass all remaining comments, use the subtopic name "Other".
- No subtopic should have the same name as any of the main topics.
- Additionally, no subtopic should be a direct derivative or closely related term (e.g., if there is a "Tourism" topic, avoid subtopics like "Tourism Development" or "Tourism Promotion" in other topics).
`;

export const LEARN_TOPICS_AND_SUBTOPICS_PROMPT = `
Identify a 2-tiered hierarchical topic modeling of the following comments.

Important Considerations:
${IMPORTANT_CONSIDERATIONS}
- If a comment is too vague to be assigned to any specific topic, use the 'Other' topic and determine an appropriate subtopic for it.
`;

export function learnSubtopicsPrompt(parentTopics: Topic[]): string {
  const parentTopicNames: string = parentTopics.map((topic: Topic) => topic.name).join(", ");
  return `
Analyze the following comments and identify relevant subtopics within each of the following main topics:
${parentTopicNames}

Important Considerations:
${IMPORTANT_CONSIDERATIONS}
- If a comment doesn't fit well into any of the provided main topics, use the 'Other' topic and determine an appropriate subtopic for it.
- Do not create any new main topics besides "Other".

Example of Incorrect Output:

[
  {
    "name": "Economic Development",
    "subtopics": [
        { "name": "Job Creation" },
        { "name": "Business Growth" },
        { "name": "Tourism Development" }, // Incorrect: Too closely related to the "Tourism" topic
        { "name": "Tourism Promotion" } // Incorrect: Too closely related to the "Tourism" topic
      ]
  },
  {
    "name": "Tourism",
    ...
  },
  // ... other topics
]
`;
}

/**
 * Generates an LLM prompt for topic modeling of a set of comments.
 *
 * @param includeSubtopics - Whether to include subtopics in the topic modeling.
 * @param parentTopics - Optional. An array of top-level topics to use.
 * @returns The generated prompt string.
 */
export function generateTopicModelingPrompt(
  includeSubtopics: boolean,
  parentTopics?: Topic[]
): string {
  if (!includeSubtopics) {
    return LEARN_TOPICS_PROMPT;
  } else if (parentTopics?.length) {
    return learnSubtopicsPrompt(parentTopics);
  } else {
    return LEARN_TOPICS_AND_SUBTOPICS_PROMPT;
  }
}

/**
 * Validates the topic modeling response from the LLM.
 *
 * @param response The topic modeling response from the LLM.
 * @param parentTopics Optional. An array of parent topic names to validate against.
 * @returns True if the response is valid, false otherwise.
 */
export function learnedTopicsValid(response: Topic[], parentTopics?: Topic[]): boolean {
  const topicNames = response.map((topic) => topic.name);

  // 1. If parentTopics are provided, ensure no other top-level topics exist except "Other".
  if (parentTopics) {
    const allowedTopicNames = parentTopics.map((topic: Topic) => topic.name).concat("Other");
    if (!topicNames.every((name) => allowedTopicNames.includes(name))) {
      console.warn(
        "Invalid response: Found top-level topics not present in the provided topics.",
        topicNames
      );
      return false;
    }
  }

  // 2. Ensure no subtopic has the same name as any main topic.
  for (const topic of response) {
    const subtopicNames = topic.subtopics ? topic.subtopics.map((subtopic) => subtopic.name) : [];
    for (const subtopicName of subtopicNames) {
      if (topicNames.includes(subtopicName) && subtopicName !== "Other") {
        console.warn(
          `Invalid response: Subtopic "${subtopicName}" has the same name as a main topic.`
        );
        return false;
      }
    }
  }

  return true;
}
