// Copyright 2023 Google LLC
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

// Module to interact with Vertex AI.

import {
  HarmBlockThreshold,
  HarmCategory,
  VertexAI,
} from "@google-cloud/vertexai";

// Initialize Vertex with your Cloud project and location
const vertex_ai = new VertexAI({
  project: "conversation-ai-experiments",
  location: "us-central1",
});
const model = "gemini-1.5-pro-001";

const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  },
];

const base_model_spec = {
  model: model,
  generationConfig: {
    maxOutputTokens: 8192,
    temperature: 1,
    topP: 0.95,
  },
  safetySettings: safetySettings,
};

const json_model_spec = {
  model: model,
  generationConfig: {
    maxOutputTokens: 8192,
    temperature: 1,
    topP: 0.95,
    response_mime_type: "application/json",
  },
  safetySettings: safetySettings,
};

// Instantiate the models
const generativeModel = vertex_ai.getGenerativeModel(base_model_spec);
const generativeJsonModel = vertex_ai.getGenerativeModel(json_model_spec);

// Topic schema (TBC... doesn't work yet; see GH issue below)
const topics_schema = {
  type: "array",
  items: {
    type: "object",
    properties: {
      name: { type: "string" },
      subtopics: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string" },
          },
        },
      },
    },
  },
};

/**
 * Combines the data and instructions into a prompt to send to Vertex.
 * @param instructions: what the model should do.
 * @param data: the data that the model should consider.
 * @returns the instructions and the data as a text
 */
export function getPrompt(instructions: string, data: string[]) {
  return `${instructions} ${data.join()}`;
}

function getRequest(instructions: string, data: string[]) {
  return {
    contents: [
      { role: "user", parts: [{ text: getPrompt(instructions, data) }] },
    ],
  };
}

class FailedExecutionError extends Error {
  constructor(public response: any, message?: string) {
    super(message || "An error occurred calling Vertex");
    this.name = "VertexAIExecutionError";
    this.response = response;
  }
}

/**
 * Lower level protocol for sending a set of instructions to an llm with
 * comments (which could contain summary information).
 */
export async function executeRequest(
  instructions: string,
  comments: string[]
): Promise<string | FailedExecutionError> {
  const req = getRequest(instructions, comments);
  const streamingResp = await generativeModel.generateContentStream(req);

  const response = await streamingResp.response;
  if (response.candidates![0].content.parts[0].text) {
    return response.candidates![0].content.parts[0].text;
  } else {
    console.warn("Malformed response: ", response);
    throw new FailedExecutionError(response);
  }
}

/**
 * Utility function for sending a set of instructions to an LLM with comments,
 * and returning the results as JSON.
 */
async function generateJSON(instructions: string, comments: string[]) {
  /**
   * this doesn't work yet, because of https://github.com/googleapis/nodejs-vertexai/issues/392
   */
  //const response = await generativeModel.generateContent({
  //responseSchema: topics_schema // or is it response_schema?
  //});

  const req = getRequest(instructions, comments);
  const streamingResp = await generativeJsonModel.generateContentStream(req);

  const response = await streamingResp.response;
  if (response.candidates![0].content.parts[0].text) {
    const responseText = response.candidates![0].content.parts[0].text;
    const generatedJSON = JSON.parse(responseText);
    return generatedJSON;
  } else {
    console.warn("Malformed response: ", response);
    throw new FailedExecutionError(response);
  }
}

interface Topic {
  name: string;
  subtopics?: Topic[];
}

export async function learnTopics(
  comments: string[],
  { depth = 1, parentTopics }: { depth?: number; parentTopics?: string[] }
): Promise<Topic[]> {
  let instructions =
    "Identify a " +
    depth +
    "-tiered hierarchical topic modeling of the following comments, and return the results as ";
  if (depth == 1) {
    instructions += "an array of strings.";
  } else if (depth == 2) {
    instructions +=
      "an array of objects with keys name and subtopics, where subtopics points to an array strings (no nested objects).";
  }
  if (parentTopics !== undefined) {
    instructions +=
      " The top level topics should be: " +
      JSON.stringify([
        "Infrastructure & Development",
        "Social & Community Issues",
        "Economy & Employment",
        "Governance & Policy",
      ]) +
      ".";
    console.log(instructions);
  }
  const response = await generateJSON(instructions, comments);
  return response;
}

/**
 * Summarizes the comments using a LLM on Vertex.
 * @param instructions: how the comments should be summarized.
 * @param comments: the data to summarize
 * @returns: the LLM's summarization.
 */
export async function summarize(
  instructions: string,
  comments: string[]
): Promise<string> {
  const summaryResponse = await executeRequest(instructions, comments);
  if (summaryResponse instanceof FailedExecutionError) {
    throw summaryResponse;
  } else {
    return summaryResponse;
  }
}

/**
 * Extracts topics from the comments using a LLM on Vertex AI.
 * @param comments The comments data for topic modeling
 * @returns: The LLM's topic modeling.
 */
export async function getTopics(comments: string[]): Promise<string> {
  const topicsResponse = await learnTopics(comments, { depth: 1 });
  if (topicsResponse instanceof FailedExecutionError) {
    throw topicsResponse;
  } else {
    return JSON.stringify(topicsResponse);
  }
}

/**
 * Categorize the comments by topics using a LLM on Vertex.
 * @param instructions How the comments should be categorized.
 * @param comments The data to summarize
 * @returns: The LLM's categorization.
 */
export async function categorize(
  instructions: string,
  comments: string[]
): Promise<string> {
  const categorizeResponse = await executeRequest(instructions, comments);
  if (categorizeResponse instanceof FailedExecutionError) {
    throw categorizeResponse;
  } else {
    return categorizeResponse;
  }
}
