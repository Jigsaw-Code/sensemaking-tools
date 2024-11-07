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

// Module to interact with models available on Google Cloud's Model Garden, including Gemini and
// Gemma models. All available models are listed here: https://cloud.google.com/model-garden?hl=en

import {
  GenerativeModel,
  HarmBlockThreshold,
  HarmCategory,
  ModelParams,
  VertexAI,
  Schema,
  SchemaType,
} from "@google-cloud/vertexai";
import { Topic, Comment, isCommentType, isTopicType } from "../types";
import { Model } from "./model";

/**
 * Class to interact with models available through Google Cloud's Model Garden.
 */
export class VertexModel extends Model {
  private vertexAI: VertexAI;
  private baseModel: GenerativeModel;
  private topicLearningModel: GenerativeModel;
  private topicAndSubtopicLearningModel: GenerativeModel;
  private topicCategorizationModel: GenerativeModel;
  private topicAndSubtopicCategorizationModel: GenerativeModel;

  /**
   * Create a model object.
   * @param project - the Google Cloud Project ID, not the numberic project name
   * @param location - The Google Cloud Project location
   * @param modelName - the name of the model to connect with, like "gemini-1.5-pro-002", see the
   * full list here: https://cloud.google.com/model-garden?hl=en
   */
  constructor(project: string, location: string, modelName: string) {
    super();
    this.vertexAI = new VertexAI({
      project: project,
      location: location,
    });

    // Instantiate the models
    this.baseModel = this.vertexAI.getGenerativeModel(
      getModelSpec(modelName) // No schema for the base model
    );
    this.topicLearningModel = this.vertexAI.getGenerativeModel(
      getModelSpec(modelName, topicLearningSchema)
    );
    this.topicAndSubtopicLearningModel = this.vertexAI.getGenerativeModel(
      getModelSpec(modelName, topicAndSubtopicLearningSchema)
    );
    this.topicCategorizationModel = this.vertexAI.getGenerativeModel(
      getModelSpec(modelName, topicCategorizationSchema)
    );
    this.topicAndSubtopicCategorizationModel = this.vertexAI.getGenerativeModel(
      getModelSpec(modelName, topicAndSubtopicCategorizationSchema)
    );
  }

  /**
   * Send a request to the model
   * @param prompt the text including instructions and/or data to give the model
   * @param model the model specification to use, by default it is unconstrained
   * @returns the model response as a string
   */
  async executeRequest(prompt: string, model: GenerativeModel = this.baseModel): Promise<string> {
    const req = getRequest(prompt);
    const streamingResp = await model.generateContentStream(req);

    const response = await streamingResp.response;
    if (response.candidates![0].content.parts[0].text) {
      return response.candidates![0].content.parts[0].text;
    } else {
      console.warn("Malformed response: ", response);
      throw new Error("Error from Generative Model, response: " + response);
    }
  }

  /**
   * Generates topics and optionally subtopics based on a prompt.
   * @param prompt includes both model instructions and comments to find topics for.
   * @param includeSubtopics when true both Topics and Subtopics will be found.
   * @returns a list of topics that are present in the input.
   */
  async generateTopics(prompt: string, includeSubtopics: boolean): Promise<Topic[]> {
    const model = includeSubtopics ? this.topicAndSubtopicLearningModel : this.topicLearningModel;
    return generateTopicsWithModel(prompt, model);
  }

  /**
   *  Categorizes the comments based on the given Topics.
   * @param prompt includes both model instructions and comments to categorize.
   * @param includeSubtopics when true both Topics and Subtopics will be found.
   * @returns a list of Comments which all contain at least one associated Topic.
   */
  async generateComments(prompt: string, includeSubtopics: boolean): Promise<Comment[]> {
    const model = includeSubtopics
      ? this.topicAndSubtopicCategorizationModel
      : this.topicCategorizationModel;
    return generateCommentsWithModel(prompt, model);
  }
}

const safetySettings = [
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
];

// RESPONSE SCHEMAS
// For details see: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output
const topicLearningSchema: Schema = {
  type: SchemaType.ARRAY,
  items: {
    type: SchemaType.OBJECT,
    properties: {
      name: { type: SchemaType.STRING },
    },
  },
};

const topicAndSubtopicLearningSchema: Schema = {
  type: SchemaType.ARRAY,
  items: {
    type: SchemaType.OBJECT,
    properties: {
      name: { type: SchemaType.STRING },
      subtopics: {
        type: SchemaType.ARRAY,
        items: {
          type: SchemaType.OBJECT,
          properties: {
            name: { type: SchemaType.STRING },
          },
        },
      },
    },
  },
};

const topicCategorizationSchema: Schema = {
  type: SchemaType.ARRAY,
  items: {
    type: SchemaType.OBJECT,
    required: ["id", "topics"],
    properties: {
      id: { type: SchemaType.STRING }, // Comment id
      topics: {
        type: SchemaType.ARRAY,
        items: {
          type: SchemaType.OBJECT,
          required: ["name"],
          properties: {
            name: { type: SchemaType.STRING }, // Topic name
          },
        },
      },
    },
  },
};

const topicAndSubtopicCategorizationSchema: Schema = {
  type: SchemaType.ARRAY,
  items: {
    type: SchemaType.OBJECT,
    required: ["id", "topics"],
    properties: {
      id: { type: SchemaType.STRING }, // Comment id
      topics: {
        type: SchemaType.ARRAY,
        items: {
          type: SchemaType.OBJECT,
          required: ["name", "subtopics"],
          properties: {
            name: { type: SchemaType.STRING }, // Topic name
            subtopics: {
              type: SchemaType.ARRAY,
              items: {
                type: SchemaType.OBJECT,
                required: ["name"],
                properties: {
                  name: { type: SchemaType.STRING }, // Subtopic name
                },
              },
            },
          },
        },
      },
    },
  },
};

/**
 * Creates a model specification object for Vertex AI generative models.
 *
 * @param schema Optional. The JSON schema for the response. Only used if responseMimeType is 'application/json'.
 * @returns A model specification object ready to be used with vertex_ai.getGenerativeModel().
 */
function getModelSpec(modelName: string, schema?: Schema): ModelParams {
  const modelParams: ModelParams = {
    model: modelName,
    generationConfig: {
      // Param docs: http://cloud/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig
      maxOutputTokens: 8192,
      temperature: 0,
      topP: 0,
    },
    safetySettings: safetySettings,
  };

  if (schema && modelParams.generationConfig) {
    modelParams.generationConfig.responseMimeType = "application/json";
    modelParams.generationConfig.responseSchema = schema;
  }
  return modelParams;
}

// The maximum number of times an API call should be retried.
export const MAX_RETRIES = 3;
// How long in miliseconds to wait between API calls.
export const RETRY_DELAY_MS = 2000; // 2 seconds. TODO: figure out how to set it to zero for tests

function getRequest(prompt: string) {
  return {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
  };
}

/**
 * Helper function for tests. Categorizes the comments based on the given Topics.
 * @param prompt includes both model instructions and comments to categorize.
 * @param model what model to call for generation.
 * @returns a list of Comments which all contain at least one associated Topic.
 */
// TODO: Restrict access to this function. It is intended to only be available for testing. It can
// be made "protected" once it is a class method.
export async function generateCommentsWithModel(
  prompt: string,
  model: GenerativeModel
): Promise<Comment[]> {
  const response = await generateJSON(prompt, model);
  // TODO: reconsider this quick fix. The model is explicitly prompted to remove the text so that
  // the output is shorter.
  const comments = response.map((comment) => {
    comment.text = comment.text || "";
    return comment;
  });
  if (!comments.every((comment) => isCommentType(comment))) {
    // TODO: Add retry logic for this error.
    throw new Error("Model response comments are not all valid, response: " + response);
  }
  return comments;
}

/**
 * Helper function for tests. Generates topics and optionally subtopics based on a prompt.
 * @param prompt includes both model instructions and comments to find topics for.
 * @param model what model to call for generation.
 * @returns a list of topics that are present in the input.
 */
// TODO: Restrict access to this function. It is intended to only be available for testing. It can
// be made "protected" once it is a class method.
export async function generateTopicsWithModel(
  prompt: string,
  model: GenerativeModel
): Promise<Topic[]> {
  const response = await generateJSON(prompt, model);
  if (!response.every((topic: Topic) => isTopicType(topic))) {
    // TODO: Add retry logic for this error.
    throw new Error("Model response topics are not all valid, response: " + response);
  }
  return response;
}

/**
 * Utility function for sending a set of instructions to an LLM with comments,
 * and returning the results as an array of JSON. It includes retry logic to handle rate limit
 * errors.
 *
 * @param instructions The instructions for the LLM on how to process the comments.
 * @param prompt The instructions for the LLM on how to process the comments.
 * @returns A Promise that resolves with the LLM's output parsed as a JSON object.
 * @throws An error if the LLM's response is malformed or if there's an error during processing.
 */
// TODO: Restrict access to this function. It is intended to only be available for testing. It can
// be made "protected" once it is a class method.
// eslint-disable-next-line  @typescript-eslint/no-explicit-any
export async function generateJSON(prompt: string, model: GenerativeModel): Promise<any[]> {
  const req = getRequest(prompt);
  let streamingResp;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      streamingResp = await model.generateContentStream(req);
      break; // Exit loop if successful
    } catch (error) {
      if (
        error instanceof Error &&
        (error.message.includes("429 Too Many Requests") ||
          error.message.includes("RESOURCE_EXHAUSTED")) &&
        attempt < MAX_RETRIES
      ) {
        console.warn(
          `Rate limit error, attempt ${attempt}. Retrying in ${RETRY_DELAY_MS / 1000} seconds...`
        );
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
      } else {
        console.error("Error during generateJSON:", error);
        throw error;
      }
    }
  }

  if (!streamingResp) {
    throw new Error("Failed to get a model response after multiple retries.");
  }

  const response = await streamingResp.response;

  if (response.candidates![0].content.parts[0].text) {
    const responseText = response.candidates![0].content.parts[0].text;
    const generatedJSON = JSON.parse(responseText);
    if (!Array.isArray(generatedJSON)) {
      throw new Error("Model response is not a list: " + response);
    }
    return generatedJSON;
  } else {
    console.warn("Malformed response: ", response);
    throw new Error("Error from Generative Model, response: " + response);
  }
}
