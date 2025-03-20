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

import pLimit from "p-limit";
import {
  GenerativeModel,
  HarmBlockThreshold,
  HarmCategory,
  ModelParams,
  Schema,
  VertexAI,
} from "@google-cloud/vertexai";
import { Model } from "./model";
import { checkDataSchema } from "../types";
import { Static, TSchema } from "@sinclair/typebox";
import { retryCall } from "../sensemaker_utils";
import { MAX_RETRIES, RETRY_DELAY_MS, SUMMARIZATION_VERTEX_PARALLELISM } from "./model_util";

/**
 * Class to interact with models available through Google Cloud's Model Garden.
 */
export class VertexModel extends Model {
  private vertexAI: VertexAI;
  private modelName: string;
  private limit: pLimit.Limit; // controls model calls concurrency on model's instance level

  /**
   * Create a model object.
   * @param project - the Google Cloud Project ID, not the numberic project name
   * @param location - The Google Cloud Project location
   * @param modelName - the name of the model from Vertex AI's Model Garden to connect with, see
   * the full list here: https://cloud.google.com/model-garden
   */
  constructor(project: string, location: string, modelName: string = "gemini-1.5-pro-002") {
    super();
    this.vertexAI = new VertexAI({
      project: project,
      location: location,
    });
    this.modelName = modelName;

    console.log(
      "Creating VertexModel with ",
      SUMMARIZATION_VERTEX_PARALLELISM,
      " parallel workers..."
    );
    this.limit = pLimit(SUMMARIZATION_VERTEX_PARALLELISM);
  }

  /**
   * Get generative model corresponding to structured data output specification as a JSON Schema specification.
   */
  getGenerativeModel(schema?: TSchema): GenerativeModel {
    return this.vertexAI.getGenerativeModel(getModelParams(this.modelName, schema));
  }

  /**
   * Generate text based on the given prompt.
   * @param prompt the text including instructions and/or data to give the model
   * @returns the model response as a string
   */
  async generateText(prompt: string): Promise<string> {
    return await this.callLLM(prompt, this.getGenerativeModel());
  }

  /**
   * Generate structured data based on the given prompt.
   * @param prompt the text including instructions and/or data to give the model
   * @param schema a JSON Schema specification (generated from TypeBox)
   * @returns the model response as data structured according to the JSON Schema specification
   */
  async generateData(prompt: string, schema: TSchema): Promise<Static<typeof schema>> {
    const response = JSON.parse(await this.callLLM(prompt, this.getGenerativeModel(schema)));
    if (!checkDataSchema(schema, response)) {
      // TODO: Add retry logic for this error.
      throw new Error("Model response does not match schema: " + response);
    }
    return response;
  }

  /**
   * Calls an LLM to generate text based on a given prompt and handles rate limiting, response validation and retries.
   *
   * @param prompt - The text prompt to send to the language model.
   * @param model - The specific language model that will be called.
   * @returns A Promise that resolves with the text generated by the language model.
   */
  async callLLM(prompt: string, model: GenerativeModel): Promise<string> {
    const req = getRequest(prompt);

    // Wrap the entire retryCall sequence with the `p-limit` limiter,
    // so we don't let other calls to start until we're done with the current one
    // (in case it's failing with rate limits error and needs to be waited on and retried first)
    const rateLimitedCall = () =>
      this.limit(async () => {
        return await retryCall(
          // call LLM
          async function () {
            console.log("Calling LLM...");
            return (await model.generateContentStream(req)).response;
          },
          // Check if the response exists and contains a text field.
          function (response): boolean {
            if (!response) {
              console.error("Failed to get a model response.");
              return false;
            }
            if (!response.candidates![0].content.parts[0].text) {
              console.error(`Model returned a malformed response: ${response}`);
              return false;
            }
            console.log(`Input token count: ${response.usageMetadata?.promptTokenCount}`);
            console.log(`Output token count: ${response.usageMetadata?.candidatesTokenCount}`);
            return true;
          },
          MAX_RETRIES,
          "Failed to get a valid model response.",
          RETRY_DELAY_MS,
          [], // Arguments for the LLM call
          [] // Arguments for the validator function
        );
      });

    const response = await rateLimitedCall();
    return response.candidates![0].content.parts[0].text!;
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

/**
 * Creates a model specification object for Vertex AI generative models.
 *
 * @param schema Optional. The JSON schema for the response. Only used if responseMimeType is 'application/json'.
 * @returns A model specification object ready to be used with vertex_ai.getGenerativeModel().
 */
function getModelParams(modelName: string, schema?: Schema): ModelParams {
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

type Request = {
  contents: {
    role: string;
    parts: { text: string }[];
  }[];
};
function getRequest(prompt: string): Request {
  return {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
  };
}
