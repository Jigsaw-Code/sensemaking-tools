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
  RequestOptions,
  Schema,
  VertexAI,
} from "@google-cloud/vertexai";
import { Model } from "./model";
import { checkDataSchema } from "../types";
import { Static, TSchema } from "@sinclair/typebox";
import { retryCall } from "../sensemaker_utils";
import { RETRY_DELAY_MS, DEFAULT_VERTEX_PARALLELISM, MAX_LLM_RETRIES } from "./model_util";

// Import localization system
import { getLanguagePrefix, type SupportedLanguage } from "../../templates/l10n";

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
  constructor(
    project: string,
    location: string,
    modelName: string = "gemini-2.5-pro-preview-06-05"
  ) {
    super();
    if (modelName === "gemini-2.5-pro-preview-06-05" && location != "global") {
      throw Error(
        "Only the 'global' location is supported for the 'gemini-2.5-pro-preview-06-05' model. See " +
          "https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro for more information."
      );
    }
    this.vertexAI = new VertexAI({
      project: project,
      location: location,
      apiEndpoint: "aiplatform.googleapis.com",
    });
    this.modelName = modelName;

    console.log("Creating VertexModel with ", DEFAULT_VERTEX_PARALLELISM, " parallel workers...");
    this.limit = pLimit(DEFAULT_VERTEX_PARALLELISM);
  }

  /**
   * Get generative model corresponding to structured data output specification as a JSON Schema specification.
   */
  getGenerativeModel(schema?: TSchema): GenerativeModel {
    const requestOptions: RequestOptions = { timeout: 150000 * 4 }; // 10 min (overrides current default of 5 mins)
    return this.vertexAI.getGenerativeModel(getModelParams(this.modelName, schema), requestOptions);
  }

  /**
   * Generate text based on the given prompt.
   * @param prompt the text including instructions and/or data to give the model
   * @param output_lang the output language for the response
   * @returns the model response as a string
   */
  async generateText(prompt: string, output_lang: SupportedLanguage = "en"): Promise<string> {
    return await this.callLLM(prompt, this.getGenerativeModel(), undefined, output_lang);
  }

  /**
   * Generate structured data based on the given prompt.
   * @param prompt the text including instructions and/or data to give the model
   * @param schema a JSON Schema specification (generated from TypeBox)
   * @param output_lang the output language for the response
   * @returns the model response as data structured according to the JSON Schema specification
   */
  async generateData(prompt: string, schema: TSchema, output_lang: SupportedLanguage = "en"): Promise<Static<typeof schema>> {
    const validateResponse = (response: string): boolean => {
      let parsedResponse;
      try {
        parsedResponse = JSON.parse(response);
      } catch (e) {
        console.error(`Model returned a non-JSON response:\n${response}\n${e}`);
        return false;
      }
      if (!checkDataSchema(schema, parsedResponse)) {
        console.error("Model response does not match schema: " + response);
        return false;
      }

      return true;
    };
    
    return JSON.parse(
      await this.callLLM(prompt, this.getGenerativeModel(schema), validateResponse, output_lang)
    );
  }

  /**
   * Calls an LLM to generate text based on a given prompt and handles rate limiting, response validation and retries.
   *
   * Concurrency: To take advantage of concurrent execution, invoke this function as a batch of callbacks,
   * and pass it to the `executeConcurrently` function. It will run multiple `callLLM` functions concurrently,
   * up to the limit set by `p-limit` in `VertexModel`'s constructor.
   *
   * @param prompt - The text prompt to send to the language model.
   * @param model - The generative model to use.
   * @param validator - optional check for the model response.
   * @param output_lang - the output language for the response
   * @returns A Promise that resolves with the text generated by the language model.
   */
  async callLLM(
    prompt: string,
    model: GenerativeModel,
    validator: (response: string) => boolean = () => true,
    output_lang: SupportedLanguage = "en"
  ): Promise<string> {
    // Get language prefix from localization system
    const languagePrefix = getLanguagePrefix(output_lang);
    
    const req = getRequest(languagePrefix, prompt);

    // Wrap the entire retryCall sequence with the `p-limit` limiter,
    // so we don't let other calls to start until we're done with the current one
    // (in case it's failing with rate limits error and needs to be waited on and retried first)
    const rateLimitedCall = () =>
      this.limit(async () => {
        return await retryCall(
          // call LLM
          async function () {
            return (await model.generateContentStream(req)).response;
          },
          // Check if the response exists and contains a text field.
          function (response): boolean {
            if (!response) {
              console.error("Failed to get a model response.");
              return false;
            }
            const responseText = response.candidates![0].content.parts[0].text;
            if (!responseText) {
              console.error(`Model returned an empty response:`, response.candidates![0].content);
              return false;
            }
            if (!validator(responseText)) {
              return false;
            }
            console.log(
              `✓ Completed LLM call (input: ${response.usageMetadata?.promptTokenCount} tokens, output: ${response.usageMetadata?.candidatesTokenCount} tokens)`
            );
            return true;
          },
          MAX_LLM_RETRIES,
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
function getRequest(languagePrefix: string, prompt: string): Request {
  return {
    contents: [{
      role: "system",
        parts: [{ text: languagePrefix }],
      }, {
       role: "user", parts: [{ text: prompt }] }],
  };
}
