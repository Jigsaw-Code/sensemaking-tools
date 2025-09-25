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

// Abstract class to interact with LLMs. Different implementations that call different LLM APIs
// will inherit this class and provide a concrete implementations that follow this structure. Then
// different models and model providers can be easily swapped in and out.

import { TSchema, type Static } from "@sinclair/typebox";

// Specify which model will be called for different tasks. The tradeoff between speed and quality
// may be different for different modeling tasks.
export interface ModelSettings {
  defaultModel: Model;
  summarizationModel?: Model;
  categorizationModel?: Model;
  groundingModel?: Model;
}

// An abstract base class that defines how to interact with models.
export abstract class Model {
  // The best batch size to use for categorization.
  public readonly categorizationBatchSize: number;

  constructor(batchSize: number = 100) {
    this.categorizationBatchSize = batchSize;
  }

  /**
   * Abstract method for generating a text response based on the given prompt.
   * @param prompt - the instructions and data to process as a prompt
   * @param output_lang - the output language for the response
   * @returns the model response
   */
  abstract generateText(prompt: string, output_lang?: string): Promise<string>;

  /**
   * Abstract method for generating structured data based on the given prompt.
   * @param prompt - the instructions and data to process as a prompt
   * @param schema - the schema to use for the structured data
   * @param output_lang - the output language for the response
   * @returns the model response
   */
  abstract generateData(prompt: string, schema: TSchema, output_lang?: string): Promise<Static<typeof schema>>;
}
