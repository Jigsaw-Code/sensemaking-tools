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

import { Topic, Comment } from "../types";

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
  public readonly categorizationBatchSize = 100;

  /**
   * A general call to the given model.
   * @param prompt - what information and instructions are given to the model.
   * @returns the model response
   */
  abstract executeRequest(prompt: string): Promise<string>;

  /**
   * Defines topics based on the information in the given prompt.
   * @param prompt - the instructions and data to categorize into topics
   * @param includeSubtopics - whether to include another layer of categorization under topics
   * @returns the topics found in the given instructions
   */
  abstract generateTopics(prompt: string, includeSubtopics: boolean): Promise<Topic[]>;

  /**
   * Categorizes the data in the prompt into the given topics.
   * @param prompt - includes instructions, comment data, and the topics to categorize the
   * comments into.
   * @param includeSubtopics - whether to include another layer of categorization under topics
   * @returns the given comments sorted into the given topics and optionally subtopics.
   */
  abstract generateComments(prompt: string, includeSubtopics: boolean): Promise<Comment[]>;
}
