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

// Module to interact with sensemaking tools.

import { Comment, SummarizationType, Summary, Topic } from "./types";
import { categorizeCommentsRecursive } from "./tasks/categorization";
import { summarizeByType } from "./tasks/summarization";
import { ModelSettings, Model } from "./models/model";
import { getUniqueTopics } from "./sensemaker_utils";
import { SupportedLanguage, translateSummary } from "../templates/l10n";

// Class to make sense of conversation data. Uses LLMs to learn what topics were discussed and
// categorize comments. Then these categorized comments can be used with optional Vote data to
// summarize a conversation.
export class Sensemaker {
  private modelSettings: ModelSettings;

  /**
   * Creates a Sensemaker object
   * @param modelSettings what models to use for what tasks, a default model can be set.
   */
  constructor(modelSettings: ModelSettings) {
    this.modelSettings = modelSettings;
  }

  /**
   * Get corresponding model from modelSettings object, or defaultModel if none specified.
   * @param modelSetting the key of the modelSettings options you want the Model for (corresponding to task)
   * @return The model to use for the corresponding ModelSetting key
   */
  getModel(modelSetting: keyof ModelSettings): Model {
    // Consider getting rid of this function once we have non default model
    // implementations, in case we want to switch to a static compilation of the correct model for each key.
    return this.modelSettings[modelSetting] || this.modelSettings.defaultModel;
  }

  /**
   * Generates a conversation summary, optionally incorporating vote data.
   *
   * It offers flexibility in how topics for the summary are determined:
   * 1. Categorized Comments: If the input `comments` are already categorized (i.e., they have a
   *    `topics` property), those topics are used directly for the summary structure.
   * 2. Provided Topics:  If `topics` are explicitly provided, they are used to categorize the
   *    comments before summarization. This ensures the summary has statistics based on the
   *    specified topics (like comments count per topic).
   * 3. Learned Topics: If neither categorized comments nor explicit topics are provided, the
   *    function will automatically learn topics from the comments using an LLM. This is the most
   *    automated option but requires more processing time.
   *
   * The function supports different summarization types (e.g., basic summarization,
   * vote-tally-based summarization), and allows for additional instructions to guide the
   * summarization process. The generated summary is then grounded in the original comments to
   * ensure accuracy and relevance.
   *
   * @param comments An array of `Comment` objects representing the public conversation comments. If
   *  these comments are already categorized (have a `topics` property), the summarization will be
   *  based on those existing categories.
   * @param summarizationType  The type of summarization to perform (e.g.,
   *  `SummarizationType.GROUP_INFORMED_CONSENSUS`).
   * @param topics  An optional array of `Topic` objects. If provided, these topics will be used for
   *  comment categorization before summarization, ensuring that the summary addresses the specified
   *  topics. If `comments` are already categorized, this parameter is ignored.
   * @param additionalContext Optional additional context to provide to the LLM for
   *  summarization. The context will be appended verbatim to the summarization prompt. This
   * should be 1-2 sentences on what the conversation is about and where it takes place.
   * @returns A Promise that resolves to a `Summary` object, containing the generated summary text
   *  and metadata.
   */
  public async summarize(
    comments: Comment[],
    summarizationType: SummarizationType = SummarizationType.AGGREGATE_VOTE,
    topics?: Topic[],
    additionalContext?: string,
    output_lang: SupportedLanguage = "en"
  ): Promise<Summary> {
    const startTime = performance.now();

    // Categories are required for summarization, this is a no-op if they already have categories.
    comments = await this.categorizeComments(comments, true, topics, additionalContext, 2, output_lang);

    const summary = await summarizeByType(
      this.getModel("summarizationModel"),
      comments,
      summarizationType,
      additionalContext,
      output_lang
    );

    console.log(`[DEBUG] summary: ${JSON.stringify(summary)}`);

    // TODO: translate the summary's "Other" to the output language
    const translatedSummary = translateSummary(summary, output_lang);
    
    console.log(`Summarization took ${(performance.now() - startTime) / (1000 * 60)} minutes.`);
    return translatedSummary;
  }

  /**
   * Extracts topics from the comments using a LLM on Vertex AI. Retries if the LLM response is invalid.
   * @param comments The comments data for topic modeling
   * @param includeSubtopics Whether to include subtopics in the topic modeling
   * @param topics Optional. The user provided top-level topics, if these are specified only
   * subtopics will be learned.
   * @param additionalContext Optional additional context to provide to the LLM for
   *  topic learning. The context will be appended verbatim to the prompt. This
   * should be 1-2 sentences on what the conversation is about and where it takes place.
   * @param topicDepth how many levels of topics to learn, from topic to sub-sub-topic
   * @returns: Topics (optionally containing subtopics) representing what is discussed in the
   * comments.
   */
  public async learnTopics(
    comments: Comment[],
    includeSubtopics: boolean,
    topics?: Topic[],
    additionalContext?: string,
    topicDepth?: 1 | 2 | 3,
    output_lang: SupportedLanguage = "en"
  ): Promise<Topic[]> {
    const startTime = performance.now();

    // Categorization learns one level of topics and categorizes them and repeats recursively. We want
    // to use this logic here as well, so just categorize the comments and take only the learned
    // topics.
    const categorizedComments = await this.categorizeComments(
      comments,
      includeSubtopics,
      topics,
      additionalContext,
      topicDepth,
      output_lang
    );
    const learnedTopics = getUniqueTopics(categorizedComments);

    console.log(`Topic learning took ${(performance.now() - startTime) / (1000 * 60)} minutes.`);

    return learnedTopics;
  }

  /**
   * Categorize the comments by topics using a LLM on Vertex.
   * @param comments The data to summarize
   * @param includeSubtopics Whether to include subtopics in the categorization.
   * @param topics The user provided topics (and optionally subtopics).
   * @param additionalContext Optional additional context to provide to the LLM for
   * categorization. The context will be appended verbatim to the prompt. This
   * should be 1-2 sentences on what the conversation is about and where it takes place.
   * @param topicDepth how many levels of topics to learn, from topic to sub-sub-topic
   * @returns: The LLM's categorization.
   */
  public async categorizeComments(
    comments: Comment[],
    includeSubtopics: boolean,
    topics?: Topic[],
    additionalContext?: string,
    topicDepth?: 1 | 2 | 3,
    output_lang: SupportedLanguage = "en"
  ): Promise<Comment[]> {
    const startTime = performance.now();
    if (!includeSubtopics && topicDepth && topicDepth > 1) {
      throw Error("topicDepth can only be set when includeSubtopics is true");
    }

    // TODO: ensure the topics argument and the topics assigned to the passed in comments are in
    // sync.
    const categorizedComments = await categorizeCommentsRecursive(
      comments,
      includeSubtopics ? topicDepth || 2 : 1,
      this.getModel("categorizationModel"),
      topics,
      additionalContext,
      output_lang
    );

    console.log(`Categorization took ${(performance.now() - startTime) / (1000 * 60)} minutes.`);
    return categorizedComments;
  }
}
