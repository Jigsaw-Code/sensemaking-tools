# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, cast

from case_studies.wtp.models.genai_model import GenaiModel
from case_studies.wtp.models import custom_types
from case_studies.wtp import runner_utils
from case_studies.wtp import sensemaker_utils
from pydantic import TypeAdapter, ValidationError

# Target a safe estimated threshold (e.g., 400k tokens) to close a chunk.
# Gemini 1.5 Pro has 1M-2M context window, but we want to be safe.
TARGET_CHUNK_TOKEN_COUNT = 200000
MAX_CHUNK_TOKEN_COUNT = 1000000  # 1M limit


def parse_response(response_text: str, schema_to_expect: Any) -> Any:
  """Parses the LLM response text into the expected schema using TypeAdapter.

  Args:
      response_text: The response text from the LLM.
      schema_to_expect: The schema to expect for the response.

  Returns:
      The parsed response.
  """
  # Handle case where response_text is already a structured object (dict or list)
  if not isinstance(response_text, str):
    try:
      return TypeAdapter(schema_to_expect).validate_python(response_text)
    except ValidationError as e:
      logging.debug(
          "Model response failed Pydantic validation for schema"
          f" {schema_to_expect}: {response_text}.\nError: {e}"
      )
      raise ValueError(f"Model response failed Pydantic validation") from e

  json_text = response_text

  # Drop markdown code block delimiters if present
  if json_text.endswith("```"):
    json_text = json_text.removesuffix("```")

  # Find the first occurrence of '{' or '[', and drop everything in front of it as not expected.
  match = re.search(r"[\[{]", json_text)
  if match:
    first_brace_index = match.start()
    json_text = json_text[first_brace_index:]

  try:
    return TypeAdapter(schema_to_expect).validate_json(json_text)
  except ValidationError as e:
    logging.debug(
        "Model response failed Pydantic validation for schema"
        f" {schema_to_expect}: {json_text}.\nError: {e}"
    )
    raise ValueError(f"Model response failed Pydantic validation") from e
  except json.JSONDecodeError as e:
    logging.debug(f"Model returned invalid JSON: {json_text}.")
    raise ValueError(f"Model returned invalid JSON") from e
  except Exception as e:
    logging.error(
        "Failed to parse or validate model response against schema"
        f" {schema_to_expect}: {json_text}.\nError: {e}"
    )
    raise ValueError(
        f"Failed to parse or validate model response: {json_text}."
    ) from e


async def create_chunks(
    model: GenaiModel,
    instructions: str,
    prompt_input_data: List[str],
    additional_context: Optional[str] = None,
) -> List[List[str]]:
  """Creates chunks of data based on token count."""
  chunks = []
  current_chunk = []
  current_chunk_estimated_tokens = 0

  # Estimate base tokens (instructions + context)
  base_prompt = sensemaker_utils.get_prompt(
      instructions, [], additional_context
  )
  base_tokens = runner_utils.estimate_tokens(base_prompt)

  current_chunk_estimated_tokens = base_tokens

  total_items = len(prompt_input_data)
  for i, item in enumerate(prompt_input_data):
    item_tokens = (
        runner_utils.estimate_tokens(item) + 5
    )  # +5 for XML tags overhead

    if current_chunk_estimated_tokens + item_tokens > TARGET_CHUNK_TOKEN_COUNT:
      # Check actual token count before closing chunk
      prompt_str = sensemaker_utils.get_prompt(
          instructions, current_chunk, additional_context
      )
      chunks.append(current_chunk)
      current_chunk = []
      current_chunk_estimated_tokens = base_tokens

    current_chunk.append(item)
    current_chunk_estimated_tokens += item_tokens

  if current_chunk:
    chunks.append(current_chunk)

  return chunks


async def generate_topics_with_chunking(
    model: GenaiModel,
    instructions: str,
    prompt_input_data: List[str],
    schema_to_expect: any,
    additional_context: Optional[str] = None,
    chunks: Optional[List[List[str]]] = None,
) -> custom_types.FlatTopicList:
  """Generates topics from a list of statements, handling token limits via chunking and merging."""

  if chunks is None:
    chunks = await create_chunks(
        model, instructions, prompt_input_data, additional_context
    )

  if not chunks:
    return custom_types.FlatTopicList(topics=[])

  # Process chunks concurrently
  prompts = []
  chunk_count = len(chunks)
  for i, chunk in enumerate(chunks):
    prompt_str = sensemaker_utils.get_prompt(
        instructions, chunk, additional_context
    )
    prompts.append({
        "prompt": prompt_str,
        "chunk": f"topic_chunk_{i}/{chunk_count}",
        "response_schema": schema_to_expect,
        "log_prefix_marker": "1",
    })

  def parser(resp, job):
    return parse_response(resp["text"], job["response_schema"])

  results_df, _ = await model.process_prompts_concurrently(
      prompts, response_parser=parser
  )

  partial_results = []
  for result in results_df["result"]:
    if result is None or (isinstance(result, dict) and result.get("error")):
      logging.warning(f"Skipping failed chunk result: {result}")
      continue

    if isinstance(result, list):
      partial_results.extend(result)
    elif isinstance(result, custom_types.FlatTopicList):
      partial_results.extend(result.topics)
    else:
      logging.warning(
          f"Unexpected result type in chunk processing: {type(result)}"
      )

  return await _merge_topics(
      model, partial_results, schema_to_expect, additional_context
  )


async def _merge_topics(
    model: GenaiModel,
    partial_results: List[custom_types.FlatTopic],
    schema_to_expect: any,
    additional_context: Optional[str] = None,
) -> custom_types.FlatTopicList:
  """Merges a list of topics into a consolidated list."""
  logging.info("Merging results from all chunks.")
  merge_instructions = f"""
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
You have been provided with multiple lists of topics that were generated by analyzing different chunks of the same dataset.
Your task is to synthesize and consolidate these lists by merging similar topics, and generate a final, deduplicated list of topics.

 ### **Consolidation Criteria**

 *   **Merge Duplicates:** Identify and merge topics that are semantically identical or highly similar.
 *   **Ensure Distinctness:** The final topics should be meaningfully different and cover separate conceptual areas.
 *   **Substantive:** Topics should contain multiple, distinct opinions. Avoid single-opinion topics.
 *   **Subject Linkage:** Topic names should be phrased as aspects of the main subject. E.g. "Freedom and Equality" subject could have the following topics: "Defining Freedom", "Defining Equality", "Barriers to Freedom and Equality", "Society's Role in Freedom and Equality", "The Individual's Role in Freedom and Equality", etc.
 *   **Maintain Consistency:** Ensure all topics in the final list are at a similar level of abstraction.
 *   **Efficiency:** Keep the number of topics as low as possible. Actively consolidate topics when their content can be logically grouped.
 *   **Concise:** The topic name should be concise.

RESPONSE STRUCTURE:
Respond with a single consolidated list of the identified topics only, nothing else.
The response should be in JSON format, that can be parse into the following class:
class FlatTopicList:
    topics: List[Topic]
class Topic:
    name: str

Do not include markdown code blocks around the JSON response, such as ```json or ```
Response example:
{{"topics": [{{"name": "Topic 1"}}, {{"name": "Topic 2"}}]}}
"""

  combined_topics = [topic.name for topic in partial_results]

  prompt_str = sensemaker_utils.get_prompt(
      merge_instructions, combined_topics, additional_context
  )

  resp = await model.call_gemini(
      prompt=prompt_str,
      run_name="merge_topics",
      response_schema=schema_to_expect,
  )

  if resp.get("error"):
    raise ValueError(f"Error calling Gemini for merge: {resp.get('error')}")

  final_response = parse_response(resp["text"], schema_to_expect)

  if isinstance(final_response, custom_types.FlatTopicList):
    return final_response

  if isinstance(final_response, list):
    return custom_types.FlatTopicList(topics=final_response)

  logging.error(
      f"LLM response for merge is not a FlatTopicList: {final_response}"
  )
  return custom_types.FlatTopicList(topics=[])


async def generate_opinions_with_chunking(
    model: GenaiModel,
    instructions: str,
    prompt_input_data: List[str],
    schema_to_expect: any,
    parent_topic: custom_types.Topic,
    additional_context: Optional[str] = None,
    chunks: Optional[List[List[str]]] = None,
) -> custom_types.NestedTopic:
  """Generates opinions from a list of statements, handling token limits via chunking and merging."""

  if chunks is None:
    chunks = await create_chunks(
        model, instructions, prompt_input_data, additional_context
    )

  if not chunks:
    return custom_types.NestedTopic(name=parent_topic.name, subtopics=[])

  # Chunking
  prompts = []
  chunk_count = len(chunks)
  for i, chunk in enumerate(chunks):
    prompt_str = sensemaker_utils.get_prompt(
        instructions, chunk, additional_context
    )
    prompts.append({
        "prompt": prompt_str,
        "chunk": f"opinion_chunk_{i}/{chunk_count}",
        "response_schema": schema_to_expect,
        "log_prefix_marker": "4",
    })

  def parser(resp, job):
    return parse_response(resp["text"], job["response_schema"])

  results_df, _ = await model.process_prompts_concurrently(
      prompts, response_parser=parser
  )

  partial_results_list = []
  for result in results_df["result"]:
    if result is None or (isinstance(result, dict) and result.get("error")):
      logging.warning(f"Skipping failed opinion chunk result: {result}")
      continue

    if isinstance(result, custom_types.NestedTopic):
      partial_results_list.append(result)
    else:
      logging.warning(
          f"Unexpected result type in opinion chunk processing: {type(result)}"
      )

  return await merge_opinions(
      model,
      partial_results_list,
      schema_to_expect,
      parent_topic,
      additional_context,
  )


def merge_opinions_prompt(parent_topic: custom_types.Topic) -> str:
  return f"""
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
You have been provided with multiple lists of opinions for the topic "{parent_topic.name}" that were generated by analyzing different chunks of the same dataset.
Your task is to synthesize and consolidate these lists by merging similar opinions, and generate a final, deduplicated list of opinions.

### **Consolidation Criteria**

  * **Distinct:** Within a single topic, opinions should represent unique viewpoints. Opinions should be meaningfully distinct and different within the topic.
  * **Substantive:** Opinions should be substantive (i.e. not single-quote opinions).
  * **Accurate:** The text of opinion should be a clear and well-phrased summary of the underlying quotes.
  * **Concise:** The opinions should be concise.
  * **Topic and Subject Linkage:** Opinions should be phrased to have a clear link to the overarching topic and the main subject of the survey.
  * **Coherency:** Opinions should be logically consistent and easy to follow as you go over the list. E.g. for "Barriers to Freedom and Equality" topic, all the opinions could start with: "A key barrier to freedom and equality is "; for "Defining Equality" topic: "Equality is ", etc.
  * **Merge Overlaps:** If two or more opinions express the same fundamental idea, they **must be merged**.
  * **Efficiency:** Keep the number of opinions as low as possible. Actively consolidate opinions when their content can be logically grouped.

When creating opinions, keep in mind that on later stages extracted quotes will be categorized into the identified opinions.
For that, the quote must holistically match the entire opinion. A partial match, where the quote only supports one piece of the opinion, is not sufficient.
To be a match, the quote must explicitly support every key concept within the opinion.
So avoid creating opinions that will be hard to completely match to quotes.

RESPONSE STRUCTURE:
Respond only with the identified opinions, where top level is the overarching topic, and opinions are subtopics.
The response should be in JSON format, that can be parse into the following class:
class Topic:
    name: str # This will be the overarching topic
    subtopics: List[Topic] # Where subtopic is class Topic {{{{ name: str }}}} (the opinions)

Do not include markdown code blocks around the JSON response, such as ```json or ```
For example:
{{
  "name": "{parent_topic.name}",
  "subtopics": [
      {{ "name": "Opinion 1" }},
      {{ "name": "Opinion 2" }}
  ]
}}
"""


async def merge_opinions(
    model: GenaiModel,
    partial_results: List[custom_types.NestedTopic],
    schema_to_expect: any,
    parent_topic: custom_types.Topic,
    additional_context: Optional[str] = None,
) -> custom_types.NestedTopic:
  """Merges a list of opinions into a consolidated list."""
  logging.info("Merging opinion results from all chunks.")
  merge_instructions = merge_opinions_prompt(parent_topic)

  combined_opinions: List[str] = []
  for nested_topic in partial_results:
    if nested_topic and nested_topic.subtopics:
      for subtopic in nested_topic.subtopics:
        combined_opinions.append(subtopic.name)

  if not combined_opinions:
    logging.warning("No opinions found in partial results to merge.")
    return custom_types.NestedTopic(name=parent_topic.name, subtopics=[])

  prompt_str = sensemaker_utils.get_prompt(
      merge_instructions, combined_opinions, additional_context
  )

  resp = await model.call_gemini(
      prompt=prompt_str,
      run_name="merge_opinions",
      response_schema=schema_to_expect,
      max_concurrent_calls=20,
  )

  if resp.get("error"):
    logging.error(
        f"Error calling Gemini for opinion merge: {resp.get('error')}"
    )
    return custom_types.NestedTopic(name=parent_topic.name, subtopics=[])

  final_response = parse_response(resp["text"], schema_to_expect)

  if isinstance(final_response, custom_types.NestedTopic):
    return final_response

  logging.error(
      f"LLM response for opinion merge is not a NestedTopic: {final_response}"
  )
  return custom_types.NestedTopic(name=parent_topic.name, subtopics=[])
