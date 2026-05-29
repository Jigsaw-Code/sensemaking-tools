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

from src.models.genai_model import GenaiModel
from src.models import custom_types
from src import runner_utils
from src import sensemaker_utils
from src import prompts
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
  prompt_jobs = []
  chunk_count = len(chunks)
  for i, chunk in enumerate(chunks):
    prompt_str = sensemaker_utils.get_prompt(
        instructions, chunk, additional_context
    )
    prompt_jobs.append({
        "prompt": prompt_str,
        "chunk": f"topic_chunk_{i}/{chunk_count}",
        "response_schema": schema_to_expect,
        "log_prefix_marker": "1 (Topic Identification)",
    })

  def parser(resp, job):
    return parse_response(resp["text"], job["response_schema"])

  results_df, _, _, _ = await model.process_prompts_concurrently(
      prompt_jobs, response_parser=parser
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
  merge_instructions = prompts.topic_modeling_merge_topics_prompt

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
  prompt_jobs = []
  chunk_count = len(chunks)
  for i, chunk in enumerate(chunks):
    prompt_str = sensemaker_utils.get_prompt(
        instructions, chunk, additional_context
    )
    prompt_jobs.append({
        "prompt": prompt_str,
        "chunk": f"opinion_chunk_{i}/{chunk_count}",
        "response_schema": schema_to_expect,
        "log_prefix_marker": "4 (Opinion Identification)",
    })

  def parser(resp, job):
    return parse_response(resp["text"], job["response_schema"])

  results_df, _, _, _ = await model.process_prompts_concurrently(
      prompt_jobs, response_parser=parser
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



async def merge_opinions(
    model: GenaiModel,
    partial_results: List[custom_types.NestedTopic],
    schema_to_expect: any,
    parent_topic: custom_types.Topic,
    additional_context: Optional[str] = None,
) -> custom_types.NestedTopic:
  """Merges a list of opinions into a consolidated list."""
  logging.info("Merging opinion results from all chunks.")
  merge_instructions = prompts.get_topic_modeling_merge_opinions_prompt(parent_topic.name)

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
