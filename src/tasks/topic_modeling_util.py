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
from collections import Counter
from typing import Any, Dict, List, Optional, cast

from src.models.genai_model import GenaiModel
from src.models import custom_types
from src import run_health
from src import runner_utils
from src import sensemaker_utils
from src import prompts
from pydantic import TypeAdapter, ValidationError

# Target a safe estimated threshold (e.g., 400k tokens) to close a chunk.
# Gemini 1.5 Pro has 1M-2M context window, but we want to be safe.
TARGET_CHUNK_TOKEN_COUNT = 2000
MAX_CHUNK_TOKEN_COUNT = 1000000  # 1M limit

_PLACEHOLDER_TOPIC_RE = re.compile(
    r"^(?:Topic|Opinion|Subtopic|Category)\s*\d+$", re.IGNORECASE
)
_PREFIX_TOPIC_RE = re.compile(
    r"^(?:Topic|Opinion|Subtopic|Category)\s*\d+\s*:\s*", re.IGNORECASE
)


def is_placeholder_topic_name(name: str) -> bool:
  """Returns True if `name` is a generic placeholder like 'Topic 1' or 'Opinion 2'."""
  cleaned = name.strip()
  return not cleaned or bool(_PLACEHOLDER_TOPIC_RE.match(cleaned))


def clean_topic_name(name: str) -> str:
  """Strips 'Topic N:' / 'Opinion N:' prefixes, truncates runaway strings, and strips punctuation."""
  cleaned = _PREFIX_TOPIC_RE.sub("", name.strip())
  cleaned = re.sub(r"\s+", " ", cleaned).strip()
  if cleaned.count(" & ") >= 3:
    parts = []
    for p in cleaned.split(" & "):
      p_clean = p.strip()
      if p_clean and p_clean not in parts:
        parts.append(p_clean)
      if len(parts) >= 3:
        break
    cleaned = " & ".join(parts)
  if len(cleaned) > 120:
    truncated = cleaned[:120].rsplit(" ", 1)[0]
    cleaned = truncated if truncated else cleaned[:120]
  return cleaned.rstrip(",.;:&- ").strip()


def clean_and_filter_topic_names(
    names: List[str], max_unique: Optional[int] = None
) -> List[str]:
  """Cleans, deduplicates (by frequency then order), and filters placeholder names.

  If all input names are placeholders (e.g., in unit test fixtures using
  'Topic 1', 'Topic 2'), preserves the deduplicated placeholders.

  Args:
      names: Raw topic or opinion names to clean and deduplicate.
      max_unique: Optional maximum number of unique names to return. If None,
        all unique cleaned names are returned.

  Returns:
      A list of cleaned, deduplicated topic or opinion names.
  """
  cleaned_all: List[str] = []
  for raw in names:
    if not isinstance(raw, str):
      continue
    s = clean_topic_name(raw)
    if s and s.lower() != "other":
      cleaned_all.append(s)

  if not cleaned_all:
    return []

  non_placeholders = [s for s in cleaned_all if not is_placeholder_topic_name(s)]
  candidates = non_placeholders if non_placeholders else cleaned_all

  counts = Counter(candidates)
  first_seen: Dict[str, int] = {}
  for idx, s in enumerate(candidates):
    if s not in first_seen:
      first_seen[s] = idx

  ordered_unique = sorted(
      first_seen.keys(), key=lambda s: (-counts[s], first_seen[s])
  )
  if max_unique is not None:
    return ordered_unique[:max_unique]
  return ordered_unique


def _repair_truncated_json(json_text: str) -> Optional[str]:
  """Attempts to repair truncated JSON objects/arrays produced when max_tokens is hit."""
  if not json_text:
    return None

  in_string = False
  escape = False
  stack: List[str] = []
  string_start_idx: Optional[int] = None
  last_valid_array_item_end: Optional[int] = None
  stack_at_last_item: Optional[List[str]] = None

  for idx, ch in enumerate(json_text):
    if in_string:
      if escape:
        escape = False
      elif ch == "\\":
        escape = True
      elif ch == '"':
        in_string = False
        string_start_idx = None
      continue

    if ch == '"':
      in_string = True
      string_start_idx = idx
    elif ch in ("{", "["):
      stack.append(ch)
    elif ch == "}":
      if not stack or stack[-1] != "{":
        return None
      stack.pop()
      if stack and stack[-1] == "[":
        last_valid_array_item_end = idx + 1
        stack_at_last_item = list(stack)
    elif ch == "]":
      if not stack or stack[-1] != "[":
        return None
      stack.pop()

  if not in_string and not stack:
    return None

  # Prefer truncating back to the last complete element in an open array.
  if last_valid_array_item_end is not None and stack_at_last_item is not None:
    prefix = json_text[:last_valid_array_item_end]
    closers = "".join("]" if c == "[" else "}" for c in reversed(stack_at_last_item))
    return prefix + closers

  # Otherwise, if truncated inside a runaway string literal, cap the string and close open containers.
  if in_string and string_start_idx is not None and stack:
    raw_str_content = json_text[string_start_idx + 1 : string_start_idx + 1 + 120]
    raw_str_content = raw_str_content.rstrip("\\")
    prefix = json_text[: string_start_idx + 1] + raw_str_content + '"'
    closers = "".join("]" if c == "[" else "}" for c in reversed(stack))
    return prefix + closers

  return None


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
  except (ValidationError, json.JSONDecodeError) as e:
    repaired = _repair_truncated_json(json_text)
    if repaired is not None:
      try:
        return TypeAdapter(schema_to_expect).validate_json(repaired)
      except Exception:
        pass
    if isinstance(e, ValidationError):
      logging.debug(
          "Model response failed Pydantic validation for schema"
          f" {schema_to_expect}: {json_text}.\nError: {e}"
      )
      raise ValueError(f"Model response failed Pydantic validation") from e
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


def _report_failed_chunks(
    stage: str,
    event: str,
    failures: List[str],
    total: int,
) -> None:
  """Logs a WARNING and records run_health events for failed chunks.

  Args:
    stage: Human-readable stage name for the log message.
    event: run_health event name (one occurrence recorded per failed chunk).
    failures: One description per failed chunk.
    total: Total number of chunks submitted.
  """
  if not failures:
    return
  run_health.warning(
      event,
      f"{stage}: failed {len(failures)} of {total} chunks; their input was"
      f" excluded from the merged result. Failures: {failures[:10]}",
      n=len(failures),
      detail=[f"{failure} (of {total} chunks)" for failure in failures],
  )


def _chunk_label(row: Dict[str, Any], index: int) -> str:
  """Returns a label identifying a chunk result row for diagnostics."""
  return str(row.get("chunk") or f"chunk_{index}")


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
    max_concurrent_calls: Optional[int] = None,
) -> custom_types.FlatTopicList:
  """Generates topics from a list of statements, handling token limits via chunking and merging.

  Args:
      model: The GenaiModel to use.
      instructions: The prompt instructions.
      prompt_input_data: The statement texts to generate topics from.
      schema_to_expect: The response schema expected from the model.
      additional_context: Optional context for the LLM prompt.
      chunks: Optional precomputed chunks of prompt_input_data. If None,
        chunks are computed here.
      max_concurrent_calls: Optional maximum number of concurrent LLM calls.
        If None, the model's default is used.

  Returns:
      The generated (and, if chunked, merged) topics.
  """

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
      prompt_jobs,
      response_parser=parser,
      max_concurrent_calls=max_concurrent_calls,
  )

  partial_results = []
  failures: List[str] = []
  rows = results_df.to_dict("records")
  for index, row in enumerate(rows):
    result = row.get("result")
    label = _chunk_label(row, index)
    if result is None or (isinstance(result, dict) and result.get("error")):
      failures.append(f"{label}: {result}")
      continue

    if isinstance(result, list):
      partial_results.extend(result)
    elif isinstance(result, custom_types.FlatTopicList):
      partial_results.extend(result.topics)
    else:
      failures.append(f"{label}: unexpected result type {type(result)}")
  if len(rows) < chunk_count:
    failures.append(f"{chunk_count - len(rows)} chunk(s) returned no result")
  _report_failed_chunks(
      "Topic identification", "topic_chunk_failed", failures, chunk_count
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

  raw_topics = [topic.name for topic in partial_results]
  had_non_placeholder = any(
      not is_placeholder_topic_name(clean_topic_name(name))
      for name in raw_topics
      if isinstance(name, str) and clean_topic_name(name).lower() != "other"
  )
  combined_topics = clean_and_filter_topic_names(raw_topics)
  if not combined_topics:
    combined_topics = raw_topics

  prompt_str = sensemaker_utils.get_prompt(
      merge_instructions, combined_topics, additional_context
  )

  resp = await model.generate_content(
      prompt=prompt_str,
      run_name="merge_topics",
      response_schema=schema_to_expect,
  )

  if resp.get("error"):
    if combined_topics:
      logging.warning(
          f"Error calling model for merge ({resp.get('error')}); falling back"
          " to discovered chunk topics."
      )
      return custom_types.FlatTopicList(
          topics=[
              custom_types.FlatTopic(name=name) for name in combined_topics
          ]
      )
    raise ValueError(f"Error calling Gemini for merge: {resp.get('error')}")

  try:
    final_response = parse_response(resp["text"], schema_to_expect)
  except Exception as e:
    if combined_topics:
      logging.warning(
          f"Failed to parse merged topics ({e}); falling back to"
          " discovered chunk topics."
      )
      return custom_types.FlatTopicList(
          topics=[
              custom_types.FlatTopic(name=name) for name in combined_topics
          ]
      )
    raise

  merged_topics: Optional[List[custom_types.FlatTopic]] = None
  if isinstance(final_response, custom_types.FlatTopicList):
    merged_topics = final_response.topics
  elif isinstance(final_response, list):
    merged_topics = final_response

  if merged_topics is not None:
    if had_non_placeholder:
      cleaned_merged = clean_and_filter_topic_names(
          [t.name for t in merged_topics]
      )
      cleaned_merged = [
          name for name in cleaned_merged if not is_placeholder_topic_name(name)
      ]
      if not cleaned_merged:
        logging.warning(
            "Merged topics only contained placeholders; falling back to"
            " discovered chunk topics."
        )
        cleaned_merged = combined_topics
      return custom_types.FlatTopicList(
          topics=[custom_types.FlatTopic(name=name) for name in cleaned_merged]
      )
    return custom_types.FlatTopicList(topics=merged_topics)

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
  failures: List[str] = []
  rows = results_df.to_dict("records")
  for index, row in enumerate(rows):
    result = row.get("result")
    label = _chunk_label(row, index)
    if result is None or (isinstance(result, dict) and result.get("error")):
      failures.append(f"{label}: {result}")
      continue

    if isinstance(result, custom_types.NestedTopic):
      partial_results_list.append(result)
    else:
      failures.append(f"{label}: unexpected result type {type(result)}")
  if len(rows) < chunk_count:
    failures.append(f"{chunk_count - len(rows)} chunk(s) returned no result")
  _report_failed_chunks(
      f"Opinion identification for topic '{parent_topic.name}'",
      "opinion_chunk_failed",
      failures,
      chunk_count,
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
  merge_instructions = prompts.get_topic_modeling_merge_opinions_prompt(
      parent_topic.name
  )

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

  resp = await model.generate_content(
      prompt=prompt_str,
      run_name="merge_opinions",
      response_schema=schema_to_expect,
      max_concurrent_calls=20,
  )

  if resp.get("error"):
    run_health.error(
        "opinion_merge_failed",
        f"Error calling Gemini for opinion merge: {resp.get('error')}",
        detail=f"topic '{parent_topic.name}': {str(resp.get('error'))[:200]}",
    )
    return custom_types.NestedTopic(name=parent_topic.name, subtopics=[])

  final_response = parse_response(resp["text"], schema_to_expect)

  if isinstance(final_response, custom_types.NestedTopic):
    return final_response

  logging.error(
      f"LLM response for opinion merge is not a NestedTopic: {final_response}"
  )
  return custom_types.NestedTopic(name=parent_topic.name, subtopics=[])
