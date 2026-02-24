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
import functools
import itertools
import json
import logging
import os
import re
import sys
import difflib
import pandas as pd
from collections import defaultdict
from typing import Any, Callable, Coroutine, Iterable, Set, Tuple, Union, cast
from more_itertools import batched
from case_studies.wtp.tasks.topic_modeling_util import parse_response
from pydantic import TypeAdapter, ValidationError
from case_studies.wtp.models.genai_model import GenaiModel, MAX_LLM_RETRIES, WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS, MAX_CONCURRENT_CALLS
from case_studies.wtp.sensemaker_utils import execute_concurrently, get_prompt
from case_studies.wtp import runner_utils
from case_studies.wtp.models.custom_types import (
    Statement,
    StatementRecord,
    Topic,
    FlatTopic,
    NestedTopic,
    StatementRecordList,
    OpinionResponseSchema,
)
from case_studies.wtp.evals.autorater_evals import (
    prepare_opinion_eval_prompts,
    parse_eval_response,
)
from case_studies.wtp.tasks.categorization_rules import OPINION_CATEGORIZATION_MAIN_RULES
from case_studies.wtp.tasks import topic_modeling, topic_modeling_util
from case_studies.wtp.tasks.topic_modeling import learn_topics, learn_opinions


# Use dynamic batching based on token count
# Target 20k tokens per batch (safe for output generation limits)
MAX_BATCH_TOKENS = 20000
MAX_ITEMS_PER_BATCH = 50


async def categorize_topics(
    statements: list[Statement],
    model: GenaiModel,
    current_topics: list[Topic] | None = None,
    additional_context: str | None = None,
) -> tuple[list[Statement], list[Topic]]:
  """Categorizes a list of statements into topics.

  This function serves as the first main step in the categorization process.
  If a list of topics is not provided, it will first learn them from the
  statements.

  Args:
      statements: The list of statements to categorize.
      model: The language model to use for categorization.
      current_topics: An optional list of predefined topics to use.
      additional_context: Optional text to provide additional context to the
        model.

  Returns:
      A tuple containing the list of statements with topics assigned and the
      list
      of topics that were used for categorization.
  """
  input_statements = [s.model_copy(deep=True) for s in statements]

  effective_topics = (
      [t.model_copy(deep=True) for t in current_topics]
      if current_topics
      else []
  )

  # If no topics are provided, learn them from the statements.
  if not effective_topics:
    logging.debug("Learning initial top-level topics.")
    learned_initial_topics = await learn_topics(
        input_statements, model, additional_context=additional_context
    )
    if not learned_initial_topics:
      raise ValueError(
          "Failed to learn initial topics. Cannot proceed with categorization."
      )
    effective_topics = learned_initial_topics

  # Ensure an "Other" category exists for statements that don't fit.
  if not any(t.name == "Other" for t in effective_topics):
    effective_topics.append(FlatTopic(name="Other"))

  logging.info(
      "Performing initial topic categorization into:\n- %s",
      "\n- ".join([t.name for t in effective_topics]),
  )
  llm_statements_with_topics = await _process_topic_categorization(
      input_statements,
      model,
      effective_topics,
      additional_context,
  )

  # Transfer the topics from the LLM statements back to the input statements.
  input_statements_map = {s.id: s for s in input_statements}
  for llm_statement in llm_statements_with_topics:
    if llm_statement.id in input_statements_map:
      input_statements_map[llm_statement.id].topics = llm_statement.topics

  return list(input_statements_map.values()), effective_topics


async def learn_global_opinions(
    statements_with_topics_and_quotes: list[Statement],
    topics_to_process: list[Topic],
    model: GenaiModel,
    additional_context: str | None = None,
) -> dict[str, Any]:
  """Learns opinions for all topics globally using concurrent batching.

  Args:
      statements_with_topics_and_quotes: The list of statements with topics and quotes.
      topics_to_process: The list of topics to learn opinions for.
      model: The language model to use for learning opinions.
      additional_context: Optional text to provide additional context to the
        model.

  Returns:
      A dictionary containing the learned opinions for each topic.
  """
  logging.info("Starting Global Opinion Learning Phase (Concurrent)...")

  # 1. Prepare Chunks for all topics concurrently
  async def _prepare_topic_chunks(topic: Topic):
    # Skip "Other" topic for opinion categorization
    if topic.name == "Other":
      return topic, [], None

    # Gather relevant quotes for this topic
    quotes = []
    for s in statements_with_topics_and_quotes:
      for q in s.quotes:
        if q.topic.name == topic.name:
          quotes.append(q)

    if not quotes:
      return topic, [], None

    prompt_input_data = [f"<quote>{q.text}</quote>" for q in quotes]
    instructions = topic_modeling.learn_opinions_prompt(topic)

    # This might call the model for token counting
    chunks = await topic_modeling_util.create_chunks(
        model, instructions, prompt_input_data, additional_context
    )
    return topic, chunks, instructions

  # Gather chunks for all topics
  logging.info(f"Preparing chunks for {len(topics_to_process)} topics...")
  preparation_tasks = [_prepare_topic_chunks(t) for t in topics_to_process]
  # We use gather here as this is primarily CPU/minor I/O or token counting
  prepared_topics = await asyncio.gather(*preparation_tasks)

  # 2. Build Generation Jobs (Phase 1)
  generation_prompts = []
  # Map topic name to list of indices in the generation_prompts list (and subsequent result DF)
  topic_job_indices: dict[str, list[int]] = {}

  for topic, chunks, instructions in prepared_topics:
    if not chunks:
      continue

    topic_job_indices[topic.name] = []
    for i, chunk in enumerate(chunks):
      prompt_str = get_prompt(instructions, chunk, additional_context)

      job_index = len(generation_prompts)
      generation_prompts.append({
          "prompt": prompt_str,
          "topic_obj": topic,  # Store full object for later use
          "chunk_index": i,
          "total_chunks": len(chunks),
          "response_schema": OpinionResponseSchema,
          "log_prefix_marker": "4",
          "progress_index": (
              i
          ),  # Approximate within topic, or could use global index
      })
      topic_job_indices[topic.name].append(job_index)

  if not generation_prompts:
    logging.warning("No chunks generated for opinion learning.")
    return {}

  logging.info(
      f"Generated {len(generation_prompts)} opinion learning batches "
      f"across {len(topic_job_indices)} active topics."
  )

  # 3. Execute Phase 1: Initial Generation
  def _parse_opinion_response(resp, job):
    return parse_response(resp["text"], job["response_schema"])

  results_df, _ = await model.process_prompts_concurrently(
      generation_prompts,
      response_parser=_parse_opinion_response,
  )

  # 4. Process Results and Prepare Merges (Phase 2)
  final_topic_map = {}
  merge_prompts = []
  # Map topic name to its merge job index
  merge_job_indices: dict[str, int] = {}

  # Group results by topic name
  # We can iterate through the dataframe and collect results
  topic_results: dict[str, list[NestedTopic]] = {}

  # Initialize list for all expected topics
  for t in topics_to_process:
    topic_results[t.name] = []

  for _, row in results_df.iterrows():
    topic_obj = row.get("topic_obj")
    if not topic_obj:
      logging.warning(f"Result missing topic_obj: {row}")
      continue

    result = row["result"]
    topic_name = topic_obj.name

    if topic_name not in topic_results:
      # Should not happen if initialized, but good safety
      topic_results[topic_name] = []

    # Ensure an "Other" opinion exists for each topic.
    if not any(sub.name == "Other" for sub in result.subtopics):
      result.subtopics.append(FlatTopic(name="Other"))
    if isinstance(result, OpinionResponseSchema):
      # Convert to NestedTopic
      topic_results[topic_name].append(
          NestedTopic(name=result.name, subtopics=result.subtopics)
      )
    elif isinstance(result, NestedTopic):
      topic_results[topic_name].append(result)
    elif isinstance(result, dict) and result.get("error"):
      logging.error(
          f"Opinion chunk failed for topic '{topic_name}':"
          f" {result.get('error')}"
      )
      raise ValueError(f"Failed to learn opinions for '{topic_name}'.")

  # Now iterate over grouped results
  for topic_name, partial_topics in topic_results.items():
    if not partial_topics:
      final_topic_map[topic_name] = NestedTopic(name=topic_name, subtopics=[])
      continue

    # Check if merge is needed
    if len(partial_topics) == 1:
      # No merge needed
      final_topic_map[topic_name] = partial_topics[0]
    else:
      # Multiple chunks -> Merge needed
      # Consolidate all opinion names
      combined_opinions = []
      for pt in partial_topics:
        if pt.subtopics:
          for sub in pt.subtopics:
            combined_opinions.append(sub.name)

      if not combined_opinions:
        final_topic_map[topic_name] = NestedTopic(name=topic_name, subtopics=[])
        continue

      # Prepare Merge Prompt
      merge_instructions = topic_modeling_util.merge_opinions_prompt(topic_obj)
      prompt_str = get_prompt(
          merge_instructions, combined_opinions, additional_context
      )

      merge_idx = len(merge_prompts)
      merge_prompts.append({
          "prompt": prompt_str,
          "topic_obj": topic_obj,
          "response_schema": (
              OpinionResponseSchema
          ),  # Merge prompt expects same structure (JSON list of opinions)
          "log_prefix_marker": "4",
          "is_merge": True,
      })
      merge_job_indices[topic_name] = merge_idx

  # 5. Execute Phase 2: Merging (if any)
  if merge_prompts:
    logging.info(f"Merging opinions for {len(merge_prompts)} topics...")

    merge_results_df, _ = await model.process_prompts_concurrently(
        merge_prompts,
        response_parser=_parse_opinion_response,
    )

    for topic_name, idx in merge_job_indices.items():
      result = merge_results_df.iloc[idx]["result"]

      final_res = None
      if isinstance(result, OpinionResponseSchema):
        final_res = NestedTopic(name=result.name, subtopics=result.subtopics)
      elif isinstance(result, NestedTopic):
        final_res = result

      if final_res:
        final_topic_map[topic_name] = final_res
      else:
        logging.error(
            f"Merge failed for topic '{topic_name}'. Returning empty."
        )
        final_topic_map[topic_name] = NestedTopic(name=topic_name, subtopics=[])

  return final_topic_map


async def categorize_opinions(
    statements_with_topics_and_quotes: list[Statement],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, Any],
    model: GenaiModel,
    additional_context: str | None = None,
    run_autoraters: bool = True,
) -> Iterable[Statement]:
  """Categorizes opinions within the provided topics for each statement.

  Args:
    statements_with_topics_and_quotes: list of statements to categorize.
    topics_to_process: list of topics to categorize opinions within.
    topic_to_opinions_map: Map of topic names to their learned opinion structures.
    model: GenaiModel instance for LLM calls.
    additional_context: Additional context to provide to the LLM.
    run_autoraters: Whether to run autorater evaluations.

  Returns:
    List of statements with categorized opinions.
  """

  logging.info("Starting Global Opinion Categorization Phase...")

  # Pre-compute valid input statements map for O(1) lookups and O(1) in-place updates
  input_statements_map = {
      s.id: s.model_copy(deep=True) for s in statements_with_topics_and_quotes
  }

  # Track work queue: Topic Name -> List of IDs to process for that topic
  # We use IDs to track statements to avoid deep copying lists repeatedly
  # Initial work queue has all statements for all topics
  topic_work_queue_ids: dict[str, list[str]] = {
      t.name: list(input_statements_map.keys()) for t in topics_to_process
  }

  autorater_retry_counts: dict[str, int] = {}
  MAX_AUTORATER_RETRIES = 3

  total_statements = len(input_statements_map)
  finalized_count_events = 0

  for attempt in range(1, MAX_LLM_RETRIES + 1):
    if not topic_work_queue_ids:
      break

    all_prompts = _prepare_opinion_prompts_for_pending_work(
        topic_work_queue_ids,
        topics_to_process,
        topic_to_opinions_map,
        input_statements_map,
        additional_context,
        attempt,
    )

    if not all_prompts:
      logging.info("No prompts generated (likely no relevant quotes left).")
      break

    logging.info(
        f"Global Categorization (Attempt {attempt}): {len(all_prompts)} batches"
        " prepared."
    )

    # Execute
    def _parser(resp, job):
      parsed_wrapper = parse_response(resp["text"], job["response_schema"])
      return parsed_wrapper.items if parsed_wrapper else []

    results_df, stats = await model.process_prompts_concurrently(
        all_prompts,
        response_parser=_parser,
        max_concurrent_calls=MAX_CONCURRENT_CALLS,
    )

    # --- Phase 1: Aggregate Candidates and Failures ---

    (
        autorater_candidates,
        immediate_retries,
        phase1_finalized_count,
    ) = _process_opinion_llm_results(
        results_df,
        input_statements_map,
        run_autoraters,
    )
    finalized_count_events += phase1_finalized_count

    # --- Phase 2: Concurrent Autorater Execution ---

    if run_autoraters and autorater_candidates:
      (
          phase2_finalized_count,
          autorater_retries,
      ) = await _run_opinion_autoraters(
          model,
          autorater_candidates,
          input_statements_map,
          autorater_retry_counts,
          MAX_AUTORATER_RETRIES,
      )
      finalized_count_events += phase2_finalized_count

      # Merge autorater retries into immediate retries
      for t_name, stats in autorater_retries.items():
        if t_name not in immediate_retries:
          immediate_retries[t_name] = []
        immediate_retries[t_name].extend(stats)

    # --- Phase 3: Construct Next Retry Queue ---

    # Process immediate retries (from LLM failures or Autorater retryable failures)
    next_topic_work_queue_ids: dict[str, list[str]] = {}
    for t_name, stats in immediate_retries.items():
      if t_name not in next_topic_work_queue_ids:
        next_topic_work_queue_ids[t_name] = []

      # Dedupe
      for s in stats:
        if s.id not in next_topic_work_queue_ids[t_name]:
          next_topic_work_queue_ids[t_name].append(s.id)

    # Optimize deduplication for next queue
    for t_name in next_topic_work_queue_ids:
      next_topic_work_queue_ids[t_name] = list(
          set(next_topic_work_queue_ids[t_name])
      )

    topic_work_queue_ids = next_topic_work_queue_ids

    if not topic_work_queue_ids:
      break

    if attempt < MAX_LLM_RETRIES:
      retry_counts = {k: len(v) for k, v in topic_work_queue_ids.items()}
      total_retrying = sum(retry_counts.values())
      logging.info(
          f"Progress: {finalized_count_events}/{total_statements} statements"
          " finalized (events)."
      )
      logging.warning(
          f"Retrying {total_retrying} statements across {len(retry_counts)}"
          f" topics. Sleeping {WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS}s..."
      )
      await asyncio.sleep(WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS)

  # Handle exhausted retries - Default to 'Other'
  _assign_defaults_for_exhausted_retries(
      topic_work_queue_ids,
      topics_to_process,
      input_statements_map,
  )

  return input_statements_map.values()


def _prepare_opinion_prompts_for_pending_work(
    topic_work_queue_ids: dict[str, list[str]],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, Any],
    input_statements_map: dict[str, Statement],
    additional_context: str | None,
    attempt: int,
) -> list[dict[str, Any]]:
  """Prepares opinion categorization prompts for all pending topics."""
  all_prompts = []

  for topic_name, pending_ids in topic_work_queue_ids.items():
    # Find the topic object
    topic = next((t for t in topics_to_process if t.name == topic_name), None)
    if not topic:
      continue

    learned_topic_result = topic_to_opinions_map.get(topic.name)
    if not learned_topic_result or not learned_topic_result.subtopics:
      logging.warning(
          f"No opinions learned for topic '{topic.name}'. Skipping."
      )
      continue

    opinions = learned_topic_result.subtopics

    # Reconstruct statement list from IDs
    pending_statements = [
        input_statements_map[sid]
        for sid in pending_ids
        if sid in input_statements_map
    ]

    if not pending_statements:
      continue

    # Prepare prompts
    opinion_categorization_prompts = _prepare_categorization_prompts(
        pending_statements,
        opinions,
        _opinion_categorization_prompt(opinions),
        StatementRecordList,
        additional_context,
        parent_topic_name=topic.name,
        is_opinion_categorization=True,
        attempt=attempt,
    )

    # Inject metadata
    for p in opinion_categorization_prompts:
      p["target_opinions"] = opinions
      p["parent_topic_obj"] = topic
      # We also need to know which topic this prompt belongs to for retry tracking
      p["work_queue_topic_name"] = topic.name

    all_prompts.extend(opinion_categorization_prompts)

  return all_prompts


def _process_opinion_llm_results(
    results_df: pd.DataFrame,
    input_statements_map: dict[str, Statement],
    run_autoraters: bool,
) -> Tuple[list[dict[str, Any]], dict[str, list[Statement]], int]:
  """Processes LLM results, identifying candidates for autoraters and failures."""
  next_topic_work_queue_ids: dict[str, list[str]] = {}

  # List of (valid_records, original_statements, topic_name, parent_topic_obj)
  autorater_candidates = []

  # Immediate failures/retries
  immediate_retries: dict[str, list[Statement]] = defaultdict(list)

  finalized_count = 0

  for i, row in results_df.iterrows():
    result = row.get("result")

    # Metadata
    target_opinions = row.get("target_opinions")
    parent_topic = row.get("parent_topic_obj")
    batch_items = row.get("batch_items")
    topic_name = row.get("work_queue_topic_name")

    if not topic_name or not parent_topic:
      continue

    if isinstance(result, list):
      processed_result = _process_categorized_llm_records(
          result,
          list(input_statements_map.values()),  # Use master list values
          batch_items,
          target_topics_or_opinions=target_opinions,
      )

      valid_records = processed_result["valid_records"]
      needs_retry = processed_result["needs_retry_statements"]

      if valid_records:
        if run_autoraters:
          autorater_candidates.append({
              "records": valid_records,
              "batch_items": batch_items,
              "target_opinions": target_opinions,
              "parent_topic_name": parent_topic.name,
              "parent_topic_obj": parent_topic,
          })
        else:
          # No autorater: immediately merge valid records
          _merge_opinions_into_statements_inplace(
              input_statements_map, valid_records, parent_topic
          )
          finalized_count += len(valid_records)

      if needs_retry:
        immediate_retries[topic_name].extend(needs_retry)

    elif isinstance(result, dict) and "error" in result:
      logging.error(f"Batch failed for {topic_name}: {result['error']}")
      immediate_retries[topic_name].extend(batch_items)
    else:
      logging.error(f"Batch failed for {topic_name}: Invalid result type")
      immediate_retries[topic_name].extend(batch_items)

  return autorater_candidates, immediate_retries, finalized_count


async def _run_opinion_autoraters(
    model: GenaiModel,
    autorater_candidates: list[dict[str, Any]],
    input_statements_map: dict[str, Statement],
    autorater_retry_counts: dict[str, int],
    max_autorater_retries: int,
) -> Tuple[int, dict[str, list[Statement]]]:
  """Runs autoraters concurrently and processes results."""
  logging.info(
      f"Running autoraters for {len(autorater_candidates)} batches"
      " concurrently..."
  )

  finalized_count = 0
  autorater_retries: dict[str, list[Statement]] = {}

  # 1. Start Workers
  (
      queue,
      workers,
      results_list,
      stats_list,
      stop_event,
  ) = model.start_concurrent_workers(
      response_parser=parse_eval_response,
      max_concurrent_calls=MAX_CONCURRENT_CALLS,
  )

  # 2. Generate and Push Prompts
  for candidate_group in autorater_candidates:
    prompts = prepare_opinion_eval_prompts(
        candidate_group["records"],
        candidate_group["batch_items"],
        candidate_group["target_opinions"],
        candidate_group["parent_topic_name"],
    )
    for p in prompts:
      # Inject metadata needed for processing later
      p["metadata"]["parent_topic_obj"] = candidate_group["parent_topic_obj"]
      p["metadata"]["parent_topic_name"] = candidate_group["parent_topic_name"]

      # Add default values to stats and retry_attempts in order to satisfy
      # the requirements of GenaiModel._api_worker_with_retry and skip
      # processing this job if they are not present.
      if "stats" not in p or p["stats"] is None:
        p["stats"] = {}
      if "retry_attempts" not in p:
        p["retry_attempts"] = MAX_LLM_RETRIES

      queue.put_nowait(p)

  # 3. Signal Completion
  for _ in range(MAX_CONCURRENT_CALLS):
    queue.put_nowait(None)

  # 4. Wait for Workers
  await asyncio.gather(*workers)

  # 5. Process Results
  for row in results_list:
    result = row.get("result")
    metadata = row.get("metadata", {})
    original_record = metadata.get("original_record")
    parent_topic_obj = metadata.get("parent_topic_obj")
    parent_topic_name = metadata.get("parent_topic_name")

    score = result.get("score") if isinstance(result, dict) else None

    if not original_record or not parent_topic_obj:
      continue

    if score is not None and score >= 4:
      # Passed
      _merge_opinions_into_statements_inplace(
          input_statements_map, [original_record], parent_topic_obj
      )
      finalized_count += 1
    else:
      # Failed
      stmt_id = original_record.id
      stat = input_statements_map.get(stmt_id)
      if not stat:
        continue

      autorater_retry_counts[stat.id] = (
          autorater_retry_counts.get(stat.id, 0) + 1
      )

      if autorater_retry_counts[stat.id] >= max_autorater_retries:
        logging.warning(
            f"Statement {stat.id} failed autorater"
            f" {max_autorater_retries} times. Defaulting to 'Other'."
        )
        # Permanently failed -> Other
        other_topic = FlatTopic(name="Other")
        other_record = StatementRecord(
            id=stat.id, quote_id=original_record.quote_id, topics=[other_topic]
        )
        _merge_opinions_into_statements_inplace(
            input_statements_map, [other_record], parent_topic_obj
        )
        finalized_count += 1
      else:
        # Retryable
        if parent_topic_name not in autorater_retries:
          autorater_retries[parent_topic_name] = []
        autorater_retries[parent_topic_name].append(stat)

  # Dedupe autorater_retries
  for t_name in autorater_retries:
    unique_stats = {s.id: s for s in autorater_retries[t_name]}.values()
    autorater_retries[t_name] = list(unique_stats)

  return finalized_count, autorater_retries


def _assign_defaults_for_exhausted_retries(
    topic_work_queue_ids: dict[str, list[str]],
    topics_to_process: list[Topic],
    input_statements_map: dict[str, Statement],
) -> None:
  """Assigns statements to 'Other' if they exhausted all retries."""
  for topic_name, failed_ids in topic_work_queue_ids.items():
    topic = next((t for t in topics_to_process if t.name == topic_name), None)
    if not topic:
      continue

    failed_stats = [
        input_statements_map[sid]
        for sid in failed_ids
        if sid in input_statements_map
    ]

    logging.warning(
        f"Exhausted retries for topic {topic_name} on {len(failed_stats)}"
        " statements. Defaulting to 'Other'."
    )

    other_topic = FlatTopic(name="Other")
    other_records = []
    for stat in failed_stats:
      if stat.quotes:
        relevant_quotes = [q for q in stat.quotes if q.topic.name == topic_name]
        for q in relevant_quotes:
          other_records.append(
              StatementRecord(id=stat.id, quote_id=q.id, topics=[other_topic])
          )
    if other_records:
      _merge_opinions_into_statements_inplace(
          input_statements_map, other_records, topic
      )

  return


def _topic_categorization_prompt(topics: list[Topic]) -> str:
  """Generates the prompt for categorizing statements into topics."""
  topics_json_list = [{"name": t.name} for t in topics]
  return f"""
For each of the following statements, identify any relevant topic from the list below.
Input Topics:
{json.dumps(topics_json_list)}

Important Considerations:
- Ensure the assigned topic accurately reflects the meaning of the statement.
- If relevant and necessary (e.g. when a statement contains multiple disjoint claims), a statement can be assigned to multiple topics.
- Prioritize using the existing topics whenever possible. Keep the "Other" topic to minimum, ideally keep it empty.
- Use "Other" topic if the statement is completely off-topic and doesn't really fit any of the topics.
- All statements must be assigned at least one existing topic.
- Do not create any new topics that are not listed in the Input Topics.
- When generating the JSON output, minimize the size of the response. For example, prefer this compact format: {{"id": "5258", "topics": [{{"name": "Arts, Culture, And Recreation"}}]}} instead of adding unnecessary whitespace or newlines.

class StatementRecordList(BaseModel):
    items: list[StatementRecord]

class StatementRecord(BaseModel):
    id: str = Field(description="The unique identifier of the statement.")
    topics: list[Topic] = Field(description="A list of topics assigned to the statement.")

class Topic(BaseModel):
    name: str

Response must be a valid JSON object matching StatementRecordList schema. Example:
{{
  "items": [
    {{
      "id": "5258",
      "topics": [{{"name": "Arts, Culture, And Recreation"}}]
    }}
  ]
}}
"""


def _opinion_categorization_prompt(opinions: list[Topic]) -> str:
  """Generates the prompt for categorizing quotes into opinions."""
  opinions_json_list = [{"name": op.name} for op in opinions]

  return f"""
Categorize the following quotes based on the provided opinions.

Input Opinions:
{json.dumps(opinions_json_list)}

{OPINION_CATEGORIZATION_MAIN_RULES}

Other rules:
- Prioritize using the existing opinions whenever possible.
- Use "Other" opinion if the quote is completely off-topic and doesn't really fit any of the opinions. Keep the "Other" opinion to minimum, ideally keep it empty.
- All quotes must be assigned at least one existing opinion.
- Do not create any new opinions that are not listed in the Input Opinions.
- Respond with a JSON array of objects, each with "id", "quote_id" (which is the id of the quote), and "topics" (A list of opinions assigned to the quote, each with a "name" key).
- When generating the JSON output, minimize the size of the response. For example, prefer this compact format: {{"id": "5258", "quote_id": "q1", "topics": [{{"name": "Opinion from the Input list"}}]}} instead of adding unnecessary whitespace or newlines.

VERY IMPORTANT:
Double check to make sure that all quote ids and topic names are in the input. For example, if an input quote_id is '1183-Defining Freedom', then the output quote_id should be the same '1183-Defining Freedom', and not anything else like '1183' or '1188-Defining Freedom'.

class StatementRecordList(BaseModel):
    items: list[StatementRecord]

class StatementRecord(BaseModel):
    id: str = Field(description="The unique identifier of the statement.")
    quote_id: str = Field(description="The unique identifier of the quote.")
    topics: list[Topic] = Field(description="A list of opinions assigned to the quote.")

class Topic(BaseModel):
    name: str

You must follow the rules for the instructions strictly.
Pay close attention to Rules "Most Literal Match" and "Holistic Match". Do not select any opinion that is only a partial match or requires an inference if a more literal match is available.

Response must be a valid JSON object matching StatementRecordList schema. Example:
{{
  "items": [
     {{
       "id": "5258",
       "quote_id": "q1",
       "topics": [
          {{"name": "An opinion assigned to the quote."}}
       ]
     }}
  ]
}}
"""


def _create_token_based_batches(
    statements: list[Statement], max_tokens: int, max_items: int = 50
) -> list[list[Statement]]:
  """Creates batches of statements that fit within a token limit."""
  batches = []
  current_batch = []
  current_tokens = 0

  effective_limit = max_tokens

  for s in statements:
    # Estimate tokens for the statement text + ID formatting overhead
    # "{id}: {text}\n" -> text length + ~10 chars overhead
    s_tokens = runner_utils.estimate_tokens(s.text) + 5

    # If a single message is huge, it still goes into its own batch (or filtered earlier)
    # Also check if adding this item would exceed the max items per batch
    if current_batch and (
        (current_tokens + s_tokens > effective_limit)
        or (len(current_batch) >= max_items)
    ):
      batches.append(current_batch)
      current_batch = []
      current_tokens = 0

    current_batch.append(s)
    current_tokens += s_tokens

  if current_batch:
    batches.append(current_batch)

  return batches


def _prepare_categorization_prompts(
    statements: list[Statement],
    target_topics_or_opinions: list[Topic],
    instructions: str,
    output_schema: Any,
    additional_context: str | None = None,
    parent_topic_name: str | None = None,
    is_opinion_categorization: bool = False,
    attempt: int = 1,
) -> list[dict[str, Any]]:
  """Prepares prompts for a batch of statements."""
  prompts = []

  # Create batches
  batches = _create_token_based_batches(
      statements, MAX_BATCH_TOKENS, max_items=MAX_ITEMS_PER_BATCH
  )

  for i, batch in enumerate(batches):
    statements_for_model_prompt_data: list[str] = []
    is_processing_quotes_for_parent = parent_topic_name and any(
        s.quotes for s in batch
    )

    # If processing opinions, format the prompt with quotes.
    if is_processing_quotes_for_parent:
      for statement_item in batch:
        if statement_item.quotes:
          relevant_quotes = [
              q
              for q in statement_item.quotes
              if q.topic.name == parent_topic_name
          ]
          for quote_obj in relevant_quotes:
            statements_for_model_prompt_data.append(
                json.dumps({
                    "id": statement_item.id,
                    "quote_id": quote_obj.id,
                    "quote_text": quote_obj.text,
                    "topic_name": parent_topic_name,
                })
            )
      if not statements_for_model_prompt_data:
        # No relevant quotes in this batch, skip prompt generation.
        continue

    # Otherwise, use the full statement text.
    if not statements_for_model_prompt_data:
      statements_for_model_prompt_data = [
          json.dumps({"id": s.id, "text": s.text}) for s in batch
      ]

    prompt_str = get_prompt(
        instructions, statements_for_model_prompt_data, additional_context
    )

    prompts.append({
        "prompt": prompt_str,
        "topic": f"{parent_topic_name or 'topics'}_batch_{i}_attempt_{attempt}",
        "batch_items": batch,
        "response_schema": output_schema,
        "log_prefix_marker": "5" if is_opinion_categorization else "2",
    })

  return prompts


async def _process_topic_categorization(
    statements_to_categorize: list[Statement],
    model: GenaiModel,
    target_topics: list[Topic],
    additional_context: str | None = None,
) -> list[StatementRecord]:
  """Performs a single round of categorization against a flat list of topics.

  This function orchestrates the categorization of a batch of statements by
  generating
  the appropriate prompt (for topics) and executing the calls to the
  language model concurrently.

  Args:
      statements_to_categorize: The statements to be categorized.
      model: The language model to use.
      target_topics: The list of topics to categorize into.
      additional_context: Optional context for the model prompt.

  Returns:
      A list of StatementRecord objects with the results from the model.
  """
  instructions = _topic_categorization_prompt(target_topics)

  uncategorized_for_retry: list[Statement] = list(statements_to_categorize)
  successfully_categorized_llm_records: list[StatementRecord] = []
  output_schema = StatementRecordList

  logging.info(f"Categorizing statements into {len(target_topics)} topics.")

  logging.info(
      "Performing categorization into topics:"
      f" {[t.name for t in target_topics[:20]]}"
  )

  for attempt in range(1, MAX_LLM_RETRIES + 1):
    if not uncategorized_for_retry:
      break

    # Create batches
    batches = _create_token_based_batches(
        uncategorized_for_retry, MAX_BATCH_TOKENS, max_items=MAX_ITEMS_PER_BATCH
    )
    prompts = []

    logging.info(
        f"Split {len(uncategorized_for_retry)} items into"
        f" {len(batches)} batches."
    )

    for i, batch in enumerate(batches):
      # Prepare the data for the model prompt.
      statements_for_model_prompt_data: list[str] = []

      # Otherwise, use the full statement text.
      if not statements_for_model_prompt_data:
        statements_for_model_prompt_data = [
            json.dumps({"id": s.id, "text": s.text}) for s in batch
        ]

      prompt_str = get_prompt(
          instructions, statements_for_model_prompt_data, additional_context
      )

      prompts.append({
          "prompt": prompt_str,
          "topic": f"topics_batch_{i}_attempt_{attempt}",
          "batch_items": batch,
          "response_schema": output_schema,
          "log_prefix_marker": "2",
      })

    if not prompts:
      if uncategorized_for_retry:
        logging.warning(
            "No prompts generated for remaining uncategorized items. Stopping"
            " retry loop."
        )
      break

    # Call GenaiModel
    def _parser(resp, job):
      parsed_wrapper = parse_response(resp["text"], job["response_schema"])
      return parsed_wrapper.items

    results_df, _ = await model.process_prompts_concurrently(
        prompts,
        response_parser=_parser,
        max_concurrent_calls=MAX_CONCURRENT_CALLS,
    )

    # Process results
    current_batch_uncategorized = []

    # We need to map results back to batches to find missing items
    # prompts list matches results_df rows order if we iterate results_df['result']?
    # process_prompts_concurrently preserves order.

    if len(results_df) != len(prompts):
      logging.error(
          f"Result count mismatch: expected {len(prompts)}, got"
          f" {len(results_df)}"
      )
      # Add all items to retry
      current_batch_uncategorized.extend(uncategorized_for_retry)
      continue

    for i, row in results_df.iterrows():
      result = row["result"]
      batch_items = prompts[i]["batch_items"]

      if isinstance(result, list):  # list[StatementRecord]
        processed_result = _process_categorized_llm_records(
            result,
            statements_to_categorize,  # original full list
            batch_items,  # current batch
            target_topics,
        )

        valid_records = processed_result["valid_records"]
        needs_retry = processed_result["needs_retry_statements"]

        successfully_categorized_llm_records.extend(valid_records)

        current_batch_uncategorized.extend(needs_retry)

      else:
        if "error" in result:
          logging.error(
              f"Batch failed or invalid result: {result['error']}."
              " Retrying entire batch."
          )
        else:
          logging.error(
              f"Batch failed or invalid result: {type(result)}."
              f" Result: {str(result)[:500]}..."
              " Retrying entire batch."
          )
        current_batch_uncategorized.extend(batch_items)

    uncategorized_for_retry = current_batch_uncategorized

    # Deduplicate retry list
    seen = set()
    deduped_retry = []
    for s in uncategorized_for_retry:
      if s.id not in seen:
        deduped_retry.append(s)
        seen.add(s.id)
    uncategorized_for_retry = deduped_retry

    if not uncategorized_for_retry:
      break

    if attempt < MAX_LLM_RETRIES:
      logging.warning(
          f"Attempt {attempt}: {len(uncategorized_for_retry)} uncategorized"
          " items need retry. Retrying in"
          f" {WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS}s..."
      )
      await asyncio.sleep(WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS)

  if uncategorized_for_retry:
    item_ids = [s.id for s in uncategorized_for_retry]
    logging.warning(
        "\n"
        + ("!" * 80)
        + f"\nMax retries reached for {len(uncategorized_for_retry)} items."
        + " Defaulting them to 'Other'.\n"
        + f"Item IDs: {item_ids[:20]}\n"
        + ("!" * 80)
    )

    other_topic = FlatTopic(name="Other")

    for statement in uncategorized_for_retry:
      new_record = StatementRecord(id=statement.id, topics=[other_topic])
      successfully_categorized_llm_records.append(new_record)

  return successfully_categorized_llm_records


def _process_categorized_llm_records(
    llm_output_records: list[StatementRecord],
    all_original_input_statements: list[Statement],
    statements_in_current_batch: list[Statement],
    target_topics_or_opinions: list[Topic],
) -> dict[str, Union[list[StatementRecord], list[Statement]]]:
  """Validates and processes the raw records from the language model.

  This function separates valid records from invalid ones, identifies which
  statements are missing from the response, and prepares a list of statements
  that need to be retried.

  Returns:
      A dictionary containing valid records and statements that need a retry.
  """
  valid_records, invalid_records = _validate_llm_records(
      llm_output_records,
      all_original_input_statements,
      target_topics_or_opinions,
  )

  missing_statements = _find_missing_from_llm_response(
      llm_output_records, statements_in_current_batch
  )

  ids_of_invalid_records = {rec.id for rec in invalid_records}
  statements_needing_retry_due_to_invalid_data = [
      s for s in statements_in_current_batch if s.id in ids_of_invalid_records
  ]

  all_needs_retry_statements = (
      missing_statements + statements_needing_retry_due_to_invalid_data
  )

  # Deduplicate the list of statements to retry.
  deduped_needs_retry_statements: list[Statement] = []
  seen_ids_for_retry: set[str] = set()
  for statement_item in all_needs_retry_statements:
    if statement_item.id not in seen_ids_for_retry:
      deduped_needs_retry_statements.append(statement_item)
      seen_ids_for_retry.add(statement_item.id)

  return {
      "valid_records": valid_records,
      "needs_retry_statements": deduped_needs_retry_statements,
  }


def _validate_llm_records(
    llm_records: list[StatementRecord],
    all_original_input_statements: list[Statement],
    target_topics_or_opinions: list[Topic],
) -> tuple[list[StatementRecord], list[StatementRecord]]:
  """Validates LLM records against a set of rules.

  Checks for:
  - Records with IDs not present in the original input.
  - Records with an empty list of topics.
  - Records with topic names that are not in the list of expected topics.

  Returns:
      A tuple of two lists: records that passed validation and those that
      failed.
  """
  passed_validation: list[StatementRecord] = []
  failed_validation: list[StatementRecord] = []
  original_statement_ids = {s.id for s in all_original_input_statements}
  # Create a set of valid topic names for quick lookup.
  valid_topic_names = {t.name for t in target_topics_or_opinions}

  for record in llm_records:
    if _is_extra_statement_record(record.id, original_statement_ids):
      continue
    if _has_empty_topics_in_record(record):
      failed_validation.append(record)
      continue
    if _has_invalid_topic_names_in_record(record, valid_topic_names):
      failed_validation.append(record)
      continue
    passed_validation.append(record)
  return passed_validation, failed_validation


def _is_extra_statement_record(
    record_id: str, input_statement_ids: set[str]
) -> bool:
  """Checks if a record ID from the LLM is an unexpected extra one."""
  if record_id not in input_statement_ids:
    logging.debug(f"LLM record for extra statement id: {record_id}")
    return True
  return False


def _has_empty_topics_in_record(record: StatementRecord) -> bool:
  """Checks if an LLM record has an empty list of topics."""
  if not record.topics:
    logging.debug(
        f"LLM record for ID {record.id} has empty topics list:"
        f" {record.model_dump_json(exclude_none=True)}"
    )
    return True
  return False


def _has_invalid_topic_names_in_record(
    record: StatementRecord, valid_topic_names: set[str]
) -> bool:
  """Checks if an LLM record contains topic names not in the valid set.

  This method validates the topics assigned to the record against the provided
  set of valid topic names. It attempts to auto-correct minor mismatches
  (e.g., case sensitivity, whitespace, or close fuzzy matches) to improve
  robustness against LLM hallucinations or typos.

  Side Effects:
      This method MODIFIES the `record` input in-place. If a close match is
      found for an invalid topic, the topic name in the record is updated to
      the canonical valid name. This is acceptable here because `record` is a
      transient data object parsed from the LLM response, and we want to
      canonicalize the data before it is merged into the main data structures.

  Returns:
      True if the record contains any topic names that are invalid and could
      not be auto-corrected. False otherwise.
  """
  for i, assigned_topic in enumerate(record.topics):
    name = assigned_topic.name
    if name == "Other":
      continue

    if name in valid_topic_names:
      continue

    # Try fuzzy matching
    # 1. Normalize (lowercase, strip)
    normalized_name = name.lower().strip()
    valid_map = {v.lower().strip(): v for v in valid_topic_names}

    if normalized_name in valid_map:
      # Exact match after normalization -> Auto-correct
      record.topics[i].name = valid_map[normalized_name]
      continue

    # 2. Difflib close match
    matches = difflib.get_close_matches(
        name, valid_topic_names, n=1, cutoff=0.9
    )
    if matches:
      # High confidence match -> Auto-correct
      logging.info(f"Auto-correcting topic '{name}' to '{matches[0]}'")
      record.topics[i].name = matches[0]
      continue

    logging.debug(
        f"LLM record for ID {record.id} (quote id:"
        f" '{record.quote_id if record.quote_id else 'N/A'}') has an invalid"
        f" topic: '{assigned_topic.name}'. Valid:"
        f" {list(valid_topic_names)[:20]}"
    )
    return True
  return False


def _find_missing_from_llm_response(
    llm_records: list[StatementRecord],
    statements_sent_to_llm_batch: list[Statement],
) -> list[Statement]:
  """Finds statements that were sent to the model but were not in the response."""
  processed_ids = {record.id for record in llm_records}
  missing_statements = [
      s for s in statements_sent_to_llm_batch if s.id not in processed_ids
  ]
  if missing_statements:
    logging.warning(
        f"Missing {len(missing_statements)} statement IDs in model's response"
    )
    logging.debug(
        "Missing statements (up to 20):"
        f" {[s.id for s in missing_statements[:20]]}"
    )

  return missing_statements


def _merge_opinions_into_statements_inplace(
    input_statements_map: dict[str, Statement],
    categorized_llm_records: list[StatementRecord],
    parent_topic: Topic,
) -> None:
  """Merges categorized opinions back into the main list of statements (in-place).

  This function updates the topic structure of quotes to reflect the newly
  assigned opinions, nesting them under the parent topic.
  """
  for llm_statement in categorized_llm_records:
    input_statement = input_statements_map.get(llm_statement.id)

    if not input_statement:
      logging.error(
          f"Statement ID {llm_statement.id} not found in map during merge."
      )
      continue

    newly_assigned_opinions = llm_statement.topics

    # Opinion categorization is done on quotes. Find the right quote and update its topic
    # to be a NestedTopic that contains the parent topic and the assigned opinions as subtopics.
    if llm_statement.quote_id and input_statement.quotes:
      found_and_updated_quote = False

      # First, try to find an exact match on quote ID
      for i, input_quote in enumerate(input_statement.quotes):
        if (
            input_quote.id == llm_statement.quote_id
            and input_quote.topic.name == parent_topic.name
        ):
          input_statement.quotes[i].topic = NestedTopic(
              name=parent_topic.name,
              subtopics=newly_assigned_opinions,
          )
          found_and_updated_quote = True
          break

      # Fallback: If no exact match, try to match by topic name if unique
      if not found_and_updated_quote:
        potential_matches = [
            (i, q)
            for i, q in enumerate(input_statement.quotes)
            if q.topic.name == parent_topic.name
        ]

        if len(potential_matches) == 1:
          i, input_quote = potential_matches[0]
          logging.warning(
              f"Loose match for statement {llm_statement.id}: LLM quote ID"
              f" '{llm_statement.quote_id}' mismatched but topic"
              f" '{parent_topic.name}' matched unique quote '{input_quote.id}'."
              " Proceeding with assignment."
          )
          input_statement.quotes[i].topic = NestedTopic(
              name=parent_topic.name,
              subtopics=newly_assigned_opinions,
          )
          found_and_updated_quote = True

      if not found_and_updated_quote:
        logging.warning(
            f"Problem with statement (id: {llm_statement.id}) and corresponding"
            " input quote_id-topic pairs:"
            + ", ".join([
                f"('{q.id}' - '{q.topic.name})'" for q in input_statement.quotes
            ])
        )
        logging.warning(
            f"Opinion categorization failed. For statement {llm_statement.id},"
            f" LLM quote '{llm_statement.quote_id}' or provided topic"
            f" '{parent_topic.name}' didn't match any input quote_id-topic"
            " pair. Skipping this opinion assignment."
        )
        continue
