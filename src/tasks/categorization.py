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
import time
import difflib
import pandas as pd
from collections import defaultdict
from typing import Any, Callable, Coroutine, Iterable, Set, Tuple, Union, cast
from more_itertools import batched
from src.tasks.topic_modeling_util import parse_response
from pydantic import TypeAdapter, ValidationError
from src import prompts
from src.models import decision
from src.models import genai_model
from src.models import systemone
from src.sensemaker_utils import execute_concurrently, get_prompt
from src import runner_utils
from src.models.custom_types import (
    Quote,
    Statement,
    StatementRecord,
    Topic,
    FlatTopic,
    NestedTopic,
    StatementRecordList,
    QuoteOpinionRecord,
    QuoteOpinionRecordList,
    OpinionResponseSchema,
    OpinionDuplicateDecision,
    OpinionDedupResponse,
)
from src.evals.autorater_evals import (
    prepare_opinion_eval_prompts,
    parse_eval_response,
)
from src.tasks import topic_modeling, topic_modeling_util
from src.tasks.topic_modeling import learn_topics, learn_opinions


# Use dynamic batching based on token count
# Target 20k tokens per batch (safe for output generation limits)
MAX_BATCH_TOKENS = int(os.getenv("MAX_BATCH_TOKENS", 20000))
MAX_ITEMS_PER_BATCH = int(os.getenv("MAX_ITEMS_PER_BATCH", 5))

# Number of autorater rejections a single quote may receive before it is
# defaulted to 'Other'. Counted per (statement id, quote id), i.e. per quote.
DEFAULT_MAX_AUTORATER_RETRIES = 3
# Minimum integer autorater score (rubric is 0-4) for an opinion assignment to
# pass.
DEFAULT_AUTORATER_MIN_SCORE = 4

# Concurrency is driven dynamically via CLI flags passed down to tasks.


def normalize_opinion_menu(opinions: list[Topic]) -> list[Topic]:
  """Returns an opinion menu with unique names and exactly one trailing 'Other'.

  Args:
    opinions: Opinion topics as stored in the topic -> opinions map.

  Returns:
    A new list: first occurrence of each distinct (case-insensitive) non-Other
    name, in original order, followed by a single FlatTopic('Other').
  """
  seen: set[str] = set()
  normalized: list[Topic] = []
  for op in opinions:
    key = op.name.strip().lower()
    if key == "other" or key in seen:
      continue
    seen.add(key)
    normalized.append(op)
  normalized.append(FlatTopic(name="Other"))
  return normalized


def is_other_opinion(name: str) -> bool:
  """Returns True if `name` denotes the 'Other' opinion (case-insensitive)."""
  return name.strip().lower() == "other"


def strip_other_opinions(opinions: list[Topic]) -> list[Topic]:
  """Returns `opinions` without any 'Other' entries, order preserved."""
  return [op for op in opinions if not is_other_opinion(op.name)]


async def categorize_topics(
    statements: list[Statement],
    model: genai_model.GenaiModel,
    current_topics: list[Topic] | None = None,
    additional_context: str | None = None,
    max_concurrent_calls: int | None = None,
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
      max_concurrent_calls: Optional maximum number of concurrent LLM calls.
        If None, the model's default is used.

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
        input_statements,
        model,
        additional_context=additional_context,
        max_concurrent_calls=max_concurrent_calls,
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
      max_concurrent_calls=max_concurrent_calls,
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
    model: genai_model.GenaiModel,
    additional_context: str | None = None,
    max_concurrent_calls: int | None = None,
    offer_other_opinion: bool = True,
) -> dict[str, Any]:
  """Learns opinions for all topics globally using concurrent batching.

  Args:
      statements_with_topics_and_quotes: The list of statements with topics and quotes.
      topics_to_process: The list of topics to learn opinions for.
      model: The language model to use for learning opinions.
      additional_context: Optional text to provide additional context to the
        model.
      max_concurrent_calls: Optional maximum number of concurrent LLM calls.
        If None, the model's default is used.
      offer_other_opinion: Whether to append an 'Other' opinion to learned menus.

  Returns:
      A dictionary containing the learned opinions for each topic.
  """
  logging.info("Starting Global Opinion Learning Phase (Concurrent)...")

  final_topic_map: dict[str, NestedTopic] = {}
  topics_needing_learning: list[Topic] = []
  for t in topics_to_process:
    if t.name == "Other":
      continue
    if isinstance(t, NestedTopic) and t.subtopics:
      if offer_other_opinion and not any(sub.name == "Other" for sub in t.subtopics):
        t.subtopics.append(FlatTopic(name="Other"))
      final_topic_map[t.name] = t
    else:
      topics_needing_learning.append(t)

  if final_topic_map and not topics_needing_learning:
    logging.info(
        "All topics already have predefined opinions. Skipping opinion"
        " learning."
    )
    return final_topic_map

  # 1. Prepare Chunks for all topics concurrently
  async def _prepare_topic_chunks(topic: Topic):
    # Gather relevant quotes for this topic
    quotes = []
    for s in statements_with_topics_and_quotes:
      for q in s.quotes:
        if q.topic.name == topic.name:
          quotes.append(q)

    if not quotes:
      return topic, [], None

    prompt_input_data = [f"<quote>{q.text}</quote>" for q in quotes]
    instructions = prompts.get_topic_modeling_opinions_prompt(topic.name)

    # This might call the model for token counting
    chunks = await topic_modeling_util.create_chunks(
        model, instructions, prompt_input_data, additional_context
    )
    return topic, chunks, instructions

  # Gather chunks for all topics
  logging.info(
      f"Preparing chunks for {len(topics_needing_learning)} topics..."
  )
  preparation_tasks = [_prepare_topic_chunks(t) for t in topics_needing_learning]
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
          "log_prefix_marker": "4 (Opinion Identification)",
          "progress_index": (
              i
          ),  # Approximate within topic, or could use global index
      })
      topic_job_indices[topic.name].append(job_index)

  if not generation_prompts:
    logging.warning("No chunks generated for opinion learning.")
    return final_topic_map

  logging.info(
      f"Generated {len(generation_prompts)} opinion learning batches "
      f"across {len(topic_job_indices)} active topics."
  )

  all_stage_stats = []
  total_wall_delay = 0.0
  start_time = time.time()

  # 3. Execute Phase 1: Initial Generation
  def _parse_opinion_response(resp, job):
    return parse_response(resp["text"], job["response_schema"])

  results_df, stats, wall_delay, _ = await model.process_prompts_concurrently(
      generation_prompts,
      response_parser=_parse_opinion_response,
      skip_log=True,
      max_concurrent_calls=max_concurrent_calls,
  )
  all_stage_stats.extend(stats.to_dict("records"))
  total_wall_delay += wall_delay

  # 4. Process Results and Prepare Merges (Phase 2)
  merge_prompts = []
  # Map topic name to its merge job index
  merge_job_indices: dict[str, int] = {}

  # Group results by topic name
  # We can iterate through the dataframe and collect results
  topic_results: dict[str, list[NestedTopic]] = defaultdict(list)
  # Map topic name to its Topic object, for use when building merge jobs.
  topic_objs_by_name: dict[str, Topic] = {}

  # Initialize list for all expected topics
  for t in topics_needing_learning:
    topic_results[t.name] = []
    topic_objs_by_name[t.name] = t

  failed_opinion_chunks: dict[str, int] = defaultdict(int)

  for _, row in results_df.iterrows():
    topic_obj = row.get("topic_obj")
    if not topic_obj:
      logging.warning(f"Result missing topic_obj: {row}")
      continue

    result = row["result"]
    topic_name = topic_obj.name
    topic_objs_by_name[topic_name] = topic_obj

    if isinstance(result, dict) and result.get("error"):
      failed_opinion_chunks[topic_name] += 1
      logging.warning(
          f"Opinion chunk failed for topic '{topic_name}':"
          f" {result.get('error')}"
      )
      continue

    parsed_topic = None
    if isinstance(result, OpinionResponseSchema):
      parsed_topic = NestedTopic(name=result.name, subtopics=result.subtopics)
    elif isinstance(result, NestedTopic):
      parsed_topic = result
    elif isinstance(result, dict) and not result.get("error"):
      subtopics = result.get("subtopics", [])
      subtopic_objs = [
          s if isinstance(s, FlatTopic) else FlatTopic(name=s.get("name", "Other") if isinstance(s, dict) else str(s))
          for s in subtopics
      ]
      parsed_topic = NestedTopic(name=result.get("name", topic_name), subtopics=subtopic_objs)

    if parsed_topic:
      # Ensure an "Other" opinion exists for each topic if offer_other_opinion is True.
      if offer_other_opinion and not any(sub.name == "Other" for sub in parsed_topic.subtopics):
        parsed_topic.subtopics.append(FlatTopic(name="Other"))
      topic_results[topic_name].append(parsed_topic)

  if failed_opinion_chunks:
    total_failed = sum(failed_opinion_chunks.values())
    logging.warning(
        f"Opinion learning failed for {total_failed} of"
        f" {len(generation_prompts)} chunks across"
        f" {len(failed_opinion_chunks)} topics: {dict(failed_opinion_chunks)}."
        " Quotes in failed chunks contribute no opinions."
    )

  # Now iterate over grouped results
  for topic_name, partial_topics in topic_results.items():
    if not partial_topics:
      final_topic_map[topic_name] = NestedTopic(
          name=topic_name,
          subtopics=[FlatTopic(name="Other")] if offer_other_opinion else [],
      )
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
        final_topic_map[topic_name] = NestedTopic(
            name=topic_name,
            subtopics=[FlatTopic(name="Other")] if offer_other_opinion else [],
        )
        continue

      # Prepare Merge Prompt
      target_topic_obj = topic_objs_by_name.get(topic_name)
      merge_instructions = prompts.get_topic_modeling_merge_opinions_prompt(
          topic_name
      )
      prompt_str = get_prompt(
          merge_instructions, combined_opinions, additional_context
      )

      merge_idx = len(merge_prompts)
      merge_prompts.append({
          "prompt": prompt_str,
          "topic_obj": target_topic_obj,
          "response_schema": (
              OpinionResponseSchema
          ),  # Merge prompt expects same structure (JSON list of opinions)
          "log_prefix_marker": "4 (Opinion Identification)",
          "is_merge": True,
      })
      merge_job_indices[topic_name] = merge_idx

  # 5. Execute Phase 2: Merging (if any)
  if merge_prompts:
    logging.info(f"Merging opinions for {len(merge_prompts)} topics...")

    merge_results_df, stats, wall_delay, _ = (
        await model.process_prompts_concurrently(
            merge_prompts,
            response_parser=_parse_opinion_response,
            skip_log=True,
            max_concurrent_calls=max_concurrent_calls,
        )
    )
    all_stage_stats.extend(stats.to_dict("records"))
    total_wall_delay += wall_delay

    for topic_name, idx in merge_job_indices.items():
      result = merge_results_df.iloc[idx]["result"]

      final_res = None
      if isinstance(result, OpinionResponseSchema):
        final_res = NestedTopic(name=result.name, subtopics=result.subtopics)
      elif isinstance(result, NestedTopic):
        final_res = result
      elif isinstance(result, dict) and not result.get("error"):
        subtopics = result.get("subtopics", [])
        subtopic_objs = [
            s if isinstance(s, FlatTopic) else FlatTopic(name=s.get("name", "Other") if isinstance(s, dict) else str(s))
            for s in subtopics
        ]
        final_res = NestedTopic(name=result.get("name", topic_name), subtopics=subtopic_objs)

      if final_res:
        if not final_res.subtopics:
          final_res.subtopics = []
        if offer_other_opinion and not any(sub.name == "Other" for sub in final_res.subtopics):
          final_res.subtopics.append(FlatTopic(name="Other"))
        final_topic_map[topic_name] = final_res
      else:
        logging.error(
            f"Merge failed for topic '{topic_name}'. Falling back to combined partials."
        )
        fallback_subs = []
        seen = set()
        for pt in topic_results.get(topic_name, []):
          for sub in (pt.subtopics or []):
            if sub.name not in seen:
              fallback_subs.append(sub)
              seen.add(sub.name)
        if offer_other_opinion and not any(s.name == "Other" for s in fallback_subs):
          fallback_subs.append(FlatTopic(name="Other"))
        final_topic_map[topic_name] = NestedTopic(
            name=topic_name, subtopics=fallback_subs
        )

  if all_stage_stats:
    model.log_stats_summary(
        all_stage_stats,
        "4 (Opinion Identification)",
        total_wall_delay,
        time.time() - start_time,
    )

  return final_topic_map


async def categorize_opinions(
    statements_with_topics_and_quotes: list[Statement],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, Any],
    model: genai_model.GenaiModel,
    additional_context: str | None = None,
    run_autoraters: bool = True,
    max_concurrent_calls: int | None = None,
    max_autorater_retries: int = DEFAULT_MAX_AUTORATER_RETRIES,
    autorater_min_score: int = DEFAULT_AUTORATER_MIN_SCORE,
    offer_other_opinion: bool = True,
) -> Iterable[Statement]:
  """Categorizes opinions within the provided topics for each statement.

  Args:
    statements_with_topics_and_quotes: list of statements to categorize.
    topics_to_process: list of topics to categorize opinions within.
    topic_to_opinions_map: Map of topic names to their learned opinion structures.
    model: GenaiModel instance for LLM calls.
    additional_context: Additional context to provide to the LLM.
    run_autoraters: Whether to run autorater evaluations.
    max_concurrent_calls: Optional maximum number of concurrent LLM calls.
    max_autorater_retries: Number of autorater rejections a single quote may
      receive before it is defaulted to 'Other'.
    autorater_min_score: Minimum integer autorater score (0-4 rubric) for an
      opinion assignment to pass.
    offer_other_opinion: Whether to offer an 'Other' option to the model.

  Returns:
    List of statements with categorized opinions.
  """
  if max_autorater_retries < 0:
    raise ValueError(
        f"max_autorater_retries must be >= 0, got {max_autorater_retries}"
    )
  if not (0 <= autorater_min_score <= 4):
    raise ValueError(
        f"autorater_min_score must be between 0 and 4, got {autorater_min_score}"
    )

  logging.info("Starting Global Opinion Categorization Phase...")

  decision_model = decision.decision_client()
  if decision_model is not None:
    logging.info(
        "Categorizing opinions with System One model %s.",
        decision_model.model,
    )
    return await _categorize_opinions_with_decision(
        statements_with_topics_and_quotes,
        topics_to_process,
        topic_to_opinions_map,
        additional_context,
        run_autoraters,
        decision_model,
        autorater_min_score=autorater_min_score,
    )

  # Pre-compute valid input statements map for O(1) lookups and O(1) in-place updates
  input_statements_map = {
      s.id: s.model_copy(deep=True) for s in statements_with_topics_and_quotes
  }

  # Track work queue: Topic Name -> List of IDs to process for that topic
  # We use IDs to track statements to avoid deep copying lists repeatedly
  # Initial work queue has all statements for all topics following this rule:
  # If topic is "education", the statement has a quote from "education" topic,
  # then it'll be included in the work queue for "education" topic.
  # However, if the statement has 2 quotes whose topics are "health" and
  # "economy", then this statment will not be included in the work queue for
  # "education", or any other topic that is not "health" or "economy".
  topic_work_queue_ids: dict[str, list[str]] = dict()
  for t in topics_to_process:
    topic_work_queue_ids[t.name] = [
        sid
        for sid, s in input_statements_map.items()
        if t.name in [q.topic.name for q in s.quotes]
    ]

  # Autorater rejections per quote, keyed by (statement id, quote id).
  autorater_retry_counts: dict[tuple[str, str], int] = {}

  total_statements = len(input_statements_map)
  finalized_count_events = 0

  all_stage_5_stats = []
  total_stage_5_wall_delay = 0.0

  all_stage_6_stats = []
  total_stage_6_wall_delay = 0.0
  total_stage_6_duration = 0.0

  stage_start_time = time.time()

  if not offer_other_opinion:
    _assign_other_for_empty_menus(
        topic_work_queue_ids,
        topics_to_process,
        topic_to_opinions_map,
        input_statements_map,
    )

  for attempt in range(1, model.max_llm_retries + 1):
    if not topic_work_queue_ids:
      break

    all_prompts = _prepare_opinion_prompts_for_pending_work(
        topic_work_queue_ids,
        topics_to_process,
        topic_to_opinions_map,
        input_statements_map,
        additional_context,
        attempt,
        offer_other_opinion=offer_other_opinion,
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

    results_df, stats, wall_delay, duration = (
        await model.process_prompts_concurrently(
            all_prompts,
            response_parser=_parser,
            skip_log=True,
            max_concurrent_calls=max_concurrent_calls,
        )
    )
    all_stage_5_stats.extend(stats.to_dict("records"))
    total_stage_5_wall_delay += wall_delay

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
          stage_6_stats,
          stage_6_wall_delay,
          stage_6_duration,
      ) = await _run_opinion_autoraters(
          model=model,
          autorater_candidates=autorater_candidates,
          input_statements_map=input_statements_map,
          autorater_retry_counts=autorater_retry_counts,
          max_autorater_retries=max_autorater_retries,
          autorater_min_score=autorater_min_score,
          max_concurrent_calls=max_concurrent_calls,
      )
      finalized_count_events += phase2_finalized_count
      all_stage_6_stats.extend(stage_6_stats)
      total_stage_6_wall_delay += stage_6_wall_delay
      total_stage_6_duration += stage_6_duration

      # Merge autorater retries into immediate retries
      for t_name, stats_item in autorater_retries.items():
        if t_name not in immediate_retries:
          immediate_retries[t_name] = []
        immediate_retries[t_name].extend(stats_item)

    # --- Phase 3: Construct Next Retry Queue ---

    # Process immediate retries (from LLM failures or Autorater retryable failures)
    next_topic_work_queue_ids: dict[str, list[str]] = {}
    for t_name, stats_item in immediate_retries.items():
      if t_name not in next_topic_work_queue_ids:
        next_topic_work_queue_ids[t_name] = []

      # Dedupe
      for s in stats_item:
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

    if attempt < model.max_llm_retries:
      retry_counts = {k: len(v) for k, v in topic_work_queue_ids.items()}
      total_retrying = sum(retry_counts.values())
      logging.info(
          f"Progress: {finalized_count_events}/{total_statements} statements"
          " finalized (events)."
      )
      logging.warning(
          f"Retrying {total_retrying} statements across {len(retry_counts)}"
          " topics. Sleeping"
          f" {genai_model.WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS}s..."
      )
      await asyncio.sleep(genai_model.WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS)

  # Handle exhausted retries - Default to 'Other'
  _assign_defaults_for_exhausted_retries(
      topic_work_queue_ids,
      topics_to_process,
      input_statements_map,
  )

  stage_duration = time.time() - stage_start_time

  if all_stage_5_stats:
    model.log_stats_summary(
        all_stage_5_stats,
        "5 (Opinion Categorization)",
        total_stage_5_wall_delay,
        stage_duration,
    )

  if all_stage_6_stats:
    model.log_stats_summary(
        all_stage_6_stats,
        "6 (Opinion Evaluation)",
        total_stage_6_wall_delay,
        total_stage_6_duration,
    )

  return input_statements_map.values()


def _assign_other_for_empty_menus(
    topic_work_queue_ids: dict[str, list[str]],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, Any],
    input_statements_map: dict[str, Statement],
) -> None:
  """Assigns 'Other' to quotes of topics with no non-Other opinions.

  Used when 'Other' is not offered: such topics have nothing to offer the LLM,
  so their quotes go straight to 'Other' and the topics are removed from the
  work queue (mutated in place).

  Args:
    topic_work_queue_ids: Topic name -> pending statement ids (mutated).
    topics_to_process: Topics being categorized.
    topic_to_opinions_map: Topic name -> NestedTopic of opinions.
    input_statements_map: Statement id -> Statement (updated in place).
  """
  for topic_name in list(topic_work_queue_ids):
    learned = topic_to_opinions_map.get(topic_name)
    menu = strip_other_opinions(
        (learned.subtopics if learned and learned.subtopics else []) or []
    )
    if menu:
      continue
    topic = next((t for t in topics_to_process if t.name == topic_name), None)
    sids = topic_work_queue_ids.pop(topic_name)
    if not topic:
      continue
    records = []
    for sid in sids:
      stmt = input_statements_map.get(sid)
      if not stmt:
        continue
      for q in stmt.quotes:
        if q.topic.name == topic_name:
          records.append(
              StatementRecord(
                  id=sid, quote_id=q.id, topics=[FlatTopic(name="Other")]
              )
          )
    logging.warning(
        f"Topic '{topic_name}' has no non-Other opinions and 'Other' is not"
        f" offered; assigning {len(records)} quotes to 'Other'."
    )
    if records:
      _merge_opinions_into_statements_inplace(
          input_statements_map, records, topic
      )


def _prepare_opinion_prompts_for_pending_work(
    topic_work_queue_ids: dict[str, list[str]],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, Any],
    input_statements_map: dict[str, Statement],
    additional_context: str | None,
    attempt: int,
    offer_other_opinion: bool = True,
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
    if offer_other_opinion:
      opinions = normalize_opinion_menu(opinions)
    else:
      opinions = strip_other_opinions(opinions)
      if not opinions:
        logging.warning(
            f"Topic '{topic.name}' has no non-Other opinions to offer."
            " Skipping."
        )
        continue

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
        prompts.get_categorization_opinion_prompt(
            json.dumps([{"name": op.name} for op in opinions])
        ),
        QuoteOpinionRecordList,
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

    logging.debug(f"Opinion results for {topic_name}: {result}")
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
    model: genai_model.GenaiModel,
    autorater_candidates: list[dict[str, Any]],
    input_statements_map: dict[str, Statement],
    autorater_retry_counts: dict[tuple[str, str], int],
    max_autorater_retries: int = DEFAULT_MAX_AUTORATER_RETRIES,
    autorater_min_score: int = DEFAULT_AUTORATER_MIN_SCORE,
    max_concurrent_calls: int | None = None,
) -> Tuple[int, dict[str, list[Statement]], list[dict], float, float]:
  """Runs autoraters concurrently and processes results."""
  logging.info(
      f"Running autoraters for {len(autorater_candidates)} batches"
      " concurrently..."
  )

  finalized_count = 0
  autorater_retries: dict[str, list[Statement]] = {}

  model.total_wall_delay = 0.0
  stage_start_time = time.time()

  # 1. Start Workers
  (
      queue,
      workers,
      results_list,
      stats_list,
      stop_event,
  ) = model.start_concurrent_workers(
      response_parser=parse_eval_response,
      max_concurrent_calls=max_concurrent_calls,
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
        p["retry_attempts"] = model.max_llm_retries

      p["log_prefix_marker"] = "6 (Opinion Evaluation)"

      queue.put_nowait(p)

  # 3. Signal Completion
  for _ in range(len(workers)):
    queue.put_nowait(None)

  # 4. Wait for Workers
  await asyncio.gather(*workers)

  stage_duration = time.time() - stage_start_time
  # Skip direct logging to allow Stage 5 to aggregate Stage 6 summaries.
  wall_delay = model.total_wall_delay

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

    if score is not None and score >= autorater_min_score:
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

      # Count rejections per quote (not per statement): a statement with
      # quotes in several topics must not exhaust one quote's budget through
      # rejections of its other quotes.
      retry_key = (stat.id, original_record.quote_id or parent_topic_name)
      autorater_retry_counts[retry_key] = (
          autorater_retry_counts.get(retry_key, 0) + 1
      )

      if autorater_retry_counts[retry_key] >= max_autorater_retries:
        logging.warning(
            f"Quote {original_record.quote_id} (statement {stat.id}) failed"
            f" autorater {max_autorater_retries} times. Defaulting to 'Other'."
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

  return (
      finalized_count,
      autorater_retries,
      stats_list,
      wall_delay,
      stage_duration,
  )


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


def _apply_dedup_decisions(
    new_opinions: list[FlatTopic],
    num_existing: int,
    decisions: list[OpinionDuplicateDecision],
) -> tuple[list[FlatTopic], dict[str, str]]:
  """Applies LLM dedup decisions to a topic's new opinions.

  Invalid references (unknown ids, self/forward references to new opinions)
  are ignored, i.e. the new opinion is kept.

  Args:
    new_opinions: The new opinions (N1..Nm).
    num_existing: Number of existing opinions (E1..En).
    decisions: LLM decisions.

  Returns:
    (kept new opinions, {dropped new opinion name: id it was merged into}).
  """
  dup_of: dict[int, str] = {}
  for d in decisions:
    m_new = re.fullmatch(r"\s*N(\d+)\s*", d.new_id or "")
    if not m_new or not d.duplicate_of:
      continue
    n_idx = int(m_new.group(1))
    if not 1 <= n_idx <= len(new_opinions):
      continue
    target = d.duplicate_of.strip().upper()
    m_e = re.fullmatch(r"E(\d+)", target)
    m_n = re.fullmatch(r"N(\d+)", target)
    if m_e and 1 <= int(m_e.group(1)) <= num_existing:
      dup_of[n_idx] = target
    elif m_n and 1 <= int(m_n.group(1)) < n_idx:
      dup_of[n_idx] = target
  kept, merged = [], {}
  for i, op in enumerate(new_opinions, start=1):
    if i in dup_of:
      merged[op.name] = dup_of[i]
    else:
      kept.append(op)
  return kept, merged


async def dedup_new_opinions_against_existing(
    new_by_topic: dict[str, list[FlatTopic]],
    existing_by_topic: dict[str, list[Topic]],
    model: genai_model.GenaiModel,
    max_concurrent_calls: int | None = None,
) -> dict[str, list[FlatTopic]]:
  """Drops new opinions that an LLM judge deems equivalent to existing ones.

  One judge call per topic (all topics concurrently). Each new opinion that
  is semantically equivalent to an existing opinion, or to an earlier new
  opinion, is dropped; its quotes are then picked up by the equivalent
  opinion during re-categorization. If the judge call fails for a topic, all
  of that topic's new opinions are kept (logged at WARNING).

  Args:
    new_by_topic: Topic name -> newly learned opinions.
    existing_by_topic: Topic name -> existing (non-Other) opinions.
    model: The model.
    max_concurrent_calls: Maximum concurrent LLM calls.

  Returns:
    Topic name -> kept new opinions.
  """
  jobs = []
  for topic_name, new_ops in new_by_topic.items():
    if not new_ops:
      continue
    existing = existing_by_topic.get(topic_name, [])
    jobs.append({
        "prompt": prompts.get_opinion_dedup_prompt(
            topic_name, [e.name for e in existing], [n.name for n in new_ops]
        ),
        "dedup_topic_name": topic_name,
        "response_schema": OpinionDedupResponse,
        "response_mime_type": "application/json",
        "log_prefix_marker": "5b (New Opinion Dedup)",
    })
  if not jobs:
    return new_by_topic

  def _parser(resp, job):
    parsed = parse_response(resp["text"], job["response_schema"])
    if parsed is None:
      raise ValueError("Empty dedup response")
    return parsed

  results_df, _, _, _ = await model.process_prompts_concurrently(
      jobs,
      response_parser=_parser,
      max_concurrent_calls=max_concurrent_calls,
      retry_attempts=3,
      skip_log=True,
  )

  out = dict(new_by_topic)
  total_new = sum(len(v) for v in new_by_topic.values())
  total_merged = 0
  for _, row in results_df.iterrows():
    topic_name = row.get("dedup_topic_name")
    result = row.get("result")
    if not isinstance(result, OpinionDedupResponse):
      logging.warning(
          f"New-opinion dedup failed for topic '{topic_name}'; keeping all"
          f" {len(new_by_topic.get(topic_name, []))} new opinions."
      )
      continue
    kept, merged = _apply_dedup_decisions(
        new_by_topic[topic_name],
        len(existing_by_topic.get(topic_name, [])),
        result.decisions,
    )
    existing = existing_by_topic.get(topic_name, [])
    new_ops = new_by_topic[topic_name]
    for name, target in merged.items():
      idx = int(target[1:]) - 1
      target_name = existing[idx].name if target[0] == "E" else new_ops[idx].name
      logging.info(
          f"Dedup [{topic_name}]: merged new opinion '{name}' into"
          f" {target} '{target_name}'."
      )
    total_merged += len(merged)
    out[topic_name] = kept
  logging.info(
      f"New-opinion dedup merged {total_merged} of {total_new} new opinions"
      " into equivalent existing/new opinions."
  )
  return out


async def recategorize_other_quotes(
    statements: list[Statement],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, NestedTopic],
    model: genai_model.GenaiModel,
    additional_context: str | None = None,
    pass_existing_opinions: bool = True,
    run_autoraters: bool = True,
    recategorize_all: bool = True,
    max_concurrent_calls: int | None = None,
    dedup_new_opinions: bool = False,
    offer_other_opinion: bool = True,
    **categorize_kwargs: Any,
) -> tuple[list[Statement], dict[str, NestedTopic]]:
  """Reruns opinion learning on 'Other' quotes and recategorizes.

  Args:
    statements: Statements after pass-1 opinion categorization.
    topics_to_process: Topics to process.
    topic_to_opinions_map: Topic name -> NestedTopic of pass-1 opinions. A new
      expanded map is returned; entries are replaced, not mutated.
    model: The model.
    additional_context: Optional additional prompt context.
    pass_existing_opinions: Tell the learner about the existing opinions.
    run_autoraters: Run autoraters during re-categorization.
    recategorize_all: Re-categorize ALL statements (else only 'Other' ones).
    max_concurrent_calls: Maximum concurrent LLM calls.
    dedup_new_opinions: If True, existing opinions are passed to the learner as
      a hard constraint and an LLM judge maps each new opinion that is equivalent
      to an existing (or earlier new) opinion onto it, dropping the duplicate.
    offer_other_opinion: Whether to append an 'Other' opinion to learned menus.
    **categorize_kwargs: Forwarded to `categorize_opinions`.

  Returns:
    (statements, expanded topic -> opinions map).
  """
  logging.info("Starting secondary opinion learning on 'Other' quotes...")

  other_quotes_by_topic: dict[str, list[Quote]] = {
      t.name: [] for t in topics_to_process if t.name != "Other"
  }
  other_statements_by_id: dict[str, Statement] = {}

  for s in statements:
    has_other = False
    for q in s.quotes:
      opinions = getattr(q.topic, "subtopics", [])
      is_other = any(o.name.lower() == "other" for o in opinions)
      if is_other and q.topic.name != "Other":
        other_quotes_by_topic.setdefault(q.topic.name, []).append(q)
        has_other = True
    if has_other:
      other_statements_by_id[s.id] = s

  if not any(other_quotes_by_topic.values()):
    logging.info(
        "No quotes found assigned to 'Other'. Skipping secondary pass."
    )
    return statements, topic_to_opinions_map

  topic_to_opinions_map = dict(topic_to_opinions_map)
  learned_new: dict[str, list[FlatTopic]] = {}
  existing_by_topic: dict[str, list[Topic]] = {}

  for t in topics_to_process:
    if t.name == "Other":
      continue
    quotes = other_quotes_by_topic.get(t.name, [])
    if not quotes:
      continue

    logging.info(
        f"Learning new opinions for {len(quotes)} 'Other' quotes in topic"
        f" '{t.name}'..."
    )

    existing_subtopics = [
        sub
        for sub in topic_to_opinions_map.get(
            t.name, NestedTopic(name=t.name, subtopics=[])
        ).subtopics
        if sub.name.lower() != "other"
    ]
    existing_by_topic[t.name] = existing_subtopics

    custom_context = additional_context or ""
    if pass_existing_opinions and existing_subtopics:
      instructions = prompts.get_topic_modeling_new_opinions_prompt(
          t.name, [sub.name for sub in existing_subtopics]
      )
    else:
      instructions = prompts.get_topic_modeling_opinions_prompt(t.name)

    prompt_input_data = [f"<quote>{q.text}</quote>" for q in quotes]
    chunks = await topic_modeling_util.create_chunks(
        model, instructions, prompt_input_data, custom_context
    )

    generation_prompts = []
    for i, chunk in enumerate(chunks):
      prompt_str = get_prompt(instructions, chunk, custom_context)
      generation_prompts.append({
          "prompt": prompt_str,
          "topic_obj": t,
          "chunk_index": i,
          "total_chunks": len(chunks),
          "response_schema": OpinionResponseSchema,
          "log_prefix_marker": f"5b (Other Opinion ID - {t.name})",
          "progress_index": i,
      })

    if not generation_prompts:
      continue

    def _parse_opinion_response(resp, job):
      return parse_response(resp["text"], job["response_schema"])

    results_df, _, _, _ = await model.process_prompts_concurrently(
        generation_prompts,
        response_parser=_parse_opinion_response,
        max_concurrent_calls=max_concurrent_calls,
        retry_attempts=3,
    )

    new_subtopics: list[FlatTopic] = []
    existing_names = {sub.name.lower() for sub in existing_subtopics}
    existing_names.add("other")

    failed_chunks = 0
    for _, row in results_df.iterrows():
      result = row["result"]
      if isinstance(result, OpinionResponseSchema):
        for sub in result.subtopics:
          if sub.name.lower() not in existing_names:
            new_subtopics.append(FlatTopic(name=sub.name))
            existing_names.add(sub.name.lower())
      else:
        failed_chunks += 1
    if failed_chunks:
      logging.warning(
          f"Secondary opinion learning failed for {failed_chunks} of"
          f" {len(generation_prompts)} chunks in topic '{t.name}'; their"
          " 'Other' quotes contribute no new opinions."
      )

    if new_subtopics:
      logging.info(
          f"Learned {len(new_subtopics)} new categories for topic '{t.name}': "
          + ", ".join(sub.name for sub in new_subtopics)
      )
      learned_new[t.name] = new_subtopics

  if dedup_new_opinions and learned_new:
    learned_new = await dedup_new_opinions_against_existing(
        learned_new,
        existing_by_topic,
        model,
        max_concurrent_calls=max_concurrent_calls,
    )

  for topic_name, new_subtopics in learned_new.items():
    if not new_subtopics:
      continue
    updated_subtopics = existing_by_topic.get(topic_name, []) + new_subtopics
    if offer_other_opinion:
      updated_subtopics.append(FlatTopic(name="Other"))
    topic_to_opinions_map[topic_name] = NestedTopic(
        name=topic_name, subtopics=updated_subtopics
    )

  if recategorize_all:
    logging.info(
        f"Re-categorizing ALL {len(statements)} statements across the newly"
        " expanded opinion set..."
    )
    statements_to_recategorize = [
        Statement(
            id=s.id,
            text=s.text,
            topics=s.topics,
            quotes=[
                Quote(
                    id=q.id,
                    text=q.text,
                    topic=(
                        q.topic.model_copy(deep=True)
                        if (q.topic and q.topic.name == "Other")
                        else FlatTopic(name=q.topic.name)
                        if q.topic
                        else None
                    ),
                )
                for q in s.quotes
            ] if s.quotes else [],
        )
        for s in statements
    ]
  else:
    statements_to_recategorize = list(other_statements_by_id.values())
    logging.info(
        f"Re-categorizing {len(statements_to_recategorize)} statements that had"
        " 'Other'..."
    )
    for s in statements_to_recategorize:
      if s.quotes:
        for q in s.quotes:
          opinions = getattr(q.topic, "subtopics", [])
          if any(o.name.lower() == "other" for o in opinions):
            q.topic = FlatTopic(name=q.topic.name)

  recategorized_statements = await categorize_opinions(
      statements_with_topics_and_quotes=statements_to_recategorize,
      topics_to_process=topics_to_process,
      topic_to_opinions_map=topic_to_opinions_map,
      model=model,
      additional_context=additional_context,
      run_autoraters=run_autoraters,
      max_concurrent_calls=max_concurrent_calls,
      offer_other_opinion=offer_other_opinion,
      **categorize_kwargs,
  )

  if recategorize_all:
    return list(recategorized_statements), topic_to_opinions_map

  recat_map = {s.id: s for s in recategorized_statements}
  final_statements = []
  for s in statements:
    if s.id in recat_map:
      final_statements.append(recat_map[s.id])
    else:
      final_statements.append(s)

  return final_statements, topic_to_opinions_map


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
  prompt_jobs = []

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

    prompt_jobs.append({
        "prompt": prompt_str,
        "topic": f"{parent_topic_name or 'topics'}_batch_{i}_attempt_{attempt}",
        "batch_items": batch,
        "response_schema": output_schema,
        "log_prefix_marker": (
            "5 (Opinion Categorization)"
            if is_opinion_categorization
            else "2 (Topic Categorization)"
        ),
    })

  return prompt_jobs


def _decision_state(
    text: str, additional_context: str | None, label: str
) -> str:
  """Prefixes optional survey context onto the text being judged."""
  if not additional_context:
    return text
  return f"{additional_context}\n\n{label}:\n{text}"


async def _assign_topics_with_decision(
    client: systemone.SystemOneClient,
    statements: list[Statement],
    target_topics: list[Topic],
    additional_context: str | None,
) -> list[StatementRecord]:
  """Assigns topics with one yes/no question per topic."""
  labels = [topic.name for topic in target_topics]
  records = []
  for statement in statements:
    chosen = await decision.assign_labels(
        client,
        _decision_state(statement.text, additional_context, "Statement"),
        labels,
        kind="topic",
    )
    records.append(
        StatementRecord(
            id=statement.id,
            topics=[FlatTopic(name=name) for name in chosen],
        )
    )
  return records


async def _categorize_opinions_with_decision(
    statements_with_topics_and_quotes: list[Statement],
    topics_to_process: list[Topic],
    topic_to_opinions_map: dict[str, Any],
    additional_context: str | None,
    run_autoraters: bool,
    client: systemone.SystemOneClient,
    autorater_min_score: int = DEFAULT_AUTORATER_MIN_SCORE,
) -> Iterable[Statement]:
  """Assigns opinions per quote. A failing autorater becomes Other once.

  System One answers are deterministic, so repeating the same judgment cannot
  change the verdict. A score below autorater_min_score is finalized as Other
  immediately, without the retries the generative path uses. A quote that
  already fell back to Other is not rated, since rejecting it would change
  nothing.
  """
  input_statements_map = {
      statement.id: statement.model_copy(deep=True)
      for statement in statements_with_topics_and_quotes
  }
  for topic in topics_to_process:
    learned = topic_to_opinions_map.get(topic.name)
    if not learned or not learned.subtopics:
      logging.warning(
          "No opinions learned for topic '%s'. Skipping.", topic.name
      )
      continue
    opinion_names = [opinion.name for opinion in learned.subtopics]
    records: list[QuoteOpinionRecord] = []
    for statement in input_statements_map.values():
      for quote in statement.quotes or []:
        if quote.topic.name != topic.name:
          continue
        chosen = await decision.assign_labels(
            client,
            _decision_state(quote.text, additional_context, "Quote"),
            opinion_names,
            kind="opinion",
        )
        if run_autoraters and chosen != [decision.OTHER]:
          level = await decision.rubric_level(
              client,
              (
                  f"Topic: {topic.name}\n"
                  f"Quote: {quote.text}\n"
                  f"Assigned opinions: {', '.join(chosen)}\n"
                  f"Available opinions: {', '.join(opinion_names)}"
              ),
              instructions=(
                  "How well does this opinion assignment perform on"
                  " correctness and minimality? The top level means every"
                  " assigned opinion is a correct match and the set is"
                  " minimal."
              ),
          )
          if level < autorater_min_score:
            logging.info(
                "Opinion assignment for %s/%s scored %s; using Other.",
                statement.id,
                quote.id,
                level,
            )
            chosen = ["Other"]
        records.append(
            QuoteOpinionRecord(
                id=statement.id,
                quote_id=quote.id,
                topics=[FlatTopic(name=name) for name in chosen],
            )
        )
    if records:
      _merge_opinions_into_statements_inplace(
          input_statements_map, records, topic
      )
  return input_statements_map.values()


async def _process_topic_categorization(
    statements_to_categorize: list[Statement],
    model: genai_model.GenaiModel,
    target_topics: list[Topic],
    additional_context: str | None = None,
    max_concurrent_calls: int | None = None,
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
  decision_model = decision.decision_client()
  if decision_model is not None:
    logging.info(
        "Assigning topics with System One model %s.", decision_model.model
    )
    return await _assign_topics_with_decision(
        decision_model,
        statements_to_categorize,
        target_topics,
        additional_context,
    )

  topics_json_str = json.dumps([{"name": t.name} for t in target_topics])
  instructions = prompts.get_topic_categorization_prompt(topics_json_str)

  uncategorized_for_retry: list[Statement] = list(statements_to_categorize)
  successfully_categorized_llm_records: list[StatementRecord] = []
  output_schema = StatementRecordList

  logging.info(f"Categorizing statements into {len(target_topics)} topics.")

  logging.info(
      "Performing categorization into topics:"
      f" {[t.name for t in target_topics[:20]]}"
  )

  all_stage_stats = []
  total_wall_delay = 0.0
  start_time = time.time()

  for attempt in range(1, model.max_llm_retries + 1):
    if not uncategorized_for_retry:
      break

    # Create batches
    batches = _create_token_based_batches(
        uncategorized_for_retry, MAX_BATCH_TOKENS, max_items=MAX_ITEMS_PER_BATCH
    )
    prompt_jobs = []

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

      prompt_jobs.append({
          "prompt": prompt_str,
          "topic": f"topics_batch_{i}_attempt_{attempt}",
          "batch_items": batch,
          "response_schema": output_schema,
          "log_prefix_marker": "2 (Topic Categorization)",
      })

    if not prompt_jobs:
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

    results_df, stats, wall_delay, _ = await model.process_prompts_concurrently(
        prompt_jobs,
        response_parser=_parser,
        skip_log=True,
        max_concurrent_calls=max_concurrent_calls,
    )
    all_stage_stats.extend(stats.to_dict("records"))
    total_wall_delay += wall_delay

    # Process results
    current_batch_uncategorized = []

    # We need to map results back to batches to find missing items
    # prompts list matches results_df rows order if we iterate results_df['result']?
    # process_prompts_concurrently preserves order.

    if len(results_df) != len(prompt_jobs):
      logging.error(
          f"Result count mismatch: expected {len(prompt_jobs)}, got"
          f" {len(results_df)}"
      )
      # Add all items to retry
      current_batch_uncategorized.extend(uncategorized_for_retry)
      continue

    for i, row in results_df.iterrows():
      result = row["result"]
      batch_items = prompt_jobs[i]["batch_items"]

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

    if attempt < model.max_llm_retries:
      logging.warning(
          f"Attempt {attempt}: {len(uncategorized_for_retry)} uncategorized"
          " items need retry. Retrying in"
          f" {genai_model.WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS}s..."
      )
      await asyncio.sleep(genai_model.WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS)

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

  if all_stage_stats:
    model.log_stats_summary(
        all_stage_stats,
        "2 (Topic Categorization)",
        total_wall_delay,
        time.time() - start_time,
    )

  return successfully_categorized_llm_records


def _process_categorized_llm_records(
    llm_output_records: list[Union[StatementRecord, QuoteOpinionRecord]],
    all_original_input_statements: list[Statement],
    statements_in_current_batch: list[Statement],
    target_topics_or_opinions: list[Topic],
) -> dict[
    str,
    Union[list[Union[StatementRecord, QuoteOpinionRecord]], list[Statement]],
]:
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
    llm_records: list[Union[StatementRecord, QuoteOpinionRecord]],
    all_original_input_statements: list[Statement],
    target_topics_or_opinions: list[Topic],
) -> tuple[
    list[Union[StatementRecord, QuoteOpinionRecord]],
    list[Union[StatementRecord, QuoteOpinionRecord]],
]:
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
    llm_records: list[Union[StatementRecord, QuoteOpinionRecord]],
    statements_sent_to_llm_batch: list[Statement],
) -> list[Statement]:
  """Finds statements that were sent to the model but were not in the response."""
  processed_ids = {record.id for record in llm_records}
  logging.debug(f"Processed IDs from LLM: {processed_ids}")
  logging.debug(
      f"IDs sent to LLM: {[s.id for s in statements_sent_to_llm_batch]}"
  )
  logging.debug(
      f"Types of processed IDs: {[type(record.id) for record in llm_records]}"
  )
  logging.debug(
      f"Types of sent IDs: {[type(s.id) for s in statements_sent_to_llm_batch]}"
  )
  missing_statements = [
      s for s in statements_sent_to_llm_batch if s.id not in processed_ids
  ]
  if missing_statements:
    logging.warning(
        f"Missing {len(missing_statements)} of"
        f" {len(statements_sent_to_llm_batch)} statement IDs in model's"
        " response"
    )
    logging.debug(
        "Missing statements (up to 20):"
        f" {[s.id for s in missing_statements[:20]]}"
    )
  else:
    logging.info("No missing statements in model's response")

  return missing_statements


def _merge_opinions_into_statements_inplace(
    input_statements_map: dict[str, Statement],
    categorized_llm_records: list[Union[StatementRecord, QuoteOpinionRecord]],
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
