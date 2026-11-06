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

"""Entry point to interact with sensemaking tools."""

import logging
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from src import checkpoint_utils
from src import prompts
from src import runner_utils
from src.models import custom_types
from src.tasks import categorization
from src.quote_extraction import quote_extraction_lib
from src.models import genai_model

# Opinion-stage checkpoint names. The *_PASS1 checkpoints hold pass-1 opinion
# learning (Step 4) and categorization (Step 5) results and are never
# overwritten by optional pass-2 'Other' recategorization (Step 5b), so Step 5b
# can be re-run (or ablated across arms) from an identical pass-1 state.
OPINIONS_CHECKPOINT = "statements_with_opinions"
LEARNED_OPINIONS_CHECKPOINT = "learned_opinions"
PASS1_OPINIONS_CHECKPOINT = "statements_with_opinions_pass1"
PASS1_LEARNED_OPINIONS_CHECKPOINT = "learned_opinions_pass1"


class Sensemaker:
  """Class to make sense of conversation data.

  Uses LLMs to learn what topics and opinions were discussed, extract quotes and
  categorize statements.
  """

  _genai_model: genai_model.GenaiModel

  def __init__(
      self,
      genai_model: genai_model.GenaiModel,
  ):
    """Creates a Sensemaker object.

    Args:
        genai_model: The LLM model instance (GenaiModel).
    """
    if not genai_model:
      raise ValueError("A genai_model instance must be provided.")
    self._genai_model = genai_model

  async def categorize_statements(
      self,
      statements: List[custom_types.Statement],
      topics: Optional[List[custom_types.Topic]] = None,
      additional_context: Optional[str] = None,
      original_csv_rows: Optional[List[Dict[str, Any]]] = None,
      output_dir: Optional[str] = None,
      run_autoraters: bool = True,
      skip_quote_extraction: bool = False,
      max_concurrent_calls: Optional[int] = None,
      max_autorater_retries: int = categorization.DEFAULT_MAX_AUTORATER_RETRIES,
      autorater_min_score: int = categorization.DEFAULT_AUTORATER_MIN_SCORE,
      opinion_matching: str = prompts.OPINION_MATCHING_SUPPORTS,
      recategorize_other: bool = False,
      pass_existing_opinions_to_recategorize: bool = True,
      recategorize_other_skip_autoraters: bool = False,
      dedup_new_opinions: bool = False,
      offer_other_opinion: bool = False,
      offer_other_opinion_on_recategorize: Optional[bool] = None,
  ) -> Iterable[custom_types.Statement]:
    """Categorize statements into topics and opinions using an LLM.

    Pipeline steps:
      - Step 1: Topic discovery and statement-to-topic categorization
        (checkpoint: `statements_with_topics_and_learned_topics`).
      - Step 2: Quote extraction per assigned topic
        (checkpoint: `statements_with_quotes`).
      - Opinion processing (Steps 4, 5, and optional Step 5b):
        - Step 4: Pass-1 global opinion learning per topic
          (checkpoints: `learned_opinions`, `learned_opinions_pass1`).
        - Step 5: Pass-1 quote-to-opinion categorization
          (checkpoint: `statements_with_opinions_pass1`, plus
          `statements_with_opinions` when Step 5b is not run).
        - Step 5b (optional, when `recategorize_other=True`): Pass-2 opinion
          learning on 'Other' quotes followed by corpus re-categorization
          (updates `learned_opinions` and `statements_with_opinions` while
          leaving `*_pass1` checkpoints intact).

    Args:
        statements: The statements to categorize.
        topics: Optional. User-provided topics. If not provided, topics are
          learned first.
        additional_context: Optional additional context for the LLM.
        original_csv_rows: Optional original CSV rows for semi-final dump.
        output_dir: Optional output directory for semi-final dump.
        run_autoraters: Whether to run autorater evaluations.
        skip_quote_extraction: Whether to skip quote extraction and use whole
          response as quote.
        max_concurrent_calls: Optional maximum number of concurrent LLM calls,
          applied to every LLM stage. If None, the model's default is used.
        max_autorater_retries: Number of autorater rejections a single quote
          may receive before it is defaulted to 'Other'.
        autorater_min_score: Minimum integer autorater score (0-4 rubric) for
          an opinion assignment to pass.
        opinion_matching: Opinion matching rules for categorization and
          autoraters: 'supports' (core claim, default) or 'identical' (every
          key concept). One of prompts.OPINION_MATCHING_MODES.
        recategorize_other: Whether to run a secondary pass on quotes assigned
          to 'Other' to discover new emergent opinions and re-categorize.
        pass_existing_opinions_to_recategorize: Whether to pass existing
          opinions as a constraint during secondary opinion learning.
        recategorize_other_skip_autoraters: Whether to skip autoraters during
          second-pass recategorization.
        dedup_new_opinions: Whether to run an LLM judge to deduplicate newly
          learned opinions against existing opinions.
        offer_other_opinion: Whether to include 'Other' in the input opinion
          menu presented to the categorizer (default: False).
        offer_other_opinion_on_recategorize: Optional override for
          offer_other_opinion specifically during second-pass recategorization.

    Returns:
        The statements with topics and opinions assigned.
    """
    start_time = time.perf_counter_ns()

    logging.debug(
        f"Starting statement categorization for {len(statements)} statements. "
        f"Provided topics: {'Yes' if topics else 'No'}."
    )

    quotes_present = any(s.quotes for s in statements)
    if topics and quotes_present:
      logging.info(
          "Topics and quotes are present. Skipping topic categorization and"
          " quote extraction."
      )
      statements_with_quotes = statements
      learned_topics = topics

    else:
      # Step 1: Topic Categorization

      topics_checkpoint_filename = "statements_with_topics_and_learned_topics"
      checkpoint_data = checkpoint_utils.load_checkpoint(
          topics_checkpoint_filename, output_dir
      )

      if checkpoint_data:
        statements_with_topics, learned_topics = checkpoint_data
        logging.info(
            "Loaded statements with topics and learned topics from checkpoint."
        )
      else:

        statements_with_topics, learned_topics = (
            await categorization.categorize_topics(
                statements=statements,
                model=self._genai_model,
                current_topics=topics,
                additional_context=additional_context,
                max_concurrent_calls=max_concurrent_calls,
            )
        )

        checkpoint_utils.save_checkpoint(
            (statements_with_topics, learned_topics),
            topics_checkpoint_filename,
            output_dir,
        )

      # Step 2: Extract quotes
      logging.info("Step 2: Extracting quotes for initialized topics...")

      quotes_checkpoint = "statements_with_quotes"
      loaded_quotes = checkpoint_utils.load_checkpoint(
          quotes_checkpoint, output_dir
      )

      if loaded_quotes:
        logging.info("Loaded statements with quotes from checkpoint.")
        statements_with_quotes = loaded_quotes
      else:
        if skip_quote_extraction:
          statements_with_quotes = quote_extraction_lib.skip_quote_extraction(
              statements=statements_with_topics
          )
        else:
          statements_with_quotes = (
              await quote_extraction_lib.extract_quotes_from_text(
                  statements=statements_with_topics,
                  model=self._genai_model,
                  additional_context=additional_context,
                  max_concurrent_calls=max_concurrent_calls,
              )
          )

        if output_dir:
          checkpoint_utils.save_checkpoint(
              statements_with_quotes, quotes_checkpoint, output_dir
          )

      if output_dir and original_csv_rows:
        logging.info("Dumping semi-final results to CSV.")
        semifinal_rows = _prepare_semifinal_csv_rows(
            original_csv_rows, statements_with_quotes
        )
        semifinal_csv_path = os.path.join(
            output_dir, "categorized_semifinal.csv"
        )
        runner_utils.write_dicts_to_csv(semifinal_rows, semifinal_csv_path)

    # Opinion Processing (Steps 4, 5, and optional Step 5b)
    logging.info("Starting opinion processing.")

    categorize_kwargs = dict(
        max_autorater_retries=max_autorater_retries,
        autorater_min_score=autorater_min_score,
        opinion_matching=opinion_matching,
        offer_other_opinion=offer_other_opinion,
    )

    loaded_opinions = checkpoint_utils.load_checkpoint(
        OPINIONS_CHECKPOINT, output_dir
    )
    pass1_statements = None
    if not loaded_opinions:
      pass1_statements = checkpoint_utils.load_checkpoint(
          PASS1_OPINIONS_CHECKPOINT, output_dir
      )

    if loaded_opinions:
      # Path F: final results exist; nothing to recompute.
      statements_with_opinions = loaded_opinions
      logging.info(
          "Opinion path: FINAL checkpoint '%s' loaded; skipping Steps 4, 5"
          " and 5b.",
          OPINIONS_CHECKPOINT,
      )
      if recategorize_other and not checkpoint_utils.checkpoint_exists(
          PASS1_OPINIONS_CHECKPOINT, output_dir
      ):
        logging.warning(
            "Step 5b was requested but SKIPPED: only the legacy final"
            " checkpoint '%s' exists and there is no pass-1 checkpoint '%s'"
            " to re-run Step 5b from.",
            OPINIONS_CHECKPOINT,
            PASS1_OPINIONS_CHECKPOINT,
        )
      elif recategorize_other:
        logging.info(
            "Step 5b already completed for this output_dir. To re-run it from"
            " the pass-1 checkpoint, delete '%s.pkl'.",
            OPINIONS_CHECKPOINT,
        )
      logging.info(
          "Topic and opinion categorization took"
          f" {(time.perf_counter_ns() - start_time) / 60e9:.2f} minutes."
      )
      return statements_with_opinions

    if pass1_statements:
      # Path R: resume from the preserved pass-1 results.
      topic_to_opinions_map = checkpoint_utils.load_checkpoint(
          PASS1_LEARNED_OPINIONS_CHECKPOINT, output_dir
      )
      if not topic_to_opinions_map:
        raise RuntimeError(
            f"Checkpoint '{PASS1_OPINIONS_CHECKPOINT}' exists but"
            f" '{PASS1_LEARNED_OPINIONS_CHECKPOINT}' is missing or unreadable"
            f" in {output_dir}; cannot resume. Copy both pass-1 checkpoints."
        )
      statements_with_opinions = pass1_statements
      logging.info(
          "Opinion path: RESUME from pass-1 checkpoints ('%s', '%s');"
          " skipping Steps 4 and 5. %s",
          PASS1_OPINIONS_CHECKPOINT,
          PASS1_LEARNED_OPINIONS_CHECKPOINT,
          "Running Step 5b."
          if recategorize_other
          else "Step 5b not requested; pass-1 results are final.",
      )
    else:
      # Path P: full pass 1.
      logging.info(
          "Opinion path: FULL (Step 4 learning, Step 5 categorization%s).",
          ", Step 5b" if recategorize_other else "",
      )
      # Step 4: Global Opinion Learning
      topic_to_opinions_map = checkpoint_utils.load_checkpoint(
          LEARNED_OPINIONS_CHECKPOINT, output_dir
      )

      if topic_to_opinions_map:
        logging.info("Loaded learned opinions from checkpoint.")
      else:
        topic_to_opinions_map = await categorization.learn_global_opinions(
            statements_with_topics_and_quotes=statements_with_quotes,
            topics_to_process=learned_topics,
            model=self._genai_model,
            additional_context=additional_context,
            max_concurrent_calls=max_concurrent_calls,
            offer_other_opinion=offer_other_opinion,
        )
        if output_dir:
          checkpoint_utils.save_checkpoint(
              topic_to_opinions_map, LEARNED_OPINIONS_CHECKPOINT, output_dir
          )

      # Step 5: Global Opinion Categorization
      statements_with_opinions = await categorization.categorize_opinions(
          statements_with_topics_and_quotes=statements_with_quotes,
          topics_to_process=learned_topics,
          topic_to_opinions_map=topic_to_opinions_map,
          model=self._genai_model,
          additional_context=additional_context,
          run_autoraters=run_autoraters,
          max_concurrent_calls=max_concurrent_calls,
          pass_label="pass1",
          **categorize_kwargs,
      )

      # Preserve pass-1 results; Step 5b never writes these checkpoints.
      if output_dir:
        checkpoint_utils.save_checkpoint(
            list(statements_with_opinions),
            PASS1_OPINIONS_CHECKPOINT,
            output_dir,
        )
        checkpoint_utils.save_checkpoint(
            topic_to_opinions_map,
            PASS1_LEARNED_OPINIONS_CHECKPOINT,
            output_dir,
        )

    # Step 5b: Secondary pass on 'Other' quotes
    if recategorize_other:
      pass2_offer_other = (
          offer_other_opinion
          if offer_other_opinion_on_recategorize is None
          else offer_other_opinion_on_recategorize
      )
      step5b_categorize_kwargs = dict(categorize_kwargs)
      step5b_categorize_kwargs["offer_other_opinion"] = pass2_offer_other
      logging.info(
          "Step 5b: Re-categorizing quotes (recategorize_all=True)..."
      )
      statements_with_opinions, topic_to_opinions_map = (
          await categorization.recategorize_other_quotes(
              statements=list(statements_with_opinions),
              topics_to_process=learned_topics,
              topic_to_opinions_map=topic_to_opinions_map,
              model=self._genai_model,
              additional_context=additional_context,
              pass_existing_opinions=pass_existing_opinions_to_recategorize,
              run_autoraters=not recategorize_other_skip_autoraters,
              recategorize_all=True,
              max_concurrent_calls=max_concurrent_calls,
              dedup_new_opinions=dedup_new_opinions,
              **step5b_categorize_kwargs,
          )
      )
      if output_dir:
        checkpoint_utils.save_checkpoint(
            topic_to_opinions_map, LEARNED_OPINIONS_CHECKPOINT, output_dir
        )

    if output_dir:
      checkpoint_utils.save_checkpoint(
          list(statements_with_opinions), OPINIONS_CHECKPOINT, output_dir
      )

    logging.info(
        "Topic and opinion categorization took"
        f" {(time.perf_counter_ns() - start_time) / 60e9:.2f} minutes."
    )

    return statements_with_opinions


def _prepare_semifinal_csv_rows(
    original_csv_rows: List[Dict[str, str]],
    statements_with_quotes: List[custom_types.Statement],
) -> List[Dict[str, str]]:
  """Prepares CSV rows for semi-final output."""
  statements_map = {s.id: s for s in statements_with_quotes}
  output_rows = []

  original_rows_map = {}
  for row in original_csv_rows:
    statement_id = row.get("participant_id")
    if statement_id and statement_id not in original_rows_map:
      original_rows_map[statement_id] = row

  for statement_id, statement in statements_map.items():
    base_row_data = original_rows_map.get(statement_id, {})

    if not statement.quotes:
      continue

    for quote in statement.quotes:
      new_row = base_row_data.copy()
      new_row["quote_id"] = str(quote.id)
      new_row["quote_with_brackets"] = quote.text
      new_row["quote"] = re.sub(r"[\[\]]", "", quote.text)
      new_row["topic"] = quote.topic.name
      output_rows.append(new_row)

  return output_rows
