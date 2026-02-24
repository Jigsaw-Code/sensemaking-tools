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

from case_studies.wtp import runner_utils, checkpoint_utils
from case_studies.wtp.models import custom_types
from case_studies.wtp.tasks import categorization
from case_studies.wtp.quote_extraction import quote_extraction_lib
from case_studies.wtp.models import genai_model


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
  ) -> Iterable[custom_types.Statement]:
    """Categorize statements into topics and opinions using an LLM.

    Args:
        statements: The statements to categorize.
        topics: Optional. User-provided topics. If not provided, topics are
          learned first.
        additional_context: Optional additional context for the LLM.
        original_csv_rows: Optional original CSV rows for semi-final dump.
        output_dir: Optional output directory for semi-final dump.
        run_autoraters: Whether to run autorater evaluations.

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
        statements_with_quotes = (
            await quote_extraction_lib.extract_quotes_from_text(
                statements=statements_with_topics,
                model=self._genai_model,
                additional_context=additional_context,
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

    # Step 3: Opinion Processing
    logging.info("Starting opinion processing.")

    opinions_checkpoint = "statements_with_opinions"
    loaded_opinions = checkpoint_utils.load_checkpoint(
        opinions_checkpoint, output_dir
    )

    if loaded_opinions:
      logging.info("Loaded statements with opinions from checkpoint.")
      statements_with_opinions = loaded_opinions
    else:
      # Step 4: Global Opinion Learning
      learned_opinions_checkpoint = "learned_opinions"
      topic_to_opinions_map = checkpoint_utils.load_checkpoint(
          learned_opinions_checkpoint, output_dir
      )

      if topic_to_opinions_map:
        logging.info("Loaded learned opinions from checkpoint.")
      else:
        topic_to_opinions_map = await categorization.learn_global_opinions(
            statements_with_topics_and_quotes=statements_with_quotes,
            topics_to_process=learned_topics,
            model=self._genai_model,
            additional_context=additional_context,
        )
        if output_dir:
          checkpoint_utils.save_checkpoint(
              topic_to_opinions_map, learned_opinions_checkpoint, output_dir
          )

      # Step 5: Global Opinion Categorization
      statements_with_opinions = await categorization.categorize_opinions(
          statements_with_topics_and_quotes=statements_with_quotes,
          topics_to_process=learned_topics,
          topic_to_opinions_map=topic_to_opinions_map,
          model=self._genai_model,
          additional_context=additional_context,
          run_autoraters=run_autoraters,
      )

      if output_dir:
        checkpoint_utils.save_checkpoint(
            list(statements_with_opinions), opinions_checkpoint, output_dir
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
    statement_id = row.get("rid")
    if statement_id and statement_id not in original_rows_map:
      original_rows_map[statement_id] = row

  for statement_id, statement in statements_map.items():
    base_row_data = original_rows_map.get(statement_id, {})

    if not statement.quotes:
      continue

    for quote in statement.quotes:
      new_row = base_row_data.copy()
      new_row["quote_id"] = str(quote.id)
      new_row["representative_text_with_brackets"] = quote.text
      new_row["representative_text"] = re.sub(r"[\[\]]", "", quote.text)
      new_row["topic"] = quote.topic.name
      output_rows.append(new_row)

  return output_rows
