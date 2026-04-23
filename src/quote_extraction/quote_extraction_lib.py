# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import re
from typing import List, Optional, cast

from src.models.genai_model import GenaiModel
from src import checkpoint_utils
from src import prompts
from src.models.custom_types import FlatTopic, NestedTopic, Quote, Statement, Topic


def _create_quote_extraction_prompt(text: str, context: str, topic: str) -> str:
  return prompts.get_quote_extraction_prompt(text, context, topic)


def _prepare_prompts(
    statements: List[Statement], additional_context: Optional[str]
) -> List[dict]:
  """Creates a list of prompts for quote extraction."""
  prompts_with_metadata = []
  for statement_obj in statements:
    if statement_obj.topics:
      for topic in statement_obj.topics:
        prompt = _create_quote_extraction_prompt(
            text=statement_obj.text,
            context=additional_context or "",
            topic=topic.name,
        )
        prompts_with_metadata.append({
            "prompt": prompt,
            "statement_id": statement_obj.id,
            "topic": topic,
            "log_prefix_marker": "3 (Quote Extraction)",
        })
  return prompts_with_metadata


async def extract_quotes_from_text(
    statements: List[Statement],
    model: GenaiModel,
    additional_context: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> List[Statement]:
  """For each statement and its assigned topics, extracts a representative quote.

  Args:
      statements: A list of Statement objects, with topics assigned.
      model: The GenaiModel to use for extraction.
      additional_context: Optional context for the LLM prompt.
      output_dir: The directory to use for checkpointing.

  Returns:
      The list of statements, updated with extracted quotes.
  """
  # Check if quote extraction was already completed
  # and if so re-use that.
  quotes_checkpoint_filename = "statements_with_quotes"
  if output_dir:
    cached_statements = checkpoint_utils.load_checkpoint(
        quotes_checkpoint_filename, output_dir
    )
    if cached_statements:
      logging.info("Loaded statements with quotes from checkpoint.")
      return cached_statements

  # Create quote extraction prompts
  prompts_with_metadata = _prepare_prompts(statements, additional_context)
  if not prompts_with_metadata:
    raise ValueError("No statement-topic pairs for quote extraction.")

  # Run all quote extraction requests using realtime API.
  statements_map_for_quote_update = {s.id: s for s in statements}
  await _get_quotes_realtime(
      model, statements_map_for_quote_update, prompts_with_metadata
  )

  checkpoint_utils.save_checkpoint(
      list(statements_map_for_quote_update.values()),
      quotes_checkpoint_filename,
      output_dir,
  )
  return statements


async def _get_quotes_realtime(
    model: GenaiModel,
    statements_map_for_quote_update: dict[str, Statement],
    prompts_with_metadata: List[dict],
):
  """Calls model for each prompt, and processes quote response."""
  logging.info(
      "Extracting quotes for"
      f" {len(prompts_with_metadata)} statement-topic pairs using concurrent"
      " processing..."
  )

  def _parser(resp, job):
    # The response should be just the quote text
    return {"text": resp["text"], "error": resp.get("error")}

  # We use process_prompts_concurrently which handles retries and rate limiting
  response_df, _, _, _ = await model.process_prompts_concurrently(
      prompts_with_metadata,
      response_parser=_parser,
  )

  # Add each quote to the statement object
  # This code does not assume response_df is in any sorted order.
  for _, row in response_df.iterrows():
    result = row["result"]
    # Reconstruct the metadata from the original job/prompt data
    # process_prompts_concurrently preserves order if we match by index,
    # but safer to read from the 'job' input that process_prompts_concurrently uses?
    # Actually process_prompts_concurrently returns a DF where each row corresponds to a job request.
    # The job request contains the metadata we passed in.

    # We need to get these back. The results_df has columns for all keys in the input dicts!
    statement_id_res = row["statement_id"]
    topic_obj = row["topic"]

    quote_str = None
    if isinstance(result, dict):
      quote_str = result.get("text")
      error = result.get("error")
      if error:
        logging.warning(
            f"Failed to extract quote for statement {statement_id_res}, topic"
            f" {topic_obj.name}: {error}"
        )
        continue
    else:
      # Should not happen with our parser
      logging.warning(f"Unexpected result format: {result}")
      continue

    if not quote_str:
      continue

    statement_to_update = statements_map_for_quote_update.get(statement_id_res)
    if statement_to_update and quote_str:
      if statement_to_update.quotes is None:
        statement_to_update.quotes = []

      statement_to_update.quotes.append(
          Quote(
              id=f"{statement_to_update.id}-{topic_obj.name}",
              text=quote_str,
              topic=(
                  NestedTopic(
                      name=topic_obj.name, subtopics=topic_obj.subtopics
                  )
                  if isinstance(topic_obj, NestedTopic)
                  else FlatTopic(name=topic_obj.name)
              ),
          )
      )
  logging.info("Quote extraction complete.")
  return


def skip_quote_extraction(statements: List[Statement]) -> List[Statement]:
  """Bypasses LLM quote extraction and uses full statement text as quote.

  Args:
      statements: A list of Statement objects, with topics assigned.

  Returns:
      The list of statements, updated with quotes.
  """
  logging.info("Skipping quote extraction, using entire response as quote.")
  for statement in statements:
    response_text = join_response_text(statement.text)
    if statement.topics:
      if statement.quotes is None:
        statement.quotes = []

      for topic in statement.topics:
        statement.quotes.append(
            Quote(
                id=f"{statement.id}-{topic.name}",
                text=response_text,
                topic=(
                    NestedTopic(name=topic.name, subtopics=topic.subtopics)
                    if isinstance(topic, NestedTopic)
                    else FlatTopic(name=topic.name)
                ),
            )
        )
  return statements


def join_response_text(survey_text):
  """Extract each response and make sure it ends with proper punctation."""
  responses = re.findall(r'<response>(.*?)</response>', survey_text, re.DOTALL)
  if not responses:
    return survey_text
  for i in range(len(responses)):
    if responses[i][-1] not in {'.', '?', '!'}:
      responses[i] += '.'
  return " ".join(responses)