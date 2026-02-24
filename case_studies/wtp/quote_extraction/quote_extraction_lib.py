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
from typing import List, Optional, cast

from case_studies.wtp.models.genai_model import GenaiModel
from case_studies.wtp import checkpoint_utils
from case_studies.wtp.models.custom_types import FlatTopic, NestedTopic, Quote, Statement, Topic


def _create_quote_extraction_prompt(text: str, context: str, topic: str) -> str:
  """Creates a prompt for extracting a quote from text."""
  context_block = (
      f"<additionalContext>\n  {context}\n</additionalContext>\n"
      if context
      else ""
  )
  return f"""{context_block}Extract the most representative quote that represents participant opinion on <topic>{topic}</topic> topic from the following text:
<text>{text}</text>

- You are a professional journalist quoting from a participant's responses in a transcript to create a coherent quotation that represents the participant's opinion on the given topic.
- Use best practices of professional journalists to achieve that. Specifically and **sparingly**, using brackets to enclose your modifications, you can lightly edit, correct misspelling, miscapitalizations or mispunctations, redact any profanity (e.g., replace a profane word with its first letter followed by dashes, like "[s---]"), or add clarifying information so that the quote is understandable even without seeing the original question. No change can be made to the response outside of brackets.
- You may also merge elements from across multiple questions for coherence. You must use ellipses to show when you are doing this, and you cannot modify the original sentence order.
- Other than the bracketed modifications, the quotation should be an ellipsis-delimited concatenation of substrings of the participant's response that obeys the original sentence order.
- We want to surface powerful and personal nuances a person shared on the opinion while keeping the quote concise and scannable. Stories about the participants' lives are especially valuable to feature. The quotation should feel punchy and profound, like an incisive portrait of the participant's humanity.
- The quote should only cover the given topic. We may extract several quotes from this transcript and must avoid redundancy.
- If there's not enough text for personal nuances, the quote should be just what the person expressed (e.g.: "I don't know"), and it's okay for it to be short.
- Do not add any extra commentary or markdown to the quote.
- Please output only the quotation. You should not enclose the quotation in quotation marks.
"""


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
            "log_prefix_marker": "3",
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
  response_df, _ = await model.process_prompts_concurrently(
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
