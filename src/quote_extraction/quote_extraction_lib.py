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
from src.models import model_factory
from src.models import systemone
from src.models.custom_types import FlatTopic, NestedTopic, Quote, Statement, Topic

# Joins non-adjacent sentences of a span quote, as an ellipsis would in print.
SPAN_QUOTE_SEPARATOR = " ... "
# A sentence ends at ., ! or ? (optionally followed by closing quotes or
# brackets) before whitespace, or at a line break.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s)|\n")


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
    max_concurrent_calls: Optional[int] = None,
) -> List[Statement]:
  """For each statement and its assigned topics, extracts a representative quote.

  Args:
      statements: A list of Statement objects, with topics assigned.
      model: The GenaiModel to use for extraction.
      additional_context: Optional context for the LLM prompt.
      output_dir: The directory to use for checkpointing.
      max_concurrent_calls: Optional maximum number of concurrent LLM calls.
        If None, the model's default is used.

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

  span_client = model_factory.get_quote_model()
  if span_client is not None:
    logging.info(
        "Extracting quotes as exact spans with System One model %s.",
        span_client.model,
    )
    await extract_span_quotes(span_client, statements, additional_context)
    checkpoint_utils.save_checkpoint(
        statements, quotes_checkpoint_filename, output_dir
    )
    return statements

  # Create quote extraction prompts
  prompts_with_metadata = _prepare_prompts(statements, additional_context)
  if not prompts_with_metadata:
    raise ValueError("No statement-topic pairs for quote extraction.")

  # Run all quote extraction requests using realtime API.
  statements_map_for_quote_update = {s.id: s for s in statements}
  await _get_quotes_realtime(
      model,
      statements_map_for_quote_update,
      prompts_with_metadata,
      max_concurrent_calls=max_concurrent_calls,
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
    max_concurrent_calls: Optional[int] = None,
):
  """Calls model for each prompt, and processes quote response.

  Args:
      model: The GenaiModel to use for extraction.
      statements_map_for_quote_update: Map of statement ID to Statement, which
        is updated in place with extracted quotes.
      prompts_with_metadata: The prompts and associated metadata to process.
      max_concurrent_calls: Optional maximum number of concurrent LLM calls.
        If None, the model's default is used.
  """
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
      max_concurrent_calls=max_concurrent_calls,
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


async def extract_span_quotes(
    client: systemone.SystemOneClient,
    statements: List[Statement],
    additional_context: Optional[str] = None,
) -> List[Statement]:
  """Extracts each topic's quote as exact sentences of the response.

  One extract question per topic marks the participant's words on that
  topic. Each marked span is widened to its whole sentences, and sentences
  that are not adjacent are joined with an ellipsis, so every piece of the
  quote is verbatim text of the response (after <response> markup is
  joined, as skip_quote_extraction does). A topic with no marked span gets
  the whole response. Host errors propagate, so a host without extract
  support fails the stage instead of silently degrading it.

  Args:
      client: A System One client whose host answers extract questions.
      statements: Statements with topics assigned; updated in place.
      additional_context: Optional survey context for the questions.

  Returns:
      The statements, updated with quotes.

  Raises:
      ValueError: If no statement has a topic.
      SystemOneError: If the host rejects or cannot answer a request.
  """
  with_topics = [s for s in statements if s.topics]
  if not with_topics:
    raise ValueError("No statement-topic pairs for quote extraction.")
  quotes = await asyncio.gather(*(
      _span_quotes_for_statement(client, s, additional_context)
      for s in with_topics
  ))
  for statement, statement_quotes in zip(with_topics, quotes):
    if statement.quotes is None:
      statement.quotes = []
    statement.quotes.extend(statement_quotes)
  logging.info("Span quote extraction complete.")
  return statements


async def _span_quotes_for_statement(
    client: systemone.SystemOneClient,
    statement: Statement,
    additional_context: Optional[str],
) -> List[Quote]:
  """Asks one extract question per topic and assembles the quotes."""
  text = join_response_text(statement.text)
  topics = statement.topics or []
  context = (
      f"Survey context: {additional_context.strip()}\n"
      if additional_context and additional_context.strip()
      else ""
  )
  questions = {
      f"t{index}": systemone.ExtractQuestion(
          instructions=(
              f"{context}Mark every sentence in which the participant gives"
              f" their view on '{topic.name}': what they want done about it,"
              " why, and any personal experience or story behind it. Do not"
              " mark text about other topics."
          )
      )
      for index, topic in enumerate(topics)
  }
  answers = await client.ask(text, questions)
  quotes = []
  for index, topic in enumerate(topics):
    answer = answers[f"t{index}"]
    if not isinstance(answer, systemone.ExtractAnswer):
      raise systemone.SystemOneError(
          systemone.FailureKind.PARSE,
          f"question t{index} did not return an extract answer.",
      )
    quote_text = assemble_span_quote(text, answer.spans)
    if not quote_text:
      logging.info(
          "No span for statement %s, topic %s; using the whole response.",
          statement.id,
          topic.name,
      )
      quote_text = text
    quotes.append(
        Quote(
            id=f"{statement.id}-{topic.name}",
            text=quote_text,
            topic=(
                NestedTopic(name=topic.name, subtopics=topic.subtopics)
                if isinstance(topic, NestedTopic)
                else FlatTopic(name=topic.name)
            ),
        )
    )
  return quotes


def sentence_bounds(text: str) -> List[tuple[int, int]]:
  """Returns the [start, end) offsets of each sentence, whitespace trimmed.

  Args:
      text: The text to split.

  Returns:
      Sentence offsets in order; together they cover every non-space
      character of text.
  """
  bounds = []
  start = 0
  for match in _SENTENCE_END.finditer(text):
    bounds.append((start, match.end()))
    start = match.end()
  bounds.append((start, len(text)))
  trimmed = []
  for begin, end in bounds:
    segment = text[begin:end]
    left = len(segment) - len(segment.lstrip())
    right = len(segment.rstrip())
    if right > left:
      trimmed.append((begin + left, begin + right))
  return trimmed


def assemble_span_quote(
    text: str, spans: tuple[systemone.Span, ...] | List[systemone.Span]
) -> str:
  """Widens spans to whole sentences and joins them into one quote.

  Args:
      text: The text the spans index into.
      spans: Marked spans, as returned by an extract question.

  Returns:
      The selected sentences in order: adjacent ones as they appear in text,
      others joined with SPAN_QUOTE_SEPARATOR. Empty when no span overlaps
      a sentence.
  """
  sentences = sentence_bounds(text)
  chosen = sorted({
      index
      for span in spans
      for index, (begin, end) in enumerate(sentences)
      if span.start < end and begin < span.end
  })
  pieces = []
  run_start = None
  for position, index in enumerate(chosen):
    if run_start is None:
      run_start = index
    next_index = chosen[position + 1] if position + 1 < len(chosen) else None
    if next_index != index + 1:
      pieces.append(text[sentences[run_start][0] : sentences[index][1]])
      run_start = None
  return SPAN_QUOTE_SEPARATOR.join(pieces)


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
    logging.debug(
        f"Skipped quote extraction for {statement.id}. Extracted text:"
        f" '{response_text}'"
    )
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
  if not survey_text:
    return ""
  responses = re.findall(r"<response>(.*?)</response>", survey_text, re.DOTALL)
  if not responses:
    return survey_text

  processed_responses = []
  for r in responses:
    r = r.strip()
    if r:
      if not r.endswith(('.', '?', '!')):
        r += "."
      processed_responses.append(r)

  return " ".join(processed_responses) if processed_responses else survey_text
