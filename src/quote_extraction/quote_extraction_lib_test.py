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
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch
from src.models.genai_model import GenaiModel
from src.models import custom_types
from src.models import systemone
from src.quote_extraction import quote_extraction_lib
import pandas as pd


def _span(text: str, phrase: str) -> dict[str, object]:
  """Returns a wire span for the first occurrence of phrase in text."""
  start = text.index(phrase)
  return {
      "text": phrase,
      "start": start,
      "end": start + len(phrase),
      "confidence": 0.9,
  }


class _ExtractTransport:
  """Fake extract host: marks the given phrases for each question id."""

  def __init__(self, phrases_by_question: dict[str, list[str]]):
    """Stores the phrases to mark, keyed by question id."""
    self.phrases_by_question = phrases_by_question
    self.bodies = []

  def __call__(self, url: str, body: bytes) -> bytes:
    """Records the request and marks the phrases in its state."""
    payload = json.loads(body)
    self.bodies.append(payload)
    state = payload["state"]
    answers = {}
    for key in payload["questions"]:
      phrases = self.phrases_by_question.get(key, [])
      answers[key] = {
          "type": "extract",
          "spans": {key: [_span(state, phrase) for phrase in phrases]},
          "not_stated": {key: 0.0 if phrases else 0.99},
      }
    return json.dumps({"model": "mercury-decide", "answers": answers}).encode()


class QuoteExtractionLibTest(unittest.IsolatedAsyncioTestCase):

  def test_extract_quotes_from_text_generates_unique_ids(self):
    # Setup
    statements = [
        custom_types.Statement(
            id="statement1",
            text="This is a statement about topic A and B.",
            topics=[
                custom_types.FlatTopic(name="Topic A"),
                custom_types.FlatTopic(name="Topic B"),
            ],
        ),
        custom_types.Statement(
            id="statement2",
            text="This is another statement about topic A.",
            topics=[custom_types.FlatTopic(name="Topic A")],
        ),
    ]

    mock_model = MagicMock()

    # Mock process_prompts_concurrently to return a DataFrame with results
    # Each result should correspond to the parsed output
    mock_results_df = pd.DataFrame([
        {
            "result": {"text": "This is a quote.", "error": None},
            "statement_id": "statement1",
            "topic": custom_types.FlatTopic(name="Topic A"),
        },
        {
            "result": {"text": "This is a quote.", "error": None},
            "statement_id": "statement1",
            "topic": custom_types.FlatTopic(name="Topic B"),
        },
        {
            "result": {"text": "This is a quote.", "error": None},
            "statement_id": "statement2",
            "topic": custom_types.FlatTopic(name="Topic A"),
        },
    ])
    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(mock_results_df, pd.DataFrame(), 0.0, 1.0)
    )

    # Execution
    result = asyncio.run(
        quote_extraction_lib.extract_quotes_from_text(
            statements=statements,
            model=mock_model,
        )
    )

    # Assertion
    expected_result = [
        custom_types.Statement(
            id="statement1",
            text="This is a statement about topic A and B.",
            topics=[
                custom_types.FlatTopic(name="Topic A"),
                custom_types.FlatTopic(name="Topic B"),
            ],
            quotes=[
                custom_types.Quote(
                    id="statement1-Topic A",
                    text="This is a quote.",
                    topic=custom_types.FlatTopic(name="Topic A"),
                ),
                custom_types.Quote(
                    id="statement1-Topic B",
                    text="This is a quote.",
                    topic=custom_types.FlatTopic(name="Topic B"),
                ),
            ],
        ),
        custom_types.Statement(
            id="statement2",
            text="This is another statement about topic A.",
            topics=[custom_types.FlatTopic(name="Topic A")],
            quotes=[
                custom_types.Quote(
                    id="statement2-Topic A",
                    text="This is a quote.",
                    topic=custom_types.FlatTopic(name="Topic A"),
                )
            ],
        ),
    ]
    self.assertEqual(result, expected_result)

    all_quote_ids = []
    for statement in result:
      self.assertIsNotNone(statement.quotes)
      for quote in statement.quotes:
        self.assertIsNotNone(quote.id)
        self.assertTrue(quote.id.startswith(statement.id))
        self.assertIn(quote.topic.name, quote.id)
        all_quote_ids.append(quote.id)

    self.assertEqual(
        len(all_quote_ids), 3
    )  # 2 for statement1, 1 for statement2
    self.assertEqual(
        len(set(all_quote_ids)), len(all_quote_ids)
    )  # Check for uniqueness

  def _make_single_statement_and_mock_model(self):
    """Builds one statement and a mock model returning a quote for it.

    Returns:
        A tuple of (statements, mock_model).
    """
    topic = custom_types.FlatTopic(name="Topic A")
    statements = [
        custom_types.Statement(
            id="statement1", text="A statement.", topics=[topic]
        )
    ]
    mock_results_df = pd.DataFrame([{
        "result": {"text": "A quote.", "error": None},
        "statement_id": "statement1",
        "topic": topic,
    }])
    mock_model = MagicMock()
    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(mock_results_df, pd.DataFrame(), 0.0, 1.0)
    )
    return statements, mock_model

  async def test_extract_quotes_forwards_max_concurrent_calls(self):
    statements, mock_model = self._make_single_statement_and_mock_model()

    await quote_extraction_lib.extract_quotes_from_text(
        statements=statements, model=mock_model, max_concurrent_calls=5
    )

    _, kwargs = mock_model.process_prompts_concurrently.call_args
    self.assertEqual(kwargs.get("max_concurrent_calls"), 5)

  async def test_extract_quotes_defaults_max_concurrent_calls_to_none(self):
    statements, mock_model = self._make_single_statement_and_mock_model()

    await quote_extraction_lib.extract_quotes_from_text(
        statements=statements, model=mock_model
    )

    _, kwargs = mock_model.process_prompts_concurrently.call_args
    self.assertIsNone(kwargs.get("max_concurrent_calls"))

  def test_assemble_span_quote_widens_and_joins_sentences(self):
    """Tests sentence widening, adjacent runs, and the ellipsis join."""
    text = "One two. Three four! Five six? Seven eight."
    spans = [
        systemone.Span(text="two", start=4, end=7, confidence=1.0),
        systemone.Span(text="Three", start=9, end=14, confidence=1.0),
        systemone.Span(text="eight", start=37, end=42, confidence=1.0),
    ]

    quote = quote_extraction_lib.assemble_span_quote(text, spans)

    self.assertEqual(quote, "One two. Three four! ... Seven eight.")
    self.assertEqual(quote_extraction_lib.assemble_span_quote(text, []), "")
    crossing = [systemone.Span(text="four! Five", start=15, end=25, confidence=1.0)]
    self.assertEqual(
        quote_extraction_lib.assemble_span_quote(text, crossing),
        "Three four! Five six?",
    )

  def test_sentence_bounds_cover_every_sentence(self):
    """Tests sentence splitting on punctuation and line breaks."""
    text = 'He said "stop." Then\nleft.  Done'
    bounds = quote_extraction_lib.sentence_bounds(text)
    self.assertEqual(
        [text[b:e] for b, e in bounds],
        ['He said "stop."', "Then", "left.", "Done"],
    )

  async def test_span_client_extracts_verbatim_quotes(self):
    """Tests the span path: verbatim sentences, fallback, no LLM call."""
    statement = custom_types.Statement(
        id="s1",
        text=(
            "<response>I bike to work</response><response>The painted lanes are"
            " scary and cars park in them. Please build protected"
            " lanes.</response>"
        ),
        topics=[
            custom_types.FlatTopic(name="Cycling"),
            custom_types.FlatTopic(name="Taxes"),
        ],
    )
    transport = _ExtractTransport(
        {"t0": ["painted lanes are scary", "protected lanes"]}
    )
    client = systemone.SystemOneClient(
        model="mercury-decide", transport=transport
    )
    model = MagicMock()
    model.process_prompts_concurrently = AsyncMock()

    with patch.object(
        quote_extraction_lib.model_factory,
        "get_quote_model",
        return_value=client,
    ):
      result = await quote_extraction_lib.extract_quotes_from_text(
          statements=[statement], model=model
      )

    cleaned = "I bike to work. The painted lanes are scary and cars park in them. Please build protected lanes."
    self.assertIsNotNone(result[0].quotes)
    quotes = {quote.topic.name: quote for quote in result[0].quotes or []}
    self.assertEqual(
        quotes["Cycling"].text,
        "The painted lanes are scary and cars park in them. Please build"
        " protected lanes.",
    )
    self.assertEqual(quotes["Cycling"].id, "s1-Cycling")
    self.assertEqual(quotes["Taxes"].text, cleaned)
    self.assertEqual(transport.bodies[0]["state"], cleaned)
    self.assertEqual(
        [q["type"] for q in transport.bodies[0]["questions"].values()],
        ["extract", "extract"],
    )
    model.process_prompts_concurrently.assert_not_called()

  async def test_span_host_error_fails_the_stage(self):
    """Tests that a host rejecting extract questions is not papered over."""

    def transport(url: str, body: bytes) -> bytes:
      """Rejects the request the way a judgment-only host does."""
      raise systemone.SystemOneError(
          systemone.FailureKind.HTTP,
          'question "t0": type must be choice, noul, or score',
          status=400,
      )

    client = systemone.SystemOneClient(model="clef", transport=transport)
    statements, model = self._make_single_statement_and_mock_model()
    with patch.object(
        quote_extraction_lib.model_factory,
        "get_quote_model",
        return_value=client,
    ):
      with self.assertRaises(systemone.SystemOneError):
        await quote_extraction_lib.extract_quotes_from_text(
            statements=statements, model=model
        )
    model.process_prompts_concurrently.assert_not_called()

  def test_join_response_text(self):
    self.assertEqual(
        quote_extraction_lib.join_response_text("<response>Hello.</response>"),
        "Hello.",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text("<response>Hello</response>"),
        "Hello.",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text(
            "<response>Hello</response> <response>World!</response>"
        ),
        "Hello. World!",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text("This text has no response tags."),
        "This text has no response tags.",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text(""),
        "",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text(None),
        "",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text("<response>  Hello  </response>"),
        "Hello.",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text(
            "<response>Hello</response> <response>   </response>"
        ),
        "Hello.",
    )
    self.assertEqual(
        quote_extraction_lib.join_response_text("<response>   </response>"),
        "<response>   </response>",
    )


if __name__ == "__main__":
  unittest.main()
