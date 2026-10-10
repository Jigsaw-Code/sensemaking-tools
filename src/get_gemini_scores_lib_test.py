# Copyright 2026 Google LLC
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

"""Tests for get_gemini_scores_lib."""

from typing import Any
import unittest
from unittest import mock

from google.api_core import exceptions as google_exceptions

from src import attribute_prompt_config
from src import get_gemini_scores_lib
from src.models import genai_model

_ATTR = "CURIOSITY_EXPERIMENTAL"
_OTHER_ATTR = "REASONING_EXPERIMENTAL"
_JOB = {"target_attr": _ATTR}


def _response(text: str | None) -> dict[str, Any]:
  """Builds a GenaiModel.call_gemini-style response dict."""
  return {
      "text": text,
      "total_token_count": 1,
      "prompt_token_count": 1,
      "candidates_token_count": 1,
      "tool_use_prompt_token_count": 0,
      "thoughts_token_count": 0,
  }


class ParseScoreResponseTest(unittest.TestCase):
  """Tests for parse_score_response."""

  def test_valid_response_returns_score(self):
    """Checks a well-formed response is parsed into a float score."""
    result = get_gemini_scores_lib.parse_score_response(
        _response('{"score": 0.75}'), _JOB
    )
    self.assertEqual(result, {_ATTR: 0.75})

  def test_malformed_responses_raise(self):
    """Checks malformed responses raise rather than defaulting to 0.0."""
    bad_texts = {
        "invalid_json": "not json",
        "missing_score": '{"other": 1}',
        "non_numeric_score": '{"score": "high"}',
        "empty_text": "",
        "none_text": None,
    }
    for name, text in bad_texts.items():
      with self.subTest(name):
        with self.assertRaisesRegex(ValueError, _ATTR):
          get_gemini_scores_lib.parse_score_response(_response(text), _JOB)

  def test_missing_text_key_raises(self):
    """Checks a response without a text key raises."""
    with self.assertRaises(ValueError):
      get_gemini_scores_lib.parse_score_response({}, _JOB)

  def test_missing_target_attr_raises(self):
    """Checks a job without target_attr fails clearly."""
    with self.assertRaises(KeyError):
      get_gemini_scores_lib.parse_score_response(
          _response('{"score": 0.5}'), {}
      )

  def test_error_message_truncates_raw_text(self):
    """Checks long raw output is truncated in the error message."""
    long_text = "x" * 5000
    with self.assertRaises(ValueError) as ctx:
      get_gemini_scores_lib.parse_score_response(_response(long_text), _JOB)
    self.assertNotIn(long_text, str(ctx.exception))
    self.assertLess(len(str(ctx.exception)), 2000)

  def test_boundary_scores_accepted(self):
    """Checks scores at and within [0.0, 1.0] are accepted."""
    cases = {
        '{"score": 0}': 0.0,
        '{"score": 0.0}': 0.0,
        '{"score": 1}': 1.0,
        '{"score": 1.0}': 1.0,
        '{"score": "0.7"}': 0.7,
    }
    for text, expected in cases.items():
      with self.subTest(text):
        result = get_gemini_scores_lib.parse_score_response(
            _response(text), _JOB
        )
        self.assertEqual(result, {_ATTR: expected})

  def test_out_of_range_or_non_finite_scores_raise(self):
    """Checks scores that are not probabilities raise so they are retried."""
    bad_texts = [
        '{"score": 1.5}',
        '{"score": -0.2}',
        '{"score": 85}',
        '{"score": NaN}',
        '{"score": Infinity}',
        '{"score": -Infinity}',
    ]
    for text in bad_texts:
      with self.subTest(text):
        with self.assertRaises(ValueError):
          get_gemini_scores_lib.parse_score_response(_response(text), _JOB)


# Prevents GenaiModel from constructing a real google.genai client.
@mock.patch("google.genai.Client")
class ContentScorerRetryTest(unittest.TestCase):
  """Runs ContentScorer through the real GenaiModel retry loop.

  Only GenaiModel.call_gemini is mocked, so these tests verify that a
  malformed response is retried by the wrapper and that exhausted retries
  produce no score rather than a 0.0.
  """

  def setUp(self):
    """Removes sleeps from the worker loop so tests run instantly.

    genai_model.asyncio is the global asyncio module, so this replaces
    asyncio.sleep process-wide for the duration of each test.
    """
    super().setUp()
    sleep_patcher = mock.patch.object(
        genai_model.asyncio, "sleep", new=mock.AsyncMock()
    )
    sleep_patcher.start()
    self.addCleanup(sleep_patcher.stop)

  def _make_scorer(
      self, responses: Any, max_llm_retries: int
  ) -> get_gemini_scores_lib.ContentScorer:
    """Builds a ContentScorer whose Gemini calls return `responses`.

    Args:
      responses: A side_effect for call_gemini: a list of responses returned
        in order, or a callable taking call_gemini's keyword arguments.
      max_llm_retries: Maximum attempts per job.

    Returns:
      The configured ContentScorer.
    """
    scorer = get_gemini_scores_lib.ContentScorer(
        gemini_api_key="test_key",
        model_name="test_model",
        max_llm_retries=max_llm_retries,
    )
    scorer.client.call_gemini = mock.AsyncMock(side_effect=responses)
    return scorer

  def test_malformed_response_is_retried(self, mock_client):
    """Checks a malformed response triggers a retry that can succeed."""
    del mock_client  # Unused.
    scorer = self._make_scorer(
        [_response("not json"), _response('{"score": 0.6}')],
        max_llm_retries=3,
    )

    results = scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, 2)
    self.assertEqual(results, [{"row_id": 0, "scores": {_ATTR: 0.6}}])

  def test_out_of_range_score_is_retried(self, mock_client):
    """Checks a percentage-style score is retried rather than recorded."""
    del mock_client  # Unused.
    scorer = self._make_scorer(
        [_response('{"score": 85}'), _response('{"score": 0.85}')],
        max_llm_retries=3,
    )

    results = scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, 2)
    self.assertEqual(results, [{"row_id": 0, "scores": {_ATTR: 0.85}}])

  def test_exhausted_retries_produce_no_score(self, mock_client):
    """Checks repeated malformed responses yield no score, not 0.0."""
    del mock_client  # Unused.
    scorer = self._make_scorer([_response("not json")] * 3, max_llm_retries=3)

    results = scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, 3)
    self.assertEqual(results, [{"row_id": 0, "scores": {}}])

  def test_default_limit_caps_billed_attempts(self, mock_client):
    """Checks a persistently bad response stops at the scoring default."""
    del mock_client  # Unused.
    default = get_gemini_scores_lib.DEFAULT_SCORING_MAX_LLM_RETRIES
    self.assertLess(default, genai_model.MAX_LLM_RETRIES)
    # Built directly, not via _make_scorer, so max_llm_retries is omitted and
    # ContentScorer's own default applies.
    scorer = get_gemini_scores_lib.ContentScorer(
        gemini_api_key="test_key", model_name="test_model"
    )
    scorer.client.call_gemini = mock.AsyncMock(
        side_effect=lambda **kwargs: _response('{"score": 85}')
    )

    results = scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, default)
    self.assertEqual(results, [{"row_id": 0, "scores": {}}])

  def test_explicit_limit_is_honored(self, mock_client):
    """Checks callers can override the scoring attempt limit."""
    del mock_client  # Unused.
    scorer = self._make_scorer(
        lambda **kwargs: _response("not json"), max_llm_retries=2
    )

    scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, 2)

  def test_service_unavailable_does_not_consume_attempts(self, mock_client):
    """Checks 503s pause and retry without counting against the limit."""
    del mock_client  # Unused.
    unavailable = google_exceptions.ServiceUnavailable("overloaded")
    scorer = self._make_scorer(
        [unavailable, unavailable, unavailable, _response('{"score": 0.3}')],
        max_llm_retries=1,
    )

    results = scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, 4)
    self.assertEqual(results, [{"row_id": 0, "scores": {_ATTR: 0.3}}])

  def test_partial_failure_keeps_successful_attribute(self, mock_client):
    """Checks one attribute failing does not affect another's score."""
    del mock_client  # Unused.
    # Match on the definition: labels such as "Reasoning" also appear in
    # every prompt's calibrated examples.
    failing_definition = attribute_prompt_config.ATTRIBUTES[_OTHER_ATTR][
        "definition"
    ]

    def respond(**kwargs):
      if failing_definition in kwargs["system_prompt"]:
        return _response("not json")
      return _response('{"score": 0.8}')

    scorer = self._make_scorer(respond, max_llm_retries=3)

    with mock.patch.object(
        get_gemini_scores_lib.logging, "warning"
    ) as mock_warning:
      results = scorer.score(
          [{"text": "hello", "row_id": 7}], [_ATTR, _OTHER_ATTR]
      )

    self.assertEqual(results, [{"row_id": 7, "scores": {_ATTR: 0.8}}])
    # One successful call plus three failed attempts.
    self.assertEqual(scorer.client.call_gemini.await_count, 4)
    mock_warning.assert_called_once()
    self.assertIn((7, _OTHER_ATTR), mock_warning.call_args.args[-1])

  def test_no_warning_when_all_scores_obtained(self, mock_client):
    """Checks the missing-score warning is not logged on full success."""
    del mock_client  # Unused.
    scorer = self._make_scorer(
        [_response('{"score": 0.2}')], max_llm_retries=3
    )

    with mock.patch.object(
        get_gemini_scores_lib.logging, "warning"
    ) as mock_warning:
      scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    mock_warning.assert_not_called()

  def test_api_error_still_retried_without_parser(self, mock_client):
    """Checks API-level errors (e.g. safety) keep their existing retries."""
    del mock_client  # Unused.
    scorer = self._make_scorer(
        [{"error": "SAFETY"}, _response('{"score": 0.4}')], max_llm_retries=3
    )

    with mock.patch.object(
        get_gemini_scores_lib, "parse_score_response",
        wraps=get_gemini_scores_lib.parse_score_response,
    ) as mock_parser:
      results = scorer.score([{"text": "hello", "row_id": 0}], [_ATTR])

    self.assertEqual(scorer.client.call_gemini.await_count, 2)
    # The parser only sees the successful response, not the API error.
    mock_parser.assert_called_once()
    self.assertEqual(results, [{"row_id": 0, "scores": {_ATTR: 0.4}}])


if __name__ == "__main__":
  unittest.main()
