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

"""Tests for get_bridging_scores."""

import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

from src import get_bridging_scores as gbs

# Deterministic fake score per text, identical across attributes.
_TEXT_SCORES = {"alpha": 0.1, "beta": 0.5, "gamma": 0.9}


def _fake_gemini_score(texts_with_ids, attributes):
  """Mimics ContentScorer.score, returning a fixed score per text.

  Results are returned in reverse order so tests catch any code that maps
  results back by position rather than by row_id.
  """
  results = [
      {
          "row_id": item["row_id"],
          "scores": {attr: _TEXT_SCORES[item["text"]] for attr in attributes},
      }
      for item in texts_with_ids
  ]
  return list(reversed(results))


def _fake_perspective_score(client, text, attributes):
  """Mimics get_perspective_scores_lib.score_text."""
  del client  # Unused.
  return {attr: _TEXT_SCORES[text] for attr in attributes}


class GetBridgingScoresTest(unittest.TestCase):
  """Tests for get_bridging_scores.get_bridging_scores."""

  def setUp(self):
    """Builds an input frame where quotes repeat across topics/opinions."""
    super().setUp()
    self.df = pd.DataFrame({
        "participant_id": ["p1", "p1", "p2", "p3", "p3", "p3"],
        "quote": ["alpha", "alpha", "beta", "gamma", "alpha", "gamma"],
        "topic": ["t1", "t2", "t1", "t1", "t3", "t2"],
    })
    patcher = patch("src.get_bridging_scores.ContentScorer")
    self.mock_scorer = patcher.start().return_value
    self.addCleanup(patcher.stop)
    self.mock_scorer.score.side_effect = _fake_gemini_score

  def _score_gemini(self, df):
    """Runs get_bridging_scores with the GEMINI backend."""
    return gbs.get_bridging_scores(df, "quote", "key", None, "GEMINI", "m")

  def _assert_scores_match_text(self, result):
    """Asserts every row carries the score for its own text."""
    for _, row in result.iterrows():
      expected = _TEXT_SCORES[row["quote"]]
      for attr in gbs.BRIDGING_ATTRIBUTES:
        self.assertAlmostEqual(row[attr], expected)
      self.assertAlmostEqual(row[gbs.AVERAGE_BRIDGING_COLUMN], expected)

  def test_gemini_scores_each_unique_text_once(self):
    """Checks Gemini receives only unique texts and all rows get scores."""
    result = self._score_gemini(self.df)

    self.mock_scorer.score.assert_called_once()
    sent, attrs = self.mock_scorer.score.call_args.args
    self.assertCountEqual([s["text"] for s in sent], ["alpha", "beta", "gamma"])
    self.assertEqual(attrs, gbs.BRIDGING_ATTRIBUTES)
    self.assertEqual(len(result), len(self.df))
    self._assert_scores_match_text(result)

  @patch("src.get_bridging_scores.get_perspective_scores_lib")
  def test_perspective_scores_each_unique_text_once(self, mock_lib):
    """Checks Perspective is called once per unique text."""
    mock_lib.score_text.side_effect = _fake_perspective_score

    result = gbs.get_bridging_scores(
        self.df, "quote", None, "key", "PERSPECTIVE", "model"
    )

    scored_texts = [c.args[1] for c in mock_lib.score_text.call_args_list]
    self.assertCountEqual(scored_texts, ["alpha", "beta", "gamma"])
    self._assert_scores_match_text(result)

  def test_preserves_rows_index_and_does_not_mutate_input(self):
    """Checks output keeps row order, columns, and index; input untouched."""
    df = self.df.copy()
    df.index = [10, 10, 3, 7, 7, 1]  # Non-unique, non-sequential index.
    original = df.copy()

    result = self._score_gemini(df)

    pd.testing.assert_frame_equal(df, original)
    self.assertEqual(list(result.index), list(df.index))
    pd.testing.assert_frame_equal(result[list(df.columns)], df)
    self._assert_scores_match_text(result)

  def test_failed_text_gets_nan(self):
    """Checks rows whose text returned no result get NaN, others unaffected."""

    def score_without_beta(texts_with_ids, attributes):
      kept = [t for t in texts_with_ids if t["text"] != "beta"]
      return _fake_gemini_score(kept, attributes)

    self.mock_scorer.score.side_effect = score_without_beta

    result = self._score_gemini(self.df)

    beta_rows = result[result["quote"] == "beta"]
    self.assertEqual(len(beta_rows), 1)
    self.assertTrue(beta_rows[gbs.BRIDGING_ATTRIBUTES].isna().all(axis=None))
    self.assertTrue(beta_rows[gbs.AVERAGE_BRIDGING_COLUMN].isna().all())
    self._assert_scores_match_text(result[result["quote"] != "beta"])

  def test_non_string_texts_are_scored_as_strings(self):
    """Checks non-string values are converted and deduplicated as strings."""
    self.mock_scorer.score.side_effect = lambda items, attrs: [
        {"row_id": i["row_id"], "scores": {a: 0.3 for a in attrs}}
        for i in items
    ]
    df = pd.DataFrame({"quote": [1, 1, 2]})

    result = self._score_gemini(df)

    sent = [s["text"] for s in self.mock_scorer.score.call_args.args[0]]
    self.assertCountEqual(sent, ["1", "2"])
    self.assertTrue((result[gbs.AVERAGE_BRIDGING_COLUMN] == 0.3).all())

  def test_existing_scores_replaced_with_warning_when_all_calls_fail(self):
    """Checks stale input scores are never kept when the scorer fails."""
    df = self.df.copy()
    for col in gbs.BRIDGING_ATTRIBUTES + [gbs.AVERAGE_BRIDGING_COLUMN]:
      df[col] = 0.99
    self.mock_scorer.score.side_effect = lambda items, attrs: []

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
      result = self._score_gemini(df)

    self.assertIn("Warning: Existing bridging score columns", stdout.getvalue())
    score_columns = gbs.BRIDGING_ATTRIBUTES + [gbs.AVERAGE_BRIDGING_COLUMN]
    self.assertTrue(result[score_columns].isna().all(axis=None))
    self.assertEqual(len(result), len(df))

  def test_partial_attributes_do_not_mix_with_existing_scores(self):
    """Checks missing attributes become NaN rather than keeping old values."""
    df = self.df.copy()
    for col in gbs.BRIDGING_ATTRIBUTES:
      df[col] = 0.99
    returned_attr = gbs.BRIDGING_ATTRIBUTES[0]
    self.mock_scorer.score.side_effect = lambda items, attrs: [
        {"row_id": i["row_id"], "scores": {returned_attr: 0.2}} for i in items
    ]

    with contextlib.redirect_stdout(io.StringIO()):
      result = self._score_gemini(df)

    self.assertTrue((result[returned_attr] == 0.2).all())
    self.assertTrue(result[gbs.BRIDGING_ATTRIBUTES[1:]].isna().all(axis=None))
    self.assertTrue((result[gbs.AVERAGE_BRIDGING_COLUMN] == 0.2).all())

  def test_output_has_exactly_bridging_columns(self):
    """Checks unexpected score keys are dropped and column order is fixed."""
    self.mock_scorer.score.side_effect = lambda items, attrs: [
        {
            "row_id": i["row_id"],
            "scores": {"UNEXPECTED": 1.0, **{a: 0.4 for a in reversed(attrs)}},
        }
        for i in items
    ]

    result = self._score_gemini(self.df)

    expected_columns = (
        list(self.df.columns)
        + gbs.BRIDGING_ATTRIBUTES
        + [gbs.AVERAGE_BRIDGING_COLUMN]
    )
    self.assertEqual(list(result.columns), expected_columns)

  def test_empty_input_returns_empty_frame_with_score_columns(self):
    """Checks an empty input returns an empty frame instead of raising."""
    self.mock_scorer.score.side_effect = lambda items, attrs: []
    df = self.df.iloc[0:0]

    result = self._score_gemini(df)

    self.assertTrue(result.empty)
    for col in gbs.BRIDGING_ATTRIBUTES + [gbs.AVERAGE_BRIDGING_COLUMN]:
      self.assertIn(col, result.columns)

  def test_unknown_scorer_type_raises(self):
    """Checks an unrecognized scorer_type raises ValueError."""
    with self.assertRaises(ValueError):
      gbs.get_bridging_scores(self.df, "quote", None, None, "OTHER", "model")


if __name__ == "__main__":
  unittest.main()
