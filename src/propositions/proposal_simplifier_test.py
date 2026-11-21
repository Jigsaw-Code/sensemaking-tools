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

"""Unit tests for proposal_simplifier."""

import asyncio
import unittest

import pandas as pd

from src.models import fake_base_model
from src.models import retry_utils
from src.propositions import proposal_simplifier

_SIMPLE_PROPOSAL = (
    "Build safe bike lanes along main roads so people can ride to work."
)
_COMPLEX_PROPOSAL = (
    "Establish an intergovernmental administrative modernization consortium"
    " dedicated to systematically restructuring bureaucratic procurement"
    " methodologies, infrastructural capitalization mechanisms, and"
    " multijurisdictional accountability frameworks."
)
_STILL_COMPLEX_REWRITE = (
    "Create an interagency administrative modernization group to restructure"
    " bureaucratic procurement methodologies and infrastructural accountability"
    " frameworks."
)
_WORSE_COMPLEX_REWRITE = (
    "Institutionalize comprehensive multijurisdictional administrative"
    " modernization consortiums specifically operationalizing bureaucratic"
    " procurement methodologies, infrastructural capitalization mechanisms,"
    " intergovernmental accountability frameworks, and regulatory compliance"
    " standardizations."
)
_COMPLIANT_REWRITE = (
    "Create a joint group to make buying rules simpler and clearer. Track how"
    " road and building funds are spent each year."
)


class ReadabilityScoringTest(unittest.TestCase):
  """Tests deterministic pyphen syllable counting and FKGL scoring."""

  def test_count_syllables_uses_pyphen(self):
    self.assertEqual(proposal_simplifier.count_syllables("cat"), 1)
    self.assertEqual(proposal_simplifier.count_syllables("water"), 2)
    self.assertEqual(proposal_simplifier.count_syllables("water,"), 2)
    self.assertEqual(proposal_simplifier.count_syllables("infrastructure"), 4)
    self.assertEqual(proposal_simplifier.count_syllables("transportation"), 4)
    self.assertEqual(proposal_simplifier.count_syllables("accountability"), 5)
    self.assertEqual(proposal_simplifier.count_syllables("2026"), 0)
    self.assertEqual(proposal_simplifier.count_syllables(""), 0)
    self.assertEqual(
        proposal_simplifier.count_syllables("computadora", lang="es"), 3
    )

  def test_score_readability_threads_lang_parameter(self):
    score_en = proposal_simplifier.score_readability("Computadora.", lang="en")
    score_es = proposal_simplifier.score_readability("Computadora.", lang="es")
    self.assertEqual(score_es.syllable_count, 3)
    self.assertGreaterEqual(score_en.syllable_count, 1)

  def test_score_readability_computes_expected_fkgl_and_averages(self):
    # "Water flows fast. Dogs run far." -> 6 words, 2 sentences, 7 syllables.
    # FKGL = 0.39 * (6 / 2) + 11.8 * (7 / 6) - 15.59 = -0.65.
    score = proposal_simplifier.score_readability(
        "Water flows fast. Dogs run far."
    )
    self.assertEqual(score.word_count, 6)
    self.assertEqual(score.sentence_count, 2)
    self.assertEqual(score.syllable_count, 7)
    self.assertEqual(score.avg_sentence_length, 3.0)
    self.assertEqual(score.avg_syllables_per_word, 1.17)
    self.assertEqual(score.fkgl, -0.65)

    complex_score = proposal_simplifier.score_readability(_COMPLEX_PROPOSAL)
    self.assertGreater(complex_score.fkgl, proposal_simplifier.DEFAULT_MAX_FKGL)

  def test_score_readability_empty_or_punctuation_only(self):
    for text in ("", "   ", "... !!!"):
      score = proposal_simplifier.score_readability(text)
      self.assertEqual(
          score,
          proposal_simplifier.ReadabilityScore(
              fkgl=0.0,
              word_count=0,
              sentence_count=0,
              syllable_count=0,
              avg_sentence_length=0.0,
              avg_syllables_per_word=0.0,
          ),
      )

  def test_check_readability_compliance_reports_violations(self):
    empty_score = proposal_simplifier.score_readability("")
    compliant, violations = proposal_simplifier.check_readability_compliance(
        empty_score
    )
    self.assertFalse(compliant)
    self.assertEqual(violations, ("Statement is empty.",))

    many_sentences_score = proposal_simplifier.ReadabilityScore(
        fkgl=12.4,
        word_count=45,
        sentence_count=8,
        syllable_count=80,
        avg_sentence_length=5.63,
        avg_syllables_per_word=1.78,
    )
    compliant, violations = proposal_simplifier.check_readability_compliance(
        many_sentences_score, max_fkgl=9.5, max_sentences=6, max_words=30
    )
    self.assertFalse(compliant)
    self.assertEqual(len(violations), 3)
    self.assertIn("12.40", violations[0])
    self.assertIn("Sentence count was 8", violations[1])
    self.assertIn("Word count was 45", violations[2])

  def test_clean_candidate_text_strips_wrappers(self):
    wrapped = (
        '```text\nRewrite: "**Transit Plan**: Build safer bus stops."\n```'
    )
    self.assertEqual(
        proposal_simplifier.clean_candidate_text(wrapped),
        "Build safer bus stops.",
    )


class SimplifyProposalTest(unittest.TestCase):
  """Tests single-proposal simplification and readability assertion retries."""

  def test_verbatim_passthrough_when_already_compliant(self):
    model = fake_base_model.FakeBaseModel([])
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            model, _SIMPLE_PROPOSAL
        )
    )
    self.assertEqual(len(model.calls), 0)
    self.assertTrue(result.compliant)
    self.assertEqual(result.attempts, 0)
    self.assertEqual(result.violations, ())
    self.assertEqual(result.pre_simplified_statement, _SIMPLE_PROPOSAL)
    self.assertEqual(result.post_simplified_statement, _SIMPLE_PROPOSAL)
    self.assertEqual(result.pre_score, result.post_score)

  def test_blank_input_returns_non_compliant_without_model_call(self):
    model = fake_base_model.FakeBaseModel([])
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(model, "   ")
    )
    self.assertEqual(len(model.calls), 0)
    self.assertFalse(result.compliant)
    self.assertEqual(result.attempts, 0)
    self.assertEqual(result.post_simplified_statement, "")
    self.assertEqual(result.violations, ("Statement is empty.",))

  def test_retries_with_violation_feedback_until_compliant(self):
    model = fake_base_model.FakeBaseModel([
        {"text": _STILL_COMPLEX_REWRITE},
        {"text": f"```\n{_COMPLIANT_REWRITE}\n```"},
    ])
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            model, _COMPLEX_PROPOSAL, max_attempts=4
        )
    )
    self.assertTrue(result.compliant)
    self.assertEqual(result.attempts, 2)
    self.assertEqual(result.violations, ())
    self.assertEqual(result.pre_simplified_statement, _COMPLEX_PROPOSAL)
    self.assertEqual(result.post_simplified_statement, _COMPLIANT_REWRITE)
    self.assertGreater(
        result.pre_score.fkgl, proposal_simplifier.DEFAULT_MAX_FKGL
    )
    self.assertLessEqual(
        result.post_score.fkgl, proposal_simplifier.DEFAULT_MAX_FKGL
    )
    # Second prompt should feed back the prior attempt and exact FKGL violation.
    self.assertEqual(len(model.calls), 2)
    self.assertIn(_STILL_COMPLEX_REWRITE, model.calls[1]["prompt"])
    self.assertIn("Flesch-Kincaid Grade Level was", model.calls[1]["prompt"])
    self.assertGreater(
        model.calls[1]["temperature"], model.calls[0]["temperature"]
    )

  def test_exhausted_attempts_returns_lowest_penalty_candidate(self):
    model = fake_base_model.FakeBaseModel([
        {"text": _STILL_COMPLEX_REWRITE},
        {"text": _WORSE_COMPLEX_REWRITE},
    ])
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            model, _COMPLEX_PROPOSAL, max_attempts=2
        )
    )
    self.assertFalse(result.compliant)
    self.assertEqual(result.attempts, 2)
    self.assertEqual(result.post_simplified_statement, _STILL_COMPLEX_REWRITE)
    self.assertTrue(result.violations)
    self.assertGreater(
        result.post_score.fkgl, proposal_simplifier.DEFAULT_MAX_FKGL
    )

  def test_recovers_from_transient_model_errors(self):
    model = fake_base_model.FakeBaseModel([
        RuntimeError("transient failure"),
        {"error": "rate limit exceeded"},
        {"text": _COMPLIANT_REWRITE},
    ])
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            model, _COMPLEX_PROPOSAL, max_attempts=3
        )
    )
    self.assertTrue(result.compliant)
    self.assertEqual(result.attempts, 1)
    self.assertEqual(result.post_simplified_statement, _COMPLIANT_REWRITE)
    self.assertEqual(len(model.calls), 3)
    # Transport retries must not consume readability iterations or bump temp.
    self.assertEqual(
        [call["temperature"] for call in model.calls], [0.1, 0.1, 0.1]
    )

  def test_exhausted_transport_errors_returns_best_candidate(self):
    # Exhausted transport errors on attempt 1 fall back to the original text.
    failing_model = fake_base_model.FakeBaseModel([
        RuntimeError("network down"),
        {"error": "503 unavailable"},
    ])
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            failing_model, _COMPLEX_PROPOSAL, max_attempts=2
        )
    )
    self.assertFalse(result.compliant)
    self.assertEqual(result.attempts, 1)
    self.assertEqual(result.post_simplified_statement, _COMPLEX_PROPOSAL)
    self.assertTrue(result.violations)
    self.assertEqual(len(failing_model.calls), 2)

    # When an earlier readability attempt improved the draft, keep it on break.
    partial_model = fake_base_model.FakeBaseModel([
        {"text": _STILL_COMPLEX_REWRITE},
        RuntimeError("timeout"),
        {"error": "503 unavailable"},
    ])
    partial_result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            partial_model, _COMPLEX_PROPOSAL, max_attempts=2
        )
    )
    self.assertFalse(partial_result.compliant)
    self.assertEqual(partial_result.attempts, 2)
    self.assertEqual(
        partial_result.post_simplified_statement, _STILL_COMPLEX_REWRITE
    )
    self.assertEqual(len(partial_model.calls), 3)

  def test_reuses_existing_retrying_model_without_double_wrapping(self):
    inner = fake_base_model.FakeBaseModel([
        RuntimeError("transient error"),
        {"error": "still failing"},
    ])
    wrapped = retry_utils.RetryingModel(inner, max_retries=2)
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            wrapped, _COMPLEX_PROPOSAL, max_attempts=4
        )
    )
    self.assertFalse(result.compliant)
    self.assertEqual(result.attempts, 1)
    self.assertEqual(result.post_simplified_statement, _COMPLEX_PROPOSAL)
    self.assertEqual(len(inner.calls), 2)

  def test_additional_context_and_custom_system_prompt_propagated(self):
    model = fake_base_model.FakeBaseModel([
        {"text": _STILL_COMPLEX_REWRITE},
        {"text": _COMPLIANT_REWRITE},
    ])
    context_note = "Focus on transit access for rural riders."
    custom_system = "Custom plain-English editor instructions."
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            model,
            _COMPLEX_PROPOSAL,
            max_attempts=3,
            additional_context=context_note,
            system_prompt=custom_system,
        )
    )
    self.assertTrue(result.compliant)
    for call in model.calls:
      self.assertIn(context_note, call["prompt"])
      self.assertEqual(call["system_prompt"], custom_system)

  def test_max_words_enforced_only_when_specified(self):
    # _SIMPLE_PROPOSAL has 13 words and low FKGL.
    model = fake_base_model.FakeBaseModel(
        [{"text": "Build safe bike lanes along main roads."}]
    )
    result = asyncio.run(
        proposal_simplifier.simplify_proposal_with_assertion(
            model, _SIMPLE_PROPOSAL, max_words=8, max_attempts=2
        )
    )
    self.assertTrue(result.compliant)
    self.assertEqual(result.attempts, 1)
    self.assertLessEqual(result.post_score.word_count, 8)
    self.assertIn("at most 8 words", model.calls[0]["prompt"])

  def test_invalid_arguments_raise_early(self):
    model = fake_base_model.FakeBaseModel([])
    with self.assertRaises(ValueError):
      asyncio.run(
          proposal_simplifier.simplify_proposal_with_assertion(
              model, _SIMPLE_PROPOSAL, max_fkgl=0.0
          )
      )
    with self.assertRaises(TypeError):
      asyncio.run(
          proposal_simplifier.simplify_proposal_with_assertion(
              model, _SIMPLE_PROPOSAL, max_fkgl=True
          )
      )
    with self.assertRaises(ValueError):
      asyncio.run(
          proposal_simplifier.simplify_proposal_with_assertion(
              model, _SIMPLE_PROPOSAL, max_sentences=0
          )
      )
    with self.assertRaises(ValueError):
      asyncio.run(
          proposal_simplifier.simplify_proposal_with_assertion(
              model, _SIMPLE_PROPOSAL, max_words=0
          )
      )
    with self.assertRaises(ValueError):
      asyncio.run(
          proposal_simplifier.simplify_proposal_with_assertion(
              model, _SIMPLE_PROPOSAL, max_attempts=0
          )
      )


class SimplifyProposalsBatchTest(unittest.TestCase):
  """Tests batch DataFrame simplification behavior."""

  def test_dataframe_simplification_preserves_columns_without_mutating_input(
      self,
  ):
    input_df = pd.DataFrame({
        "proposal_id": ["p1", "p2", "p3"],
        "proposition": [_SIMPLE_PROPOSAL, _COMPLEX_PROPOSAL, None],
    })
    original_columns = list(input_df.columns)
    model = fake_base_model.FakeBaseModel([{"text": _COMPLIANT_REWRITE}])

    output_df = asyncio.run(
        proposal_simplifier.simplify_proposals_dataframe(
            model, input_df, max_concurrent_calls=2
        )
    )

    # Input DataFrame must remain untouched.
    self.assertEqual(list(input_df.columns), original_columns)
    self.assertEqual(len(model.calls), 1)
    self.assertEqual(list(output_df["proposal_id"]), ["p1", "p2", "p3"])
    self.assertEqual(
        list(output_df["simplification"]),
        [_SIMPLE_PROPOSAL, _COMPLIANT_REWRITE, ""],
    )
    self.assertEqual(
        list(output_df["post_simplified_statement"]),
        [_SIMPLE_PROPOSAL, _COMPLIANT_REWRITE, ""],
    )
    self.assertEqual(
        list(output_df["pre_simplified_statement"]),
        [_SIMPLE_PROPOSAL, _COMPLEX_PROPOSAL, ""],
    )
    self.assertEqual(
        list(output_df["readability_compliant"]), [True, True, False]
    )
    self.assertEqual(list(output_df["simplification_attempts"]), [0, 1, 0])
    self.assertGreater(
        output_df["pre_fkgl"].iloc[1], proposal_simplifier.DEFAULT_MAX_FKGL
    )
    self.assertLessEqual(
        output_df["post_fkgl"].iloc[1], proposal_simplifier.DEFAULT_MAX_FKGL
    )

  def test_dataframe_empty_and_missing_column(self):
    model = fake_base_model.FakeBaseModel([])
    empty_df = pd.DataFrame({"proposition": pd.Series(dtype="object")})
    result_df = asyncio.run(
        proposal_simplifier.simplify_proposals_dataframe(model, empty_df)
    )
    self.assertTrue(result_df.empty)
    for col in (
        "simplification",
        "pre_simplified_statement",
        "post_simplified_statement",
        "pre_fkgl",
        "post_fkgl",
        "readability_compliant",
        "simplification_attempts",
    ):
      self.assertIn(col, result_df.columns)

    wrong_col_df = pd.DataFrame({"other": ["text"]})
    with self.assertRaises(ValueError):
      asyncio.run(
          proposal_simplifier.simplify_proposals_dataframe(
              model, wrong_col_df, statement_column="proposition"
          )
      )


if __name__ == "__main__":
  unittest.main()
