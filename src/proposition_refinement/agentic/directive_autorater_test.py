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

"""Unit tests for the post-hoc directive and atomicity autorater."""

import asyncio
from collections.abc import Sequence
import json
import pathlib
import tempfile
from typing import Any
import unittest

import pandas as pd

from src.models import fake_base_model
from src.proposition_refinement.agentic import directive_autorater

_SEED_STATEMENT = (
    "Add dedicated bus lanes along main evening commuter corridors. Fund"
    " signage and paint from the existing street maintenance budget."
)
_FOCUSED_CANDIDATE = (
    "Add dedicated bus lanes along main evening commuter corridors. Fund"
    " signage and paint from the existing street maintenance budget. Publish"
    " quarterly on-time bus speed reports online."
)
_STACKED_SENTENCES_CANDIDATE = (
    "Add dedicated bus lanes along main evening commuter corridors. Fund"
    " signage and paint from the existing street maintenance budget. Publish"
    " quarterly bus speed reports online. Install solar shelters at every"
    " stop. Launch a citywide fare subsidy for evening riders."
)
_PRE_SIMPLIFIED = (
    "Add dedicated bus lanes along main evening commuter corridors. Fund"
    " signage and paint from the existing street maintenance budget and"
    " publish quarterly performance audits."
)
_POST_SIMPLIFIED_CLEAN = (
    "Add dedicated bus lanes on main evening commuter roads. Pay for signs"
    " and paint from the current street upkeep budget and post quarterly"
    " performance checks online."
)
_POST_SIMPLIFIED_DRIFTED = (
    "Add dedicated bus lanes on main evening commuter roads using a new"
    " commuter toll."
)


def _make_rubric_json(**overrides: Any) -> str:
  """Builds a valid JSON response string matching DIRECTIVE_RESPONSE_SCHEMA."""
  payload: dict[str, Any] = {
      "added_components": [{
          "text": "Publish quarterly on-time bus speed reports online",
          "kind": "safeguard_or_oversight",
          "relation": "directly_addresses_target_friction",
          "borrowed_from_companion": False,
      }],
      "n_primary_new_components": 1,
      "preserves_core_purpose": True,
      "is_atomic": True,
      "has_unrelated_addition": False,
      "companion_substitution": False,
      "self_contained": True,
      "simplifier_drift": "not_applicable",
      "drift_note": "",
      "rationale": (
          "Keeps the seed core purpose intact and adds one focused safeguard."
      ),
  }
  payload.update(overrides)
  return json.dumps(payload)


class _DelayedFakeModel(fake_base_model.FakeBaseModel):
  """FakeBaseModel subclass adding sleep delay and concurrency tracking."""

  def __init__(
      self,
      responses: Sequence[dict[str, Any] | Exception],
      *,
      delay_seconds: float,
  ) -> None:
    super().__init__(responses)
    self._delay_seconds = delay_seconds
    self.active_calls: int = 0
    self.peak_concurrency: int = 0

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: dict[str, Any] | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    self.active_calls += 1
    self.peak_concurrency = max(self.peak_concurrency, self.active_calls)
    try:
      if self._delay_seconds > 0:
        await asyncio.sleep(self._delay_seconds)
      return await super().generate_content(
          prompt=prompt,
          run_name=run_name,
          temperature=temperature,
          system_prompt=system_prompt,
          response_mime_type=response_mime_type,
          response_schema=response_schema,
          **kwargs,
      )
    finally:
      self.active_calls -= 1


class PromptAndMetricsTest(unittest.TestCase):
  """Tests deterministic sentence metrics and prompt construction."""

  def test_deterministic_metrics_use_candidate_drafts_helpers(self) -> None:
    item = directive_autorater.DirectiveAuditItem(
        item_id="stacked-1",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_STACKED_SENTENCES_CANDIDATE,
        declared_new_components=(
            "- Quarterly reports\n- Solar shelters\n- Evening fare subsidy\n"
        ),
    )
    metrics = directive_autorater.compute_deterministic_metrics(item)
    self.assertEqual(metrics["seed_sentence_count"], 2)
    self.assertEqual(metrics["candidate_sentence_count"], 5)
    self.assertEqual(metrics["added_sentences"], 3)
    self.assertTrue(metrics["exceeds_sentence_limit"])
    self.assertEqual(metrics["declared_component_count"], 3)
    self.assertTrue(metrics["exceeds_declared_component_limit"])

  def test_build_prompt_includes_companions_simplifier_and_context(
      self,
  ) -> None:
    item = directive_autorater.DirectiveAuditItem(
        item_id="v01",
        proposal_title="Evening Bus Priority",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_PRE_SIMPLIFIED,
        target_friction="Skepticism about bus lane effectiveness",
        refinement_archetype="Performance Transparency",
        companion_proposals=(
            directive_autorater.CompanionContext(
                proposal_id="shuttle-1",
                title="Neighborhood Shuttle",
                statement="Launch a two-year neighborhood shuttle pilot.",
            ),
        ),
        pre_simplified_statement=_PRE_SIMPLIFIED,
        post_simplified_statement=_POST_SIMPLIFIED_CLEAN,
    )
    prompt = directive_autorater.build_directive_prompt(
        item, additional_context="Focus on transit operational commitments."
    )
    self.assertIn("<additionalContext>", prompt)
    self.assertIn("Focus on transit operational commitments.", prompt)
    self.assertIn("Neighborhood Shuttle", prompt)
    self.assertIn("PRE-SIMPLIFIED STATEMENT:", prompt)
    self.assertIn("POST-SIMPLIFIED STATEMENT:", prompt)
    self.assertIn(_POST_SIMPLIFIED_CLEAN, prompt)

    no_simplifier_item = directive_autorater.DirectiveAuditItem(
        item_id="v02",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_FOCUSED_CANDIDATE,
    )
    prompt_no_pair = directive_autorater.build_directive_prompt(
        no_simplifier_item
    )
    self.assertNotIn("PRE-SIMPLIFIED STATEMENT:", prompt_no_pair)
    self.assertIn("not_applicable", prompt_no_pair)


class DirectiveEvaluationTest(unittest.TestCase):
  """Tests pass/fail verdicts, drift audits, and error recovery."""

  def test_passing_verdict_normalizes_absent_simplifier_drift(self) -> None:
    model = fake_base_model.FakeBaseModel([{
        "text": _make_rubric_json(simplifier_drift="wording_only"),
        "error": None,
    }])
    item = directive_autorater.DirectiveAuditItem(
        item_id="pass-1",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_FOCUSED_CANDIDATE,
    )
    verdict = asyncio.run(
        directive_autorater.evaluate_directive_item(
            model, item, retry_delay_seconds=0.0
        )
    )
    self.assertTrue(verdict.passed)
    self.assertEqual(verdict.failure_reasons, ())
    self.assertTrue(verdict.preserves_core_purpose)
    self.assertTrue(verdict.is_atomic)
    self.assertEqual(verdict.n_primary_new_components, 1)
    self.assertEqual(verdict.added_sentences, 1)
    self.assertFalse(verdict.exceeds_sentence_limit)
    self.assertEqual(verdict.simplifier_drift, "not_applicable")
    self.assertEqual(verdict.drift_note, "")
    self.assertEqual(verdict.attempts, 1)
    self.assertIsNone(verdict.error)
    self.assertEqual(
        model.calls[0]["response_schema"],
        directive_autorater.DIRECTIVE_RESPONSE_SCHEMA,
    )

  def test_failing_verdicts_cover_purpose_stacking_and_sentence_limit(
      self,
  ) -> None:
    model = fake_base_model.FakeBaseModel([{
        "text": _make_rubric_json(
            n_primary_new_components=3,
            preserves_core_purpose=False,
            is_atomic=True,
            has_unrelated_addition=True,
            companion_substitution=True,
            self_contained=False,
        ),
        "error": None,
    }])
    item = directive_autorater.DirectiveAuditItem(
        item_id="fail-multi",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_STACKED_SENTENCES_CANDIDATE,
        declared_new_components="Reports\nShelters\nFare subsidy",
    )
    verdict = asyncio.run(
        directive_autorater.evaluate_directive_item(
            model, item, retry_delay_seconds=0.0
        )
    )
    self.assertFalse(verdict.passed)
    self.assertFalse(verdict.is_atomic)
    self.assertTrue(verdict.exceeds_sentence_limit)
    self.assertTrue(verdict.exceeds_declared_component_limit)
    self.assertCountEqual(
        verdict.failure_reasons,
        [
            "does_not_preserve_core_purpose",
            "companion_substitution",
            "not_self_contained",
            "has_unrelated_addition",
            "too_many_new_components (3 > 2)",
            "exceeds_sentence_limit (3 > 2)",
            "exceeds_declared_component_limit (3 > 2)",
        ],
    )

  def test_simplifier_drift_passes_wording_only_and_fails_substantive(
      self,
  ) -> None:
    model = fake_base_model.FakeBaseModel([
        {
            "text": _make_rubric_json(
                simplifier_drift="wording_only",
                drift_note="Replaced jargon with plain phrasing only.",
            ),
            "error": None,
        },
        {
            "text": _make_rubric_json(
                simplifier_drift="substantive_change",
                drift_note=(
                    "Dropped street maintenance budget and added a new toll."
                ),
            ),
            "error": None,
        },
    ])
    clean_item = directive_autorater.DirectiveAuditItem(
        item_id="simp-clean",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_PRE_SIMPLIFIED,
        pre_simplified_statement=_PRE_SIMPLIFIED,
        post_simplified_statement=_POST_SIMPLIFIED_CLEAN,
    )
    drifted_item = directive_autorater.DirectiveAuditItem(
        item_id="simp-drift",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_PRE_SIMPLIFIED,
        pre_simplified_statement=_PRE_SIMPLIFIED,
        post_simplified_statement=_POST_SIMPLIFIED_DRIFTED,
    )

    clean_verdict = asyncio.run(
        directive_autorater.evaluate_directive_item(
            model, clean_item, retry_delay_seconds=0.0
        )
    )
    drifted_verdict = asyncio.run(
        directive_autorater.evaluate_directive_item(
            model, drifted_item, retry_delay_seconds=0.0
        )
    )
    self.assertTrue(clean_verdict.passed)
    self.assertEqual(clean_verdict.simplifier_drift, "wording_only")
    self.assertFalse(drifted_verdict.passed)
    self.assertEqual(drifted_verdict.simplifier_drift, "substantive_change")
    self.assertIn(
        "substantive_simplifier_drift", drifted_verdict.failure_reasons
    )
    self.assertIn("toll", drifted_verdict.drift_note)

  def test_malformed_json_and_model_errors_retry_and_recover(self) -> None:
    fenced_valid = f"```json\n{_make_rubric_json()}\n```"
    model = fake_base_model.FakeBaseModel([
        {"text": "{not valid json", "error": None},
        {"text": "", "error": "503 transient overload"},
        {"text": fenced_valid, "error": None},
    ])
    item = directive_autorater.DirectiveAuditItem(
        item_id="recover-1",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_FOCUSED_CANDIDATE,
    )
    verdict = asyncio.run(
        directive_autorater.evaluate_directive_item(
            model, item, max_attempts=3, retry_delay_seconds=0.0
        )
    )
    self.assertTrue(verdict.passed)
    self.assertEqual(verdict.attempts, 3)
    self.assertIsNone(verdict.error)

  def test_exhausted_errors_return_error_verdict(self) -> None:
    model = fake_base_model.FakeBaseModel([
        RuntimeError("socket timeout"),
        {
            "text": json.dumps({"n_primary_new_components": "bad"}),
            "error": None,
        },
    ])
    item = directive_autorater.DirectiveAuditItem(
        item_id="err-1",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_FOCUSED_CANDIDATE,
    )
    verdict = asyncio.run(
        directive_autorater.evaluate_directive_item(
            model, item, max_attempts=2, retry_delay_seconds=0.0
        )
    )
    self.assertFalse(verdict.passed)
    self.assertEqual(verdict.attempts, 2)
    self.assertIsNotNone(verdict.error)
    self.assertTrue(verdict.failure_reasons[0].startswith("autorater_error:"))

  def test_validation_rejects_empty_statements_and_bad_limits(self) -> None:
    model = fake_base_model.FakeBaseModel([])
    bad_item = directive_autorater.DirectiveAuditItem(
        item_id="empty", seed_proposal="   ", candidate_proposal="Valid."
    )
    with self.assertRaises(ValueError):
      asyncio.run(directive_autorater.evaluate_directive_item(model, bad_item))
    with self.assertRaises(ValueError):
      asyncio.run(
          directive_autorater.evaluate_directive_batch(
              model, [], max_concurrent_calls=0
          )
      )


class BatchAndRunnerTest(unittest.TestCase):
  """Tests bounded batch execution, refiner JSON loading, and file runner."""

  def test_batch_enforces_semaphore_cap_and_preserves_order(self) -> None:
    items = [
        directive_autorater.DirectiveAuditItem(
            item_id=f"item-{idx}",
            seed_proposal=_SEED_STATEMENT,
            candidate_proposal=_FOCUSED_CANDIDATE,
        )
        for idx in range(5)
    ]
    responses = [
        {
            "text": _make_rubric_json(
                is_atomic=(idx % 2 == 0),
                n_primary_new_components=1 if idx % 2 == 0 else 3,
            ),
            "error": None,
        }
        for idx in range(5)
    ]
    model = _DelayedFakeModel(responses, delay_seconds=0.02)
    verdicts = asyncio.run(
        directive_autorater.evaluate_directive_batch(
            model,
            items,
            max_concurrent_calls=2,
            retry_delay_seconds=0.0,
        )
    )
    self.assertEqual(model.peak_concurrency, 2)
    self.assertEqual([v.item_id for v in verdicts], [i.item_id for i in items])
    summary = directive_autorater.summarize_verdicts(verdicts)
    self.assertEqual(summary["total"], 5)
    self.assertEqual(summary["passed"], 3)
    self.assertEqual(summary["failed"], 2)
    self.assertEqual(summary["non_atomic"], 2)

  def test_runner_loads_refiner_json_and_csv_and_writes_outputs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      refiner_json = root / "refined_variations.json"
      refiner_json.write_text(
          json.dumps([{
              "proposal_id": "bus-priority",
              "proposal_title": "Evening Bus Priority",
              "original_statement": _SEED_STATEMENT,
              "variations": [{
                  "variation_id": "v01",
                  "description": _POST_SIMPLIFIED_CLEAN,
                  "pre_simplified_description": _PRE_SIMPLIFIED,
                  "target_friction_resolved": "Route transparency",
                  "refinement_archetype": "Public Reporting",
              }],
          }])
      )
      out_json = root / "verdicts.json"
      fake_model = fake_base_model.FakeBaseModel([{
          "text": _make_rubric_json(simplifier_drift="wording_only"),
          "error": None,
      }])
      args = directive_autorater.get_args([
          "--input_path",
          str(refiner_json),
          "--output_path",
          str(out_json),
          "--additional_context",
          "Transit context.",
          "--max_concurrent_calls",
          "2",
      ])
      config = directive_autorater.config_from_args(args)
      verdicts = asyncio.run(
          directive_autorater.run_directive_audit(config, model=fake_model)
      )

      self.assertEqual(len(verdicts), 1)
      self.assertEqual(verdicts[0].item_id, "bus-priority_v01")
      self.assertTrue(verdicts[0].passed)
      self.assertIn("Transit context.", fake_model.calls[0]["prompt"])
      written = json.loads(out_json.read_text())
      self.assertEqual(written["summary"]["passed"], 1)
      self.assertEqual(written["verdicts"][0]["item_id"], "bus-priority_v01")

      # Verify CSV round-trip loading and writing with an explicit config.
      in_csv = root / "audit_items.csv"
      pd.DataFrame([{
          "item_id": "csv-1",
          "seed_proposal": _SEED_STATEMENT,
          "candidate_proposal": _FOCUSED_CANDIDATE,
      }]).to_csv(in_csv, index=False)
      out_csv = root / "verdicts.csv"
      csv_model = fake_base_model.FakeBaseModel(
          [{"text": _make_rubric_json(), "error": None}]
      )
      csv_config = directive_autorater.DirectiveAuditConfig(
          input_path=in_csv,
          output_path=out_csv,
          retry_delay_seconds=0.0,
      )
      asyncio.run(
          directive_autorater.run_directive_audit(csv_config, model=csv_model)
      )
      out_df = pd.read_csv(out_csv)
      self.assertEqual(out_df.shape[0], 1)
      self.assertEqual(out_df.loc[0, "item_id"], "csv-1")
      self.assertTrue(bool(out_df.loc[0, "passed"]))


if __name__ == "__main__":
  unittest.main()
