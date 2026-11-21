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

"""Unit tests for agentic proposal refinement runner."""

import asyncio
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import enum
import importlib
import json
import pathlib
import tempfile
from typing import Any
import unittest
from unittest import mock

import pandas as pd

from src.models import base_model
from src.proposition_refinement.agentic import candidate_drafts
from src.proposition_refinement.agentic import refiner
from src.propositions import proposal_simplifier

_FOCAL_PROPOSAL = refiner.ProposalSpec(
    proposal_id="bus-priority",
    title="Evening Bus Priority",
    statement=(
        "Add dedicated bus lanes along main evening commuter corridors. Fund"
        " signage and paint from the existing street maintenance budget."
    ),
    topic="Evening Transit Reliability",
    companion_proposals=(
        refiner.CompanionProposal(
            proposal_id="shuttle-pilot",
            title="Neighborhood Shuttle Pilot",
            statement="Launch a two-year neighborhood shuttle pilot.",
        ),
    ),
)

_SIMPLE_VARIATION_STMT = (
    "Add dedicated bus lanes along main evening commuter corridors. Fund"
    " signage and paint from the existing street maintenance budget. Publish"
    " quarterly bus speed reports online."
)

_COMPLEX_VARIATION_STMT = (
    "Institutionalize dedicated transit prioritization corridors along primary"
    " metropolitan commuter thoroughfares utilizing reallocated infrastructural"
    " maintenance appropriations. Operationalize comprehensive quarterly"
    " performance accountability disclosures evaluating schedule adherence."
)

_SIMPLIFIED_VARIATION_STMT = (
    "Add dedicated bus lanes along main commuter roads using existing road"
    " repair funds. Publish clear quarterly bus speed reports online."
)


class _FakeSimplifierModel(base_model.BaseModel):
  """Deterministic fake BaseModel for testing optional simplification."""

  def __init__(self, responses: Sequence[dict[str, Any]]):
    super().__init__()
    self._responses = list(responses)
    self.calls: list[dict[str, Any]] = []

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
    self.calls.append({"prompt": prompt, "run_name": run_name})
    if not self._responses:
      raise AssertionError("FakeSimplifierModel exhausted scripted responses.")
    return self._responses.pop(0)

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: Callable[[str, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs: Any,
  ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    raise NotImplementedError("Not used by refiner tests.")


class _StubBuiltinTools(enum.Enum):
  """Minimal stand-in for Antigravity BuiltinTools."""

  FINISH = "finish"
  VIEW_FILE = "view_file"


@dataclasses.dataclass
class _StubCapabilitiesConfig:
  enabled_tools: list[Any]
  enable_subagents: bool = True


@dataclasses.dataclass
class _StubCustomSystemInstructions:
  text: str


@dataclasses.dataclass
class _StubLocalAgentConfig:
  model: str
  capabilities: _StubCapabilitiesConfig
  tools: list[Callable[..., Any]]
  response_schema: dict[str, Any]
  system_instructions: _StubCustomSystemInstructions
  save_dir: str
  api_key: str | None = None


class _StubAntigravityModule:
  """Stand-in namespace mimicking the subset of google.antigravity used."""

  BuiltinTools = _StubBuiltinTools
  CapabilitiesConfig = _StubCapabilitiesConfig
  CustomSystemInstructions = _StubCustomSystemInstructions
  LocalAgentConfig = _StubLocalAgentConfig


def _make_variation(
    variation_id: str,
    candidate_id: str,
    description: str,
    *,
    title: str = "Focused variation",
) -> dict[str, str]:
  """Creates a schema-valid variation dictionary for tests."""
  return {
      "variation_id": variation_id,
      "candidate_id": candidate_id,
      "title": title,
      "description": description,
      "target_friction_resolved": (
          "Evening commuters face unpredictable delays."
      ),
      "refinement_archetype": "Public Performance Reporting",
      "feasibility_assessment": (
          "Checked reference corpus; reporting fits existing staff capacity."
      ),
  }


def _verify_in_slate(
    slate: candidate_drafts.CandidateDraftSlate,
    candidate_id: str,
    statement: str,
    *,
    title: str = "Focused variation",
    new_components: str = "Quarterly public speed report",
) -> str:
  """Writes and verifies a candidate draft in `slate`."""
  return slate.write(
      candidate_id=candidate_id,
      title=title,
      statement=statement,
      target_friction_resolved="Evening commuters face unpredictable delays.",
      feasibility_assessment=(
          "Checked reference corpus; reporting fits existing staff capacity."
      ),
      status="verified",
      new_components=new_components,
      seed_purpose_preserved=(
          "Yes: keeps dedicated evening bus lanes funded from street"
          " maintenance."
      ),
  )


class PromptAndSchemaTest(unittest.TestCase):
  """Tests instruction-sandwich prompting, feedback formatting, and schemas."""

  def test_build_refinement_prompt_sandwich_and_additional_context(self):
    feedback = refiner.format_feedback_context(
        _FOCAL_PROPOSAL,
        [
            refiner.FeedbackStatement(
                speaker_id="p01",
                text="Buses are stuck behind delivery trucks at 6 PM.",
                group_id="01",
            ),
            refiner.FeedbackStatement(
                speaker_id="p02",
                text="Show data proving lanes actually speed up commutes.",
                group_id="02",
            ),
        ],
    )
    self.assertIn("2 independent discussion groups", feedback)
    self.assertIn("Neighborhood Shuttle Pilot", feedback)
    self.assertIn("Group 01", feedback)

    without_ctx = refiner.build_refinement_prompt(
        _FOCAL_PROPOSAL,
        feedback,
        min_variations=2,
        max_variations=5,
        additional_context=None,
        has_corpus=False,
    )
    self.assertNotIn("<additionalContext>", without_ctx)
    self.assertNotIn("list_corpus_files", without_ctx)
    # Task block appears both before and after the feedback dossier.
    first_task = without_ctx.find("### REFINEMENT GENERATION TASK:")
    feedback_pos = without_ctx.find("Buses are stuck behind delivery trucks")
    second_task = without_ctx.find(
        "### REFINEMENT GENERATION TASK:", feedback_pos
    )
    workflow_pos = without_ctx.find("### WORKFLOW:")
    self.assertGreaterEqual(first_task, 0)
    self.assertGreater(feedback_pos, first_task)
    self.assertGreater(second_task, feedback_pos)
    self.assertGreater(workflow_pos, second_task)

    with_ctx = refiner.build_refinement_prompt(
        _FOCAL_PROPOSAL,
        feedback,
        additional_context="Prioritize accessibility for night-shift workers.",
        has_corpus=True,
    )
    self.assertIn("<additionalContext>", with_ctx)
    self.assertIn("night-shift workers", with_ctx)
    self.assertIn("list_corpus_files", with_ctx)

  def test_flat_response_schema_and_post_validation(self):
    schema = refiner.build_response_schema(min_variations=2, max_variations=4)
    schema_text = json.dumps(schema)
    self.assertNotIn("$defs", schema_text)
    self.assertNotIn("$ref", schema_text)
    self.assertEqual(schema["properties"]["variations"]["minItems"], 2)
    self.assertEqual(schema["properties"]["variations"]["maxItems"], 4)

    valid_payload = {
        "proposal_id": _FOCAL_PROPOSAL.proposal_id,
        "variations": [
            _make_variation("v01", "c01", _SIMPLE_VARIATION_STMT),
            _make_variation("v02", "c02", _SIMPLE_VARIATION_STMT),
        ],
    }
    valid_items, errors = refiner.validate_variation_payload(
        valid_payload, min_variations=2, max_variations=4
    )
    self.assertEqual(len(valid_items), 2)
    self.assertEqual(errors, [])

    # Cardinality below min_variations and missing required fields are caught.
    bad_payload = {
        "proposal_id": _FOCAL_PROPOSAL.proposal_id,
        "variations": [
            _make_variation("v01", "c01", _SIMPLE_VARIATION_STMT),
            {"variation_id": "v02", "title": "Incomplete"},
        ],
    }
    valid_items, errors = refiner.validate_variation_payload(
        bad_payload, min_variations=2, max_variations=4
    )
    self.assertEqual(len(valid_items), 1)
    self.assertTrue(any("missing non-empty string" in e for e in errors))
    self.assertTrue(any("Expected at least 2" in e for e in errors))

    # Fenced JSON text extraction works when structured output is None.
    fenced = f"```json\n{json.dumps(valid_payload)}\n```"
    extracted, source = refiner.extract_payload(None, fenced)
    self.assertEqual(source, "text")
    self.assertEqual(extracted["proposal_id"], _FOCAL_PROPOSAL.proposal_id)


class AntigravityConfigAndImportGuardTest(unittest.TestCase):
  """Tests lazy SDK loading, helpful ImportError, and CapabilitiesConfig."""

  def test_missing_antigravity_raises_helpful_import_error(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.object(
          importlib,
          "import_module",
          side_effect=ModuleNotFoundError(
              "No module named 'google.antigravity'"
          ),
      ):
        with self.assertRaises(ImportError) as ctx:
          asyncio.run(
              refiner.refine_proposal(
                  _FOCAL_PROPOSAL,
                  "Feedback text.",
                  save_dir=pathlib.Path(tmpdir) / "sessions",
              )
          )
        self.assertIn("google-antigravity", str(ctx.exception))
        self.assertIn("session_executor", str(ctx.exception))

  def test_build_agent_config_enforces_finish_capability_and_save_dir(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      save_path = pathlib.Path(tmpdir) / "persisted_session"
      slate = candidate_drafts.CandidateDraftSlate(
          seed_statement=_FOCAL_PROPOSAL.statement
      )
      request = refiner.SessionRequest(
          proposal=_FOCAL_PROPOSAL,
          prompt="Refine this proposal.",
          system_instructions=refiner.AGENTIC_SYSTEM_INSTRUCTIONS,
          model_name="gemini-3.8-flash",
          min_variations=2,
          max_variations=5,
          response_schema=refiner.build_response_schema(2, 5),
          save_dir=save_path,
          tools=tuple(candidate_drafts.make_draft_slate_tools(slate)),
          slate=slate,
          label="session",
          attempt=1,
          api_key="test-key",
      )
      with mock.patch.object(
          importlib, "import_module", return_value=_StubAntigravityModule
      ):
        config = refiner.build_agent_config(request)

      self.assertEqual(
          config.capabilities.enabled_tools, [_StubBuiltinTools.FINISH]
      )
      self.assertFalse(config.capabilities.enable_subagents)
      self.assertEqual(config.save_dir, str(save_path))
      self.assertEqual(config.api_key, "test-key")
      self.assertEqual(len(config.tools), 2)

      bad_request = dataclasses.replace(request, save_dir=pathlib.Path(""))
      with self.assertRaises(ValueError):
        refiner.build_agent_config(bad_request)


class RefinerExecutionBehaviorTest(unittest.TestCase):
  """Tests retries, timeouts, slate gating, decomposition, and simplifier."""

  def test_outer_fresh_attempt_retry_recovers_from_503_400_cascade(self):
    seen_dirs: list[pathlib.Path] = []
    seen_slates: list[candidate_drafts.CandidateDraftSlate] = []

    async def fake_executor(
        request: refiner.SessionRequest,
    ) -> refiner.SessionResult:
      seen_dirs.append(request.save_dir)
      seen_slates.append(request.slate)
      if request.attempt == 1:
        raise RuntimeError(
            "503: This model is currently experiencing high demand"
            " (PREFILL_QUEUE_PREEMPTED)"
        )
      if request.attempt == 2:
        raise RuntimeError(
            "400: Requests ending with a model turn are not supported"
        )
      _verify_in_slate(request.slate, "c01", _SIMPLE_VARIATION_STMT)
      return refiner.SessionResult(
          structured_output={
              "proposal_id": request.proposal.proposal_id,
              "variations": [
                  _make_variation("v01", "c01", _SIMPLE_VARIATION_STMT)
              ],
          }
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      save_dir = pathlib.Path(tmpdir) / "sessions"
      out_dir = pathlib.Path(tmpdir) / "output"
      result = asyncio.run(
          refiner.refine_proposal(
              _FOCAL_PROPOSAL,
              "Riders want quarterly speed reporting.",
              save_dir=save_dir,
              output_dir=out_dir,
              min_variations=1,
              max_variations=3,
              max_attempts=4,
              retry_delay_seconds=0.0,
              session_executor=fake_executor,
          )
      )

      self.assertEqual(result["attempts"], 3)
      self.assertEqual(result["num_variations"], 1)
      # Each outer attempt must receive its own persistent dir and fresh slate.
      self.assertEqual(len(set(seen_dirs)), 3)
      self.assertTrue(all(d.is_dir() for d in seen_dirs))
      self.assertEqual(len({id(s) for s in seen_slates}), 3)
      self.assertTrue(
          (out_dir / f"{_FOCAL_PROPOSAL.proposal_id}_variations.json").is_file()
      )

  def test_timeout_triggers_retry_and_raises_when_exhausted(self):
    attempts_seen: list[int] = []

    async def slow_executor(
        request: refiner.SessionRequest,
    ) -> refiner.SessionResult:
      attempts_seen.append(request.attempt)
      await asyncio.sleep(0.2)
      return refiner.SessionResult()

    with tempfile.TemporaryDirectory() as tmpdir:
      with self.assertRaises(TimeoutError):
        asyncio.run(
            refiner.refine_proposal(
                _FOCAL_PROPOSAL,
                "Feedback.",
                save_dir=pathlib.Path(tmpdir) / "sessions",
                min_variations=1,
                max_attempts=2,
                timeout_seconds=0.03,
                retry_delay_seconds=0.0,
                session_executor=slow_executor,
            )
        )
    self.assertEqual(attempts_seen, [1, 2])

  def test_slate_gate_and_second_pass_decomposition_split_stacked_drafts(self):
    labels_seen: list[str] = []
    decomp_prompts: list[str] = []
    split_stmt_a = (
        f"{_FOCAL_PROPOSAL.statement} Install curbside cameras to enforce bus"
        " lanes."
    )
    split_stmt_b = (
        f"{_FOCAL_PROPOSAL.statement} Add real-time arrival signs at major"
        " stops."
    )

    async def fake_executor(
        request: refiner.SessionRequest,
    ) -> refiner.SessionResult:
      labels_seen.append(request.label)
      if request.label == "session":
        # One clean verified draft, one unverified draft, one stacked draft.
        _verify_in_slate(request.slate, "c01", _SIMPLE_VARIATION_STMT)
        request.slate.write(
            candidate_id="c02",
            title="Unverified idea",
            statement=f"{_FOCAL_PROPOSAL.statement} Unverified extra sentence.",
            target_friction_resolved="Draft only.",
            status="draft",
        )
        request.slate.write(
            candidate_id="c03",
            title="Kitchen sink transit package",
            statement=(
                f"{_FOCAL_PROPOSAL.statement} Install curbside cameras. Add"
                " real-time arrival signs. Create a new transit tax district."
            ),
            target_friction_resolved="Enforcement and rider information.",
            feasibility_assessment="Checked transit brief.",
            status="verified",
            new_components="Curbside cameras\nArrival signs\nNew tax district",
            seed_purpose_preserved="Yes: keeps evening bus lanes.",
        )
        return refiner.SessionResult(
            structured_output={
                "proposal_id": request.proposal.proposal_id,
                "variations": [
                    _make_variation("v01", "c01", _SIMPLE_VARIATION_STMT),
                    # Unverified variation should be dropped by slate gate.
                    _make_variation(
                        "v02",
                        "c02",
                        f"{_FOCAL_PROPOSAL.statement} Unverified extra"
                        " sentence.",
                    ),
                ],
            }
        )

      decomp_prompts.append(request.prompt)
      _verify_in_slate(
          request.slate,
          "c03-a",
          split_stmt_a,
          title="Curbside camera enforcement",
          new_components="Curbside enforcement cameras",
      )
      _verify_in_slate(
          request.slate,
          "c03-b",
          split_stmt_b,
          title="Real-time arrival signs",
          new_components="Real-time arrival signs",
      )
      return refiner.SessionResult(
          structured_output={
              "proposal_id": request.proposal.proposal_id,
              "variations": [
                  _make_variation("v01", "c03-a", split_stmt_a),
                  _make_variation("v02", "c03-b", split_stmt_b),
              ],
          }
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      result = asyncio.run(
          refiner.refine_proposal(
              _FOCAL_PROPOSAL,
              "Participants asked for enforcement and arrival displays.",
              save_dir=pathlib.Path(tmpdir) / "sessions",
              min_variations=2,
              max_variations=3,
              max_attempts=2,
              retry_delay_seconds=0.0,
              require_verified_drafts=True,
              decompose_stacked=True,
              session_executor=fake_executor,
          )
      )

    self.assertEqual(labels_seen, ["session", "decompose"])
    self.assertEqual(len(decomp_prompts), 1)
    self.assertIn("Kitchen sink transit package", decomp_prompts[0])
    self.assertEqual(result["num_variations"], 3)
    self.assertEqual(result["grounding_audit"]["n_unverified"], 1)
    self.assertEqual(result["grounding_audit"]["decomposition"]["n_added"], 2)
    decomposed = [v for v in result["variations"] if v.get("decomposed")]
    self.assertCountEqual(
        [v["description"] for v in decomposed], [split_stmt_a, split_stmt_b]
    )

  def test_optional_readability_simplification_integrates_with_simplifier(self):
    async def fake_executor(
        request: refiner.SessionRequest,
    ) -> refiner.SessionResult:
      _verify_in_slate(request.slate, "c01", _SIMPLE_VARIATION_STMT)
      _verify_in_slate(request.slate, "c02", _COMPLEX_VARIATION_STMT)
      return refiner.SessionResult(
          structured_output={
              "proposal_id": request.proposal.proposal_id,
              "variations": [
                  _make_variation("v01", "c01", _SIMPLE_VARIATION_STMT),
                  _make_variation("v02", "c02", _COMPLEX_VARIATION_STMT),
              ],
          }
      )

    simplifier = _FakeSimplifierModel([{"text": _SIMPLIFIED_VARIATION_STMT}])
    with tempfile.TemporaryDirectory() as tmpdir:
      result = asyncio.run(
          refiner.refine_proposal(
              _FOCAL_PROPOSAL,
              "Make wording plain and clear.",
              save_dir=pathlib.Path(tmpdir) / "sessions",
              min_variations=2,
              max_variations=4,
              max_attempts=1,
              retry_delay_seconds=0.0,
              simplifier_model=simplifier,
              session_executor=fake_executor,
          )
      )

    # Already-compliant statement passes through with 0 simplifier calls.
    self.assertEqual(len(simplifier.calls), 1)
    by_id = {v["variation_id"]: v for v in result["variations"]}
    self.assertEqual(by_id["v01"]["simplification_attempts"], 0)
    self.assertEqual(by_id["v01"]["description"], _SIMPLE_VARIATION_STMT)
    self.assertEqual(by_id["v02"]["simplification_attempts"], 1)
    self.assertEqual(
        by_id["v02"]["pre_simplified_description"], _COMPLEX_VARIATION_STMT
    )
    self.assertEqual(by_id["v02"]["description"], _SIMPLIFIED_VARIATION_STMT)
    self.assertTrue(by_id["v02"]["readability_compliant"])
    self.assertLessEqual(
        by_id["v02"]["post_fkgl"], proposal_simplifier.DEFAULT_MAX_FKGL
    )


class CliAndBatchLoaderTest(unittest.TestCase):
  """Tests JSON/CSV proposal loading, corpus tools wiring, and batch CLI."""

  def test_run_cli_loads_csv_feedback_corpus_and_writes_combined_output(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      root = pathlib.Path(tmpdir)
      csv_path = root / "proposals.csv"
      pd.DataFrame([
          {
              "proposal_id": "p1",
              "title": "Bus Lanes",
              "statement": _FOCAL_PROPOSAL.statement,
          },
          {
              "proposal_id": "p2",
              "title": "Tree Canopy",
              "statement": (
                  "Plant shade trees along major sidewalks. Prioritize blocks"
                  " with the highest summer temperatures."
              ),
          },
      ]).to_csv(csv_path, index=False)

      feedback_dir = root / "feedback"
      feedback_dir.mkdir()
      (feedback_dir / "p1.md").write_text("Commuters want speed reports.\n")
      (feedback_dir / "p2.md").write_text("Residents want tree watering.\n")

      corpus_dir = root / "corpus"
      corpus_dir.mkdir()
      (corpus_dir / "budget.md").write_text(
          "Street maintenance funds cover lane paint and signage.\n"
      )

      out_dir = root / "output"
      save_dir = root / "sessions"
      tool_names_seen: list[list[str]] = []

      async def fake_executor(
          request: refiner.SessionRequest,
      ) -> refiner.SessionResult:
        tool_names_seen.append([fn.__name__ for fn in request.tools])
        # Exercise an attached corpus tool to confirm access-log recording.
        search_fn = next(
            fn for fn in request.tools if fn.__name__ == "search_corpus"
        )
        search_fn("Street maintenance")
        stmt = f"{request.proposal.statement} Publish an annual status report."
        _verify_in_slate(request.slate, "c01", stmt)
        return refiner.SessionResult(
            structured_output={
                "proposal_id": request.proposal.proposal_id,
                "variations": [_make_variation("v01", "c01", stmt)],
            }
        )

      args = refiner.get_args([
          "--input_proposals_json",
          str(csv_path),
          "--feedback_dir",
          str(feedback_dir),
          "--corpus_dir",
          str(corpus_dir),
          "--output_dir",
          str(out_dir),
          "--save_dir",
          str(save_dir),
          "--min_variations",
          "1",
          "--max_variations",
          "2",
          "--max_concurrent_sessions",
          "2",
          "--max_attempts",
          "1",
          "--additional_context",
          "Keep examples neutral and accessible.",
      ])
      results = asyncio.run(
          refiner.run_cli(args, session_executor=fake_executor)
      )

      self.assertEqual([r["proposal_id"] for r in results], ["p1", "p2"])
      for names in tool_names_seen:
        self.assertCountEqual(
            names,
            [
                "list_corpus_files",
                "search_corpus",
                "read_corpus_file",
                "write_candidate_draft",
                "view_candidate_drafts",
            ],
        )
      # Access log recorded budget.md via corpus tool during session.
      self.assertEqual(
          results[0]["grounding_audit"]["n_accessed_corpus_files"], 1
      )
      combined = json.loads((out_dir / "refined_variations.json").read_text())
      self.assertEqual(len(combined), 2)

  def test_load_proposals_supports_json_list_and_object_wrapper(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      json_path = pathlib.Path(tmpdir) / "proposals.json"
      json_path.write_text(
          json.dumps({
              "proposals": [{
                  "slug": "bike-lanes",
                  "title": "Protected Bike Lanes",
                  "text": "Build protected bike lanes on main streets.",
                  "companion_proposals": [{
                      "slug": "sidewalks",
                      "title": "Sidewalk Repair",
                      "text": "Repair broken neighborhood sidewalks.",
                  }],
              }]
          })
      )
      loaded = refiner.load_proposals(json_path)
      self.assertEqual(len(loaded), 1)
      self.assertEqual(loaded[0].proposal_id, "bike-lanes")
      self.assertEqual(len(loaded[0].companion_proposals), 1)

  def test_hard_filter_over_sentence_drop_triggers_decomposition(self):
    over_sentence = (
        f"{_FOCAL_PROPOSAL.statement} Extra one. Extra two. Extra three."
    )
    split_stmt = f"{_FOCAL_PROPOSAL.statement} Add signal priority at night."
    labels_seen: list[str] = []

    async def fake_executor(
        request: refiner.SessionRequest,
    ) -> refiner.SessionResult:
      labels_seen.append(request.label)
      if request.label == "session":
        return refiner.SessionResult(
            structured_output={
                "proposal_id": request.proposal.proposal_id,
                "variations": [_make_variation("v01", "c01", over_sentence)],
            }
        )
      _verify_in_slate(request.slate, "c01-a", split_stmt)
      return refiner.SessionResult(
          structured_output={
              "proposal_id": request.proposal.proposal_id,
              "variations": [_make_variation("v01", "c01-a", split_stmt)],
          }
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      result = asyncio.run(
          refiner.refine_proposal(
              _FOCAL_PROPOSAL,
              "Feedback.",
              save_dir=pathlib.Path(tmpdir) / "sessions",
              min_variations=1,
              max_variations=2,
              max_attempts=1,
              retry_delay_seconds=0.0,
              require_verified_drafts=False,
              decompose_stacked=True,
              session_executor=fake_executor,
          )
      )

    self.assertEqual(labels_seen, ["session", "decompose"])
    self.assertEqual(len(result["grounding_audit"]["hard_filter_dropped"]), 1)
    self.assertEqual(result["variations"][0]["description"], split_stmt)
    self.assertTrue(result["variations"][0]["decomposed"])

  def test_refine_proposals_enforces_semaphore_concurrency_cap(self):
    proposals = [
        refiner.ProposalSpec(
            proposal_id=f"p{i}",
            title=f"Proposal {i}",
            statement=f"Improve service route {i}.",
        )
        for i in range(5)
    ]
    active = 0
    peak = 0

    async def fake_executor(
        request: refiner.SessionRequest,
    ) -> refiner.SessionResult:
      nonlocal active, peak
      active += 1
      peak = max(peak, active)
      await asyncio.sleep(0.02)
      active -= 1
      stmt = f"{request.proposal.statement} Publish monthly updates."
      _verify_in_slate(request.slate, "c01", stmt)
      return refiner.SessionResult(
          structured_output={
              "proposal_id": request.proposal.proposal_id,
              "variations": [_make_variation("v01", "c01", stmt)],
          }
      )

    with tempfile.TemporaryDirectory() as tmpdir:
      results = asyncio.run(
          refiner.refine_proposals(
              proposals,
              {},
              save_dir=pathlib.Path(tmpdir) / "sessions",
              min_variations=1,
              max_variations=2,
              max_concurrent_sessions=2,
              max_attempts=1,
              retry_delay_seconds=0.0,
              session_executor=fake_executor,
          )
      )

    self.assertEqual(len(results), 5)
    self.assertLessEqual(peak, 2)
    self.assertEqual(peak, 2)


if __name__ == "__main__":
  unittest.main()
