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

"""Unit tests for the session-DB grounding autorater."""

import asyncio
from collections.abc import Sequence
import json
import pathlib
import sqlite3
import tempfile
from typing import Any
import unittest
from unittest import mock

import pandas as pd

from src.models import fake_base_model
from src.proposition_refinement.agentic import grounding_autorater
from src.proposition_refinement.agentic import session_trace_reader

_SEED_STATEMENT = (
    "Add dedicated bus lanes along main evening commuter corridors. Fund"
    " signage and paint from the existing street maintenance budget."
)
_GROUNDED_CANDIDATE = (
    "Add dedicated bus lanes along Main Ave using $400,000 from the existing"
    " $14.2 million annual street maintenance budget. Publish quarterly bus"
    " speed reports online."
)
_HALLUCINATED_CANDIDATE = (
    "Add dedicated bus lanes along Main Ave funded by a $50 million transit"
    " surplus that cuts evening commute times by 65%."
)


def _encode_varint(value: int) -> bytes:
  """Encodes a non-negative integer as a protobuf base-128 varint."""
  out = bytearray()
  remaining = value
  while True:
    chunk = remaining & 0x7F
    remaining >>= 7
    if remaining:
      out.append(chunk | 0x80)
    else:
      out.append(chunk)
      return bytes(out)


def _make_call_blob(
    call_id: str, tool_name: str, args: dict[str, Any]
) -> bytes:
  """Creates a synthetic step_type=15 or step_type=2 blob."""
  call_bytes = call_id.encode("utf-8")
  tool_bytes = tool_name.encode("utf-8")
  args_bytes = json.dumps(args).encode("utf-8")
  return (
      b"\x08\x0f\n"
      + _encode_varint(len(call_bytes))
      + call_bytes
      + b"\x12"
      + bytes([len(tool_bytes)])
      + tool_bytes
      + b"\x1a"
      + _encode_varint(len(args_bytes))
      + args_bytes
  )


def _make_result_blob(
    call_id: str, tool_name: str, args: dict[str, Any], result_text: str
) -> bytes:
  """Creates a synthetic step_type=103 blob with ToolResponse framing."""
  marker = b"antigravity.localharness.ToolResponse\x12\x10\n"
  response_json = json.dumps({"result": result_text}).encode("utf-8")
  return (
      _make_call_blob(call_id, tool_name, args)
      + b"\xff\xfeencrypted-noise\x00"
      + marker
      + response_json
  )


def _write_synthetic_session_db(
    db_path: pathlib.Path,
    *,
    research_results: Sequence[tuple[str, dict[str, Any], str]],
    finish_variations: Sequence[dict[str, Any]] = (),
) -> pathlib.Path:
  """Writes a synthetic Antigravity SQLite session DB to `db_path`."""
  db_path.parent.mkdir(parents=True, exist_ok=True)
  rows: list[tuple[int, int, bytes]] = []
  idx = 1
  for call_num, (tool_name, args, result_text) in enumerate(
      research_results, start=1
  ):
    call_id = f"call_{100 + call_num}"
    rows.append((
        idx,
        session_trace_reader.STEP_MODEL_TURN,
        _make_call_blob(call_id, tool_name, args),
    ))
    idx += 1
    rows.append((
        idx,
        session_trace_reader.STEP_TOOL_RESULT,
        _make_result_blob(call_id, tool_name, args, result_text),
    ))
    idx += 1
  if finish_variations:
    rows.append((
        idx,
        session_trace_reader.STEP_FINISH,
        _make_call_blob(
            "call_999",
            "finish",
            {
                "proposal_id": "bus-priority",
                "variations": list(finish_variations),
            },
        ),
    ))
  conn = sqlite3.connect(str(db_path))
  try:
    conn.execute(
        "CREATE TABLE steps ("
        "idx INTEGER PRIMARY KEY, "
        "step_type INTEGER NOT NULL, "
        "step_payload BLOB"
        ")"
    )
    conn.executemany(
        "INSERT INTO steps (idx, step_type, step_payload) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
  finally:
    conn.close()
  return db_path


def _make_grounding_json(**overrides: Any) -> str:
  """Builds a valid JSON response matching GROUNDING_RESPONSE_SCHEMA."""
  payload: dict[str, Any] = {
      "claims": [{
          "claim": (
              "Dedicated bus lanes cost $400,000 from a $14.2 million"
              " street maintenance budget."
          ),
          "claim_type": "funding_or_resource_limit",
          "explicit_in_statement": True,
          "inherited_from_seed": False,
          "status": "supported_by_seen_evidence",
          "evidence_snippet": (
              "Annual street maintenance budget is $14.2 million;"
              " markings cost $400,000 per corridor."
          ),
      }],
      "grounding_quality": "grounded",
      "red_flags": [],
      "rationale": "All added budget figures match retrieved corpus excerpts.",
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


class EvidencePacketTest(unittest.TestCase):
  """Tests evidence recovery and priority-aware character budgeting."""

  def test_build_evidence_packet_prioritizes_matching_blocks_on_truncation(
      self,
  ) -> None:
    trace = session_trace_reader.SessionTrace(
        db_path=None,
        total_steps=4,
        events=(
            session_trace_reader.ToolEvent(
                step_idx=1,
                step_type=session_trace_reader.STEP_TOOL_RESULT,
                call_id="call_1",
                tool_name="search_corpus",
                args={"query": "filler"},
                result="[docs/filler.md]\n" + ("unrelated text " * 60),
            ),
            session_trace_reader.ToolEvent(
                step_idx=2,
                step_type=session_trace_reader.STEP_TOOL_RESULT,
                call_id="call_2",
                tool_name="read_corpus_file",
                args={"rel_path": "docs/budget.md"},
                result=(
                    "=== docs/budget.md (Lines 1..2 of 2) ===\n"
                    "1: Budget cap is $14.2 million."
                ),
            ),
        ),
        thought_summaries=(),
        finish_event=None,
        finish_payload=None,
    )
    packet = grounding_autorater.build_evidence_packet(
        trace, priority_terms=("$14.2 million",), char_budget=400
    )
    self.assertTrue(packet.truncated)
    self.assertEqual(packet.n_research_calls, 2)
    self.assertEqual(packet.n_search_calls, 1)
    self.assertEqual(packet.n_read_calls, 1)
    self.assertEqual(
        packet.files_accessed, ("docs/filler.md", "docs/budget.md")
    )
    self.assertIn("Budget cap is $14.2 million.", packet.text)
    self.assertIn("[...truncated...]", packet.text)


class GroundingEvaluationTest(unittest.TestCase):
  """Tests grounding audit verdicts against synthetic SQLite session traces."""

  def test_grounded_proposal_passes_using_synthetic_sqlite_db(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      db_path = _write_synthetic_session_db(
          pathlib.Path(tmp) / "session.db",
          research_results=[
              (
                  "search_corpus",
                  {"query": "street maintenance budget"},
                  (
                      "[briefs/transit_budget.md]\n"
                      "  L10: Annual street maintenance budget is $14.2"
                      " million; markings cost $400,000 per corridor."
                  ),
              ),
          ],
          finish_variations=[{"description": _GROUNDED_CANDIDATE}],
      )
      model = fake_base_model.FakeBaseModel(
          [{"text": _make_grounding_json(), "error": None}]
      )
      item = grounding_autorater.GroundingAuditItem(
          item_id="bus-v01",
          proposal_title="Evening Bus Priority",
          seed_proposal=_SEED_STATEMENT,
          candidate_proposal=_GROUNDED_CANDIDATE,
          feasibility_assessment=(
              "Verified $14.2M maintenance budget in briefs/transit_budget.md."
          ),
          session_db_path=db_path,
      )
      verdict = asyncio.run(
          grounding_autorater.evaluate_grounding_item(
              model,
              item,
              additional_context="Verify exact funding figures.",
              retry_delay_seconds=0.0,
          )
      )

    self.assertTrue(verdict.passed)
    self.assertEqual(verdict.grounding_quality, "grounded")
    self.assertEqual(verdict.failure_reasons, ())
    self.assertEqual(verdict.n_supported_claims, 1)
    self.assertEqual(verdict.n_contradicted_claims, 0)
    self.assertEqual(verdict.n_ungrounded_added_claims, 0)
    self.assertEqual(verdict.n_research_calls, 1)
    self.assertEqual(verdict.files_accessed, ("briefs/transit_budget.md",))
    prompt_sent = model.calls[0]["prompt"]
    self.assertIn("<additionalContext>", prompt_sent)
    self.assertIn("Unverified Claim, NOT Evidence", prompt_sent)
    self.assertIn("$14.2 million", prompt_sent)

  def test_flags_ungrounded_and_contradicted_claims(self) -> None:
    response_json = _make_grounding_json(
        claims=[
            {
                "claim": "Funded by a $50 million transit surplus.",
                "claim_type": "funding_or_resource_limit",
                "explicit_in_statement": True,
                "inherited_from_seed": False,
                "status": "contradicted_by_seen_evidence",
                "evidence_snippet": "Transit reserve balance is $3.1 million.",
            },
            {
                "claim": "Cuts evening commute times by 65%.",
                "claim_type": "operational_or_impact_claim",
                "explicit_in_statement": True,
                "inherited_from_seed": False,
                "status": "not_supported_in_session",
                "evidence_snippet": "",
            },
        ],
        grounding_quality="ungrounded_or_contradicted",
        red_flags=["Claimed $50M surplus exceeds $3.1M reserve balance"],
        rationale=(
            "Surplus claim contradicts retrieved reserve figures and 65%"
            " reduction is unsupported."
        ),
    )
    model = fake_base_model.FakeBaseModel(
        [{"text": response_json, "error": None}]
    )
    empty_trace = session_trace_reader.SessionTrace(
        db_path=None,
        total_steps=0,
        events=(),
        thought_summaries=(),
        finish_event=None,
        finish_payload=None,
    )
    item = grounding_autorater.GroundingAuditItem(
        item_id="bus-hallucinated",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_HALLUCINATED_CANDIDATE,
        session_trace=empty_trace,
    )
    verdict = asyncio.run(
        grounding_autorater.evaluate_grounding_item(
            model, item, retry_delay_seconds=0.0
        )
    )
    self.assertFalse(verdict.passed)
    self.assertEqual(verdict.n_contradicted_claims, 1)
    self.assertEqual(verdict.n_ungrounded_added_claims, 1)
    self.assertCountEqual(
        verdict.failure_reasons,
        [
            "contradicted_claims (1)",
            "ungrounded_added_claims (1)",
            "grounding_quality=ungrounded_or_contradicted",
            "no_session_research_evidence",
        ],
    )
    self.assertEqual(
        verdict.red_flags,
        ("Claimed $50M surplus exceeds $3.1M reserve balance",),
    )

  def test_retries_malformed_json_and_returns_error_on_exhaustion(self) -> None:
    item = grounding_autorater.GroundingAuditItem(
        item_id="retry-1",
        seed_proposal=_SEED_STATEMENT,
        candidate_proposal=_GROUNDED_CANDIDATE,
    )
    recover_model = fake_base_model.FakeBaseModel([
        {"text": "{broken", "error": None},
        {"text": f"```json\n{_make_grounding_json()}\n```", "error": None},
    ])
    recovered = asyncio.run(
        grounding_autorater.evaluate_grounding_item(
            recover_model, item, max_attempts=2, retry_delay_seconds=0.0
        )
    )
    self.assertEqual(recovered.attempts, 2)
    self.assertIsNone(recovered.error)

    failing_model = fake_base_model.FakeBaseModel([
        RuntimeError("transient disconnect"),
        {"text": json.dumps({"claims": "wrong-type"}), "error": None},
    ])
    failed = asyncio.run(
        grounding_autorater.evaluate_grounding_item(
            failing_model, item, max_attempts=2, retry_delay_seconds=0.0
        )
    )
    self.assertFalse(failed.passed)
    self.assertIsNotNone(failed.error)
    self.assertEqual(failed.attempts, 2)


class BatchAndRunnerGroundingTest(unittest.TestCase):
  """Tests semaphore batch execution, trace caching, and file runner."""

  def test_batch_enforces_concurrency_and_caches_shared_session_db(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      db_path = _write_synthetic_session_db(
          pathlib.Path(tmp) / "shared.db",
          research_results=[(
              "read_corpus_file",
              {"rel_path": "docs/budget.md"},
              "=== docs/budget.md (Lines 1..1 of 1) ===\n1: $14.2 million",
          )],
      )
      items = [
          grounding_autorater.GroundingAuditItem(
              item_id=f"var-{idx}",
              seed_proposal=_SEED_STATEMENT,
              candidate_proposal=_GROUNDED_CANDIDATE,
              session_db_path=db_path,
          )
          for idx in range(4)
      ]
      model = _DelayedFakeModel(
          [{"text": _make_grounding_json(), "error": None} for _ in range(4)],
          delay_seconds=0.02,
      )
      with mock.patch.object(
          grounding_autorater.session_trace_reader,
          "load_session_trace",
          wraps=session_trace_reader.load_session_trace,
      ) as spy_load:
        verdicts = asyncio.run(
            grounding_autorater.evaluate_grounding_batch(
                model, items, max_concurrent_calls=2, retry_delay_seconds=0.0
            )
        )

    self.assertEqual(spy_load.call_count, 1)
    self.assertEqual(model.peak_concurrency, 2)
    self.assertEqual([v.item_id for v in verdicts], [i.item_id for i in items])
    summary = grounding_autorater.summarize_grounding_verdicts(verdicts)
    self.assertEqual(summary["total"], 4)
    self.assertEqual(summary["passed"], 4)
    self.assertEqual(summary["grounded"], 4)

  def test_runner_resolves_refiner_session_dirs_and_writes_json_and_csv(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      session_dir = root / "sessions" / "bus-priority" / "session01"
      _write_synthetic_session_db(
          session_dir / "trace.db",
          research_results=[(
              "search_corpus",
              {"query": "bus lane cost"},
              "[briefs/transit.md]\n  L1: Cost is $400,000 per corridor.",
          )],
          finish_variations=[{"description": _GROUNDED_CANDIDATE}],
      )
      refiner_json = root / "refined_variations.json"
      refiner_json.write_text(
          json.dumps([{
              "proposal_id": "bus-priority",
              "proposal_title": "Evening Bus Priority",
              "original_statement": _SEED_STATEMENT,
              "session_dirs": [str(session_dir)],
              "variations": [{
                  "variation_id": "v01",
                  "description": _GROUNDED_CANDIDATE,
                  "feasibility_assessment": "Checked briefs/transit.md",
              }],
          }])
      )
      out_json = root / "grounding.json"
      fake_model = fake_base_model.FakeBaseModel(
          [{"text": _make_grounding_json(), "error": None}]
      )
      args = grounding_autorater.get_args([
          "--input_path",
          str(refiner_json),
          "--output_path",
          str(out_json),
          "--max_concurrent_calls",
          "2",
      ])
      config = grounding_autorater.config_from_args(args)
      verdicts = asyncio.run(
          grounding_autorater.run_grounding_audit(config, model=fake_model)
      )

      self.assertEqual(len(verdicts), 1)
      self.assertTrue(verdicts[0].passed)
      self.assertEqual(verdicts[0].files_accessed, ("briefs/transit.md",))
      written = json.loads(out_json.read_text())
      self.assertEqual(written["summary"]["passed"], 1)

      # Verify CSV loading via sessions_dir fallback and CSV output writing.
      in_csv = root / "grounding_items.csv"
      pd.DataFrame([{
          "item_id": "csv-v01",
          "seed_proposal": _SEED_STATEMENT,
          "candidate_proposal": _GROUNDED_CANDIDATE,
      }]).to_csv(in_csv, index=False)
      out_csv = root / "grounding.csv"
      csv_model = fake_base_model.FakeBaseModel(
          [{"text": _make_grounding_json(), "error": None}]
      )
      csv_config = grounding_autorater.GroundingAuditConfig(
          input_path=in_csv,
          output_path=out_csv,
          sessions_dir=root / "sessions",
          retry_delay_seconds=0.0,
      )
      csv_verdicts = asyncio.run(
          grounding_autorater.run_grounding_audit(csv_config, model=csv_model)
      )
      self.assertEqual(csv_verdicts[0].files_accessed, ("briefs/transit.md",))
      out_df = pd.read_csv(out_csv)
      self.assertEqual(out_df.shape[0], 1)
      self.assertEqual(out_df.loc[0, "item_id"], "csv-v01")
      self.assertTrue(bool(out_df.loc[0, "passed"]))


if __name__ == "__main__":
  unittest.main()
