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

"""Tests for stance_classifier."""

import asyncio
from collections.abc import Callable
import json
from typing import Any
import unittest

import pandas as pd

from src.models import base_model
from src.transcripts import stance_classifier


class _FakeModel(base_model.BaseModel):
  """Deterministic fake BaseModel with call and concurrency tracking."""

  def __init__(
      self,
      responder: Callable[[str, str], dict[str, Any]],
      delay_seconds: float = 0.0,
  ):
    super().__init__()
    self._responder = responder
    self._delay_seconds = delay_seconds
    self.calls: list[tuple[str, str]] = []
    self._active_calls = 0
    self.peak_concurrent_calls = 0

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: dict[str, Any] | None = None,
      **kwargs,
  ) -> dict[str, Any]:
    del temperature, system_prompt, response_mime_type, response_schema, kwargs
    self._active_calls += 1
    self.peak_concurrent_calls = max(
        self.peak_concurrent_calls, self._active_calls
    )
    self.calls.append((run_name, prompt))
    try:
      if self._delay_seconds > 0:
        await asyncio.sleep(self._delay_seconds)
      return self._responder(prompt, run_name)
    finally:
      self._active_calls -= 1

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: Callable[[Any, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs,
  ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    del prompts, response_parser, max_concurrent_calls, retry_attempts, kwargs
    raise NotImplementedError("stance_classifier uses generate_content.")


def _make_sample_proposal() -> stance_classifier.Proposal:
  return stance_classifier.Proposal(
      proposal_id="bus_lanes",
      title="Dedicated Bus Lanes",
      text="Convert one travel lane on major corridors to bus-only lanes.",
      topic="Transit Reliability",
  )


def _make_sample_statements() -> list[stance_classifier.DeliberationStatement]:
  return [
      stance_classifier.DeliberationStatement(
          statement_id="s1",
          participant_id="P1",
          text="Faster buses help commuters reach work on time.",
          room_id="Room 01",
      ),
      stance_classifier.DeliberationStatement(
          statement_id="s2",
          participant_id="P2",
          text="Removing a car lane could increase congestion at rush hour.",
          room_id="Room 01",
      ),
      stance_classifier.DeliberationStatement(
          statement_id="s3",
          participant_id="P3",
          text="Try peak-hour bus lanes first before making them permanent.",
          room_id="Room 02",
      ),
      stance_classifier.DeliberationStatement(
          statement_id="s4",
          participant_id="P4",
          text="Library weekend hours should be extended.",
          room_id="Room 02",
      ),
  ]


class StanceClassifierPureHelpersTest(unittest.TestCase):
  """Tests for label normalization, batching, parsing, and dossier rendering."""

  def test_normalize_stance_label_accepts_case_and_whitespace(self):
    self.assertEqual(
        stance_classifier.normalize_stance_label("  SUPPORT "),
        stance_classifier.StanceLabel.SUPPORT,
    )
    self.assertEqual(
        stance_classifier.normalize_stance_label("Critique"),
        stance_classifier.StanceLabel.CRITIQUE,
    )
    for label in stance_classifier.STANCE_LABELS:
      self.assertEqual(
          stance_classifier.normalize_stance_label(label.value.upper()),
          label,
      )

  def test_normalize_stance_label_rejects_unknown_value(self):
    with self.assertRaisesRegex(ValueError, "Unknown stance label"):
      stance_classifier.normalize_stance_label("neutral")

  def test_statements_from_room_summaries(self):
    rooms = [
        {
            "room_header": "===== Room: Transit =====",
            "summaries": {
                "Participant 1": "Expand bus frequency.",
                "Participant 2": "Keep parking available.",
            },
        },
        {
            "room_header": "",
            "summaries": {"Participant 3": "Add shaded shelters."},
        },
    ]
    statements = stance_classifier.statements_from_room_summaries(rooms)
    self.assertEqual(len(statements), 3)
    self.assertCountEqual(
        [s.participant_id for s in statements],
        ["Participant 1", "Participant 2", "Participant 3"],
    )
    self.assertEqual(statements[0].room_id, "===== Room: Transit =====")
    self.assertEqual(statements[2].room_id, "room_02")

  def test_build_statement_batches_groups_by_room_and_splits_large_rooms(self):
    statements = [
        stance_classifier.DeliberationStatement(
            statement_id=f"a{i}",
            participant_id=f"P{i}",
            text=f"Room A text {i}",
            room_id="Room A",
        )
        for i in range(3)
    ] + [
        stance_classifier.DeliberationStatement(
            statement_id="b1",
            participant_id="P9",
            text="Room B text",
            room_id="Room B",
        )
    ]
    room_batches = stance_classifier.build_statement_batches(
        statements, group_by_room=True, max_batch_size=2
    )
    self.assertEqual(len(room_batches), 3)
    self.assertEqual(
        [[s.statement_id for s in batch] for batch in room_batches],
        [["a0", "a1"], ["a2"], ["b1"]],
    )

    flat_batches = stance_classifier.build_statement_batches(
        statements, group_by_room=False, max_batch_size=2
    )
    self.assertEqual(len(flat_batches), 2)
    self.assertEqual(
        [[s.statement_id for s in batch] for batch in flat_batches],
        [["a0", "a1"], ["a2", "b1"]],
    )

  def test_build_statement_batches_validates_inputs(self):
    dup_statements = [
        stance_classifier.DeliberationStatement("dup", "P1", "First."),
        stance_classifier.DeliberationStatement("dup", "P2", "Second."),
    ]
    with self.assertRaisesRegex(ValueError, "Duplicate statement_id"):
      stance_classifier.build_statement_batches(dup_statements)
    with self.assertRaisesRegex(ValueError, "max_batch_size"):
      stance_classifier.build_statement_batches([], max_batch_size=0)

  def test_parse_stance_batch_response_handles_fenced_json_and_ordering(self):
    raw_text = (
        "```json\n"
        '{"ratings": ['
        '{"statement_id": "s2", "stance": "CRITIQUE"}, '
        '{"statement_id": "s1", "stance": "support"}'
        "]}\n```"
    )
    parsed = stance_classifier.parse_stance_batch_response(
        raw_text, ["s1", "s2"]
    )
    self.assertEqual(
        parsed,
        [
            stance_classifier.ClassifiedStatement(
                "s1", stance_classifier.StanceLabel.SUPPORT
            ),
            stance_classifier.ClassifiedStatement(
                "s2", stance_classifier.StanceLabel.CRITIQUE
            ),
        ],
    )

  def test_parse_stance_batch_response_detects_missing_and_unexpected_ids(self):
    raw_text = json.dumps({
        "ratings": [
            {"statement_id": "s1", "stance": "support"},
            {"statement_id": "hallucinated", "stance": "critique"},
        ]
    })
    with self.assertRaises(stance_classifier.CoverageValidationError) as ctx:
      stance_classifier.parse_stance_batch_response(raw_text, ["s1", "s2"])
    self.assertEqual(ctx.exception.missing_ids, frozenset({"s2"}))
    self.assertEqual(ctx.exception.unexpected_ids, frozenset({"hallucinated"}))

  def test_parse_stance_batch_response_rejects_duplicate_ids(self):
    raw_text = json.dumps({
        "ratings": [
            {"statement_id": "s1", "stance": "support"},
            {"statement_id": "s1", "stance": "critique"},
        ]
    })
    with self.assertRaisesRegex(ValueError, "Duplicate statement_id"):
      stance_classifier.parse_stance_batch_response(raw_text, ["s1"])

  def test_prompt_uses_proposal_terminology_and_lists_companions(self):
    focal = _make_sample_proposal()
    companion = stance_classifier.Proposal(
        proposal_id="signal_priority",
        title="Transit Signal Priority",
        text="Give buses green-light priority at congested intersections.",
    )
    prompt = stance_classifier.build_stance_classification_prompt(
        focal,
        _make_sample_statements()[:2],
        companion_proposals=[companion],
    )
    self.assertIn("Dedicated Bus Lanes", prompt)
    self.assertIn("Transit Signal Priority", prompt)
    self.assertIn("[ID: s1]", prompt)
    self.assertNotIn("policy", prompt.lower())

  def test_format_classified_stance_dossier_endogenous_counts(self):
    focal = _make_sample_proposal()
    statements = _make_sample_statements()
    classifications = {
        "s1": stance_classifier.ClassifiedStatement(
            "s1", stance_classifier.StanceLabel.SUPPORT
        ),
        "s2": stance_classifier.ClassifiedStatement(
            "s2", stance_classifier.StanceLabel.CRITIQUE
        ),
        "s3": stance_classifier.ClassifiedStatement(
            "s3", stance_classifier.StanceLabel.SUGGESTION
        ),
        "s4": stance_classifier.ClassifiedStatement(
            "s4", stance_classifier.StanceLabel.UNRELATED
        ),
    }

    dossier = stance_classifier.format_classified_stance_dossier(
        focal, statements, classifications
    )
    self.assertIn(
        "3 proposal-relevant: 1 Support, 1 Critique, 1 Suggestion,"
        " 0 Observation; 1 Unrelated excluded",
        dossier,
    )
    self.assertIn("#### Support (1 statements)", dossier)
    self.assertIn("#### Critiques & Alternatives (1 statements)", dossier)
    self.assertIn("#### Suggestions for Amendment (1 statements)", dossier)
    self.assertNotIn("Library weekend hours", dossier)
    self.assertNotIn("---", dossier)

    dossier_with_unrelated = stance_classifier.format_classified_stance_dossier(
        focal, statements, classifications, include_unrelated=True
    )
    self.assertIn("1 Unrelated included", dossier_with_unrelated)
    self.assertIn("Library weekend hours", dossier_with_unrelated)

  def test_format_classified_stance_dossier_sorts_rooms_naturally(self):
    focal = _make_sample_proposal()
    room_names = ["Room 21", "Room 2", "Room 11", "Room 1"]
    statements = [
        stance_classifier.DeliberationStatement(
            statement_id=f"s{idx}",
            participant_id=f"P{idx}",
            text=f"Statement from {room_name}.",
            room_id=room_name,
        )
        for idx, room_name in enumerate(room_names, start=1)
    ]
    classifications = {
        stmt.statement_id: stance_classifier.ClassifiedStatement(
            stmt.statement_id, stance_classifier.StanceLabel.SUPPORT
        )
        for stmt in statements
    }

    dossier = stance_classifier.format_classified_stance_dossier(
        focal, statements, classifications
    )
    room_headers = [
        line for line in dossier.splitlines() if line.startswith("**Room ")
    ]
    self.assertEqual(
        room_headers,
        ["**Room 1**", "**Room 2**", "**Room 11**", "**Room 21**"],
    )

  def test_extract_valid_partial_ratings(self):
    raw_with_unexpected = json.dumps({
        "ratings": [
            {"statement_id": "s1", "stance": "support"},
            {"statement_id": "hallucinated", "stance": "critique"},
        ]
    })
    self.assertEqual(
        stance_classifier._extract_valid_partial_ratings(
            raw_with_unexpected, ["s1", "s2"]
        ),
        [
            stance_classifier.ClassifiedStatement(
                "s1", stance_classifier.StanceLabel.SUPPORT
            )
        ],
    )
    raw_with_dup = json.dumps({
        "ratings": [
            {"statement_id": "s1", "stance": "support"},
            {"statement_id": "s1", "stance": "critique"},
        ]
    })
    self.assertEqual(
        stance_classifier._extract_valid_partial_ratings(
            raw_with_dup, ["s1", "s2"]
        ),
        [],
    )
    self.assertEqual(
        stance_classifier._extract_valid_partial_ratings("not-json", ["s1"]),
        [],
    )


class StanceClassifierAsyncTest(unittest.IsolatedAsyncioTestCase):
  """Async tests for batching, semaphore throttling, and coverage retries."""

  async def test_classify_statement_batch_retries_only_missing_ids(self):
    focal = _make_sample_proposal()
    statements = _make_sample_statements()[:3]
    seen_prompts: list[str] = []

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del run_name
      seen_prompts.append(prompt)
      if len(seen_prompts) == 1:
        # Omit s3 on first attempt to verify targeted retry recovery.
        return {
            "text": json.dumps({
                "ratings": [
                    {"statement_id": "s1", "stance": "support"},
                    {"statement_id": "s2", "stance": "critique"},
                ]
            })
        }
      return {
          "text": json.dumps(
              {"ratings": [{"statement_id": "s3", "stance": "suggestion"}]}
          )
      }

    model = _FakeModel(responder)
    results = await stance_classifier.classify_statement_batch(
        model, focal, statements, max_retries=2
    )

    self.assertEqual(len(seen_prompts), 2)
    self.assertNotIn("[ID: s1]", seen_prompts[1])
    self.assertIn("[ID: s3]", seen_prompts[1])
    self.assertEqual(
        [r.statement_id for r in results],
        ["s1", "s2", "s3"],
    )
    self.assertEqual(
        results[2].stance, stance_classifier.StanceLabel.SUGGESTION
    )

  async def test_classify_statement_batch_raises_when_missing_ids_persist(self):
    focal = _make_sample_proposal()
    statements = _make_sample_statements()[:2]

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del prompt, run_name
      return {
          "text": json.dumps(
              {"ratings": [{"statement_id": "s1", "stance": "support"}]}
          )
      }

    model = _FakeModel(responder)
    with self.assertRaises(stance_classifier.CoverageValidationError) as ctx:
      await stance_classifier.classify_statement_batch(
          model, focal, statements, max_retries=2
      )
    self.assertEqual(ctx.exception.missing_ids, frozenset({"s2"}))
    self.assertEqual(len(model.calls), 2)

  async def test_classify_statement_batch_recovers_from_api_error(self):
    focal = _make_sample_proposal()
    statements = _make_sample_statements()[:1]
    attempt_counter = {"count": 0}

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del prompt, run_name
      attempt_counter["count"] += 1
      if attempt_counter["count"] == 1:
        return {"error": "Transient timeout"}
      return {
          "text": json.dumps(
              {"ratings": [{"statement_id": "s1", "stance": "observation"}]}
          )
      }

    model = _FakeModel(responder)
    results = await stance_classifier.classify_statement_batch(
        model, focal, statements, max_retries=2
    )
    self.assertEqual(
        results,
        [
            stance_classifier.ClassifiedStatement(
                "s1", stance_classifier.StanceLabel.OBSERVATION
            )
        ],
    )

  async def test_classify_statement_batch_raises_coverage_error_on_api_failure(
      self,
  ):
    focal = _make_sample_proposal()
    statements = _make_sample_statements()[:2]

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del prompt, run_name
      return {"error": "Persistent service unavailable"}

    model = _FakeModel(responder)
    with self.assertRaises(stance_classifier.CoverageValidationError) as ctx:
      await stance_classifier.classify_statement_batch(
          model, focal, statements, max_retries=2
      )
    self.assertEqual(ctx.exception.missing_ids, frozenset({"s1", "s2"}))
    self.assertEqual(len(model.calls), 2)

  async def test_classify_statements_for_proposal_respects_concurrency_cap(
      self,
  ):
    focal = _make_sample_proposal()
    statements = [
        stance_classifier.DeliberationStatement(
            statement_id=f"stmt_{i}",
            participant_id=f"P{i}",
            text=f"Statement {i}",
            room_id=f"Room {i}",
        )
        for i in range(6)
    ]

    def responder(prompt: str, run_name: str) -> dict[str, Any]:
      del run_name
      ratings = []
      for stmt in statements:
        if f"[ID: {stmt.statement_id}]" in prompt:
          ratings.append(
              {"statement_id": stmt.statement_id, "stance": "support"}
          )
      return {"text": json.dumps({"ratings": ratings})}

    model = _FakeModel(responder, delay_seconds=0.02)
    classified = await stance_classifier.classify_statements_for_proposal(
        model=model,
        focal_proposal=focal,
        statements=statements,
        group_by_room=True,
        max_batch_size=1,
        max_concurrent_calls=2,
    )

    self.assertLessEqual(model.peak_concurrent_calls, 2)
    self.assertEqual(len(model.calls), 6)
    self.assertCountEqual(
        classified.keys(), [stmt.statement_id for stmt in statements]
    )


if __name__ == "__main__":
  unittest.main()
