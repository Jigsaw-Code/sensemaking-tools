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

"""Unit tests for the Antigravity SQLite session trace reader."""

import json
import pathlib
import sqlite3
import tempfile
from typing import Any
import unittest

from src.proposition_refinement.agentic import session_trace_reader


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
    call_id: str,
    tool_name: str,
    args: dict[str, Any] | str,
    *,
    thought_summary: str = "",
) -> bytes:
  """Builds a synthetic protobuf-like step_type=15 or step_type=2 blob."""
  parts = [b"\x08\x0f\x20\x03"]
  if thought_summary:
    thought_bytes = thought_summary.encode("utf-8")
    parts.append(b"\x1a" + _encode_varint(len(thought_bytes)) + thought_bytes)
  args_text = args if isinstance(args, str) else json.dumps(args)
  args_bytes = args_text.encode("utf-8")
  tool_bytes = tool_name.encode("utf-8")
  call_bytes = call_id.encode("utf-8")
  parts.append(
      b"\n"
      + _encode_varint(len(call_bytes))
      + call_bytes
      + b"\x12"
      + bytes([len(tool_bytes)])
      + tool_bytes
      + b"\x1a"
      + _encode_varint(len(args_bytes))
      + args_bytes
  )
  return b"".join(parts)


def _make_tool_result_blob(
    call_id: str,
    tool_name: str,
    args: dict[str, Any],
    response_obj: dict[str, Any] | str,
) -> bytes:
  """Builds a synthetic step_type=103 blob with binary noise + ToolResponse."""
  call_part = _make_call_blob(call_id, tool_name, args)
  encrypted_noise = b"\xff\xfe\x80\x00\x99encrypted-blob-noise\x00\xff"
  if isinstance(response_obj, str):
    payload_bytes = response_obj.encode("utf-8")
  else:
    payload_bytes = json.dumps(response_obj).encode("utf-8")
  marker = b"antigravity.localharness.ToolResponse\x12\x10\n"
  return call_part + encrypted_noise + marker + payload_bytes


def _create_steps_db(
    connection: sqlite3.Connection,
    rows: list[tuple[int, int, bytes | str | None]],
) -> None:
  """Creates the synthetic Antigravity steps table and inserts test rows."""
  connection.execute(
      "CREATE TABLE steps ("
      "idx INTEGER PRIMARY KEY, "
      "step_type INTEGER NOT NULL, "
      "step_payload BLOB"
      ")"
  )
  connection.executemany(
      "INSERT INTO steps (idx, step_type, step_payload) VALUES (?, ?, ?)",
      rows,
  )
  connection.commit()


class SessionTraceParsingTest(unittest.TestCase):
  """Tests reconstructing tool calls, results, thoughts, and finish steps."""

  def test_reconstructs_full_trajectory_from_in_memory_db(self) -> None:
    thought = (
        "**Checking Budget Constraints**\n\n"
        "Inspecting the capital plan to verify maintenance reserve limits."
    )
    search_result = (
        "[briefs/transit_budget.md]\n"
        "  L10: Annual street maintenance budget is $14.2 million.\n\n"
        "[briefs/transit_budget.md]\n"
        "  L25: Dedicated markings cost $400,000 per corridor."
    )
    read_result = (
        "=== data/corridors.csv (Lines 1..3 of 12) ===\n"
        "1: corridor,evening_delay_min\n"
        "2: Main Ave,9.5"
    )
    finish_args = {
        "proposal_id": "bus-priority",
        "variations": [{
            "variation_id": "v01",
            "description": (
                "Add dedicated bus lanes on Main Ave using existing"
                " street maintenance funds."
            ),
        }],
    }

    rows: list[tuple[int, int, bytes | str | None]] = [
        (0, session_trace_reader.STEP_INITIAL_PROMPT, b"initial prompt blob"),
        (
            1,
            session_trace_reader.STEP_MODEL_TURN,
            _make_call_blob(
                "call_101",
                "search_corpus",
                {"query": "maintenance budget", "scope": "all"},
                thought_summary=thought,
            ),
        ),
        (
            2,
            session_trace_reader.STEP_TOOL_RESULT,
            _make_tool_result_blob(
                "call_101",
                "search_corpus",
                {"query": "maintenance budget", "scope": "all"},
                {"result": search_result},
            ),
        ),
        (
            3,
            session_trace_reader.STEP_MODEL_TURN,
            _make_call_blob(
                "call_102",
                "read_corpus_file",
                {"rel_path": "data/corridors.csv", "start_line": 1},
            ),
        ),
        (
            4,
            session_trace_reader.STEP_TOOL_RESULT,
            _make_tool_result_blob(
                "call_102",
                "read_corpus_file",
                {"rel_path": "data/corridors.csv", "start_line": 1},
                {"output": read_result},
            ),
        ),
        (
            5,
            session_trace_reader.STEP_MODEL_TURN,
            _make_call_blob(
                "call_103",
                "finish",
                finish_args,
                thought_summary=(
                    "**Finalizing Verified Candidates**\n\n"
                    "Confirming all claims match the retrieved budget text."
                ),
            ),
        ),
        (
            6,
            session_trace_reader.STEP_FINISH,
            _make_call_blob("call_103", "finish", finish_args),
        ),
    ]

    conn = sqlite3.connect(":memory:")
    try:
      _create_steps_db(conn, rows)
      trace = session_trace_reader.load_session_trace_from_connection(conn)
    finally:
      conn.close()

    self.assertTrue(trace.finished)
    self.assertEqual(trace.total_steps, 7)
    self.assertEqual(trace.parse_warnings, ())
    self.assertEqual(trace.unknown_step_types, ())
    self.assertEqual(len(trace.events), 3)

    search_event = trace.events[0]
    self.assertEqual(search_event.call_id, "call_101")
    self.assertEqual(search_event.tool_name, "search_corpus")
    self.assertEqual(search_event.call_step_idx, 1)
    self.assertEqual(search_event.result_step_idx, 2)
    self.assertEqual(search_event.result, search_result)

    read_event = trace.events[1]
    self.assertEqual(read_event.call_id, "call_102")
    self.assertEqual(read_event.result, read_result)

    self.assertEqual(
        trace.files_accessed(),
        ("briefs/transit_budget.md", "data/corridors.csv"),
    )
    self.assertEqual(trace.queries(), ("maintenance budget",))
    self.assertEqual(trace.read_paths(), ("data/corridors.csv",))
    self.assertEqual(trace.finish_payload, finish_args)
    self.assertEqual(len(trace.pre_finish_events), 2)
    self.assertEqual(len(trace.thought_summaries), 2)
    self.assertEqual(
        trace.thought_summaries[0].headings, ("Checking Budget Constraints",)
    )
    self.assertIn("maintenance reserve limits", trace.thought_summaries[0].text)

  def test_gracefully_handles_unknown_types_and_corrupted_blobs(self) -> None:
    rows: list[tuple[int, int, bytes | str | None]] = [
        (0, 17, b"\x00\x01unknown-step-type"),
        (1, 999, b"another-unknown-type"),
        (2, session_trace_reader.STEP_MODEL_TURN, b"\xff\xfe\x00\x01corrupt"),
        (
            3,
            session_trace_reader.STEP_MODEL_TURN,
            _make_call_blob("call_201", "search_corpus", "{truncated json"),
        ),
        (
            4,
            session_trace_reader.STEP_TOOL_RESULT,
            _make_tool_result_blob(
                "call_201",
                "search_corpus",
                {"query": "transit"},
                '{"result": "unterminated',
            ),
        ),
        (
            5,
            session_trace_reader.STEP_TOOL_RESULT,
            b"\x00\xffno-call-markers-here",
        ),
    ]
    conn = sqlite3.connect(":memory:")
    try:
      _create_steps_db(conn, rows)
      trace = session_trace_reader.load_session_trace_from_connection(conn)
    finally:
      conn.close()

    self.assertFalse(trace.finished)
    self.assertIsNone(trace.finish_payload)
    self.assertEqual(trace.unknown_step_types, (17, 999))
    self.assertEqual(len(trace.events), 1)
    self.assertEqual(trace.events[0].call_id, "call_201")
    self.assertEqual(trace.events[0].args, {"query": "transit"})
    self.assertIsNone(trace.events[0].result)
    self.assertGreaterEqual(len(trace.parse_warnings), 3)

  def test_thought_summary_fallback_handles_predecoded_strings(self) -> None:
    raw = (
        "\x00\x01**Evaluating Safety Trade-Offs**\n\n"
        "Checking crash reduction benchmarks before finalizing.\x00\xfftail"
    )
    summaries = session_trace_reader.extract_thought_summaries(raw)
    self.assertEqual(len(summaries), 1)
    self.assertIn("Evaluating Safety Trade-Offs", summaries[0])
    self.assertNotIn("tail", summaries[0])


class DiskLoadAndSessionSelectionTest(unittest.TestCase):
  """Tests on-disk DB loading, schema checks, and matching session selection."""

  def test_load_session_trace_from_disk_and_select_matching_session(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      dir_old = root / "model__session01__20260925T010000Z"
      dir_new = root / "model__session02__20260925T020000Z"
      dir_old.mkdir()
      dir_new.mkdir()

      old_db = dir_old / "aaa.db"
      new_db = dir_new / "bbb.db"

      for db_path, desc in (
          (old_db, "Older draft variation text."),
          (new_db, "Final accepted variation text."),
      ):
        conn = sqlite3.connect(str(db_path))
        try:
          _create_steps_db(
              conn,
              [(
                  1,
                  session_trace_reader.STEP_FINISH,
                  _make_call_blob(
                      "call_900",
                      "finish",
                      {
                          "proposal_id": "p1",
                          "variations": [{"description": desc}],
                      },
                  ),
              )],
          )
        finally:
          conn.close()

      found = session_trace_reader.find_session_dbs(root)
      self.assertEqual(found, [old_db, new_db])

      traces = [session_trace_reader.load_session_trace(p) for p in found]
      selected, method = session_trace_reader.select_matching_session(
          traces, expected_descriptions=["Final accepted variation text."]
      )
      self.assertEqual(method, "finish_payload_exact_match")
      self.assertIsNotNone(selected)
      self.assertEqual(selected.db_path, new_db)

      fallback, fb_method = session_trace_reader.select_matching_session(
          traces, expected_descriptions=["Unmatched statement."]
      )
      self.assertEqual(fb_method, "last_finished_fallback")
      self.assertIsNotNone(fallback)
      self.assertEqual(fallback.db_path, new_db)

  def test_missing_file_or_schema_raises(self) -> None:
    with self.assertRaises(FileNotFoundError):
      session_trace_reader.load_session_trace("/nonexistent/session.db")

    conn = sqlite3.connect(":memory:")
    try:
      conn.execute("CREATE TABLE other_table (id INTEGER)")
      with self.assertRaises(ValueError):
        session_trace_reader.load_session_trace_from_connection(conn)
    finally:
      conn.close()


if __name__ == "__main__":
  unittest.main()
