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

"""Stdlib-only trace reader for persisted Antigravity SQLite session databases.

WARNING (SDK Implementation Fragility):
  This module reconstructs tool calls, tool results, finish payloads, and model
  thought summaries by inspecting internal ``google-antigravity`` SQLite storage
  tables (``steps(idx, step_type, step_payload)``). It relies on undocumented
  SDK storage implementation details observed in ``google-antigravity`` 0.1.5,
  including numeric ``step_type`` codes (``15`` for model turns, ``103`` for
  tool results, ``2`` for finish steps), protobuf wire markers (``(call_\\d+)``
  with field tags and length-delimited thought strings), and
  ``antigravity.localharness.ToolResponse`` JSON framing. These internal
  structures are not a stable public API and may shift or break across SDK
  versions. Callers should treat parsed traces as best-effort diagnostic audits
  and inspect ``SessionTrace.parse_warnings`` and
  ``SessionTrace.unknown_step_types`` for signs of format drift.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
import re
import sqlite3
from typing import Any

# Internal google-antigravity step_type codes observed in SDK 0.1.5.
STEP_FINISH: int = 2
STEP_INITIAL_PROMPT: int = 14
STEP_MODEL_TURN: int = 15
STEP_TOOL_RESULT: int = 103

_KNOWN_STEP_TYPES: frozenset[int] = frozenset({
    STEP_FINISH,
    STEP_INITIAL_PROMPT,
    STEP_MODEL_TURN,
    STEP_TOOL_RESULT,
})

DEFAULT_RESEARCH_TOOLS: tuple[str, ...] = (
    "search_corpus",
    "read_corpus_file",
)
DEFAULT_LIST_TOOLS: tuple[str, ...] = ("list_corpus_files",)
DEFAULT_DRAFT_TOOLS: tuple[str, ...] = ("write_candidate_draft",)

_TOOL_RESPONSE_MARKER: str = "antigravity.localharness.ToolResponse"
_CALL_RE = re.compile(r"(call_\d+)\x12[\s\S](\w+)\x1a")
_SEARCH_FILE_RE = re.compile(r"^\[([^\]\n]+)\]", re.MULTILINE)
_READ_FILE_RE = re.compile(
    r"^=== (.+?) \(Lines \d+\.\.\d+ of \d+\) ===", re.MULTILINE
)
_THOUGHT_HEADING_RE = re.compile(r"\*\*([A-Z][^*\n]{2,120})\*\*")
_CONTROL_BREAK_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]")
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_ARGS_OFFSET: int = 10
_MIN_THOUGHT_BYTES: int = 12
_JSON_DECODER = json.JSONDecoder()
_REQUIRED_STEPS_COLUMNS: frozenset[str] = frozenset(
    {"idx", "step_type", "step_payload"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class ToolEvent:
  """One reconstructed tool invocation and optional result from a session DB."""

  step_idx: int
  step_type: int
  call_id: str
  tool_name: str
  args: dict[str, Any] | None = None
  result: str | None = None
  call_step_idx: int | None = None
  result_step_idx: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ThoughtSummary:
  """Model thought summary paragraphs extracted from a model-turn step."""

  step_idx: int
  text: str
  headings: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class SessionTrace:
  """Reconstructed trajectory from one persisted Antigravity session DB."""

  db_path: pathlib.Path | None
  total_steps: int
  events: tuple[ToolEvent, ...]
  thought_summaries: tuple[ThoughtSummary, ...]
  finish_event: ToolEvent | None
  finish_payload: dict[str, Any] | None
  unknown_step_types: tuple[int, ...] = ()
  parse_warnings: tuple[str, ...] = ()

  @property
  def finished(self) -> bool:
    """Returns True when a finish step was recovered from the session."""
    return self.finish_event is not None

  @property
  def pre_finish_events(self) -> tuple[ToolEvent, ...]:
    """Returns tool events occurring before the finish step."""
    if self.finish_event is None:
      return tuple(e for e in self.events if e.tool_name != "finish")
    finish_idx = self.finish_event.step_idx
    return tuple(
        e
        for e in self.events
        if e.tool_name != "finish" and e.step_idx < finish_idx
    )

  def research_events(
      self, research_tools: Sequence[str] = DEFAULT_RESEARCH_TOOLS
  ) -> tuple[ToolEvent, ...]:
    """Returns pre-finish tool events matching `research_tools`."""
    allowed = frozenset(research_tools)
    return tuple(e for e in self.pre_finish_events if e.tool_name in allowed)

  def files_accessed(
      self, research_tools: Sequence[str] = DEFAULT_RESEARCH_TOOLS
  ) -> tuple[str, ...]:
    """Returns ordered distinct corpus paths observed in research results."""
    seen: dict[str, None] = {}
    for event in self.research_events(research_tools):
      for rel_path in files_in_tool_result(event.tool_name, event.result):
        seen[rel_path] = None
    return tuple(seen)

  def queries(self, search_tool: str = "search_corpus") -> tuple[str, ...]:
    """Returns ordered search queries issued before finish."""
    queries: list[str] = []
    for event in self.pre_finish_events:
      if event.tool_name == search_tool and isinstance(event.args, Mapping):
        query = str(event.args.get("query") or "").strip()
        if query:
          queries.append(query)
    return tuple(queries)

  def read_paths(self, read_tool: str = "read_corpus_file") -> tuple[str, ...]:
    """Returns ordered distinct relative file paths requested via read tool."""
    paths: dict[str, None] = {}
    for event in self.pre_finish_events:
      if event.tool_name == read_tool and isinstance(event.args, Mapping):
        rel_path = str(event.args.get("rel_path") or "").strip()
        if rel_path:
          paths[rel_path] = None
    return tuple(paths)


def _normalize_text(text: str) -> str:
  """Collapses whitespace and lowercases text for description matching."""
  return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


def _decode_protobuf_varint(data: bytes, offset: int) -> tuple[int, int] | None:
  """Decodes an unsigned base-128 protobuf varint starting at `offset`."""
  value = 0
  shift = 0
  pos = offset
  limit = min(len(data), offset + 10)
  while pos < limit:
    byte = data[pos]
    value |= (byte & 0x7F) << shift
    pos += 1
    if not (byte & 0x80):
      return value, pos
    shift += 7
  return None


def extract_thought_summaries(payload: bytes | str) -> list[str]:
  """Extracts bold-headed model thought summaries from a model-turn payload.

  WARNING: Relies on internal ``google-antigravity`` protobuf wire framing
  (length-delimited field tag ``0x1a`` containing markdown thought summaries
  headed by ``**...**``) and may need adjustment if SDK serialization changes.

  Args:
    payload: Raw ``step_payload`` bytes (or pre-decoded string) from a
      ``step_type=15`` row.

  Returns:
    Ordered, deduplicated list of extracted thought summary text blocks.
  """
  if not payload:
    return []

  summaries: list[str] = []
  seen: set[str] = set()

  if isinstance(payload, (bytes, bytearray)):
    raw = bytes(payload)
    search_pos = 0
    while True:
      tag_pos = raw.find(b"\x1a", search_pos)
      if tag_pos < 0:
        break
      decoded_varint = _decode_protobuf_varint(raw, tag_pos + 1)
      search_pos = tag_pos + 1
      if decoded_varint is None:
        continue
      length, start = decoded_varint
      end = start + length
      if length < _MIN_THOUGHT_BYTES or end > len(raw):
        continue
      if raw[start : start + 2] != b"**":
        continue
      try:
        candidate = raw[start:end].decode("utf-8")
      except UnicodeDecodeError:
        continue
      cleaned = candidate.strip()
      if (
          cleaned
          and _THOUGHT_HEADING_RE.search(cleaned)
          and not _CONTROL_BREAK_RE.search(cleaned)
          and cleaned not in seen
      ):
        seen.add(cleaned)
        summaries.append(cleaned)
        search_pos = end

  if summaries:
    return summaries

  # Fallback scan for pre-decoded strings or shifted protobuf field tags.
  text = (
      payload.decode("utf-8", errors="replace")
      if isinstance(payload, (bytes, bytearray))
      else str(payload)
  )
  pos = 0
  while True:
    match = _THOUGHT_HEADING_RE.search(text, pos)
    if match is None:
      break
    start = match.start()
    break_match = _CONTROL_BREAK_RE.search(text, start)
    end = break_match.start() if break_match is not None else len(text)
    segment = text[start:end].strip()
    pos = max(end, match.end())
    # Ignore short heading-only fragments with no body text.
    if len(segment) > len(match.group(0)) + 4 and segment not in seen:
      seen.add(segment)
      summaries.append(segment)
  return summaries


def parse_tool_response(text: str) -> tuple[str | None, str | None]:
  """Extracts plaintext JSON output after the localharness ToolResponse marker.

  WARNING: Relies on internal ``antigravity.localharness.ToolResponse`` JSON
  framing (``{"result": ...}`` or ``{"output": ...}``) embedded inside
  ``step_type=103`` blobs and ignores adjacent encrypted payloads.

  Args:
    text: UTF-8 decoded step payload text.

  Returns:
    Tuple ``(result_text, warning)``. ``warning`` is populated when a marker was
    present but its JSON payload could not be decoded.
  """
  marker_pos = text.find(_TOOL_RESPONSE_MARKER)
  if marker_pos < 0:
    return None, None
  brace_pos = text.find('{"', marker_pos)
  if brace_pos < 0:
    return None, "ToolResponse marker found without JSON object payload."
  try:
    decoded, _ = _JSON_DECODER.raw_decode(text[brace_pos:])
  except json.JSONDecodeError as exc:
    return None, f"Malformed ToolResponse JSON: {exc}"
  if isinstance(decoded, Mapping):
    value = decoded.get("result", decoded.get("output"))
  else:
    value = decoded
  if value is None:
    return "", None
  if isinstance(value, str):
    return value, None
  return json.dumps(value), None


def parse_step_tool_calls(
    payload: bytes | str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
  """Parses tool call invocations and optional tool results from a step blob.

  WARNING: Relies on internal ``google-antigravity`` protobuf wire markers
  (``(call_\\d+)\\x12[\\s\\S](\\w+)\\x1a``) and ``ToolResponse`` JSON framing
  observed in SDK 0.1.5, which may shift in future SDK releases.

  Args:
    payload: Raw ``step_payload`` bytes or string from SQLite ``steps``.

  Returns:
    Tuple ``(calls, warnings)`` where each call dict has keys ``call_id``,
    ``tool_name``, ``args`` (dict or None), and ``result`` (str or None).
  """
  if not payload:
    return [], []
  text = (
      payload.decode("utf-8", errors="replace")
      if isinstance(payload, (bytes, bytearray))
      else str(payload)
  )
  warnings: list[str] = []
  result_text, result_warning = parse_tool_response(text)
  if result_warning:
    warnings.append(result_warning)

  calls: list[dict[str, Any]] = []
  seen_ids: set[str] = set()
  for match in _CALL_RE.finditer(text):
    call_id = match.group(1)
    tool_name = match.group(2)
    if call_id in seen_ids:
      continue
    seen_ids.add(call_id)

    args: dict[str, Any] | None = None
    brace_pos = text.find("{", match.end())
    if 0 <= brace_pos - match.end() <= _MAX_ARGS_OFFSET:
      try:
        decoded_args, _ = _JSON_DECODER.raw_decode(text[brace_pos:])
        if isinstance(decoded_args, dict):
          args = decoded_args
        else:
          warnings.append(
              f"Tool call {call_id} ({tool_name}) args decoded as"
              f" {type(decoded_args).__name__}, expected dict."
          )
      except json.JSONDecodeError as exc:
        warnings.append(
            f"Tool call {call_id} ({tool_name}) had malformed JSON args: {exc}"
        )

    calls.append({
        "call_id": call_id,
        "tool_name": tool_name,
        "args": args,
        "result": result_text,
    })
  return calls, warnings


def files_in_tool_result(tool_name: str, result: str | None) -> list[str]:
  """Extracts ordered distinct corpus file paths referenced in a tool result.

  Recognizes both search-snippet headers (``[rel_path]``) and bounded read
  headers (``=== rel_path (Lines a..b of N) ===``) emitted by ``corpus_tools``.

  Args:
    tool_name: Name of the tool that produced `result`.
    result: Plaintext tool output string.

  Returns:
    Ordered list of distinct stripped relative file paths.
  """
  if not result:
    return []
  lower_name = (tool_name or "").lower()
  found: list[str] = []
  if "search" in lower_name:
    found.extend(_SEARCH_FILE_RE.findall(result))
  elif "read" in lower_name:
    found.extend(_READ_FILE_RE.findall(result))
  else:
    found.extend(_SEARCH_FILE_RE.findall(result))
    found.extend(_READ_FILE_RE.findall(result))
  return list(dict.fromkeys(path.strip() for path in found if path.strip()))


def _verify_steps_schema(connection: sqlite3.Connection) -> None:
  """Verifies that the SQLite connection has the expected `steps` columns."""
  columns = {
      row[1]
      for row in connection.execute("PRAGMA table_info(steps)").fetchall()
  }
  if not columns:
    raise ValueError("SQLite session DB is missing the 'steps' table.")
  missing = sorted(_REQUIRED_STEPS_COLUMNS - columns)
  if missing:
    raise ValueError(
        f"SQLite 'steps' table is missing required column(s): {missing}."
    )


def load_session_trace_from_connection(
    connection: sqlite3.Connection,
    *,
    db_path: pathlib.Path | None = None,
) -> SessionTrace:
  """Reconstructs a ``SessionTrace`` from an open SQLite DB connection.

  WARNING: Parses internal ``google-antigravity`` SQLite ``steps`` blobs
  (``step_type`` 15 model turns, ``103`` tool results, ``2`` finish steps).
  Format drift across SDK versions may cause partial extraction and will be
  surfaced via ``parse_warnings`` and ``unknown_step_types``.

  Args:
    connection: Open ``sqlite3.Connection`` containing a ``steps`` table.
    db_path: Optional filesystem path recorded for provenance.

  Returns:
    Immutable ``SessionTrace`` with merged tool events, thought summaries,
    finish payload, and non-fatal parse diagnostics.

  Raises:
    ValueError: If the ``steps`` table or required columns are absent.
  """
  _verify_steps_schema(connection)
  rows = connection.execute(
      "SELECT idx, step_type, step_payload FROM steps ORDER BY idx"
  ).fetchall()

  events_by_call_id: dict[str, ToolEvent] = {}
  call_order: list[str] = []
  thought_summaries: list[ThoughtSummary] = []
  unknown_types: set[int] = set()
  warnings: list[str] = []
  finish_event: ToolEvent | None = None

  for idx, step_type, payload in rows:
    if step_type == STEP_INITIAL_PROMPT:
      continue
    if step_type not in _KNOWN_STEP_TYPES:
      unknown_types.add(int(step_type))
      continue

    if step_type == STEP_MODEL_TURN:
      for thought_text in extract_thought_summaries(payload):
        headings = tuple(_THOUGHT_HEADING_RE.findall(thought_text))
        thought_summaries.append(
            ThoughtSummary(step_idx=idx, text=thought_text, headings=headings)
        )

    calls, step_warnings = parse_step_tool_calls(payload)
    for warning in step_warnings:
      warnings.append(f"step idx={idx} (type={step_type}): {warning}")

    if step_type in (STEP_TOOL_RESULT, STEP_FINISH) and not calls and payload:
      warnings.append(
          f"step idx={idx} (type={step_type}): no recognizable tool call"
          " markers found in payload."
      )

    for call in calls:
      cid = call["call_id"]
      tool_name = call["tool_name"]
      existing = events_by_call_id.get(cid)
      if existing is None:
        call_order.append(cid)
        event = ToolEvent(
            step_idx=idx,
            step_type=step_type,
            call_id=cid,
            tool_name=tool_name,
            args=call["args"],
            result=call["result"] if step_type == STEP_TOOL_RESULT else None,
            call_step_idx=idx if step_type == STEP_MODEL_TURN else None,
            result_step_idx=idx if step_type == STEP_TOOL_RESULT else None,
        )
      else:
        merged_args = (
            call["args"] if call["args"] is not None else existing.args
        )
        merged_result = (
            call["result"]
            if step_type == STEP_TOOL_RESULT and call["result"] is not None
            else existing.result
        )
        merged_type = (
            step_type
            if step_type in (STEP_TOOL_RESULT, STEP_FINISH)
            else existing.step_type
        )
        merged_step_idx = idx if step_type == STEP_FINISH else existing.step_idx
        event = ToolEvent(
            step_idx=merged_step_idx,
            step_type=merged_type,
            call_id=cid,
            tool_name=tool_name or existing.tool_name,
            args=merged_args,
            result=merged_result,
            call_step_idx=(
                existing.call_step_idx
                if existing.call_step_idx is not None
                else (idx if step_type == STEP_MODEL_TURN else None)
            ),
            result_step_idx=(
                idx
                if step_type == STEP_TOOL_RESULT
                else existing.result_step_idx
            ),
        )
      events_by_call_id[cid] = event
      if step_type == STEP_FINISH or tool_name == "finish":
        finish_event = event

  ordered_events = tuple(
      sorted(
          (events_by_call_id[cid] for cid in call_order),
          key=lambda ev: ev.step_idx,
      )
  )
  finish_payload = (
      dict(finish_event.args)
      if finish_event is not None and isinstance(finish_event.args, Mapping)
      else None
  )
  return SessionTrace(
      db_path=db_path,
      total_steps=len(rows),
      events=ordered_events,
      thought_summaries=tuple(thought_summaries),
      finish_event=finish_event,
      finish_payload=finish_payload,
      unknown_step_types=tuple(sorted(unknown_types)),
      parse_warnings=tuple(warnings),
  )


def load_session_trace(db_path: pathlib.Path | str) -> SessionTrace:
  """Opens an Antigravity SQLite session DB read-only and parses its trace.

  WARNING: Parses internal ``google-antigravity`` SQLite storage structures
  (``steps(idx, step_type, step_payload)``) that may change across SDK versions.

  Args:
    db_path: Path to a persisted ``<hash>.db`` session database.

  Returns:
    Reconstructed ``SessionTrace``.

  Raises:
    FileNotFoundError: If `db_path` does not exist.
    sqlite3.DatabaseError: If `db_path` is not a readable SQLite database.
    ValueError: If the database lacks the expected ``steps`` schema.
  """
  resolved = pathlib.Path(db_path)
  if not resolved.is_file():
    raise FileNotFoundError(f"Session DB file not found: {resolved}")

  try:
    connection = sqlite3.connect(
        f"{resolved.resolve().as_uri()}?mode=ro", uri=True
    )
  except sqlite3.OperationalError:
    logging.debug(
        "Read-only URI open failed for %s; falling back to standard connect.",
        resolved,
    )
    connection = sqlite3.connect(str(resolved))
  try:
    return load_session_trace_from_connection(connection, db_path=resolved)
  finally:
    connection.close()


def find_session_dbs(session_root: pathlib.Path | str) -> list[pathlib.Path]:
  """Finds persisted `.db` session files under `session_root`, sorted."""
  root = pathlib.Path(session_root)
  if root.is_file() and root.suffix.lower() == ".db":
    return [root]
  if not root.is_dir():
    return []
  return sorted(p for p in root.rglob("*.db") if p.is_file())


def _finish_descriptions(trace: SessionTrace) -> list[str]:
  """Extracts normalized variation descriptions from a trace's finish args."""
  if not isinstance(trace.finish_payload, Mapping):
    return []
  variations = trace.finish_payload.get("variations")
  if not isinstance(variations, list):
    return []
  return [
      _normalize_text(str(item.get("description") or ""))
      for item in variations
      if isinstance(item, Mapping)
  ]


def select_matching_session(
    traces: Sequence[SessionTrace],
    expected_descriptions: Sequence[str] = (),
) -> tuple[SessionTrace | None, str]:
  """Selects the session trace most likely to have produced the final output.

  Prefers the latest finished session whose FINISH payload descriptions match
  `expected_descriptions` exactly, then falls back to partial match, latest
  finished session, or latest unfinished session with research activity.

  Args:
    traces: Chronologically ordered ``SessionTrace`` objects.
    expected_descriptions: Optional refined proposal descriptions to match
      against FINISH payloads.

  Returns:
    Tuple ``(selected_trace, selection_method)``.
  """
  if not traces:
    return None, "no_session_trace"

  finished = [trace for trace in traces if trace.finished]
  target = sorted(
      _normalize_text(d) for d in expected_descriptions if d.strip()
  )
  if target and finished:
    exact = [
        trace
        for trace in finished
        if sorted(_finish_descriptions(trace)) == target
    ]
    if exact:
      return exact[-1], "finish_payload_exact_match"
    target_set = set(target)
    partial = [
        trace
        for trace in finished
        if target_set & set(_finish_descriptions(trace))
    ]
    if partial:
      return partial[-1], "finish_payload_partial_match"

  if finished:
    return finished[-1], "last_finished_fallback"

  return traces[-1], "unfinished_fallback"
