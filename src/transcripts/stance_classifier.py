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

"""Multi-proposal stance classification for deliberation room statements.

Classifies participant deliberation statements relative to a focal actionable
proposal into five stance labels (`support`, `critique`, `suggestion`,
`observation`, `unrelated`), enforces full-statement coverage across batched
async model calls, and formats classified evidence dossiers with endogenous
stance-count headers.
"""

import argparse
import asyncio
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import enum
import itertools
import json
import logging
import os
import re
import types
from typing import Any

from src import runner_utils
from src.models import base_model
from src.models import model_factory
from src.models import retry_utils
from src.transcripts import summarize_rooms

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_CONCURRENT_CALLS = 5
DEFAULT_MAX_RETRIES = 3

_CODE_FENCE_START_RE = re.compile(r"^```(?:json)?\s*")
_CODE_FENCE_END_RE = re.compile(r"\s*```$")
_NATURAL_SORT_RE = re.compile(r"([0-9]+)")


class StanceLabel(str, enum.Enum):
  """Stance labels for participant statements relative to a focal proposal."""

  SUPPORT = "support"
  CRITIQUE = "critique"
  SUGGESTION = "suggestion"
  OBSERVATION = "observation"
  UNRELATED = "unrelated"


STANCE_LABELS: tuple[StanceLabel, ...] = (
    StanceLabel.SUPPORT,
    StanceLabel.CRITIQUE,
    StanceLabel.SUGGESTION,
    StanceLabel.OBSERVATION,
    StanceLabel.UNRELATED,
)

RELEVANT_STANCE_LABELS: tuple[StanceLabel, ...] = (
    StanceLabel.SUPPORT,
    StanceLabel.CRITIQUE,
    StanceLabel.SUGGESTION,
    StanceLabel.OBSERVATION,
)

VALID_STANCE_VALUES: frozenset[str] = frozenset(
    label.value for label in STANCE_LABELS
)

STANCE_TITLES: Mapping[StanceLabel, str] = types.MappingProxyType({
    StanceLabel.SUPPORT: "Support",
    StanceLabel.CRITIQUE: "Critiques & Alternatives",
    StanceLabel.SUGGESTION: "Suggestions for Amendment",
    StanceLabel.OBSERVATION: "General Observations",
    StanceLabel.UNRELATED: "Unrelated",
})

STANCE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ratings"],
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["statement_id", "stance"],
                "properties": {
                    "statement_id": {"type": "string"},
                    "stance": {
                        "type": "string",
                        "enum": [label.value for label in STANCE_LABELS],
                    },
                },
            },
        }
    },
}

_CLASSIFIER_ROLE_HEADER = (
    "You are an expert deliberation analyst. Your task is to classify"
    " participant statements from a group discussion based on how each"
    " statement relates to a specific focal actionable proposal."
)

_CLASSIFICATION_RULES_LINES: tuple[str, ...] = (
    "### Classification Rules",
    (
        "For each statement below, assign exactly one of the following stance"
        " labels relative to the Focal Proposal:"
    ),
    "- `support` -- the statement supports the focal proposal.",
    (
        "- `critique` -- any reason not to adopt the focal proposal. This"
        " explicitly includes statements arguing that a companion proposal in"
        " the bundle would be preferable."
    ),
    "- `suggestion` -- a specific suggestion for amending the focal proposal.",
    (
        "- `observation` -- catch-all for anything else that still touches on"
        " the focal proposal."
    ),
    (
        "- `unrelated` -- ONLY if the statement does not touch on the focal"
        " proposal at all."
    ),
)

_CLASSIFIER_OUTPUT_INSTRUCTION = (
    "Return your classifications as a JSON object with a `ratings` array"
    " containing one `{statement_id, stance}` entry for every statement ID"
    " listed above."
)


class CoverageValidationError(ValueError):
  """Raised when model output omits or hallucinates statement IDs."""

  def __init__(
      self,
      message: str,
      *,
      missing_ids: frozenset[str] = frozenset(),
      unexpected_ids: frozenset[str] = frozenset(),
  ):
    super().__init__(message)
    self.missing_ids = missing_ids
    self.unexpected_ids = unexpected_ids


@dataclasses.dataclass(frozen=True)
class Proposal:
  """An actionable proposal evaluated during deliberation."""

  proposal_id: str
  title: str
  text: str
  topic: str = ""


@dataclasses.dataclass(frozen=True)
class DeliberationStatement:
  """A participant's summarized contribution from a deliberation room."""

  statement_id: str
  participant_id: str
  text: str
  room_id: str = ""
  topic: str = ""


@dataclasses.dataclass(frozen=True)
class ClassifiedStatement:
  """Stance classification for one statement relative to a focal proposal."""

  statement_id: str
  stance: StanceLabel


def strip_markdown_fences(raw_text: str) -> str:
  """Strips optional leading and trailing markdown code fences."""
  cleaned = raw_text.strip()
  if cleaned.startswith("```"):
    cleaned = _CODE_FENCE_START_RE.sub("", cleaned)
    cleaned = _CODE_FENCE_END_RE.sub("", cleaned)
  return cleaned.strip()


def normalize_stance_label(raw_label: str) -> StanceLabel:
  """Normalizes a raw stance string into a canonical `StanceLabel`.

  Args:
    raw_label: Raw stance string returned by a model or loaded from disk.

  Returns:
    The matching `StanceLabel` enum member.

  Raises:
    ValueError: If `raw_label` is not a recognized stance label.
  """
  if not isinstance(raw_label, str):
    raise ValueError(
        f"Stance label must be a string, got {type(raw_label).__name__}."
    )
  normalized = raw_label.strip().lower()
  try:
    return StanceLabel(normalized)
  except ValueError as exc:
    allowed = ", ".join(label.value for label in STANCE_LABELS)
    raise ValueError(
        f"Unknown stance label {raw_label!r}. Expected one of: {allowed}."
    ) from exc


def statements_from_room_summaries(
    rooms: Sequence[Mapping[str, Any]],
) -> list[DeliberationStatement]:
  """Converts loaded room summary entries into `DeliberationStatement` items.

  Args:
    rooms: Room entries such as those returned by
      `summarize_rooms.load_room_summaries`.

  Returns:
    A deterministic list of `DeliberationStatement` instances.
  """
  statements: list[DeliberationStatement] = []
  for room_idx, room in enumerate(rooms, start=1):
    room_header = str(room.get("room_header", "")).strip()
    room_id = room_header or f"room_{room_idx:02d}"
    summaries = room.get("summaries", {})
    if not isinstance(summaries, Mapping):
      continue
    for stmt_idx, (participant, text) in enumerate(summaries.items(), start=1):
      clean_participant = str(participant).strip()
      clean_text = str(text).strip() if isinstance(text, str) else ""
      if not clean_participant or not clean_text:
        continue
      statement_id = f"R{room_idx:02d}_S{stmt_idx:02d}"
      statements.append(
          DeliberationStatement(
              statement_id=statement_id,
              participant_id=clean_participant,
              text=clean_text,
              room_id=room_id,
          )
      )
  return statements


def build_statement_batches(
    statements: Sequence[DeliberationStatement],
    *,
    group_by_room: bool = True,
    max_batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[DeliberationStatement]]:
  """Partitions statements into bounded batches for stance classification.

  Args:
    statements: Deliberation statements to batch.
    group_by_room: If True, groups statements by `room_id` before chunking so
      conversational room context stays together.
    max_batch_size: Maximum number of statements per batch (must be >= 1).

  Returns:
    List of non-empty statement batches.

  Raises:
    ValueError: If `max_batch_size < 1` or duplicate statement IDs are found.
  """
  if max_batch_size < 1:
    raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}.")
  if not statements:
    return []

  seen_ids: set[str] = set()
  for stmt in statements:
    if not stmt.statement_id:
      raise ValueError("Statement ID must be a non-empty string.")
    if stmt.statement_id in seen_ids:
      raise ValueError(
          f"Duplicate statement_id detected: {stmt.statement_id!r}."
      )
    seen_ids.add(stmt.statement_id)

  batches: list[list[DeliberationStatement]] = []
  if group_by_room:
    by_room: dict[str, list[DeliberationStatement]] = collections.defaultdict(
        list
    )
    for stmt in statements:
      by_room[stmt.room_id].append(stmt)
    for room_stmts in by_room.values():
      batches.extend(
          list(batch) for batch in itertools.batched(room_stmts, max_batch_size)
      )
  else:
    batches.extend(
        list(batch) for batch in itertools.batched(statements, max_batch_size)
    )
  return batches


def build_stance_classification_prompt(
    focal_proposal: Proposal,
    statements: Sequence[DeliberationStatement],
    *,
    companion_proposals: Sequence[Proposal] = (),
) -> str:
  """Builds the prompt instructing a model to classify statement stances.

  Args:
    focal_proposal: The actionable proposal to evaluate statements against.
    statements: The batch of participant statements to classify.
    companion_proposals: Optional companion proposals deliberated in the same
      comparative bundle.

  Returns:
    Prompt text for JSON stance classification.
  """
  lines = [
      _CLASSIFIER_ROLE_HEADER,
      "",
      "### Focal Proposal",
      f"- **Title:** {focal_proposal.title}",
      f"- **Text:** {focal_proposal.text}",
      "",
  ]
  if companion_proposals:
    lines.append("### Companion Proposals (Deliberated Simultaneously)")
    for companion in companion_proposals:
      lines.append(f"- **{companion.title}:** {companion.text}")
    lines.append("")

  lines.extend(_CLASSIFICATION_RULES_LINES)
  lines.append("")
  lines.append("### Statements to Classify")
  for stmt in statements:
    clean_text = stmt.text.strip().replace("\n", " ")
    lines.append(
        f'- [ID: {stmt.statement_id}] [{stmt.participant_id}]: "{clean_text}"'
    )
  lines.extend(["", _CLASSIFIER_OUTPUT_INSTRUCTION])
  return "\n".join(lines)


def _parse_ratings_entries(raw_text: str) -> list[ClassifiedStatement]:
  """Parses JSON ratings array into `ClassifiedStatement` objects."""
  cleaned = strip_markdown_fences(raw_text)
  try:
    payload = json.loads(cleaned)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Failed to parse model output as JSON: {exc}") from exc

  if not isinstance(payload, dict) or not isinstance(
      payload.get("ratings"), list
  ):
    raise ValueError(
        "Model output must be a JSON object with a 'ratings' list."
    )

  parsed: list[ClassifiedStatement] = []
  for item in payload["ratings"]:
    if not isinstance(item, dict):
      raise ValueError(f"Each rating item must be an object, got {item!r}.")
    raw_id = item.get("statement_id")
    if not isinstance(raw_id, str) or not raw_id.strip():
      raise ValueError(f"Rating item missing valid statement_id: {item!r}.")
    raw_stance = item.get("stance")
    if raw_stance is None:
      raw_stance = item.get("category")
    if raw_stance is None:
      raise ValueError(f"Rating item missing stance label: {item!r}.")
    parsed.append(
        ClassifiedStatement(
            statement_id=raw_id.strip(),
            stance=normalize_stance_label(raw_stance),
        )
    )
  return parsed


def _index_unique_ratings(raw_text: str) -> dict[str, ClassifiedStatement]:
  """Parses ratings JSON into a unique `statement_id`-keyed mapping."""
  parsed = _parse_ratings_entries(raw_text)
  by_id: dict[str, ClassifiedStatement] = {}
  for entry in parsed:
    if entry.statement_id in by_id:
      raise ValueError(
          f"Duplicate statement_id in model response: {entry.statement_id!r}."
      )
    by_id[entry.statement_id] = entry
  return by_id


def _validate_ratings_json(raw_text: str) -> str:
  """Validates ratings JSON syntax, stance labels, and ID uniqueness."""
  _index_unique_ratings(raw_text)
  return raw_text


def parse_stance_batch_response(
    raw_text: str,
    expected_ids: Sequence[str],
) -> list[ClassifiedStatement]:
  """Parses and strictly validates one batch of stance classifications.

  Args:
    raw_text: Raw JSON string returned by the model.
    expected_ids: Ordered statement IDs expected in the response.

  Returns:
    Validated `ClassifiedStatement` instances ordered to match `expected_ids`.

  Raises:
    ValueError: If JSON decoding fails, labels are invalid, or duplicate IDs
      appear in the response.
    CoverageValidationError: If any expected statement IDs are missing or
      unexpected IDs are returned.
  """
  by_id = _index_unique_ratings(raw_text)

  expected_set = frozenset(expected_ids)
  returned_set = frozenset(by_id.keys())
  missing_ids = expected_set - returned_set
  unexpected_ids = returned_set - expected_set
  if missing_ids or unexpected_ids:
    raise CoverageValidationError(
        "Statement ID coverage mismatch."
        f" Missing: {sorted(missing_ids)};"
        f" Unexpected: {sorted(unexpected_ids)}.",
        missing_ids=missing_ids,
        unexpected_ids=unexpected_ids,
    )

  return [by_id[statement_id] for statement_id in expected_ids]


def _extract_valid_partial_ratings(
    raw_text: str,
    pending_ids: Sequence[str],
) -> list[ClassifiedStatement]:
  """Extracts unique valid ratings matching `pending_ids` from partial output.

  Returns an empty list if JSON parsing fails or if duplicate statement IDs
  appear in the response.

  Args:
    raw_text: Raw JSON string returned by the model.
    pending_ids: Statement IDs still awaiting classification.

  Returns:
    Valid `ClassifiedStatement` entries whose IDs are in `pending_ids`.
  """
  try:
    partial = _parse_ratings_entries(raw_text)
  except ValueError:
    return []

  pending_id_set = frozenset(pending_ids)
  seen_ids: set[str] = set()
  valid_partial: list[ClassifiedStatement] = []
  for entry in partial:
    if entry.statement_id in seen_ids:
      return []
    seen_ids.add(entry.statement_id)
    if entry.statement_id in pending_id_set:
      valid_partial.append(entry)
  return valid_partial


async def classify_statement_batch(
    model: base_model.BaseModel,
    focal_proposal: Proposal,
    statements: Sequence[DeliberationStatement],
    *,
    companion_proposals: Sequence[Proposal] = (),
    max_retries: int = DEFAULT_MAX_RETRIES,
    semaphore: asyncio.Semaphore | None = None,
    temperature: float = 0.0,
    retry_delay_seconds: float = 0.0,
) -> list[ClassifiedStatement]:
  """Classifies one batch of statements with strict full-coverage retry logic.

  If an attempt classifies a valid subset of the batch and omits some statement
  IDs, already-classified statements are preserved and subsequent retries query
  only the remaining missing statements.

  Args:
    model: Initialized `BaseModel` instance.
    focal_proposal: The actionable proposal to classify against.
    statements: Batch of statements to classify.
    companion_proposals: Optional companion proposals from the same bundle.
    max_retries: Maximum number of model attempts (must be >= 1).
    semaphore: Optional semaphore bounding concurrent model calls.
    temperature: Sampling temperature passed to `generate_content`.
    retry_delay_seconds: Optional sleep delay between failed attempts.

  Returns:
    List of `ClassifiedStatement` objects aligned with `statements`.

  Raises:
    ValueError: If `max_retries < 1`.
    CoverageValidationError: If any statement IDs remain unclassified after
      exhausting `max_retries`.
  """
  if max_retries < 1:
    raise ValueError(f"max_retries must be >= 1, got {max_retries}.")
  if not statements:
    return []

  classified_by_id: dict[str, ClassifiedStatement] = {}
  pending = list(statements)
  last_error: Exception | None = None

  for attempt in range(max_retries):
    prompt = build_stance_classification_prompt(
        focal_proposal,
        pending,
        companion_proposals=companion_proposals,
    )
    run_name = (
        f"stance_{focal_proposal.proposal_id}_{pending[0].statement_id}"
        f"_pass{attempt + 1}"
    )

    try:
      raw_text = await retry_utils.generate_with_retry(
          model,
          prompt=prompt,
          run_name=run_name,
          max_retries=max_retries,
          retry_delay_seconds=retry_delay_seconds,
          temperature=temperature,
          response_mime_type="application/json",
          response_schema=STANCE_RESPONSE_SCHEMA,
          semaphore=semaphore,
          parser=_validate_ratings_json,
      )
    except RuntimeError as call_err:
      last_error = call_err
      break

    pending_ids = [stmt.statement_id for stmt in pending]
    try:
      batch_results = parse_stance_batch_response(raw_text, pending_ids)
      for result in batch_results:
        classified_by_id[result.statement_id] = result
      pending = []
      break
    except CoverageValidationError as cov_err:
      last_error = cov_err
      # Preserve valid entries for pending IDs and retry only missing IDs.
      for entry in _extract_valid_partial_ratings(raw_text, pending_ids):
        classified_by_id[entry.statement_id] = entry
      pending = [
          stmt
          for stmt in statements
          if stmt.statement_id not in classified_by_id
      ]
      if not pending:
        break
      logging.warning(
          "Stance batch %s attempt %d/%d missing %d statement IDs: %s",
          run_name,
          attempt + 1,
          max_retries,
          len(pending),
          [stmt.statement_id for stmt in pending],
      )

    if attempt + 1 < max_retries and retry_delay_seconds > 0:
      await asyncio.sleep(retry_delay_seconds)

  if pending:
    missing = frozenset(stmt.statement_id for stmt in pending)
    raise CoverageValidationError(
        f"Failed to classify {len(missing)} statement(s) for proposal"
        f" {focal_proposal.proposal_id!r} after {max_retries} attempt(s)."
        f" Missing IDs: {sorted(missing)}. Last error: {last_error}",
        missing_ids=missing,
    )

  return [classified_by_id[stmt.statement_id] for stmt in statements]


async def classify_statements_for_proposal(
    model: base_model.BaseModel,
    focal_proposal: Proposal,
    statements: Sequence[DeliberationStatement],
    *,
    companion_proposals: Sequence[Proposal] = (),
    group_by_room: bool = True,
    max_batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = 0.0,
    retry_delay_seconds: float = 0.0,
) -> dict[str, ClassifiedStatement]:
  """Classifies all statements for a focal proposal across concurrent batches.

  Args:
    model: Initialized `BaseModel` instance.
    focal_proposal: The actionable proposal to classify against.
    statements: All participant statements to classify.
    companion_proposals: Optional companion proposals deliberated alongside the
      focal proposal.
    group_by_room: Whether to keep statements from the same room together.
    max_batch_size: Maximum statements per model call.
    max_concurrent_calls: Concurrency cap enforced with `asyncio.Semaphore`.
    max_retries: Maximum attempts per batch before raising on missing IDs.
    temperature: Sampling temperature passed to `generate_content`.
    retry_delay_seconds: Optional delay between retries.

  Returns:
    Mapping of `statement_id` to `ClassifiedStatement` with verified full
    coverage across `statements`.

  Raises:
    CoverageValidationError: If any statement is not classified after retries.
  """
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  batches = build_statement_batches(
      statements,
      group_by_room=group_by_room,
      max_batch_size=max_batch_size,
  )
  if not batches:
    return {}

  semaphore = asyncio.Semaphore(validated_concurrency)
  batch_results = await asyncio.gather(*(
      classify_statement_batch(
          model=model,
          focal_proposal=focal_proposal,
          statements=batch,
          companion_proposals=companion_proposals,
          max_retries=max_retries,
          semaphore=semaphore,
          temperature=temperature,
          retry_delay_seconds=retry_delay_seconds,
      )
      for batch in batches
  ))

  classifications: dict[str, ClassifiedStatement] = {}
  for batch_items in batch_results:
    for item in batch_items:
      classifications[item.statement_id] = item

  expected_ids = frozenset(stmt.statement_id for stmt in statements)
  missing_ids = expected_ids - frozenset(classifications.keys())
  if missing_ids:
    raise CoverageValidationError(
        f"Incomplete coverage for proposal {focal_proposal.proposal_id!r}."
        f" Missing IDs: {sorted(missing_ids)}.",
        missing_ids=missing_ids,
    )
  return classifications


def count_stances(
    statements: Sequence[DeliberationStatement],
    classifications: Mapping[str, ClassifiedStatement],
) -> dict[StanceLabel, int]:
  """Counts classified stances strictly across the provided statements."""
  counts = {label: 0 for label in STANCE_LABELS}
  for stmt in statements:
    classified = classifications.get(stmt.statement_id)
    if classified is not None:
      counts[classified.stance] += 1
  return counts


def _natural_sort_key(text: str) -> list[int | str]:
  """Returns a case-insensitive natural sort key splitting numeric runs."""
  return [
      int(part) if part.isascii() and part.isdigit() else part.lower()
      for part in _NATURAL_SORT_RE.split(text)
  ]


def format_classified_stance_dossier(
    focal_proposal: Proposal,
    statements: Sequence[DeliberationStatement],
    classifications: Mapping[str, ClassifiedStatement],
    *,
    companion_proposals: Sequence[Proposal] = (),
    include_unrelated: bool = False,
) -> str:
  """Renders classified statements by stance with endogenous stance counts.

  Args:
    focal_proposal: The focal actionable proposal.
    statements: Participant statements evaluated against the focal proposal.
    classifications: Mapping from `statement_id` to `ClassifiedStatement`.
    companion_proposals: Optional companion proposals from the same bundle.
    include_unrelated: Whether to include `UNRELATED` statements in the body.

  Returns:
    Markdown dossier grouped by stance category (and room when present).
  """
  grouped: dict[StanceLabel, list[DeliberationStatement]] = (
      collections.defaultdict(list)
  )
  for stmt in statements:
    classified = classifications.get(stmt.statement_id)
    if classified is not None:
      grouped[classified.stance].append(stmt)

  stance_counts = count_stances(statements, classifications)
  relevant_total = sum(stance_counts[label] for label in RELEVANT_STANCE_LABELS)
  unrelated_count = stance_counts[StanceLabel.UNRELATED]

  lines = [
      (
          f"## Focal Proposal: {focal_proposal.title}"
          f" (`{focal_proposal.proposal_id}`)"
      ),
      "",
      f'- **Actionable Proposal:** "{focal_proposal.text}"',
  ]
  if focal_proposal.topic:
    lines.append(f"- **Topic:** {focal_proposal.topic}")

  if companion_proposals:
    lines.extend(["", "### Companion Proposals"])
    for companion in companion_proposals:
      lines.append(
          f"- **{companion.title}** (`{companion.proposal_id}`):"
          f' "{companion.text}"'
      )

  summary_parts = ", ".join(
      f"{stance_counts[label]} {label.value.title()}"
      for label in RELEVANT_STANCE_LABELS
  )
  unrelated_note = (
      f"{unrelated_count} Unrelated included"
      if include_unrelated
      else f"{unrelated_count} Unrelated excluded"
  )
  lines.extend([
      "",
      (
          f"### Classified Participant Statements ({relevant_total}"
          f" proposal-relevant: {summary_parts}; {unrelated_note})"
      ),
      "",
  ])

  labels_to_render = (
      STANCE_LABELS if include_unrelated else RELEVANT_STANCE_LABELS
  )
  for stance in labels_to_render:
    stance_stmts = grouped.get(stance, [])
    if not stance_stmts:
      continue
    lines.append(
        f"#### {STANCE_TITLES[stance]} ({len(stance_stmts)} statements)"
    )
    lines.append("")
    has_rooms = any(stmt.room_id for stmt in stance_stmts)
    if has_rooms:
      by_room: dict[str, list[DeliberationStatement]] = collections.defaultdict(
          list
      )
      for stmt in stance_stmts:
        by_room[stmt.room_id or "Unspecified Room"].append(stmt)
      for room_id in sorted(by_room, key=_natural_sort_key):
        lines.append(f"**{room_id}**")
        for stmt in by_room[room_id]:
          clean_text = stmt.text.strip().replace("\n", " ")
          lines.append(f'- **[{stmt.participant_id}]:** "{clean_text}"')
        lines.append("")
    else:
      for stmt in stance_stmts:
        clean_text = stmt.text.strip().replace("\n", " ")
        lines.append(f'- **[{stmt.participant_id}]:** "{clean_text}"')
      lines.append("")

  return "\n".join(lines).strip()


def _load_proposals(path: str | os.PathLike[str]) -> list[Proposal]:
  """Loads actionable proposals from a JSON list file."""
  with open(path, "r") as f:
    raw_items = json.load(f)
  if not isinstance(raw_items, list):
    raise ValueError("Proposals JSON must contain a top-level list.")
  proposals: list[Proposal] = []
  for item in raw_items:
    if not isinstance(item, dict):
      raise ValueError(f"Expected proposal object, got {item!r}.")
    proposals.append(
        Proposal(
            proposal_id=str(item["proposal_id"]),
            title=str(item["title"]),
            text=str(item["text"]),
            topic=str(item.get("topic", "")),
        )
    )
  return proposals


def main() -> None:
  """Command-line entry point for multi-proposal stance classification."""
  parser = argparse.ArgumentParser(
      description=(
          "Classify deliberation room statements by stance per proposal."
      )
  )
  parser.add_argument(
      "--room_summaries",
      type=str,
      required=True,
      help="Path to room summaries JSON from summarize_rooms.",
  )
  parser.add_argument(
      "--proposals",
      type=str,
      required=True,
      help="Path to JSON list of actionable proposals.",
  )
  parser.add_argument(
      "--proposal_id",
      type=str,
      default=None,
      help="Optional single proposal_id to classify; defaults to all.",
  )
  parser.add_argument(
      "--output",
      type=str,
      required=True,
      help="Path to write JSON stance classifications.",
  )
  parser.add_argument(
      "--model",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help="Name of the model to use.",
  )
  parser.add_argument(
      "--max_batch_size",
      type=int,
      default=DEFAULT_BATCH_SIZE,
      help="Maximum statements per model call.",
  )
  parser.add_argument(
      "--max_concurrent_calls",
      type=runner_utils.max_concurrent_calls_arg,
      default=DEFAULT_MAX_CONCURRENT_CALLS,
      help="Maximum concurrent model calls.",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  rooms = summarize_rooms.load_room_summaries(args.room_summaries)
  statements = statements_from_room_summaries(rooms)
  proposals = _load_proposals(args.proposals)
  if args.proposal_id is not None:
    target_proposals = [
        prop for prop in proposals if prop.proposal_id == args.proposal_id
    ]
    if not target_proposals:
      raise ValueError(f"Proposal ID {args.proposal_id!r} not found.")
  else:
    target_proposals = proposals

  model = model_factory.get_model(args.model)

  async def _run_all() -> dict[str, dict[str, str]]:
    all_output: dict[str, dict[str, str]] = {}
    for focal in target_proposals:
      companions = [
          prop for prop in proposals if prop.proposal_id != focal.proposal_id
      ]
      classified = await classify_statements_for_proposal(
          model=model,
          focal_proposal=focal,
          statements=statements,
          companion_proposals=companions,
          max_batch_size=args.max_batch_size,
          max_concurrent_calls=args.max_concurrent_calls,
      )
      all_output[focal.proposal_id] = {
          stmt_id: entry.stance.value for stmt_id, entry in classified.items()
      }
    return all_output

  results = asyncio.run(_run_all())
  output_dir = os.path.dirname(args.output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
  logging.info("Saved stance classifications to %s", args.output)


if __name__ == "__main__":
  main()
