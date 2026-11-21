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

"""Segment-conditioned clustering, opinion trees, and worldview synthesis.

Generalizes hierarchical sub-opinion tree discovery and segment-conditioned
deliberation evidence builders onto polymorphic `BaseModel` instances.
Supports arbitrary participant segmentations (e.g., demographic cohorts or
unsupervised opinion clusters) via `SegmentSpec` and optional
`segmentation_context`.
"""

import argparse
import asyncio
import collections
from collections import abc
import dataclasses
import enum
import json
import logging
import os
from typing import Any, TextIO

import pandas as pd

from src import input_parsing_lib
from src import runner_utils
from src.models import base_model
from src.models import model_factory
from src.models import retry_utils
from src.transcripts import stance_classifier
from src.transcripts import summarize_rooms

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
DEFAULT_ASSIGNMENT_BATCH_SIZE = 15
DEFAULT_MAX_CONCURRENT_CALLS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_SMALL_CLUSTER_THRESHOLD = 3
DEFAULT_TOP_QUOTES_PER_CLUSTER = 6
DEFAULT_MAX_STATEMENTS_PER_STANCE = 6

OPINION_DISCOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["opinions"],
    "properties": {
        "opinions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["opinion_id", "opinion_statement"],
                "properties": {
                    "opinion_id": {"type": "string"},
                    "opinion_statement": {"type": "string"},
                },
            },
        }
    },
}

MULTI_ASSIGNMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["assignments"],
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["statement_id", "assigned_opinions"],
                "properties": {
                    "statement_id": {"type": "string"},
                    "assigned_opinions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        }
    },
}

CLUSTER_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["summary"],
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Concise 1-2 sentence summary of participant reasoning."
            ),
        }
    },
}

SMALL_SEGMENT_CLUSTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["opinion_statement", "summary"],
    "properties": {
        "opinion_statement": {"type": "string"},
        "summary": {"type": "string"},
    },
}

SEGMENT_CLUSTERING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["clusters"],
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "opinion_id",
                    "opinion_statement",
                    "summary",
                    "statement_ids",
                ],
                "properties": {
                    "opinion_id": {"type": "string"},
                    "opinion_statement": {"type": "string"},
                    "summary": {"type": "string"},
                    "statement_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "other_statement_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "other_summary": {"type": "string"},
    },
}

WORLDVIEW_SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "worldview_and_support_drivers",
        "primary_friction_points",
        "key_conditions_for_support",
    ],
    "properties": {
        "worldview_and_support_drivers": {"type": "string"},
        "primary_friction_points": {
            "type": "array",
            "items": {"type": "string"},
        },
        "key_conditions_for_support": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_DISCOVERY_INSTRUCTIONS: tuple[str, ...] = (
    "## Instructions",
    (
        "1. Analyze the statements above and discover 3 to 6 distinct,"
        " recurring sub-opinions or themes (or 1 to 3 when fewer than 6"
        " statements are present)."
    ),
    (
        "2. Each sub-opinion must capture a coherent stance, recurring"
        " priority, or operational rationale expressed by participants."
    ),
    (
        "3. Assign a short identifier (e.g., 'op_1', 'op_2') and a concise,"
        " declarative opinion statement."
    ),
    (
        "4. Do NOT include an 'Other' opinion; residual remarks are assigned"
        " to an implicit 'Other' bucket in the next stage."
    ),
    (
        "5. Return strict JSON matching the schema with a non-empty"
        " 'opinions' array."
    ),
)

_ASSIGNMENT_INSTRUCTIONS: tuple[str, ...] = (
    "## Instructions",
    (
        "1. For each statement, determine which candidate sub-opinion IDs it"
        " expresses or supports."
    ),
    (
        "2. A statement may be assigned to multiple opinion IDs when it"
        " expresses multiple distinct themes."
    ),
    (
        "3. If a statement does not clearly align with any discovered"
        " sub-opinion, assign ['other']."
    ),
    (
        "4. Return strict JSON with an 'assignments' array of"
        " {statement_id, assigned_opinions} objects."
    ),
)

_SUMMARY_INSTRUCTIONS: tuple[str, ...] = (
    "## Instructions",
    (
        "1. Synthesize a concise 1-2 sentence summary explaining why"
        " participants hold this viewpoint and their core reasoning or"
        " trade-offs."
    ),
    (
        "2. Ground the summary strictly in the provided statements without"
        " inventing outside facts."
    ),
    '3. Return strict JSON: {"summary": "<1-2 sentences>"}.',
)

_WORLDVIEW_INSTRUCTIONS: tuple[str, ...] = (
    "## Synthesis Instructions",
    (
        "Synthesize this segment's perspective into an executive profile with"
        " three fields:"
    ),
    (
        "1. 'worldview_and_support_drivers': A 2-3 sentence paragraph"
        " summarizing what participants in this segment value about the"
        " actionable proposal and want preserved."
    ),
    (
        "2. 'primary_friction_points': 2 to 4 concise items detailing their"
        " specific objections, risk concerns, equity questions, or trade-off"
        " redlines."
    ),
    (
        "3. 'key_conditions_for_support': 2 to 4 concise items detailing"
        " concrete amendments, safeguards, or conditions that would broaden"
        " their support."
    ),
    "Ground every claim strictly in the recorded statements above.",
)


class SegmentViewStructure(str, enum.Enum):
  """Supported evidence layouts for segment-conditioned views."""

  FLAT_QUOTES = "flat_quotes"
  SUBTREE_ALL_QUOTES = "subtree_all_quotes"
  SUBTREE_TOP_QUOTES = "subtree_top_quotes"
  WORLDVIEW_SYNTHESIS = "worldview_synthesis"


@dataclasses.dataclass(frozen=True)
class SegmentSpec:
  """Metadata describing a participant segment."""

  id: str
  label: str
  description: str = ""


@dataclasses.dataclass(frozen=True)
class DiscoveredOpinion:
  """A discovered sub-opinion theme within a stance category."""

  opinion_id: str
  opinion_statement: str


@dataclasses.dataclass(frozen=True)
class SubOpinionCluster:
  """A cluster of statements grouped under a sub-opinion with a summary."""

  opinion_id: str
  opinion_statement: str
  summary: str
  statement_ids: tuple[str, ...]
  is_other: bool = False


@dataclasses.dataclass(frozen=True)
class StanceOpinionTree:
  """Hierarchical sub-opinion tree for one stance category."""

  stance: stance_classifier.StanceLabel
  stance_title: str
  total_statements: int
  clusters: tuple[SubOpinionCluster, ...]


@dataclasses.dataclass(frozen=True)
class ProposalOpinionTree:
  """Complete hierarchical sub-opinion tree for a focal proposal."""

  proposal_id: str
  categories: tuple[StanceOpinionTree, ...]


@dataclasses.dataclass(frozen=True)
class SegmentWorldviewProfile:
  """Synthesized worldview, friction points, and conditions for one segment."""

  segment_id: str
  worldview_and_support_drivers: str
  primary_friction_points: tuple[str, ...]
  key_conditions_for_support: tuple[str, ...]


StanceStatements = abc.Mapping[
    stance_classifier.StanceLabel,
    abc.Sequence[stance_classifier.DeliberationStatement],
]
SegmentStanceStatements = abc.Mapping[str, StanceStatements]
SegmentSubtrees = abc.Mapping[
    str,
    abc.Mapping[stance_classifier.StanceLabel, abc.Sequence[SubOpinionCluster]],
]


def _validate_unique_segment_specs(
    specs: abc.Sequence[SegmentSpec],
) -> list[SegmentSpec]:
  """Validates that every segment spec has a non-empty unique ID."""
  validated: list[SegmentSpec] = []
  seen_ids: set[str] = set()
  for spec in specs:
    clean_id = spec.id.strip()
    clean_label = spec.label.strip() or clean_id
    clean_desc = spec.description.strip()
    if not clean_id:
      raise ValueError("SegmentSpec.id must be a non-empty string.")
    if clean_id in seen_ids:
      raise ValueError(f"Duplicate SegmentSpec id detected: {clean_id!r}.")
    seen_ids.add(clean_id)
    validated.append(
        SegmentSpec(id=clean_id, label=clean_label, description=clean_desc)
    )
  return validated


def segment_specs_from_columns(columns: abc.Sequence[str]) -> list[SegmentSpec]:
  """Auto-derives bare `SegmentSpec(id=col, label=col)` entries from names.

  Args:
    columns: Column or segment names.

  Returns:
    Validated list of `SegmentSpec` objects preserving input order.

  Raises:
    ValueError: If any column name is blank or duplicated.
  """
  raw_specs = [
      SegmentSpec(id=str(col).strip(), label=str(col).strip())
      for col in columns
  ]
  return _validate_unique_segment_specs(raw_specs)


def _parse_segment_spec_collection(raw: Any) -> list[SegmentSpec]:
  """Parses a list or mapping of raw segment definitions into `SegmentSpec`s."""
  specs: list[SegmentSpec] = []
  if isinstance(raw, abc.Mapping):
    for seg_id, entry in raw.items():
      if isinstance(entry, str):
        specs.append(SegmentSpec(id=str(seg_id), label=entry))
      elif isinstance(entry, abc.Mapping):
        specs.append(
            SegmentSpec(
                id=str(entry.get("id", seg_id)),
                label=str(entry.get("label", seg_id)),
                description=str(entry.get("description", "")),
            )
        )
      else:
        raise ValueError(
            f"Unsupported segment specification for {seg_id!r}: {entry!r}."
        )
  elif isinstance(raw, abc.Sequence) and not isinstance(raw, (str, bytes)):
    for item in raw:
      if isinstance(item, str):
        specs.append(SegmentSpec(id=item, label=item))
      elif isinstance(item, abc.Mapping):
        if "id" not in item:
          raise ValueError(f"Segment spec item missing 'id': {item!r}.")
        seg_id = str(item["id"])
        specs.append(
            SegmentSpec(
                id=seg_id,
                label=str(item.get("label", seg_id)),
                description=str(item.get("description", "")),
            )
        )
      else:
        raise ValueError(f"Unsupported segment spec entry: {item!r}.")
  else:
    raise ValueError(
        "Segment specifications must be a mapping or sequence of objects."
    )
  return _validate_unique_segment_specs(specs)


def load_segment_specs(stream: TextIO) -> tuple[list[SegmentSpec], str]:
  """Loads `SegmentSpec` definitions and optional `segmentation_context`.

  Reads a JSON payload from `stream` in any of the following shapes:
    - A mapping with `"segments"` (list or dict) and optional
      `"segmentation_context"` string.
    - A flat mapping of `segment_id -> label` or `segment_id -> {label, ...}`.
    - A sequence of `{id, label, description?}` dicts.

  Args:
    stream: Readable text stream containing JSON segment specifications.

  Returns:
    A `(segment_specs, segmentation_context)` tuple.

  Raises:
    ValueError: If the JSON or specification format is invalid, or IDs are
      duplicated.
  """
  raw_payload = json.load(stream)

  if isinstance(raw_payload, abc.Mapping) and "segments" in raw_payload:
    context = str(raw_payload.get("segmentation_context", "")).strip()
    specs = _parse_segment_spec_collection(raw_payload["segments"])
    return specs, context

  specs = _parse_segment_spec_collection(raw_payload)
  return specs, ""


def map_participants_to_segments(
    participants_df: pd.DataFrame,
    participant_id_column: str,
    *,
    segment_column: str | None = None,
    segment_columns: abc.Sequence[str] | None = None,
    segment_specs: abc.Sequence[SegmentSpec] | None = None,
) -> tuple[dict[str, frozenset[str]], list[SegmentSpec]]:
  """Maps participants to segments from categorical or binary columns.

  Exactly one of `segment_column` (single categorical column) or
  `segment_columns` (one or more overlapping binary indicator columns) must be
  provided.

  Args:
    participants_df: DataFrame containing participant IDs and segment columns.
    participant_id_column: Column holding participant identifiers.
    segment_column: Name of a single categorical segment column.
    segment_columns: Names of overlapping binary indicator columns.
    segment_specs: Optional explicit `SegmentSpec` list. When omitted, bare
      specs are derived automatically from the discovered segment values or
      column names.

  Returns:
    A tuple of `(participant_to_segments, resolved_segment_specs)`.

  Raises:
    ValueError: If column arguments are missing, ambiguous, or absent from
      `participants_df`, or if a binary column contains non-binary values.
  """
  if (segment_column is None) == (segment_columns is None):
    raise ValueError(
        "Specify exactly one of 'segment_column' or 'segment_columns'."
    )
  if participant_id_column not in participants_df.columns:
    raise ValueError(
        f"Participant ID column {participant_id_column!r} not found in"
        " DataFrame."
    )

  memberships: dict[str, set[str]] = collections.defaultdict(set)

  if segment_column is not None:
    if segment_column not in participants_df.columns:
      raise ValueError(
          f"Segment column {segment_column!r} not found in DataFrame."
      )
    if segment_specs is not None:
      resolved_specs = _validate_unique_segment_specs(segment_specs)
      allowed_ids = frozenset(spec.id for spec in resolved_specs)
    else:
      cleaned_vals = (
          participants_df[segment_column].dropna().astype(str).str.strip()
      )
      unique_vals = (
          cleaned_vals[cleaned_vals != ""]
          .drop_duplicates()
          .sort_values()
          .tolist()
      )
      resolved_specs = segment_specs_from_columns(unique_vals)
      allowed_ids = frozenset(spec.id for spec in resolved_specs)

    for raw_pid, raw_seg in participants_df[
        [participant_id_column, segment_column]
    ].itertuples(index=False, name=None):
      if pd.isna(raw_pid):
        continue
      pid = str(raw_pid).strip()
      if not pid:
        continue
      memberships.setdefault(pid, set())
      if pd.isna(raw_seg):
        continue
      seg_val = str(raw_seg).strip()
      if seg_val and seg_val in allowed_ids:
        memberships[pid].add(seg_val)
  else:
    assert segment_columns is not None
    if not segment_columns:
      raise ValueError("segment_columns must contain at least one column name.")
    missing_cols = [
        col for col in segment_columns if col not in participants_df.columns
    ]
    if missing_cols:
      raise ValueError(
          f"Segment columns not found in DataFrame: {missing_cols}."
      )

    if segment_specs is not None:
      resolved_specs = _validate_unique_segment_specs(segment_specs)
      allowed_ids = frozenset(spec.id for spec in resolved_specs)
    else:
      resolved_specs = segment_specs_from_columns(segment_columns)
      allowed_ids = frozenset(spec.id for spec in resolved_specs)

    active_cols = [col for col in segment_columns if col in allowed_ids]
    for row_idx in range(participants_df.shape[0]):
      raw_pid = participants_df[participant_id_column].iloc[row_idx]
      if pd.isna(raw_pid):
        continue
      pid = str(raw_pid).strip()
      if not pid:
        continue
      memberships.setdefault(pid, set())
      for col in active_cols:
        cell = participants_df[col].iloc[row_idx]
        if input_parsing_lib.parse_binary_indicator(cell, col):
          memberships[pid].add(col)

  frozen_map = {pid: frozenset(segs) for pid, segs in memberships.items()}
  return frozen_map, resolved_specs


def compute_segment_counts(
    participant_to_segments: abc.Mapping[str, abc.Set[str]],
    segments: abc.Sequence[SegmentSpec],
) -> dict[str, int]:
  """Computes participant counts per segment from membership mappings."""
  return {
      spec.id: sum(spec.id in segs for segs in participant_to_segments.values())
      for spec in segments
  }


def partition_statements_by_segment_and_stance(
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    classifications: abc.Mapping[str, stance_classifier.ClassifiedStatement],
    participant_to_segments: abc.Mapping[str, abc.Set[str]],
    segments: abc.Sequence[SegmentSpec],
) -> dict[
    str,
    dict[
        stance_classifier.StanceLabel,
        list[stance_classifier.DeliberationStatement],
    ],
]:
  """Partitions proposal-relevant statements by segment ID and stance label.

  Excludes `UNRELATED`, unclassified statements, and participants without
  segment memberships. Overlapping segments receive every statement from their
  member participants.

  Args:
    statements: Participant deliberation statements.
    classifications: Mapping of `statement_id` to `ClassifiedStatement`.
    participant_to_segments: Mapping of `participant_id` to segment ID sets.
    segments: Ordered `SegmentSpec` definitions to populate.

  Returns:
    Nested dictionary `result[segment_id][stance] -> list[statement]`.
  """
  relevant_set = frozenset(stance_classifier.RELEVANT_STANCE_LABELS)
  partitioned = {
      spec.id: {
          stance: [] for stance in stance_classifier.RELEVANT_STANCE_LABELS
      }
      for spec in segments
  }

  for stmt in statements:
    member_segs = participant_to_segments.get(stmt.participant_id)
    if not member_segs:
      continue
    classified = classifications.get(stmt.statement_id)
    if classified is None or classified.stance not in relevant_set:
      continue
    for seg_id in member_segs:
      if seg_id in partitioned:
        partitioned[seg_id][classified.stance].append(stmt)
  return partitioned


def _parse_json_object(raw_text: str) -> dict[str, Any]:
  """Extracts and parses a JSON object from model output text."""
  cleaned = stance_classifier.strip_markdown_fences(raw_text)
  start = cleaned.find("{")
  end = cleaned.rfind("}")
  if start != -1 and end >= start:
    cleaned = cleaned[start : end + 1]
  try:
    parsed = json.loads(cleaned)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Failed to decode JSON object: {exc}") from exc
  if not isinstance(parsed, dict):
    raise ValueError(
        f"Expected JSON object from model, got {type(parsed).__name__}."
    )
  return parsed


async def _call_json_model(
    model: base_model.BaseModel,
    prompt: str,
    run_name: str,
    response_schema: dict[str, Any],
    *,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = 0.0,
    retry_delay_seconds: float = 0.0,
) -> dict[str, Any]:
  """Calls `BaseModel.generate_content` with retry and JSON validation."""
  return await retry_utils.generate_with_retry(
      model,
      prompt=prompt,
      run_name=run_name,
      max_retries=max_retries,
      retry_delay_seconds=retry_delay_seconds,
      temperature=temperature,
      response_mime_type="application/json",
      response_schema=response_schema,
      semaphore=semaphore,
      parser=_parse_json_object,
  )


def _format_statement_bullet(
    stmt: stance_classifier.DeliberationStatement,
    *,
    include_statement_id: bool = False,
) -> str:
  """Formats one participant statement as a single-line markdown bullet."""
  clean_text = stmt.text.strip().replace("\n", " ")
  id_prefix = (
      f"[Statement ID: {stmt.statement_id}] " if include_statement_id else ""
  )
  if stmt.room_id:
    attribution = f"[{stmt.room_id}, {stmt.participant_id}]"
  else:
    attribution = f"[{stmt.participant_id}]"
  return f'- {id_prefix}{attribution}: "{clean_text}"'


def _format_segment_context_lines(
    segment: SegmentSpec,
    segmentation_context: str = "",
) -> list[str]:
  """Formats segment metadata and segmentation context for LLM prompts."""
  lines = [f"## Participant Segment: {segment.label} (`{segment.id}`)"]
  if segment.description:
    lines.append(f"Segment Description: {segment.description}")
  if segmentation_context.strip():
    lines.append(f"Segmentation Context: {segmentation_context.strip()}")
  return lines


# ---------------------------------------------------------------------------
# Per-Proposal Hierarchical Sub-Opinion Trees
# ---------------------------------------------------------------------------


def build_opinion_discovery_prompt(
    focal_proposal: stance_classifier.Proposal,
    stance: stance_classifier.StanceLabel,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
) -> str:
  """Builds the Step A prompt to discover 3-6 sub-opinions in a stance."""
  stance_title = stance_classifier.STANCE_TITLES[stance]
  lines = [
      "# Task: Deliberation Sub-Opinion Discovery",
      "",
      (
          "You are an expert deliberation analyst examining participant"
          " feedback on an actionable proposal."
      ),
      "",
      f"## Focal Proposal: {focal_proposal.title}",
      f"Proposal Text: {focal_proposal.text}",
  ]
  if focal_proposal.topic:
    lines.append(f"Deliberation Topic: {focal_proposal.topic}")
  lines.extend([
      "",
      f"## Stance Category: {stance_title} ({len(statements)} statements)",
      "Below are participant statements classified into this stance category:",
      "",
  ])
  for stmt in statements:
    lines.append(_format_statement_bullet(stmt, include_statement_id=True))
  lines.append("")
  lines.extend(_DISCOVERY_INSTRUCTIONS)
  return "\n".join(lines)


def build_multi_assignment_prompt(
    focal_proposal: stance_classifier.Proposal,
    stance: stance_classifier.StanceLabel,
    opinions: abc.Sequence[DiscoveredOpinion],
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
) -> str:
  """Builds the Step B prompt to assign statements to sub-opinions or Other."""
  stance_title = stance_classifier.STANCE_TITLES[stance]
  lines = [
      "# Task: Multi-Assignment Sub-Opinion Classification",
      "",
      (
          "You are an expert deliberation analyst assigning participant"
          " statements to discovered sub-opinion clusters."
      ),
      "",
      f"## Focal Proposal: {focal_proposal.title}",
      f"Proposal Text: {focal_proposal.text}",
      f"## Stance Category: {stance_title}",
      "",
      "## Candidate Sub-Opinions",
  ]
  for opinion in opinions:
    lines.append(f"- {opinion.opinion_id}: {opinion.opinion_statement}")
  lines.append(
      "- other: Residual or general remarks not matching the options above"
  )
  lines.extend([
      "",
      f"## Statements to Assign ({len(statements)} statements)",
  ])
  for stmt in statements:
    lines.append(_format_statement_bullet(stmt, include_statement_id=True))
  lines.append("")
  lines.extend(_ASSIGNMENT_INSTRUCTIONS)
  return "\n".join(lines)


def build_cluster_summary_prompt(
    focal_proposal: stance_classifier.Proposal,
    stance: stance_classifier.StanceLabel,
    opinion: DiscoveredOpinion,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
) -> str:
  """Builds the Step C prompt to summarize participant reasoning per cluster."""
  stance_title = stance_classifier.STANCE_TITLES[stance]
  lines = [
      "# Task: Opinion Cluster Reasoning Synthesis",
      "",
      (
          "You are an expert deliberation analyst summarizing participant"
          " reasoning for one sub-opinion cluster."
      ),
      "",
      f"## Focal Proposal: {focal_proposal.title}",
      f"Proposal Text: {focal_proposal.text}",
      f"## Stance Category: {stance_title}",
      f"## Sub-Opinion: {opinion.opinion_statement}",
      f"Assigned Statements: {len(statements)}",
      "",
      "## Constituent Statements",
  ]
  for stmt in statements:
    lines.append(_format_statement_bullet(stmt))
  lines.append("")
  lines.extend(_SUMMARY_INSTRUCTIONS)
  return "\n".join(lines)


def parse_discovered_opinions(
    payload: abc.Mapping[str, Any],
) -> list[DiscoveredOpinion]:
  """Parses discovered sub-opinions from a Step A JSON payload."""
  raw_opinions = payload.get("opinions", [])
  if not isinstance(raw_opinions, list):
    return []
  opinions: list[DiscoveredOpinion] = []
  seen_ids: set[str] = set()
  for item in raw_opinions:
    if not isinstance(item, abc.Mapping):
      continue
    op_id = str(item.get("opinion_id", "")).strip()
    op_stmt = str(item.get("opinion_statement", "")).strip()
    if not op_id or not op_stmt or op_id.lower() == "other":
      continue
    if op_id in seen_ids:
      continue
    seen_ids.add(op_id)
    opinions.append(
        DiscoveredOpinion(opinion_id=op_id, opinion_statement=op_stmt)
    )
  return opinions


def _sort_clusters(
    clusters: abc.Sequence[SubOpinionCluster],
) -> tuple[SubOpinionCluster, ...]:
  """Sorts regular clusters by descending quote count with Other last.

  Relies on Python's guaranteed-stable Timsort: sorting from least- to
  most-significant key (`opinion_id` asc, quote count desc, `is_other` asc)
  preserves tie-breaking order across mixed sort directions.
  """
  ordered = sorted(clusters, key=lambda c: c.opinion_id)
  ordered.sort(key=lambda c: len(c.statement_ids), reverse=True)
  ordered.sort(key=lambda c: c.is_other)
  return tuple(ordered)


async def discover_sub_opinions(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    stance: stance_classifier.StanceLabel,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    *,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> list[DiscoveredOpinion]:
  """Discovers 3-6 recurring sub-opinions for one stance category."""
  if not statements:
    return []
  prompt = build_opinion_discovery_prompt(focal_proposal, stance, statements)
  payload = await _call_json_model(
      model=model,
      prompt=prompt,
      run_name=f"discover_{focal_proposal.proposal_id}_{stance.value}",
      response_schema=OPINION_DISCOVERY_SCHEMA,
      semaphore=semaphore,
      max_retries=max_retries,
      retry_delay_seconds=retry_delay_seconds,
  )
  opinions = parse_discovered_opinions(payload)
  if opinions:
    return opinions
  stance_title = stance_classifier.STANCE_TITLES[stance]
  return [
      DiscoveredOpinion(
          opinion_id="op_1",
          opinion_statement=f"General {stance_title.lower()} perspectives",
      )
  ]


async def assign_statements_to_opinions(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    stance: stance_classifier.StanceLabel,
    opinions: abc.Sequence[DiscoveredOpinion],
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    *,
    batch_size: int = DEFAULT_ASSIGNMENT_BATCH_SIZE,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> dict[str, tuple[str, ...]]:
  """Multi-assigns statements to sub-opinions or residual `'other'`."""
  if batch_size < 1:
    raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
  if not statements:
    return {}

  valid_ids = frozenset(opinion.opinion_id for opinion in opinions)
  chunks = [
      statements[start : start + batch_size]
      for start in range(0, len(statements), batch_size)
  ]

  payloads = await asyncio.gather(*(
      _call_json_model(
          model=model,
          prompt=build_multi_assignment_prompt(
              focal_proposal, stance, opinions, chunk
          ),
          run_name=f"assign_{focal_proposal.proposal_id}_{stance.value}_b{idx}",
          response_schema=MULTI_ASSIGNMENT_SCHEMA,
          semaphore=semaphore,
          max_retries=max_retries,
          retry_delay_seconds=retry_delay_seconds,
      )
      for idx, chunk in enumerate(chunks)
  ))

  assigned_by_stmt: dict[str, list[str]] = collections.defaultdict(list)
  for payload in payloads:
    raw_assignments = payload.get("assignments", [])
    if not isinstance(raw_assignments, list):
      continue
    for item in raw_assignments:
      if not isinstance(item, abc.Mapping):
        continue
      stmt_id = item.get("statement_id")
      raw_ops = item.get("assigned_opinions", [])
      if not isinstance(stmt_id, str) or not isinstance(raw_ops, list):
        continue
      for raw_op in raw_ops:
        op_str = str(raw_op).strip()
        if op_str in valid_ids:
          assigned_by_stmt[stmt_id].append(op_str)
        elif op_str.lower() == "other":
          assigned_by_stmt[stmt_id].append("other")

  # Guarantee 100% statement coverage by routing unassigned items to 'other'.
  final_assignments: dict[str, tuple[str, ...]] = {}
  for stmt in statements:
    deduped = tuple(dict.fromkeys(assigned_by_stmt.get(stmt.statement_id, ())))
    final_assignments[stmt.statement_id] = deduped or ("other",)
  return final_assignments


async def summarize_opinion_cluster(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    stance: stance_classifier.StanceLabel,
    opinion: DiscoveredOpinion,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    *,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> str:
  """Synthesizes a 1-2 sentence summary for one sub-opinion cluster."""
  if not statements:
    return ""
  prompt = build_cluster_summary_prompt(
      focal_proposal, stance, opinion, statements
  )
  payload = await _call_json_model(
      model=model,
      prompt=prompt,
      run_name=(
          f"summarize_{focal_proposal.proposal_id}_{stance.value}"
          f"_{opinion.opinion_id}"
      ),
      response_schema=CLUSTER_SUMMARY_SCHEMA,
      semaphore=semaphore,
      max_retries=max_retries,
      retry_delay_seconds=retry_delay_seconds,
  )
  summary = str(payload.get("summary", "")).strip()
  if summary:
    return summary
  return f"Participant reasoning regarding {opinion.opinion_statement}."


async def build_proposal_opinion_tree(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    classifications: abc.Mapping[str, stance_classifier.ClassifiedStatement],
    *,
    batch_size: int = DEFAULT_ASSIGNMENT_BATCH_SIZE,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> ProposalOpinionTree:
  """Builds a hierarchical sub-opinion tree across all non-empty stances.

  Args:
    model: Initialized `BaseModel` instance.
    focal_proposal: Focal actionable proposal.
    statements: Deliberation statements to cluster.
    classifications: Stance classifications keyed by `statement_id`.
    batch_size: Max statements per Step B multi-assignment request.
    max_concurrent_calls: Semaphore cap on concurrent model requests.
    max_retries: Maximum retries per model call.
    retry_delay_seconds: Optional sleep delay between retries.

  Returns:
    A deterministic `ProposalOpinionTree` with `Other` sorted last per stance.
  """
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  relevant_set = frozenset(stance_classifier.RELEVANT_STANCE_LABELS)
  grouped: dict[
      stance_classifier.StanceLabel,
      list[stance_classifier.DeliberationStatement],
  ] = collections.defaultdict(list)
  for stmt in statements:
    classified = classifications.get(stmt.statement_id)
    if classified is not None and classified.stance in relevant_set:
      grouped[classified.stance].append(stmt)

  active_stances = [
      stance
      for stance in stance_classifier.RELEVANT_STANCE_LABELS
      if grouped.get(stance)
  ]
  if not active_stances:
    return ProposalOpinionTree(
        proposal_id=focal_proposal.proposal_id, categories=()
    )

  semaphore = asyncio.Semaphore(validated_concurrency)

  # Step A: Discover sub-opinions concurrently across active stances.
  discovered_list = await asyncio.gather(*(
      discover_sub_opinions(
          model=model,
          focal_proposal=focal_proposal,
          stance=stance,
          statements=grouped[stance],
          semaphore=semaphore,
          max_retries=max_retries,
          retry_delay_seconds=retry_delay_seconds,
      )
      for stance in active_stances
  ))
  opinions_by_stance = dict(zip(active_stances, discovered_list))

  # Step B: Multi-assign statements to discovered sub-opinions or Other.
  assignments_list = await asyncio.gather(*(
      assign_statements_to_opinions(
          model=model,
          focal_proposal=focal_proposal,
          stance=stance,
          opinions=opinions_by_stance[stance],
          statements=grouped[stance],
          batch_size=batch_size,
          semaphore=semaphore,
          max_retries=max_retries,
          retry_delay_seconds=retry_delay_seconds,
      )
      for stance in active_stances
  ))
  assignments_by_stance = dict(zip(active_stances, assignments_list))

  # Step C: Summarize every non-empty cluster concurrently.
  summary_tasks: list[Any] = []
  cluster_specs: list[
      tuple[
          stance_classifier.StanceLabel,
          DiscoveredOpinion,
          list[stance_classifier.DeliberationStatement],
          bool,
      ]
  ] = []
  for stance in active_stances:
    stance_stmts = grouped[stance]
    assigned_map = assignments_by_stance[stance]
    for opinion in opinions_by_stance[stance]:
      members = [
          stmt
          for stmt in stance_stmts
          if opinion.opinion_id in assigned_map[stmt.statement_id]
      ]
      if members:
        cluster_specs.append((stance, opinion, members, False))
        summary_tasks.append(
            summarize_opinion_cluster(
                model=model,
                focal_proposal=focal_proposal,
                stance=stance,
                opinion=opinion,
                statements=members,
                semaphore=semaphore,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
            )
        )
    other_members = [
        stmt
        for stmt in stance_stmts
        if "other" in assigned_map[stmt.statement_id]
    ]
    if other_members:
      other_opinion = DiscoveredOpinion(
          opinion_id="other", opinion_statement="Other"
      )
      cluster_specs.append((stance, other_opinion, other_members, True))
      summary_tasks.append(
          summarize_opinion_cluster(
              model=model,
              focal_proposal=focal_proposal,
              stance=stance,
              opinion=other_opinion,
              statements=other_members,
              semaphore=semaphore,
              max_retries=max_retries,
              retry_delay_seconds=retry_delay_seconds,
          )
      )

  summaries = await asyncio.gather(*summary_tasks) if summary_tasks else []
  clusters_by_stance: dict[
      stance_classifier.StanceLabel, list[SubOpinionCluster]
  ] = collections.defaultdict(list)
  for (stance, opinion, members, is_other), summary in zip(
      cluster_specs, summaries
  ):
    clusters_by_stance[stance].append(
        SubOpinionCluster(
            opinion_id=opinion.opinion_id,
            opinion_statement=opinion.opinion_statement,
            summary=summary,
            statement_ids=tuple(stmt.statement_id for stmt in members),
            is_other=is_other,
        )
    )

  categories = tuple(
      StanceOpinionTree(
          stance=stance,
          stance_title=stance_classifier.STANCE_TITLES[stance],
          total_statements=len(grouped[stance]),
          clusters=_sort_clusters(clusters_by_stance[stance]),
      )
      for stance in active_stances
  )
  return ProposalOpinionTree(
      proposal_id=focal_proposal.proposal_id,
      categories=categories,
  )


def render_opinion_tree_markdown(
    tree: ProposalOpinionTree,
    *,
    include_summaries: bool = False,
) -> str:
  """Renders a `ProposalOpinionTree` as deterministic markdown.

  Follows the unnumbered topic-tree convention (`## <Category> (<N> quotes)`),
  with ordered sub-opinions and bolding applied strictly to `**Other**` so
  residual unclustered volume stands out immediately.

  Args:
    tree: The hierarchical opinion tree to render.
    include_summaries: Whether to append each cluster's synthesis summary.

  Returns:
    Formatted markdown string.
  """
  blocks: list[str] = []
  for category in tree.categories:
    lines = [
        f"## {category.stance_title} ({category.total_statements} quotes)",
        "",
    ]
    sorted_clusters = _sort_clusters(category.clusters)
    for index, cluster in enumerate(sorted_clusters, start=1):
      count = len(cluster.statement_ids)
      label = "**Other**" if cluster.is_other else cluster.opinion_statement
      entry = f"{index}. {label} ({count} quotes)"
      if include_summaries and cluster.summary.strip():
        entry = f"{entry} -- {cluster.summary.strip()}"
      lines.append(entry)
    blocks.append("\n".join(lines))
  return "\n\n".join(blocks).strip()


# ---------------------------------------------------------------------------
# Segment-Conditioned Sub-Tree Clustering & Worldview/Friction Synthesis
# ---------------------------------------------------------------------------


def build_segment_small_cluster_prompt(
    focal_proposal: stance_classifier.Proposal,
    segment: SegmentSpec,
    stance: stance_classifier.StanceLabel,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    *,
    segmentation_context: str = "",
) -> str:
  """Builds prompt summarizing a small (1-3 statement) segment bucket."""
  stance_title = stance_classifier.STANCE_TITLES[stance]
  lines = [
      "# Task: Segment Sub-Opinion Summary",
      "",
      (
          "You are an expert deliberation analyst summarizing participant"
          " feedback on an actionable proposal."
      ),
      "",
      f"## Focal Proposal: {focal_proposal.title}",
      f"Proposal Text: {focal_proposal.text}",
      "",
  ]
  lines.extend(_format_segment_context_lines(segment, segmentation_context))
  lines.extend([
      f"## Stance Category: {stance_title} ({len(statements)} statements)",
      "",
  ])
  for stmt in statements:
    lines.append(_format_statement_bullet(stmt))
  lines.extend([
      "",
      "## Instructions",
      (
          "1. Provide a concise 3-7 word 'opinion_statement' capturing the core"
          " theme of these statement(s)."
      ),
      (
          "2. Provide a 1-2 sentence 'summary' of the participant reasoning,"
          " grounded strictly in the statements above."
      ),
      '3. Return strict JSON: {"opinion_statement": "...", "summary": "..."}.',
  ])
  return "\n".join(lines)


def build_segment_clustering_prompt(
    focal_proposal: stance_classifier.Proposal,
    segment: SegmentSpec,
    stance: stance_classifier.StanceLabel,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    *,
    segmentation_context: str = "",
) -> str:
  """Builds prompt clustering >=4 statements in one segment-stance bucket."""
  stance_title = stance_classifier.STANCE_TITLES[stance]
  lines = [
      "# Task: Segment Sub-Opinion Clustering & Synthesis",
      "",
      (
          "You are an expert deliberation analyst clustering participant"
          " feedback on an actionable proposal."
      ),
      "",
      f"## Focal Proposal: {focal_proposal.title}",
      f"Proposal Text: {focal_proposal.text}",
      "",
  ]
  lines.extend(_format_segment_context_lines(segment, segmentation_context))
  lines.extend([
      f"## Stance Category: {stance_title} ({len(statements)} statements)",
      "",
  ])
  for stmt in statements:
    lines.append(_format_statement_bullet(stmt, include_statement_id=True))
  lines.extend([
      "",
      "## Instructions",
      (
          "1. Discover 2 to 4 recurring, distinct thematic sub-opinions"
          " expressed by participants in this segment."
      ),
      (
          "2. For each cluster, return 'opinion_id' (e.g., 'op_1'), concise"
          " declarative 'opinion_statement', 1-2 sentence 'summary', and"
          " matching 'statement_ids'."
      ),
      (
          "3. Route any residual or outlier statement IDs into"
          " 'other_statement_ids' with a brief 'other_summary'."
      ),
      "4. Return strict JSON matching the provided schema.",
  ])
  return "\n".join(lines)


async def cluster_segment_stance(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    segment: SegmentSpec,
    stance: stance_classifier.StanceLabel,
    statements: abc.Sequence[stance_classifier.DeliberationStatement],
    *,
    segmentation_context: str = "",
    small_cluster_threshold: int = DEFAULT_SMALL_CLUSTER_THRESHOLD,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> list[SubOpinionCluster]:
  """Clusters and summarizes statements for one `(segment, stance)` bucket."""
  if not statements:
    return []

  stance_title = stance_classifier.STANCE_TITLES[stance]
  run_name = (
      f"segcluster_{focal_proposal.proposal_id}_{segment.id}_{stance.value}"
  )

  if len(statements) <= small_cluster_threshold:
    prompt = build_segment_small_cluster_prompt(
        focal_proposal,
        segment,
        stance,
        statements,
        segmentation_context=segmentation_context,
    )
    payload = await _call_json_model(
        model=model,
        prompt=prompt,
        run_name=run_name,
        response_schema=SMALL_SEGMENT_CLUSTER_SCHEMA,
        semaphore=semaphore,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    opinion_stmt = str(payload.get("opinion_statement", "")).strip() or (
        f"{stance_title} on {focal_proposal.title}"
    )
    summary = str(payload.get("summary", "")).strip() or (
        f"Participants in {segment.label} expressed"
        f" {stance_title.lower()} regarding {focal_proposal.title}."
    )
    return [
        SubOpinionCluster(
            opinion_id="op_1",
            opinion_statement=opinion_stmt,
            summary=summary,
            statement_ids=tuple(stmt.statement_id for stmt in statements),
            is_other=False,
        )
    ]

  prompt = build_segment_clustering_prompt(
      focal_proposal,
      segment,
      stance,
      statements,
      segmentation_context=segmentation_context,
  )
  payload = await _call_json_model(
      model=model,
      prompt=prompt,
      run_name=run_name,
      response_schema=SEGMENT_CLUSTERING_SCHEMA,
      semaphore=semaphore,
      max_retries=max_retries,
      retry_delay_seconds=retry_delay_seconds,
  )

  raw_clusters = payload.get("clusters", [])
  raw_other_ids = payload.get("other_statement_ids", [])
  other_summary = (
      str(payload.get("other_summary", "")).strip()
      or "Other participant remarks."
  )

  valid_ids = frozenset(stmt.statement_id for stmt in statements)
  assigned_ids: set[str] = set()
  clusters: list[SubOpinionCluster] = []

  if isinstance(raw_clusters, list):
    for idx, item in enumerate(raw_clusters, start=1):
      if not isinstance(item, abc.Mapping):
        continue
      member_ids = tuple(
          dict.fromkeys(
              str(sid)
              for sid in item.get("statement_ids", [])
              if str(sid) in valid_ids
          )
      )
      if not member_ids:
        continue
      assigned_ids.update(member_ids)
      op_id = str(item.get("opinion_id", "")).strip() or f"op_{idx}"
      op_stmt = str(item.get("opinion_statement", "")).strip() or f"Theme {idx}"
      summary = (
          str(item.get("summary", "")).strip()
          or "Participant reasoning on this theme."
      )
      clusters.append(
          SubOpinionCluster(
              opinion_id=op_id,
              opinion_statement=op_stmt,
              summary=summary,
              statement_ids=member_ids,
              is_other=False,
          )
      )

  explicit_other = (
      [str(sid) for sid in raw_other_ids if str(sid) in valid_ids]
      if isinstance(raw_other_ids, list)
      else []
  )
  unassigned = [
      stmt.statement_id
      for stmt in statements
      if stmt.statement_id not in assigned_ids
  ]
  all_other_ids = tuple(dict.fromkeys(unassigned + explicit_other))
  if all_other_ids:
    clusters.append(
        SubOpinionCluster(
            opinion_id="other",
            opinion_statement="Other",
            summary=other_summary,
            statement_ids=all_other_ids,
            is_other=True,
        )
    )

  return list(_sort_clusters(clusters))


async def build_segment_subtrees(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    segments: abc.Sequence[SegmentSpec],
    segment_stance_statements: SegmentStanceStatements,
    *,
    segmentation_context: str = "",
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> dict[str, dict[stance_classifier.StanceLabel, list[SubOpinionCluster]]]:
  """Builds sub-opinion clusters across segments and stances concurrently."""
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  semaphore = asyncio.Semaphore(validated_concurrency)

  trees: dict[
      str, dict[stance_classifier.StanceLabel, list[SubOpinionCluster]]
  ] = {
      spec.id: {
          stance: [] for stance in stance_classifier.RELEVANT_STANCE_LABELS
      }
      for spec in segments
  }

  tasks: list[Any] = []
  task_keys: list[tuple[str, stance_classifier.StanceLabel]] = []
  for spec in segments:
    stance_map = segment_stance_statements.get(spec.id, {})
    for stance in stance_classifier.RELEVANT_STANCE_LABELS:
      stmts = stance_map.get(stance, ())
      if not stmts:
        continue
      task_keys.append((spec.id, stance))
      tasks.append(
          cluster_segment_stance(
              model=model,
              focal_proposal=focal_proposal,
              segment=spec,
              stance=stance,
              statements=stmts,
              segmentation_context=segmentation_context,
              semaphore=semaphore,
              max_retries=max_retries,
              retry_delay_seconds=retry_delay_seconds,
          )
      )

  if tasks:
    results = await asyncio.gather(*tasks)
    for (seg_id, stance), cluster_list in zip(task_keys, results):
      trees[seg_id][stance] = cluster_list
  return trees


def build_worldview_synthesis_prompt(
    focal_proposal: stance_classifier.Proposal,
    segment: SegmentSpec,
    total_participants: int,
    stance_statements: StanceStatements,
    *,
    segmentation_context: str = "",
    max_statements_per_stance: int = DEFAULT_MAX_STATEMENTS_PER_STANCE,
) -> str:
  """Builds prompt to synthesize worldview, friction, and support conditions."""
  if max_statements_per_stance < 1:
    raise ValueError(
        "max_statements_per_stance must be >= 1, got"
        f" {max_statements_per_stance}."
    )

  stance_blocks: list[str] = []
  for stance in stance_classifier.RELEVANT_STANCE_LABELS:
    stmts = stance_statements.get(stance, ())
    if not stmts:
      continue
    selected = stmts[:max_statements_per_stance]
    stance_title = stance_classifier.STANCE_TITLES[stance]
    header = (
        f"### {stance_title} ({len(stmts)} statements in segment,"
        f" {len(selected)} shown)"
    )
    bullets = "\n".join(_format_statement_bullet(stmt) for stmt in selected)
    stance_blocks.append(f"{header}\n{bullets}")

  evidence_body = (
      "\n\n".join(stance_blocks)
      if stance_blocks
      else "No proposal-relevant statements recorded for this segment."
  )

  lines = [
      "# Task: Segment Worldview & Friction Synthesis",
      "",
      (
          "You are an expert deliberation analyst synthesizing how a"
          " participant segment responded to an actionable proposal."
      ),
      "",
      f"## Focal Proposal: {focal_proposal.title}",
      f'Proposal Statement: "{focal_proposal.text}"',
      "",
  ]
  lines.extend(_format_segment_context_lines(segment, segmentation_context))
  lines.extend([
      f"Participants in Segment: {total_participants}",
      "",
      "## Categorized Statements from This Segment",
      evidence_body,
      "",
  ])
  lines.extend(_WORLDVIEW_INSTRUCTIONS)
  return "\n".join(lines)


async def synthesize_segment_worldview(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    segment: SegmentSpec,
    total_participants: int,
    stance_statements: StanceStatements,
    *,
    segmentation_context: str = "",
    max_statements_per_stance: int = DEFAULT_MAX_STATEMENTS_PER_STANCE,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> SegmentWorldviewProfile:
  """Synthesizes a worldview, friction, and conditions profile for a segment."""
  total_stmts = sum(
      len(stance_statements.get(stance, ()))
      for stance in stance_classifier.RELEVANT_STANCE_LABELS
  )
  if total_stmts == 0:
    return SegmentWorldviewProfile(
        segment_id=segment.id,
        worldview_and_support_drivers=(
            "No proposal-relevant statements were recorded for"
            f" {segment.label}."
        ),
        primary_friction_points=(),
        key_conditions_for_support=(),
    )

  prompt = build_worldview_synthesis_prompt(
      focal_proposal=focal_proposal,
      segment=segment,
      total_participants=total_participants,
      stance_statements=stance_statements,
      segmentation_context=segmentation_context,
      max_statements_per_stance=max_statements_per_stance,
  )
  payload = await _call_json_model(
      model=model,
      prompt=prompt,
      run_name=f"worldview_{focal_proposal.proposal_id}_{segment.id}",
      response_schema=WORLDVIEW_SYNTHESIS_SCHEMA,
      semaphore=semaphore,
      max_retries=max_retries,
      retry_delay_seconds=retry_delay_seconds,
  )

  worldview = str(payload.get("worldview_and_support_drivers", "")).strip()
  raw_frictions = payload.get("primary_friction_points", [])
  raw_conditions = payload.get("key_conditions_for_support", [])
  frictions = tuple(
      str(item).strip()
      for item in (raw_frictions if isinstance(raw_frictions, list) else ())
      if str(item).strip()
  )
  conditions = tuple(
      str(item).strip()
      for item in (raw_conditions if isinstance(raw_conditions, list) else ())
      if str(item).strip()
  )
  return SegmentWorldviewProfile(
      segment_id=segment.id,
      worldview_and_support_drivers=worldview,
      primary_friction_points=frictions,
      key_conditions_for_support=conditions,
  )


async def synthesize_all_segment_worldviews(
    model: base_model.BaseModel,
    focal_proposal: stance_classifier.Proposal,
    segments: abc.Sequence[SegmentSpec],
    segment_counts: abc.Mapping[str, int],
    segment_stance_statements: SegmentStanceStatements,
    *,
    segmentation_context: str = "",
    max_statements_per_stance: int = DEFAULT_MAX_STATEMENTS_PER_STANCE,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
) -> dict[str, SegmentWorldviewProfile]:
  """Synthesizes worldview profiles across all segments concurrently."""
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  semaphore = asyncio.Semaphore(validated_concurrency)
  profiles = await asyncio.gather(*(
      synthesize_segment_worldview(
          model=model,
          focal_proposal=focal_proposal,
          segment=spec,
          total_participants=segment_counts.get(spec.id, 0),
          stance_statements=segment_stance_statements.get(spec.id, {}),
          segmentation_context=segmentation_context,
          max_statements_per_stance=max_statements_per_stance,
          semaphore=semaphore,
          max_retries=max_retries,
          retry_delay_seconds=retry_delay_seconds,
      )
      for spec in segments
  ))
  return {profile.segment_id: profile for profile in profiles}


# ---------------------------------------------------------------------------
# Segment-Conditioned View Renderers (Endogenous Stance-Count Headers)
# ---------------------------------------------------------------------------


def format_segment_header(
    segment: SegmentSpec,
    total_participants: int,
    stance_counts: abc.Mapping[stance_classifier.StanceLabel, int],
) -> str:
  """Formats a segment heading with endogenous stance counts."""
  count_parts = ", ".join(
      f"{stance_counts.get(stance, 0)} {stance.value.title()}"
      for stance in stance_classifier.RELEVANT_STANCE_LABELS
  )
  return (
      f"## {segment.label} ({total_participants} participants |"
      f" {count_parts} statements)"
  )


def _append_segmentation_context_header(
    lines: list[str], segmentation_context: str
) -> None:
  """Appends a segmentation context section when context is non-empty."""
  cleaned = segmentation_context.strip()
  if cleaned:
    lines.extend(["### Segmentation Context", "", cleaned, ""])


def _append_segment_intro(
    lines: list[str],
    segment: SegmentSpec,
    total_participants: int,
    stance_map: StanceStatements,
) -> None:
  """Appends the endogenous segment heading and optional description."""
  stance_counts = {
      stance: len(stance_map.get(stance, ()))
      for stance in stance_classifier.RELEVANT_STANCE_LABELS
  }
  lines.append(
      format_segment_header(segment, total_participants, stance_counts)
  )
  lines.append("")
  if segment.description:
    lines.append(f"Segment description: {segment.description}")
    lines.append("")


def format_flat_stance_view(
    segments: abc.Sequence[SegmentSpec],
    segment_stance_statements: SegmentStanceStatements,
    segment_counts: abc.Mapping[str, int],
    *,
    segmentation_context: str = "",
) -> str:
  """Renders flat stance-grouped quotes per segment with endogenous counts."""
  lines: list[str] = []
  _append_segmentation_context_header(lines, segmentation_context)

  for spec in segments:
    stance_map = segment_stance_statements.get(spec.id, {})
    _append_segment_intro(
        lines, spec, segment_counts.get(spec.id, 0), stance_map
    )
    for stance in stance_classifier.RELEVANT_STANCE_LABELS:
      stmts = stance_map.get(stance, ())
      if not stmts:
        continue
      stance_title = stance_classifier.STANCE_TITLES[stance]
      lines.extend([f"### {stance_title} ({len(stmts)} statements)", ""])
      for stmt in stmts:
        lines.append(_format_statement_bullet(stmt))
      lines.append("")

  return "\n".join(lines).strip()


def format_subtree_view(
    segments: abc.Sequence[SegmentSpec],
    segment_subtrees: SegmentSubtrees,
    segment_stance_statements: SegmentStanceStatements,
    segment_counts: abc.Mapping[str, int],
    *,
    max_quotes_per_cluster: int | None = None,
    segmentation_context: str = "",
) -> str:
  """Renders segment sub-opinion trees with all quotes or capped top quotes.

  Args:
    segments: Ordered `SegmentSpec` definitions.
    segment_subtrees: Precomputed clusters keyed by `(segment_id, stance)`.
    segment_stance_statements: Statements keyed by `(segment_id, stance)`.
    segment_counts: Participant count per segment ID.
    max_quotes_per_cluster: Optional per-cluster quote cap (`None` keeps all).
    segmentation_context: Optional description of how segments were derived.

  Returns:
    Markdown evidence block using flat top-level quote bullets and bolding only
    `**Other**` among cluster labels.
  """
  if max_quotes_per_cluster is not None and max_quotes_per_cluster < 0:
    raise ValueError(
        "max_quotes_per_cluster must be >= 0 or None, got"
        f" {max_quotes_per_cluster}."
    )

  stmt_lookup: dict[str, stance_classifier.DeliberationStatement] = {}
  for stance_map in segment_stance_statements.values():
    for stmts in stance_map.values():
      for stmt in stmts:
        stmt_lookup[stmt.statement_id] = stmt

  lines: list[str] = []
  _append_segmentation_context_header(lines, segmentation_context)

  for spec in segments:
    stance_map = segment_stance_statements.get(spec.id, {})
    _append_segment_intro(
        lines, spec, segment_counts.get(spec.id, 0), stance_map
    )
    seg_tree = segment_subtrees.get(spec.id, {})
    for stance in stance_classifier.RELEVANT_STANCE_LABELS:
      clusters = _sort_clusters(seg_tree.get(stance, ()))
      if not clusters:
        continue
      stance_title = stance_classifier.STANCE_TITLES[stance]
      total_in_stance = len(stance_map.get(stance, ()))
      lines.extend([f"### {stance_title} ({total_in_stance} statements)", ""])
      for index, cluster in enumerate(clusters, start=1):
        label = "**Other**" if cluster.is_other else cluster.opinion_statement
        count = len(cluster.statement_ids)
        heading = f"{index}. {label} ({count} quotes)"
        if cluster.summary.strip():
          heading = f"{heading} -- {cluster.summary.strip()}"
        lines.append(heading)
        lines.append("")
        selected_ids = (
            cluster.statement_ids[:max_quotes_per_cluster]
            if max_quotes_per_cluster is not None
            else cluster.statement_ids
        )
        for stmt_id in selected_ids:
          stmt = stmt_lookup.get(stmt_id)
          if stmt is not None:
            lines.append(_format_statement_bullet(stmt))
        if selected_ids:
          lines.append("")

  return "\n".join(lines).strip()


def format_worldview_synthesis_view(
    segments: abc.Sequence[SegmentSpec],
    worldviews: abc.Mapping[str, SegmentWorldviewProfile],
    segment_stance_statements: SegmentStanceStatements,
    segment_counts: abc.Mapping[str, int],
    *,
    segmentation_context: str = "",
) -> str:
  """Renders segment worldview, friction points, and support conditions."""
  lines: list[str] = []
  _append_segmentation_context_header(lines, segmentation_context)

  for spec in segments:
    stance_map = segment_stance_statements.get(spec.id, {})
    _append_segment_intro(
        lines, spec, segment_counts.get(spec.id, 0), stance_map
    )
    profile = worldviews.get(spec.id)
    if profile is None:
      continue
    lines.append(
        "**Worldview & Support Drivers:**"
        f" {profile.worldview_and_support_drivers}"
    )
    lines.append("")
    if profile.primary_friction_points:
      lines.append("**Primary Friction Points:**")
      for friction in profile.primary_friction_points:
        lines.append(f"- {friction}")
      lines.append("")
    if profile.key_conditions_for_support:
      lines.append("**Key Conditions for Support:**")
      for condition in profile.key_conditions_for_support:
        lines.append(f"- {condition}")
      lines.append("")

  return "\n".join(lines).strip()


def format_segment_conditioned_view(
    structure: SegmentViewStructure,
    segments: abc.Sequence[SegmentSpec],
    segment_stance_statements: SegmentStanceStatements,
    segment_counts: abc.Mapping[str, int],
    *,
    segmentation_context: str = "",
    segment_subtrees: SegmentSubtrees | None = None,
    worldviews: abc.Mapping[str, SegmentWorldviewProfile] | None = None,
    max_quotes_per_cluster: int = DEFAULT_TOP_QUOTES_PER_CLUSTER,
) -> str:
  """Renders a segment-conditioned evidence view for the requested structure."""
  if structure == SegmentViewStructure.FLAT_QUOTES:
    return format_flat_stance_view(
        segments=segments,
        segment_stance_statements=segment_stance_statements,
        segment_counts=segment_counts,
        segmentation_context=segmentation_context,
    )
  if structure == SegmentViewStructure.SUBTREE_ALL_QUOTES:
    if segment_subtrees is None:
      raise ValueError(
          "segment_subtrees is required for SUBTREE_ALL_QUOTES view."
      )
    return format_subtree_view(
        segments=segments,
        segment_subtrees=segment_subtrees,
        segment_stance_statements=segment_stance_statements,
        segment_counts=segment_counts,
        max_quotes_per_cluster=None,
        segmentation_context=segmentation_context,
    )
  if structure == SegmentViewStructure.SUBTREE_TOP_QUOTES:
    if segment_subtrees is None:
      raise ValueError(
          "segment_subtrees is required for SUBTREE_TOP_QUOTES view."
      )
    return format_subtree_view(
        segments=segments,
        segment_subtrees=segment_subtrees,
        segment_stance_statements=segment_stance_statements,
        segment_counts=segment_counts,
        max_quotes_per_cluster=max_quotes_per_cluster,
        segmentation_context=segmentation_context,
    )
  if structure == SegmentViewStructure.WORLDVIEW_SYNTHESIS:
    if worldviews is None:
      raise ValueError("worldviews is required for WORLDVIEW_SYNTHESIS view.")
    return format_worldview_synthesis_view(
        segments=segments,
        worldviews=worldviews,
        segment_stance_statements=segment_stance_statements,
        segment_counts=segment_counts,
        segmentation_context=segmentation_context,
    )
  raise ValueError(f"Unsupported SegmentViewStructure: {structure!r}.")


def main() -> None:
  """Command-line entry point for segment-conditioned worldview synthesis."""
  parser = argparse.ArgumentParser(
      description="Synthesize segment-conditioned views for a focal proposal."
  )
  parser.add_argument(
      "--room_summaries",
      type=str,
      required=True,
      help="Path to room summaries JSON from summarize_rooms.",
  )
  parser.add_argument(
      "--classifications",
      type=str,
      required=True,
      help="Path to stance classifications JSON from stance_classifier.",
  )
  parser.add_argument(
      "--proposal_json",
      type=str,
      required=True,
      help="Path to JSON file describing the focal proposal.",
  )
  parser.add_argument(
      "--participants_csv",
      type=str,
      required=True,
      help="Path to CSV mapping participants to segment columns.",
  )
  parser.add_argument(
      "--participant_id_column",
      type=str,
      default="participant_id",
      help="Column in participants_csv containing participant IDs.",
  )
  parser.add_argument(
      "--segment_column",
      type=str,
      default=None,
      help="Categorical segment column name (mutually exclusive with binary).",
  )
  parser.add_argument(
      "--segment_columns",
      type=str,
      nargs="+",
      default=None,
      help="Overlapping binary segment indicator column names.",
  )
  parser.add_argument(
      "--segment_specs_json",
      type=str,
      default=None,
      help="Optional JSON defining segment labels, descriptions, and context.",
  )
  parser.add_argument(
      "--structure",
      type=str,
      choices=[item.value for item in SegmentViewStructure],
      default=SegmentViewStructure.WORLDVIEW_SYNTHESIS.value,
      help="Output view structure to render.",
  )
  parser.add_argument(
      "--output",
      type=str,
      required=True,
      help="Path to write the rendered markdown view.",
  )
  parser.add_argument(
      "--model",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help="Name of the model to use.",
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
  statements = stance_classifier.statements_from_room_summaries(rooms)

  with open(args.proposal_json, "r") as f:
    prop_data = json.load(f)
  focal_proposal = stance_classifier.Proposal(
      proposal_id=str(prop_data["proposal_id"]),
      title=str(prop_data["title"]),
      text=str(prop_data["text"]),
      topic=str(prop_data.get("topic", "")),
  )

  with open(args.classifications, "r") as f:
    raw_classifications = json.load(f)
  proposal_ratings = raw_classifications.get(
      focal_proposal.proposal_id, raw_classifications
  )
  classifications = {
      stmt_id: stance_classifier.ClassifiedStatement(
          statement_id=stmt_id,
          stance=stance_classifier.normalize_stance_label(raw_label),
      )
      for stmt_id, raw_label in proposal_ratings.items()
  }

  explicit_specs: list[SegmentSpec] | None = None
  segmentation_context = ""
  if args.segment_specs_json:
    with open(args.segment_specs_json, "r") as f:
      explicit_specs, segmentation_context = load_segment_specs(f)

  participants_df = pd.read_csv(args.participants_csv)
  participant_to_segments, resolved_specs = map_participants_to_segments(
      participants_df=participants_df,
      participant_id_column=args.participant_id_column,
      segment_column=args.segment_column,
      segment_columns=args.segment_columns,
      segment_specs=explicit_specs,
  )
  segment_counts = compute_segment_counts(
      participant_to_segments, resolved_specs
  )
  partitioned = partition_statements_by_segment_and_stance(
      statements=statements,
      classifications=classifications,
      participant_to_segments=participant_to_segments,
      segments=resolved_specs,
  )

  structure = SegmentViewStructure(args.structure)
  model = (
      None
      if structure == SegmentViewStructure.FLAT_QUOTES
      else model_factory.get_model(args.model)
  )

  async def _render() -> str:
    subtrees = None
    worldviews = None
    if structure in (
        SegmentViewStructure.SUBTREE_ALL_QUOTES,
        SegmentViewStructure.SUBTREE_TOP_QUOTES,
    ):
      assert model is not None
      subtrees = await build_segment_subtrees(
          model=model,
          focal_proposal=focal_proposal,
          segments=resolved_specs,
          segment_stance_statements=partitioned,
          segmentation_context=segmentation_context,
          max_concurrent_calls=args.max_concurrent_calls,
      )
    elif structure == SegmentViewStructure.WORLDVIEW_SYNTHESIS:
      assert model is not None
      worldviews = await synthesize_all_segment_worldviews(
          model=model,
          focal_proposal=focal_proposal,
          segments=resolved_specs,
          segment_counts=segment_counts,
          segment_stance_statements=partitioned,
          segmentation_context=segmentation_context,
          max_concurrent_calls=args.max_concurrent_calls,
      )
    return format_segment_conditioned_view(
        structure=structure,
        segments=resolved_specs,
        segment_stance_statements=partitioned,
        segment_counts=segment_counts,
        segmentation_context=segmentation_context,
        segment_subtrees=subtrees,
        worldviews=worldviews,
    )

  rendered = asyncio.run(_render())
  output_dir = os.path.dirname(args.output)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(args.output, "w") as f:
    f.write(rendered)
  logging.info("Saved segment synthesis view to %s", args.output)


if __name__ == "__main__":
  main()
