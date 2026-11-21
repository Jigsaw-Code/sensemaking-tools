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

"""Post-hoc session-DB grounding autorater for agentic proposal refinement.

Compares factual, numeric, governance, funding, and operational claims in a
refined actionable proposal against the exact corpus search and read tool
results recovered from the agent's persisted SQLite session database via
``session_trace_reader``. Flags ungrounded claims, hallucinations, and claims
contradicted by evidence the agent actually retrieved during its session.

WARNING (SDK Implementation Fragility):
  Evidence extraction delegates to ``session_trace_reader``, which parses
  internal ``google-antigravity`` SQLite ``steps`` blobs (numeric ``step_type``
  codes, protobuf wire markers, and ``antigravity.localharness.ToolResponse``
  JSON framing observed in SDK 0.1.5). Because those internal storage layouts
  are not a stable public API and may shift across SDK versions, grounding
  audits should be treated as best-effort diagnostic checks and callers should
  inspect recovered research-call counts and trace warnings when interpreting
  results.
"""

import argparse
import asyncio
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import logging
import math
import pathlib
import re
import sqlite3
from typing import Any

import pandas as pd

from src import runner_utils
from src.models import base_model
from src.models import model_factory
from src.proposition_refinement.agentic import session_trace_reader

DEFAULT_MODEL_NAME: str = "gemini-3.8-flash"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_CONCURRENT_CALLS: int = 16
DEFAULT_MAX_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAY_SECONDS: float = 1.0
DEFAULT_EVIDENCE_CHAR_BUDGET: int = 120_000

CLAIM_TYPES: tuple[str, ...] = (
    "authority_or_governance",
    "funding_or_resource_limit",
    "numeric_rate_or_estimate",
    "external_approval_requirement",
    "operational_or_impact_claim",
    "other",
)
CLAIM_STATUSES: tuple[str, ...] = (
    "supported_by_seen_evidence",
    "contradicted_by_seen_evidence",
    "not_supported_in_session",
)
GROUNDING_QUALITY_LEVELS: tuple[str, ...] = (
    "grounded",
    "partly_ungrounded",
    "ungrounded_or_contradicted",
)

_CLAIM_TYPES_SET: frozenset[str] = frozenset(CLAIM_TYPES)
_CLAIM_STATUSES_SET: frozenset[str] = frozenset(CLAIM_STATUSES)
_QUALITY_LEVELS_SET: frozenset[str] = frozenset(GROUNDING_QUALITY_LEVELS)

_PRIORITY_TERM_RE = re.compile(
    r"(?:\$?\d[\d,.]*(?:\s*(?:million|billion|thousand|percent|%))?"
    r"|[\w-]+\.(?:md|txt|csv|json))",
    re.IGNORECASE,
)
_EXCLUDED_TOOL_ARG_KEYS: frozenset[str] = frozenset(
    {"toolAction", "toolSummary"}
)

GROUNDING_SYSTEM_PROMPT: str = (
    "You are a strict post-hoc grounding auditor verifying whether factual and"
    " operational claims in a refined actionable proposal are supported by the"
    " tool-result evidence an agent actually retrieved during its session."
    " Judge strictly against the supplied evidence excerpts; never use outside"
    " knowledge to mark a claim supported. Return pure valid JSON matching the"
    " requested schema."
)

GROUNDING_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "claim_type": {"type": "string", "enum": list(CLAIM_TYPES)},
                    "explicit_in_statement": {"type": "boolean"},
                    "inherited_from_seed": {"type": "boolean"},
                    "status": {"type": "string", "enum": list(CLAIM_STATUSES)},
                    "evidence_snippet": {"type": "string"},
                },
                "required": [
                    "claim",
                    "claim_type",
                    "explicit_in_statement",
                    "inherited_from_seed",
                    "status",
                    "evidence_snippet",
                ],
            },
        },
        "grounding_quality": {
            "type": "string",
            "enum": list(GROUNDING_QUALITY_LEVELS),
        },
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["claims", "grounding_quality", "red_flags", "rationale"],
}

_ADDITIONAL_CONTEXT_BLOCK: str = (
    "<additionalContext>\n{additional_context}\n</additionalContext>\n\n"
)

_GROUNDING_PROMPT_TEMPLATE: str = (
    "{context_block}You are auditing whether a refined actionable proposal's"
    " factual, numeric, governance, funding, and operational claims are"
    " grounded in the corpus search and file-read results retrieved during the"
    " agent's session. Judge ONLY against the recovered session evidence below;"
    " do not use outside world knowledge to mark any claim supported.\n\n"
    "## Original Seed Proposal{title_suffix}\n"
    "{seed_proposal}\n\n"
    "## Refined Actionable Proposal Under Audit\n"
    "{candidate_proposal}\n"
    "{assessment_block}\n"
    "## Tool-Result Evidence Retrieved in the Agent Session\n"
    "{evidence_block}\n\n"
    "## Rating Rubric\n"
    "1. `claims`: List every substantive factual, numeric, legal/governance,"
    " funding/resource, external-approval, or operational/impact claim.\n"
    "   - Set `explicit_in_statement=true` for claims asserted directly in the"
    " refined statement; set `false` for implicit operational or authority"
    " dependencies or claims made only in the agent's feasibility note.\n"
    "   - Set `inherited_from_seed=true` if the claim already appears verbatim"
    " or substantively in the original seed proposal.\n"
    "   - Assign `status` strictly from the retrieved session evidence:\n"
    "     * `supported_by_seen_evidence`: A passage in the evidence above"
    " directly supports the claim. Quote the shortest decisive passage verbatim"
    " in `evidence_snippet`. Numbers invented or derived without a supporting"
    " basis in the retrieved passages are NOT supported.\n"
    "     * `contradicted_by_seen_evidence`: A passage in the evidence above"
    " conflicts with the claim in a way that undermines feasibility (for"
    " example, over-promising revenue, exceeding a stated legal or budget cap,"
    " using restricted funds impermissibly, or omitting a required approval)."
    " Quote the conflicting passage in `evidence_snippet`.\n"
    "     * `not_supported_in_session`: No retrieved passage above supports or"
    " refutes the claim (leave `evidence_snippet` empty).\n"
    "2. `grounding_quality`:\n"
    "   - `grounded`: Every explicit non-seed claim is"
    " `supported_by_seen_evidence` and no claim is contradicted.\n"
    "   - `partly_ungrounded`: Some claims are supported, but at least one"
    " explicit non-seed claim is `not_supported_in_session` (and none are"
    " contradicted).\n"
    "   - `ungrounded_or_contradicted`: No supporting session evidence was"
    " retrieved for added claims, OR any claim is"
    " `contradicted_by_seen_evidence`.\n"
    "3. `red_flags`: Concise strings flagging concrete grounding or feasibility"
    " defects (such as hallucinated figures, unsupported citations, or"
    " contradicted constraints). Empty array if none.\n"
    "4. `rationale`: One or two concise sentences summarizing the verdict."
)


@dataclasses.dataclass(frozen=True, slots=True)
class GroundingAuditItem:
  """Input record for post-hoc session-DB grounding verification."""

  item_id: str
  seed_proposal: str
  candidate_proposal: str
  proposal_title: str = ""
  feasibility_assessment: str = ""
  session_db_path: pathlib.Path | None = None
  session_trace: session_trace_reader.SessionTrace | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ClaimAudit:
  """Grounding verification outcome for one extracted proposal claim."""

  claim: str
  claim_type: str
  explicit_in_statement: bool
  inherited_from_seed: bool
  status: str
  evidence_snippet: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class EvidencePacket:
  """Budgeted concatenation of corpus tool results recovered from a trace."""

  text: str
  n_research_calls: int
  n_search_calls: int
  n_read_calls: int
  files_accessed: tuple[str, ...]
  evidence_chars_total: int
  evidence_chars_kept: int
  truncated: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GroundingVerdict:
  """Structured grounding audit verdict for one refined actionable proposal."""

  item_id: str
  passed: bool
  grounding_quality: str
  failure_reasons: tuple[str, ...]
  claims: tuple[ClaimAudit, ...]
  n_supported_claims: int
  n_contradicted_claims: int
  n_ungrounded_added_claims: int
  red_flags: tuple[str, ...]
  rationale: str
  n_research_calls: int
  files_accessed: tuple[str, ...]
  evidence_truncated: bool
  session_db_path: str | None
  trace_warnings: tuple[str, ...]
  attempts: int
  error: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Serializes the grounding verdict into JSON-compatible primitives."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class GroundingAuditConfig:
  """Immutable configuration for running a file-based grounding audit."""

  input_path: pathlib.Path
  output_path: pathlib.Path
  sessions_dir: pathlib.Path | None = None
  temperature: float = DEFAULT_TEMPERATURE
  evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET
  max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
  max_attempts: int = DEFAULT_MAX_ATTEMPTS
  retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
  require_evidence_for_added_claims: bool = True
  additional_context: str | None = None


def _validate_positive_int(name: str, value: int) -> int:
  """Validates that `value` is an integer >= 1 (excluding bools)."""
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
  if value < 1:
    raise ValueError(f"{name} must be >= 1, got {value}.")
  return value


def _validate_non_negative_float(name: str, value: float) -> float:
  """Validates that `value` is a finite number >= 0 (excluding bools)."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
  if not math.isfinite(value) or value < 0:
    raise ValueError(f"{name} must be a finite number >= 0, got {value}.")
  return float(value)


def _positive_int_arg(value: str) -> int:
  """Argparse `type=` validating an integer >= 1."""
  try:
    return _validate_positive_int("value", int(value))
  except (TypeError, ValueError) as exc:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    ) from exc


def _non_negative_float_arg(value: str) -> float:
  """Argparse `type=` validating a finite float >= 0."""
  try:
    return _validate_non_negative_float("value", float(value))
  except (TypeError, ValueError) as exc:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be a finite number >= 0."
    ) from exc


def _clean_json_text(raw_text: str) -> str:
  """Strips optional markdown code fences around a JSON response."""
  text = (raw_text or "").strip()
  if text.startswith("```json"):
    text = text.removeprefix("```json")
  elif text.startswith("```"):
    text = text.removeprefix("```")
  if text.endswith("```"):
    text = text.removesuffix("```")
  return text.strip()


def extract_priority_terms(text: str) -> list[str]:
  """Extracts numbers, amounts, percentages, and filenames to prioritize."""
  terms = [m.group(0).strip().lower() for m in _PRIORITY_TERM_RE.finditer(text)]
  return list(dict.fromkeys(t for t in terms if t))


def build_evidence_packet(
    trace: session_trace_reader.SessionTrace | None,
    *,
    research_tools: Sequence[str] = session_trace_reader.DEFAULT_RESEARCH_TOOLS,
    priority_terms: Sequence[str] = (),
    char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET,
) -> EvidencePacket:
  """Formats recovered session research tool results within a character budget.

  WARNING: Operates on traces decoded by ``session_trace_reader``, which parses
  internal ``google-antigravity`` SQLite ``steps`` blobs and may be affected by
  SDK storage changes across versions.

  Args:
    trace: Reconstructed ``SessionTrace`` (or None if unavailable).
    research_tools: Tool names whose results count as corpus evidence.
    priority_terms: Lowercase substrings prioritizing relevant tool blocks when
      truncation is required.
    char_budget: Maximum character length of the assembled evidence text (>= 1).

  Returns:
    Immutable ``EvidencePacket`` with formatted evidence text and stats.
  """
  _validate_positive_int("char_budget", char_budget)
  if trace is None:
    return EvidencePacket(
        text="",
        n_research_calls=0,
        n_search_calls=0,
        n_read_calls=0,
        files_accessed=(),
        evidence_chars_total=0,
        evidence_chars_kept=0,
        truncated=False,
    )

  research = trace.research_events(research_tools)
  n_search = sum("search" in ev.tool_name.lower() for ev in research)
  n_read = sum("read" in ev.tool_name.lower() for ev in research)
  files_accessed = trace.files_accessed(research_tools)

  blocks: list[tuple[str, str]] = []
  for idx, event in enumerate(research, start=1):
    cleaned_args = {
        k: v
        for k, v in (event.args or {}).items()
        if k not in _EXCLUDED_TOOL_ARG_KEYS
    }
    header = (
        f"--- RESULT {idx}: {event.tool_name}({json.dumps(cleaned_args)}) ---"
    )
    body = event.result or ""
    blocks.append((header, body))

  total_chars = sum(len(head) + len(body) + 2 for head, body in blocks)
  if total_chars <= char_budget:
    packet_text = "\n".join(f"{head}\n{body}" for head, body in blocks)
    return EvidencePacket(
        text=packet_text,
        n_research_calls=len(research),
        n_search_calls=n_search,
        n_read_calls=n_read,
        files_accessed=files_accessed,
        evidence_chars_total=total_chars,
        evidence_chars_kept=len(packet_text),
        truncated=False,
    )

  terms = [t.lower() for t in priority_terms if t]
  prioritized = [
      any(term in (head + body).lower() for term in terms)
      for head, body in blocks
  ]
  prio_len = sum(
      len(body) for (_, body), is_prio in zip(blocks, prioritized) if is_prio
  )
  rest_len = sum(
      len(body)
      for (_, body), is_prio in zip(blocks, prioritized)
      if not is_prio
  )
  headers_len = sum(len(head) + 2 for head, _ in blocks)
  available = max(0, char_budget - headers_len)
  prio_share = min(prio_len, int(available * 0.75)) if prio_len else 0
  rest_share = available - prio_share

  rendered: list[str] = []
  for (head, body), is_prio in zip(blocks, prioritized):
    pool, share = (prio_len, prio_share) if is_prio else (rest_len, rest_share)
    keep = len(body) if pool <= share else int(len(body) * share / max(1, pool))
    suffix = "\n[...truncated...]" if keep < len(body) else ""
    rendered.append(f"{head}\n{body[:keep]}{suffix}")
  packet_text = "\n".join(rendered)
  return EvidencePacket(
      text=packet_text,
      n_research_calls=len(research),
      n_search_calls=n_search,
      n_read_calls=n_read,
      files_accessed=files_accessed,
      evidence_chars_total=total_chars,
      evidence_chars_kept=len(packet_text),
      truncated=True,
  )


def build_grounding_prompt(
    item: GroundingAuditItem,
    evidence: EvidencePacket,
    *,
    additional_context: str | None = None,
) -> str:
  """Builds the Check-B grounding audit prompt for one refined proposal."""
  context_block = ""
  if additional_context and additional_context.strip():
    context_block = _ADDITIONAL_CONTEXT_BLOCK.format(
        additional_context=additional_context.strip()
    )
  title_suffix = (
      f": {item.proposal_title.strip()}" if item.proposal_title.strip() else ""
  )
  assessment_block = ""
  if item.feasibility_assessment and item.feasibility_assessment.strip():
    assessment_block = (
        "\n## Agent-Authored Feasibility Note (Unverified Claim, NOT"
        f" Evidence)\n{item.feasibility_assessment.strip()}\n"
    )
  evidence_block = (
      evidence.text
      if evidence.text.strip()
      else (
          "(No corpus search or file-read tool results were recovered from"
          " this session.)"
      )
  )
  return _GROUNDING_PROMPT_TEMPLATE.format(
      context_block=context_block,
      title_suffix=title_suffix,
      seed_proposal=item.seed_proposal.strip(),
      candidate_proposal=item.candidate_proposal.strip(),
      assessment_block=assessment_block,
      evidence_block=evidence_block,
  )


def _require_bool(mapping: Mapping[str, Any], field: str) -> bool:
  """Extracts a strict boolean field from a parsed JSON mapping."""
  value = mapping.get(field)
  if not isinstance(value, bool):
    raise ValueError(
        f"Field '{field}' must be a boolean, got {type(value).__name__}."
    )
  return value


def _require_str(mapping: Mapping[str, Any], field: str) -> str:
  """Extracts a string field from a parsed JSON mapping."""
  value = mapping.get(field)
  if not isinstance(value, str):
    raise ValueError(
        f"Field '{field}' must be a string, got {type(value).__name__}."
    )
  return value.strip()


def parse_grounding_response(raw_text: str) -> dict[str, Any]:
  """Parses and validates a structured JSON grounding audit response.

  Args:
    raw_text: Raw judge output text (may include markdown code fences).

  Returns:
    Dictionary containing validated ``claims`` (tuple of ``ClaimAudit``),
    ``grounding_quality``, ``red_flags`` (tuple of str), and ``rationale``.

  Raises:
    ValueError: If `raw_text` is empty, malformed JSON, or violates schema
      types or enum constraints.
  """
  cleaned = _clean_json_text(raw_text)
  if not cleaned:
    raise ValueError("Empty response from grounding autorater model.")
  try:
    decoded = json.loads(cleaned)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Malformed JSON from grounding autorater: {exc}") from exc
  if not isinstance(decoded, Mapping):
    raise ValueError("Grounding autorater response must be a JSON object.")

  raw_claims = decoded.get("claims")
  if not isinstance(raw_claims, list):
    raise ValueError("Field 'claims' must be a JSON array.")

  claims: list[ClaimAudit] = []
  for idx, entry in enumerate(raw_claims):
    if not isinstance(entry, Mapping):
      raise ValueError(f"claims[{idx}] must be a JSON object.")
    claim_type = _require_str(entry, "claim_type")
    if claim_type not in _CLAIM_TYPES_SET:
      raise ValueError(
          f"claims[{idx}].claim_type '{claim_type}' is not in {CLAIM_TYPES}."
      )
    status = _require_str(entry, "status")
    if status not in _CLAIM_STATUSES_SET:
      raise ValueError(
          f"claims[{idx}].status '{status}' is not in {CLAIM_STATUSES}."
      )
    claims.append(
        ClaimAudit(
            claim=_require_str(entry, "claim"),
            claim_type=claim_type,
            explicit_in_statement=_require_bool(entry, "explicit_in_statement"),
            inherited_from_seed=_require_bool(entry, "inherited_from_seed"),
            status=status,
            evidence_snippet=_require_str(entry, "evidence_snippet"),
        )
    )

  quality = _require_str(decoded, "grounding_quality")
  if quality not in _QUALITY_LEVELS_SET:
    raise ValueError(
        f"Field 'grounding_quality' '{quality}' is not in"
        f" {GROUNDING_QUALITY_LEVELS}."
    )
  raw_flags = decoded.get("red_flags")
  if not isinstance(raw_flags, list) or not all(
      isinstance(flag, str) for flag in raw_flags
  ):
    raise ValueError("Field 'red_flags' must be a JSON array of strings.")

  return {
      "claims": tuple(claims),
      "grounding_quality": quality,
      "red_flags": tuple(flag.strip() for flag in raw_flags if flag.strip()),
      "rationale": _require_str(decoded, "rationale"),
  }


def build_grounding_verdict(
    item: GroundingAuditItem,
    parsed: Mapping[str, Any],
    evidence: EvidencePacket,
    *,
    trace_warnings: Sequence[str] = (),
    attempts: int,
    require_evidence_for_added_claims: bool = True,
) -> GroundingVerdict:
  """Combines session evidence stats and parsed claims into a final verdict."""
  claims: tuple[ClaimAudit, ...] = tuple(parsed["claims"])
  n_supported = sum(c.status == "supported_by_seen_evidence" for c in claims)
  n_contradicted = sum(
      c.status == "contradicted_by_seen_evidence" for c in claims
  )
  added_explicit = [
      c for c in claims if c.explicit_in_statement and not c.inherited_from_seed
  ]
  n_ungrounded_added = sum(
      c.status == "not_supported_in_session" for c in added_explicit
  )
  quality = str(parsed["grounding_quality"])

  reasons: list[str] = []
  if n_contradicted > 0:
    reasons.append(f"contradicted_claims ({n_contradicted})")
  if require_evidence_for_added_claims and n_ungrounded_added > 0:
    reasons.append(f"ungrounded_added_claims ({n_ungrounded_added})")
  if quality != "grounded":
    reasons.append(f"grounding_quality={quality}")
  if evidence.n_research_calls == 0 and added_explicit:
    reasons.append("no_session_research_evidence")

  return GroundingVerdict(
      item_id=item.item_id,
      passed=not reasons,
      grounding_quality=quality,
      failure_reasons=tuple(reasons),
      claims=claims,
      n_supported_claims=n_supported,
      n_contradicted_claims=n_contradicted,
      n_ungrounded_added_claims=n_ungrounded_added,
      red_flags=tuple(parsed["red_flags"]),
      rationale=str(parsed["rationale"]),
      n_research_calls=evidence.n_research_calls,
      files_accessed=evidence.files_accessed,
      evidence_truncated=evidence.truncated,
      session_db_path=(
          str(item.session_db_path) if item.session_db_path else None
      ),
      trace_warnings=tuple(trace_warnings),
      attempts=attempts,
      error=None,
  )


def _error_verdict(
    item: GroundingAuditItem,
    evidence: EvidencePacket,
    *,
    trace_warnings: Sequence[str],
    error_message: str,
    attempts: int,
) -> GroundingVerdict:
  """Builds a failing GroundingVerdict when evaluation exhausts retries."""
  return GroundingVerdict(
      item_id=item.item_id,
      passed=False,
      grounding_quality="ungrounded_or_contradicted",
      failure_reasons=(f"autorater_error: {error_message}",),
      claims=(),
      n_supported_claims=0,
      n_contradicted_claims=0,
      n_ungrounded_added_claims=0,
      red_flags=(),
      rationale="",
      n_research_calls=evidence.n_research_calls,
      files_accessed=evidence.files_accessed,
      evidence_truncated=evidence.truncated,
      session_db_path=(
          str(item.session_db_path) if item.session_db_path else None
      ),
      trace_warnings=tuple(trace_warnings),
      attempts=attempts,
      error=error_message,
  )


def _resolve_session_trace(
    item: GroundingAuditItem,
    trace_cache: dict[pathlib.Path, session_trace_reader.SessionTrace] | None,
) -> tuple[session_trace_reader.SessionTrace | None, list[str]]:
  """Resolves the session trace for `item`, reusing batch-local cache."""
  if item.session_trace is not None:
    return item.session_trace, list(item.session_trace.parse_warnings)
  if item.session_db_path is None:
    return None, []
  db_path = pathlib.Path(item.session_db_path)
  if trace_cache is not None and db_path in trace_cache:
    cached = trace_cache[db_path]
    return cached, list(cached.parse_warnings)
  trace = session_trace_reader.load_session_trace(db_path)
  if trace_cache is not None:
    trace_cache[db_path] = trace
  return trace, list(trace.parse_warnings)


async def _evaluate_grounding_item_with_cache(
    model: base_model.BaseModel,
    item: GroundingAuditItem,
    *,
    research_tools: Sequence[str] = session_trace_reader.DEFAULT_RESEARCH_TOOLS,
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET,
    require_evidence_for_added_claims: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
    trace_cache: (
        dict[pathlib.Path, session_trace_reader.SessionTrace] | None
    ) = None,
) -> GroundingVerdict:
  """Audits one refined proposal using an optional batch-local trace cache."""
  _validate_positive_int("evidence_char_budget", evidence_char_budget)
  _validate_positive_int("max_attempts", max_attempts)
  _validate_non_negative_float("temperature", temperature)
  _validate_non_negative_float("retry_delay_seconds", retry_delay_seconds)
  if not (item.seed_proposal or "").strip():
    raise ValueError(f"Item '{item.item_id}' has an empty seed_proposal.")
  if not (item.candidate_proposal or "").strip():
    raise ValueError(f"Item '{item.item_id}' has an empty candidate_proposal.")

  trace, trace_warnings = _resolve_session_trace(item, trace_cache)
  evidence = build_evidence_packet(
      trace,
      research_tools=research_tools,
      priority_terms=extract_priority_terms(item.candidate_proposal),
      char_budget=evidence_char_budget,
  )
  prompt = build_grounding_prompt(
      item, evidence, additional_context=additional_context
  )
  last_error = "unknown error"

  for attempt in range(1, max_attempts + 1):
    try:
      response = await model.generate_content(
          prompt=prompt,
          run_name=f"grounding_audit_{item.item_id}_attempt_{attempt}",
          temperature=temperature,
          system_prompt=GROUNDING_SYSTEM_PROMPT,
          response_mime_type="application/json",
          response_schema=GROUNDING_RESPONSE_SCHEMA,
      )
      response_error = (
          response.get("error") if isinstance(response, Mapping) else None
      )
      if response_error:
        raise RuntimeError(str(response_error))
      raw_text = (
          str(response.get("text") or "")
          if isinstance(response, Mapping)
          else ""
      )
      parsed = parse_grounding_response(raw_text)
      return build_grounding_verdict(
          item,
          parsed,
          evidence,
          trace_warnings=trace_warnings,
          attempts=attempt,
          require_evidence_for_added_claims=require_evidence_for_added_claims,
      )
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
      last_error = str(exc)
      logging.warning(
          "[%s] Grounding audit attempt %d/%d failed: %s",
          item.item_id,
          attempt,
          max_attempts,
          last_error,
      )
      if attempt < max_attempts and retry_delay_seconds > 0:
        await asyncio.sleep(retry_delay_seconds)

  return _error_verdict(
      item,
      evidence,
      trace_warnings=trace_warnings,
      error_message=last_error,
      attempts=max_attempts,
  )


async def evaluate_grounding_item(
    model: base_model.BaseModel,
    item: GroundingAuditItem,
    *,
    research_tools: Sequence[str] = session_trace_reader.DEFAULT_RESEARCH_TOOLS,
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET,
    require_evidence_for_added_claims: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
) -> GroundingVerdict:
  """Audits one refined proposal's claims against its session-DB evidence.

  WARNING: When `item.session_db_path` is supplied, session evidence is parsed
  by ``session_trace_reader`` from internal ``google-antigravity`` SQLite
  ``steps`` blobs, which may change across SDK versions.

  Args:
    model: Polymorphic ``BaseModel`` instance.
    item: Grounding audit record pointing to a ``SessionTrace`` or SQLite DB.
    research_tools: Tool names whose results count as corpus evidence.
    evidence_char_budget: Maximum character length of evidence excerpts (>= 1).
    require_evidence_for_added_claims: When True, unsupported explicit added
      claims fail the verdict.
    temperature: Judge sampling temperature (>= 0).
    max_attempts: Maximum attempts before returning an error verdict (>= 1).
    retry_delay_seconds: Delay between failed attempts (>= 0).
    additional_context: Optional domain or output-language context.

  Returns:
    Immutable ``GroundingVerdict`` for `item`.
  """
  return await _evaluate_grounding_item_with_cache(
      model,
      item,
      research_tools=research_tools,
      evidence_char_budget=evidence_char_budget,
      require_evidence_for_added_claims=require_evidence_for_added_claims,
      temperature=temperature,
      max_attempts=max_attempts,
      retry_delay_seconds=retry_delay_seconds,
      additional_context=additional_context,
      trace_cache=None,
  )


async def evaluate_grounding_batch(
    model: base_model.BaseModel,
    items: Sequence[GroundingAuditItem],
    *,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    research_tools: Sequence[str] = session_trace_reader.DEFAULT_RESEARCH_TOOLS,
    evidence_char_budget: int = DEFAULT_EVIDENCE_CHAR_BUDGET,
    require_evidence_for_added_claims: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
) -> list[GroundingVerdict]:
  """Evaluates a batch of grounding items concurrently with a semaphore."""
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  _validate_positive_int("evidence_char_budget", evidence_char_budget)
  _validate_positive_int("max_attempts", max_attempts)
  _validate_non_negative_float("temperature", temperature)
  _validate_non_negative_float("retry_delay_seconds", retry_delay_seconds)
  if not items:
    return []

  semaphore = asyncio.Semaphore(validated_concurrency)
  trace_cache: dict[pathlib.Path, session_trace_reader.SessionTrace] = {}

  async def _evaluate_one(item: GroundingAuditItem) -> GroundingVerdict:
    async with semaphore:
      return await _evaluate_grounding_item_with_cache(
          model,
          item,
          research_tools=research_tools,
          evidence_char_budget=evidence_char_budget,
          require_evidence_for_added_claims=require_evidence_for_added_claims,
          temperature=temperature,
          max_attempts=max_attempts,
          retry_delay_seconds=retry_delay_seconds,
          additional_context=additional_context,
          trace_cache=trace_cache,
      )

  return list(await asyncio.gather(*(_evaluate_one(item) for item in items)))


def summarize_grounding_verdicts(
    verdicts: Sequence[GroundingVerdict],
) -> dict[str, Any]:
  """Computes aggregate grounding pass/fail and claim counts."""
  total = len(verdicts)
  passed = sum(v.passed for v in verdicts)
  return {
      "total": total,
      "passed": passed,
      "failed": total - passed,
      "errors": sum(v.error is not None for v in verdicts),
      "grounded": sum(
          v.error is None and v.grounding_quality == "grounded"
          for v in verdicts
      ),
      "partly_ungrounded": sum(
          v.error is None and v.grounding_quality == "partly_ungrounded"
          for v in verdicts
      ),
      "ungrounded_or_contradicted": sum(
          v.error is None
          and v.grounding_quality == "ungrounded_or_contradicted"
          for v in verdicts
      ),
      "items_with_contradicted_claims": sum(
          v.n_contradicted_claims > 0 for v in verdicts
      ),
      "items_with_ungrounded_added_claims": sum(
          v.n_ungrounded_added_claims > 0 for v in verdicts
      ),
  }


def _optional_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
  """Returns the first non-empty string found under `keys`, or None."""
  for key in keys:
    value = mapping.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
      continue
    text = str(value).strip()
    if text:
      return text
  return None


def _resolve_db_for_refiner_record(
    record: Mapping[str, Any],
    sessions_dir: pathlib.Path | None,
) -> pathlib.Path | None:
  """Selects the matching Antigravity session DB for a refiner output record."""
  db_candidates: list[pathlib.Path] = []
  raw_dirs = record.get("session_dirs")
  if isinstance(raw_dirs, list):
    for raw_dir in raw_dirs:
      if isinstance(raw_dir, str) and raw_dir.strip():
        db_candidates.extend(
            session_trace_reader.find_session_dbs(pathlib.Path(raw_dir))
        )
  proposal_id = str(record.get("proposal_id") or "").strip()
  if not db_candidates and sessions_dir is not None:
    search_root = (
        sessions_dir / proposal_id
        if proposal_id and (sessions_dir / proposal_id).is_dir()
        else sessions_dir
    )
    db_candidates.extend(session_trace_reader.find_session_dbs(search_root))

  if not db_candidates:
    return None

  unique_dbs = list(dict.fromkeys(db_candidates))
  traces: list[session_trace_reader.SessionTrace] = []
  for db_path in unique_dbs:
    try:
      traces.append(session_trace_reader.load_session_trace(db_path))
    except (OSError, ValueError, sqlite3.Error) as exc:
      logging.warning("Skipping unreadable session DB %s: %s", db_path, exc)

  variations = record.get("variations") or []
  descriptions = [
      str(
          v.get("pre_simplified_description") or v.get("description") or ""
      ).strip()
      for v in variations
      if isinstance(v, Mapping)
  ]
  selected, _ = session_trace_reader.select_matching_session(
      traces, expected_descriptions=descriptions
  )
  return selected.db_path if selected is not None else None


def load_grounding_items(
    input_path: pathlib.Path,
    *,
    sessions_dir: pathlib.Path | None = None,
) -> list[GroundingAuditItem]:
  """Loads grounding audit items from `.json` or `.csv` files.

  Args:
    input_path: Path to a `.json` or `.csv` file.
    sessions_dir: Optional fallback directory containing Antigravity session
      DBs when records do not specify explicit ``session_db_path`` values.

  Returns:
    Ordered list of ``GroundingAuditItem`` records.

  Raises:
    FileNotFoundError: If `input_path` does not exist.
    ValueError: If the extension is unsupported or required fields are missing.
  """
  path = pathlib.Path(input_path)
  if not path.is_file():
    raise FileNotFoundError(f"Input grounding file not found: {path}")

  suffix = path.suffix.lower()
  if suffix == ".json":
    raw = json.loads(path.read_text())
    if isinstance(raw, Mapping) and "variations" in raw:
      records: list[Any] = [raw]
    elif isinstance(raw, Mapping) and isinstance(raw.get("items"), list):
      records = raw["items"]
    elif isinstance(raw, list):
      records = raw
    else:
      raise ValueError(
          "JSON input must be a list, contain 'items', or contain 'variations':"
          f" {path}"
      )
  elif suffix == ".csv":
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    records = df.to_dict(orient="records")
  else:
    raise ValueError(
        f"Unsupported input file extension '{suffix}'; expected .json or .csv."
    )

  items: list[GroundingAuditItem] = []
  for idx, rec in enumerate(records):
    if not isinstance(rec, Mapping):
      raise ValueError(f"Input record at index {idx} must be an object.")
    if "variations" in rec:
      proposal_id = str(rec.get("proposal_id") or "").strip()
      proposal_title = str(
          rec.get("proposal_title") or rec.get("title") or proposal_id
      ).strip()
      seed = _optional_text(
          rec, "original_statement", "seed_proposal", "statement"
      )
      if not seed:
        raise ValueError(
            f"Proposal '{proposal_id or '<unknown>'}' is missing"
            " 'original_statement' or 'seed_proposal'."
        )
      matched_db = _resolve_db_for_refiner_record(rec, sessions_dir)
      for v_idx, var in enumerate(rec.get("variations") or []):
        if not isinstance(var, Mapping):
          continue
        var_id = str(var.get("variation_id") or f"v{v_idx + 1:02d}").strip()
        item_id = f"{proposal_id}_{var_id}" if proposal_id else var_id
        candidate = _optional_text(
            var,
            "post_simplified_statement",
            "post_simplified_description",
            "description",
            "statement",
        )
        if not candidate:
          raise ValueError(f"Variation '{item_id}' is missing candidate text.")
        explicit_db = _optional_text(var, "session_db_path")
        db_path = pathlib.Path(explicit_db) if explicit_db else matched_db
        items.append(
            GroundingAuditItem(
                item_id=item_id,
                seed_proposal=seed,
                candidate_proposal=candidate,
                proposal_title=proposal_title,
                feasibility_assessment=_optional_text(
                    var, "feasibility_assessment"
                )
                or "",
                session_db_path=db_path,
            )
        )
    else:
      item_id = (
          _optional_text(rec, "item_id", "variation_id", "id")
          or f"item_{idx + 1}"
      )
      seed = _optional_text(
          rec, "seed_proposal", "original_statement", "seed_statement"
      )
      candidate = _optional_text(
          rec, "candidate_proposal", "description", "statement"
      )
      if not seed or not candidate:
        raise ValueError(
            f"Record '{item_id}' requires both 'seed_proposal' and"
            " 'candidate_proposal'."
        )
      explicit_db = _optional_text(rec, "session_db_path")
      db_path = pathlib.Path(explicit_db) if explicit_db else None
      if db_path is None and sessions_dir is not None:
        dbs = session_trace_reader.find_session_dbs(sessions_dir)
        db_path = dbs[-1] if dbs else None
      items.append(
          GroundingAuditItem(
              item_id=item_id,
              seed_proposal=seed,
              candidate_proposal=candidate,
              proposal_title=_optional_text(rec, "proposal_title", "title")
              or "",
              feasibility_assessment=_optional_text(
                  rec, "feasibility_assessment"
              )
              or "",
              session_db_path=db_path,
          )
      )
  return items


def write_grounding_verdicts(
    verdicts: Sequence[GroundingVerdict], output_path: pathlib.Path
) -> None:
  """Writes grounding verdicts and aggregate summary to `.json` or `.csv`."""
  path = pathlib.Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  suffix = path.suffix.lower()
  if suffix == ".json":
    payload = {
        "summary": summarize_grounding_verdicts(verdicts),
        "verdicts": [v.to_dict() for v in verdicts],
    }
    path.write_text(json.dumps(payload, indent=2))
  elif suffix == ".csv":
    rows: list[dict[str, Any]] = []
    for verdict in verdicts:
      row = verdict.to_dict()
      for col in (
          "failure_reasons",
          "claims",
          "red_flags",
          "files_accessed",
          "trace_warnings",
      ):
        row[col] = json.dumps(row[col])
      rows.append(row)
    columns = [field.name for field in dataclasses.fields(GroundingVerdict)]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
  else:
    raise ValueError(
        f"Unsupported output file extension '{suffix}'; expected .json or .csv."
    )
  logging.info("Wrote %d grounding verdict(s) to %s.", len(verdicts), path)


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the CLI argument parser for the session-DB grounding autorater."""
  parser = argparse.ArgumentParser(
      description=(
          "Audit refined actionable proposals against corpus tool results"
          " recovered from persisted Antigravity SQLite session databases."
      )
  )
  parser.add_argument(
      "--input_path",
      type=pathlib.Path,
      required=True,
      help=(
          "Path to input refined proposals (.json or .csv, including refiner.py"
          " output JSON)."
      ),
  )
  parser.add_argument(
      "--output_path",
      type=pathlib.Path,
      required=True,
      help="Destination path for grounding verdicts (.json or .csv).",
  )
  parser.add_argument(
      "--sessions_dir",
      type=pathlib.Path,
      default=None,
      help=(
          "Optional directory containing persisted Antigravity session DBs"
          " when input records do not embed session_db_path or session_dirs."
      ),
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help=f"Judge model name. Default: {DEFAULT_MODEL_NAME}.",
  )
  parser.add_argument(
      "--gemini_api_key",
      type=str,
      default=None,
      help="Optional Gemini API key passed to model_factory.get_model.",
  )
  parser.add_argument(
      "--temperature",
      type=_non_negative_float_arg,
      default=DEFAULT_TEMPERATURE,
      help=f"Judge sampling temperature. Default: {DEFAULT_TEMPERATURE}.",
  )
  parser.add_argument(
      "--evidence_char_budget",
      type=_positive_int_arg,
      default=DEFAULT_EVIDENCE_CHAR_BUDGET,
      help=(
          "Maximum character budget for recovered session evidence. Default:"
          f" {DEFAULT_EVIDENCE_CHAR_BUDGET}."
      ),
  )
  parser.add_argument(
      "-c",
      "--max_concurrent_calls",
      type=runner_utils.max_concurrent_calls_arg,
      default=DEFAULT_MAX_CONCURRENT_CALLS,
      help=(
          "Maximum concurrent judge LLM calls. Default:"
          f" {DEFAULT_MAX_CONCURRENT_CALLS}."
      ),
  )
  parser.add_argument(
      "--max_attempts",
      type=_positive_int_arg,
      default=DEFAULT_MAX_ATTEMPTS,
      help=f"Maximum attempts per item. Default: {DEFAULT_MAX_ATTEMPTS}.",
  )
  parser.add_argument(
      "--require_evidence_for_added_claims",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Fail verdicts whenever an explicit non-seed claim lacks supporting"
          " session evidence."
      ),
  )
  runner_utils.add_additional_context_args(parser)
  return parser


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments for the session-DB grounding autorater."""
  return build_arg_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GroundingAuditConfig:
  """Constructs an immutable ``GroundingAuditConfig`` from parsed CLI args."""
  return GroundingAuditConfig(
      input_path=args.input_path,
      output_path=args.output_path,
      sessions_dir=args.sessions_dir,
      temperature=args.temperature,
      evidence_char_budget=args.evidence_char_budget,
      max_concurrent_calls=args.max_concurrent_calls,
      max_attempts=args.max_attempts,
      require_evidence_for_added_claims=args.require_evidence_for_added_claims,
      additional_context=runner_utils.get_additional_context(args),
  )


async def run_grounding_audit(
    config: GroundingAuditConfig,
    *,
    model: base_model.BaseModel,
) -> list[GroundingVerdict]:
  """Loads grounding items, evaluates them with `model`, and writes verdicts."""
  items = load_grounding_items(
      config.input_path, sessions_dir=config.sessions_dir
  )
  verdicts = await evaluate_grounding_batch(
      model,
      items,
      max_concurrent_calls=config.max_concurrent_calls,
      evidence_char_budget=config.evidence_char_budget,
      require_evidence_for_added_claims=(
          config.require_evidence_for_added_claims
      ),
      temperature=config.temperature,
      max_attempts=config.max_attempts,
      retry_delay_seconds=config.retry_delay_seconds,
      additional_context=config.additional_context,
  )
  write_grounding_verdicts(verdicts, config.output_path)
  return verdicts


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for session-DB grounding verification."""
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
  )
  args = get_args(argv)
  config = config_from_args(args)
  model = model_factory.get_model(
      model_name=args.model_name,
      gemini_api_key=args.gemini_api_key,
  )
  asyncio.run(run_grounding_audit(config, model=model))


if __name__ == "__main__":
  main()
