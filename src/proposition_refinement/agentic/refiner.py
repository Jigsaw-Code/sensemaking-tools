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

"""Agentic proposal refinement runner with draft-slate gating and decomposition.

Refines actionable proposals against participant feedback and an optional local
reference corpus using the optional ``google-antigravity`` SDK:
- Configures an explicit ``CapabilitiesConfig`` allowlist with only
  ``BuiltinTools.FINISH`` plus pure-Python draft-slate and corpus tools.
- Binds a flat hand-written JSON ``response_schema`` and post-validates schema
  fields and array cardinality in Python.
- Persists every agent session under a caller-owned ``save_dir`` and bounds
  every session with ``asyncio.wait_for``.
- Uses simple ``asyncio.Semaphore`` concurrency and an outer retry loop with a
  fresh ``Agent`` instance per attempt to recover from transient 503/400
  session failures.
- Runs a second-pass decomposition session for over-stacked drafts that fail
  the two-component / two-sentence draft-slate gate, and optionally simplifies
  final statements via ``src.propositions.proposal_simplifier``.
"""

import argparse
import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import importlib
import json
import logging
import math
import pathlib
import re
from typing import Any

import pandas as pd

from src import runner_utils
from src.models import base_model
from src.models import model_factory
from src.proposition_refinement.agentic import candidate_drafts
from src.proposition_refinement.agentic import corpus_tools
from src.propositions import proposal_simplifier

DEFAULT_MODEL_NAME: str = "gemini-3.8-flash"
DEFAULT_MIN_VARIATIONS: int = 3
DEFAULT_MAX_VARIATIONS: int = 10
DEFAULT_MAX_CONCURRENT_SESSIONS: int = 4
DEFAULT_MAX_ATTEMPTS: int = 6
DEFAULT_TIMEOUT_SECONDS: float = 600.0
DEFAULT_RETRY_DELAY_SECONDS: float = 2.0
DEFAULT_MAX_STACKED_FOR_DECOMPOSITION: int = 6

REQUIRED_VARIATION_FIELDS: tuple[str, ...] = (
    "variation_id",
    "candidate_id",
    "title",
    "description",
    "target_friction_resolved",
    "refinement_archetype",
    "feasibility_assessment",
)

_TRANSIENT_ERROR_MARKERS: frozenset[str] = frozenset({
    "503",
    "429",
    "resource_exhausted",
    "exceeded your current quota",
    "rate limit",
    "unavailable",
    "high demand",
    "prefill_queue_preempted",
    "decode_preempted",
    "capacity",
    "requests ending with a model turn are not supported",
    "connection error",
    "connection reset",
    "econnreset",
    "server error",
    "handshake",
})

_SLUG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")

AGENTIC_SYSTEM_INSTRUCTIONS: str = (
    "You are an expert deliberative analyst and civic facilitation assistant."
    " Your objective is to refine deliberated actionable proposals so that they"
    " address participant concerns, mitigate perceived risks, verify"
    " operational, legal, and fiscal feasibility against available reference"
    " materials, and earn broader cross-cutting consensus while preserving each"
    " proposal's essential core purpose. Draft every candidate statement into"
    " your candidate draft slate, check each candidate individually, and revise"
    " or reject it as needed. Only return candidates that you have verified as"
    " feasible and focused; returning fewer variations is always better than"
    " returning an unverified or over-stuffed one. Always output pure valid"
    " JSON matching the requested schema."
)

_ADDITIONAL_CONTEXT_TEMPLATE: str = (
    "<additionalContext>\n{additional_context}\n</additionalContext>"
)

_DELIBERATION_FRAMING_TEMPLATE: str = (
    "### Deliberation Context & Evidence Structure\n\n"
    "- **Parallel Discussion Groups:** Participants deliberated across"
    " {group_count} independent discussion groups covering the shared topic:"
    ' **"{topic}"**.\n'
    "- **Comparative Bundle Reasoning:** When companion proposals were"
    " discussed alongside the focal proposal, comparative remarks provide"
    " signal about participant trade-offs and priorities. Keep every"
    " refinement anchored on the focal proposal itself."
)

_REFINEMENT_TASK_TEMPLATE: str = (
    "### REFINEMENT GENERATION TASK:\n"
    "Generate no more than {max_variations} (and at least {min_variations})"
    " distinct, implementable, and focused variations for the focal actionable"
    " proposal '{title}' (`{proposal_id}`).\n\n"
    "**Only feasible, implementable variations may be returned.** Every"
    " returned variation must be individually verified following the WORKFLOW"
    " below{corpus_clause}. If a candidate cannot be made feasible, drop it;"
    " returning fewer variations is always better than returning an infeasible"
    " or unverified one.\n\n"
    "Every variation must:\n"
    "1. **Build on and Reuse Original Proposal Text (Complete &"
    " Self-Contained):**\n"
    "   - Address critiques by building on or amending the existing focal"
    " proposal rather than writing from scratch.\n"
    '   - Reuse literal text from the original proposal ("{statement}")'
    " wherever consistent with the intended refinement.\n"
    "   - Write each statement as a complete, self-contained actionable"
    " proposal (not a delta or amendment that only makes sense alongside the"
    " original).\n"
    "   - Preserve the original proposal's core purpose (what it does and what"
    " its resources or funding are for) in every variation.\n"
    "   - Avoid solving unrelated problems unless participants specifically"
    " linked them to improving this proposal.\n"
    "2. **Ground Directly in Participant Objections, Critiques, Trade-Offs, and"
    " Suggestions:**\n"
    "   - Address specific concerns, trade-offs, and improvement suggestions"
    " surfaced in the participant feedback directly within the proposal's"
    " design.\n"
    "   - Preserve the strengths that existing supporters value while resolving"
    " friction for hesitant or skeptical groups.\n"
    "3. **Limit New Components to Focused, Cohesive Amendments (Avoid Mechanism"
    " Stacking):**\n"
    "   - Introduce at most one primary new mechanism, safeguard, or component"
    " (or two only if closely related and cohesive in execution).\n"
    "   - **Hard limit:** Never add more than **two new components** or more"
    " than **two additional sentences** beyond the original proposal. Split"
    " distinct participant ideas into separate variations so reviewers can"
    " evaluate each change independently.\n"
    "4. **Respect Governing Authority, Funding Constraints, and Trade-Off"
    " Transparency:**\n"
    "   - Prioritize actions the responsible authority can carry out within its"
    " own jurisdiction and funding rules.\n"
    "   - If a participant-requested change requires external approval, voter"
    " authorization, or enabling action, state the trade-offs, delays, or"
    " conditions explicitly in the proposal.\n"
    "   - Be transparent about funding: state whether costs, taxes, or fees"
    " increase and who pays, and if funds are reallocated, state what they"
    " would otherwise cover.\n"
    "5. **Plain-English Readability and Imperative Mood:**\n"
    "   - Concise title and straightforward vocabulary avoiding unnecessary"
    " jargon.\n"
    "   - Target an ~8th-grade reading level (Flesch-Kincaid Grade Level <="
    " 9.5) and 2-4 sentences when possible (6 sentences max for already long"
    " originals).\n"
    "   - Write in the imperative mood opening directly with an action verb"
    " (such as 'Establish...', 'Require...', 'Set aside...', 'Publish...')"
    " without boilerplate subject prefixes."
)

_WORKFLOW_CORPUS_HEADER: str = (
    "### WORKFLOW:\n"
    "You have access to the local reference corpus via `list_corpus_files`,"
    " `search_corpus`, and `read_corpus_file`, and a candidate draft slate via"
    " `write_candidate_draft` and `view_candidate_drafts`. Follow this"
    " workflow before emitting your final answer:\n"
    "0. **Map the Reference Corpus First:** Call `list_corpus_files` to inspect"
    " available documents and datasets before searching. Use `search_corpus`"
    " (with `file_filter` when useful) and `read_corpus_file` to verify"
    " authority, budgets, operational rules, and constraints.\n"
)

_WORKFLOW_NO_CORPUS_HEADER: str = (
    "### WORKFLOW:\n"
    "You have access to a candidate draft slate via `write_candidate_draft` and"
    " `view_candidate_drafts`. Follow this workflow before emitting your final"
    " answer:\n"
)

_WORKFLOW_BODY_TEMPLATE: str = (
    "1. **Write Your Draft Slate:** Draft each candidate refinement with"
    " `write_candidate_draft` (status `draft`), one call per candidate using"
    " stable ids (`c01`, `c02`, ...). Write the complete actionable proposal"
    " statement verbatim. Draft at least {min_variations} candidates; you may"
    " draft more than {max_variations} so failing ones can be rejected.\n"
    "2. **Check Each Candidate Individually:** Verify every candidate against"
    " all directives:\n"
    "   - **(a) Focus (hard limit):** At most two declared new components and"
    " at most two more sentences than the original proposal.\n"
    "   - **(b) Relevance & Core Purpose:** Each change addresses participant"
    " feedback on THIS proposal and keeps its core purpose intact.\n"
    "   - **(c) Feasibility & Funding Honesty:** {feasibility_step} State"
    " funding sources, cost impacts, and any external approval trade-offs"
    " clearly.\n"
    "3. **Revise, Split, Verify, or Reject in the Slate:** Call"
    " `write_candidate_draft` with the same id:\n"
    "   - **Revise** statements to resolve feasibility or clarity issues.\n"
    "   - **Split** over-stacked candidates into separate focused candidates.\n"
    "   - **Verify** (status `verified`) only when every check passes, passing"
    " `new_components` (one per line), `seed_purpose_preserved` ('yes: ...'),"
    " and `feasibility_assessment`.\n"
    "   - **Reject** (status `rejected` with reasons in"
    " `feasibility_assessment`) any candidate that cannot pass, and draft a"
    " replacement.\n"
    "4. **Review the Slate:** Call `view_candidate_drafts` to confirm you have"
    " at least {min_variations} distinct verified candidates.\n"
    "5. **Return Only Verified Candidates:** Emit JSON with at most"
    " {max_variations} verified candidates. Set `candidate_id` to the verified"
    " slate id, copy its statement verbatim into `description`, and copy its"
    " `feasibility_assessment`."
)

_OUTPUT_FORMAT_TEMPLATE: str = (
    "### OUTPUT FORMAT:\n"
    "Return a valid JSON object matching this schema:\n"
    "{{\n"
    '  "proposal_id": "{proposal_id}",\n'
    '  "variations": [\n'
    "    {{\n"
    '      "variation_id": "v01",\n'
    '      "candidate_id": "c01",\n'
    '      "title": "Short Refined Proposal Title",\n'
    '      "description": "Verified draft statement copied verbatim.",\n'
    '      "target_friction_resolved": "Specific participant concern'
    ' addressed.",\n'
    '      "refinement_archetype": "Focused refinement mechanism name.",\n'
    '      "feasibility_assessment": "Checks performed and findings for each'
    ' change."\n'
    "    }}\n"
    "  ]\n"
    "}}"
)

_DECOMPOSITION_PROMPT_TEMPLATE: str = (
    "### DECOMPOSITION TASK:\nThe candidate refinements below of focal"
    " actionable proposal '{title}' (`{proposal_id}`) each add too many"
    " components or sentences beyond the original proposal. Split each one into"
    " smaller, focused candidates that reviewers can evaluate"
    " independently.{context_block}\n\nOriginal proposal statement:"
    ' "{statement}"\n\nOver-stacked candidates:\n\n{stacked_items}\n\nFor'
    " each over-stacked candidate, write separate new candidates that:\n- Reuse"
    " the original proposal's literal text and preserve its core purpose.\n-"
    " Add only ONE focused idea (at most two new components and at most two"
    " more sentences than the original proposal).\n- Address the participant"
    " concern the idea targeted without bundling unrelated changes.\n- Use"
    " plain-English imperative drafting (~8th-grade reading level, 2-4"
    " sentences when possible, 6 max) and state funding trade-offs"
    " honestly.\n\n### WORKFLOW:\n{tool_guidance}1. Draft each split candidate"
    " with `write_candidate_draft` (status `draft`) using ids formed from the"
    " source id plus a letter (for example `{example_id}-a`,"
    " `{example_id}-b`).\n2. Check each split candidate's feasibility"
    " individually.\n3. Mark it `verified` only when every check passes,"
    " passing `new_components`, `seed_purpose_preserved` ('yes: ...'), and"
    " `feasibility_assessment`. Reject any split that cannot pass.\n4. Return"
    " at most {max_out} `verified` split candidates matching the JSON schema"
    " (`proposal_id`: `{proposal_id}`), copying each verified statement"
    " verbatim into `description` and its slate id into `candidate_id`."
)


@dataclasses.dataclass(frozen=True, slots=True)
class CompanionProposal:
  """A companion proposal deliberated alongside the focal proposal."""

  proposal_id: str
  title: str
  statement: str


@dataclasses.dataclass(frozen=True, slots=True)
class ProposalSpec:
  """Specification of a focal actionable proposal to refine."""

  proposal_id: str
  title: str
  statement: str
  topic: str = ""
  companion_proposals: tuple[CompanionProposal, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class FeedbackStatement:
  """One participant feedback statement from deliberation."""

  speaker_id: str
  text: str
  group_id: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class SessionRequest:
  """All parameters required to execute a single agentic refinement session."""

  proposal: ProposalSpec
  prompt: str
  system_instructions: str
  model_name: str
  min_variations: int
  max_variations: int
  response_schema: dict[str, Any]
  save_dir: pathlib.Path
  tools: tuple[Callable[..., str], ...]
  slate: candidate_drafts.CandidateDraftSlate
  label: str
  attempt: int
  api_key: str | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class SessionResult:
  """Raw output returned by one agentic refinement session."""

  structured_output: Any = None
  raw_text: str = ""


SessionExecutor = Callable[[SessionRequest], Awaitable[SessionResult]]


def _validate_positive_int(name: str, value: int) -> int:
  """Validates that `value` is an integer >= 1 (excluding bools)."""
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
  if value < 1:
    raise ValueError(f"{name} must be >= 1, got {value}.")
  return value


def _validate_positive_float(name: str, value: float) -> float:
  """Validates that `value` is a finite number > 0."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
  if not math.isfinite(value) or value <= 0:
    raise ValueError(f"{name} must be a finite number > 0, got {value}.")
  return float(value)


def _validate_non_negative_float(name: str, value: float) -> float:
  """Validates that `value` is a finite number >= 0."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
  if not math.isfinite(value) or value < 0:
    raise ValueError(f"{name} must be a finite number >= 0, got {value}.")
  return float(value)


def _positive_int_arg(value: str) -> int:
  """Argparse `type=` validating an integer >= 1."""
  try:
    parsed = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    ) from exc
  if parsed < 1:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    )
  return parsed


def _positive_float_arg(value: str) -> float:
  """Argparse `type=` validating a finite float > 0."""
  try:
    parsed = float(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be a finite number > 0."
    ) from exc
  if not math.isfinite(parsed) or parsed <= 0:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be a finite number > 0."
    )
  return parsed


def build_response_schema(
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
) -> dict[str, Any]:
  """Builds a flat JSON schema binding variation fields and count range.

  Produces a hand-written schema with no ``$defs`` or ``$ref`` indirection for
  reliable constrained decoding. Because constrained decoders do not always
  enforce ``minItems``/``maxItems``, callers must post-validate outputs via
  ``validate_variation_payload``.

  Args:
    min_variations: Minimum number of variations requested (>= 1).
    max_variations: Maximum number of variations requested (>= min_variations).

  Returns:
    Flat JSON schema dictionary suitable for ``LocalAgentConfig``.
  """
  _validate_positive_int("min_variations", min_variations)
  _validate_positive_int("max_variations", max_variations)
  if max_variations < min_variations:
    raise ValueError(
        f"max_variations ({max_variations}) must be >= min_variations"
        f" ({min_variations})."
    )
  return {
      "type": "object",
      "required": ["proposal_id", "variations"],
      "properties": {
          "proposal_id": {"type": "string"},
          "variations": {
              "type": "array",
              "minItems": min_variations,
              "maxItems": max_variations,
              "items": {
                  "type": "object",
                  "required": list(REQUIRED_VARIATION_FIELDS),
                  "properties": {
                      "variation_id": {"type": "string"},
                      "candidate_id": {
                          "type": "string",
                          "description": (
                              "Id of the verified draft in the candidate slate"
                              " that this variation reproduces."
                          ),
                      },
                      "title": {"type": "string"},
                      "description": {
                          "type": "string",
                          "description": (
                              "Complete, self-contained actionable proposal"
                              " statement at ~8th-grade reading level copied"
                              " verbatim from the verified draft."
                          ),
                      },
                      "target_friction_resolved": {"type": "string"},
                      "refinement_archetype": {"type": "string"},
                      "feasibility_assessment": {
                          "type": "string",
                          "description": (
                              "Verification findings explaining what was"
                              " checked for each new component or change."
                          ),
                      },
                  },
              },
          },
      },
  }


def clean_json_response(raw_text: str) -> str:
  """Strips optional markdown code fences from an LLM text response."""
  text = (raw_text or "").strip()
  if text.startswith("```json"):
    text = text.removeprefix("```json")
  elif text.startswith("```"):
    text = text.removeprefix("```")
  if text.endswith("```"):
    text = text.removesuffix("```")
  return text.strip()


def extract_payload(structured_output: Any, raw_text: str) -> tuple[Any, str]:
  """Extracts a JSON payload from structured output or fenced text fallback.

  Args:
    structured_output: Parsed structured output from the agent finish step.
    raw_text: Fallback text output from the agent session.

  Returns:
    Tuple ``(payload, source)`` where ``source`` is ``"structured"``,
    ``"text"``, or ``"empty"``.
  """
  if isinstance(structured_output, dict):
    return structured_output, "structured"
  cleaned = clean_json_response(raw_text)
  if not cleaned:
    return None, "empty"
  try:
    return json.loads(cleaned), "text"
  except json.JSONDecodeError:
    return None, "empty"


def validate_variation_payload(
    payload: Any,
    *,
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
) -> tuple[list[dict[str, Any]], list[str]]:
  """Post-validates schema structure, required fields, and array cardinality.

  Args:
    payload: Decoded JSON object returned by an agent session.
    min_variations: Minimum required number of valid variations.
    max_variations: Maximum allowed number of variations.

  Returns:
    Tuple ``(valid_variations, validation_errors)``. Valid items are copied as
    plain dicts; ``validation_errors`` is empty only when structure, required
    string fields, and array cardinality all pass.
  """
  _validate_positive_int("min_variations", min_variations)
  _validate_positive_int("max_variations", max_variations)
  if max_variations < min_variations:
    raise ValueError(
        f"max_variations ({max_variations}) must be >= min_variations"
        f" ({min_variations})."
    )

  if not isinstance(payload, Mapping):
    return [], ["Response payload must be a JSON object."]

  errors: list[str] = []
  proposal_id = payload.get("proposal_id")
  if not isinstance(proposal_id, str) or not proposal_id.strip():
    errors.append("Payload field 'proposal_id' must be a non-empty string.")

  raw_variations = payload.get("variations")
  if not isinstance(raw_variations, list):
    errors.append("Payload field 'variations' must be a JSON array.")
    return [], errors

  valid: list[dict[str, Any]] = []
  for idx, item in enumerate(raw_variations):
    if not isinstance(item, Mapping):
      errors.append(f"variations[{idx}] must be a JSON object.")
      continue
    missing = [
        field
        for field in REQUIRED_VARIATION_FIELDS
        if not isinstance(item.get(field), str) or not item[field].strip()
    ]
    if missing:
      errors.append(
          f"variations[{idx}] missing non-empty string field(s):"
          f" {', '.join(missing)}."
      )
      continue
    valid.append(dict(item))

  if len(valid) < min_variations:
    errors.append(
        f"Expected at least {min_variations} valid variations, got"
        f" {len(valid)}."
    )
  if len(raw_variations) > max_variations:
    errors.append(
        f"Expected at most {max_variations} variations, got"
        f" {len(raw_variations)}."
    )
  return valid, errors


def is_transient_session_error(exc: Exception) -> bool:
  """Returns True when `exc` indicates a transient infrastructure/API error."""
  message = str(exc).lower()
  return any(marker in message for marker in _TRANSIENT_ERROR_MARKERS)


def _format_additional_context(additional_context: str | None) -> str:
  """Formats optional additional context into a prompt XML block."""
  if not additional_context or not additional_context.strip():
    return ""
  return _ADDITIONAL_CONTEXT_TEMPLATE.format(
      additional_context=additional_context.strip()
  )


def format_proposal_bundle(focal: ProposalSpec) -> str:
  """Formats the focal proposal and any companion proposals as markdown."""
  lines = [
      "### Actionable Proposal Bundle",
      "",
      "#### Focal Proposal to Refine",
      f"- **Proposal Identifier:** `{focal.proposal_id}`",
      f"- **Title:** {focal.title}",
      f'- **Original Operative Statement:** "{focal.statement}"',
  ]
  if focal.companion_proposals:
    lines.extend([
        "",
        "#### Companion Proposals Deliberated in the Same Bundle",
        (
            "These companion proposals were presented alongside the focal"
            " proposal in comparative discussion:"
        ),
        "",
    ])
    for companion in focal.companion_proposals:
      lines.extend([
          f"- **{companion.title}** (`{companion.proposal_id}`):",
          f'  "{companion.statement}"',
          "",
      ])
  return "\n".join(lines).rstrip()


def format_feedback_context(
    focal: ProposalSpec,
    statements: Sequence[FeedbackStatement],
    *,
    topic: str = "",
) -> str:
  """Formats participant feedback statements grouped by discussion group.

  Args:
    focal: Focal actionable proposal being refined.
    statements: Participant feedback statements.
    topic: Optional deliberation topic override (defaults to `focal.topic`).

  Returns:
    Formatted markdown dossier ready for inclusion in the refinement prompt.
  """
  grouped: dict[str, list[FeedbackStatement]] = {}
  for stmt in statements:
    group_key = stmt.group_id.strip() or "General"
    grouped.setdefault(group_key, []).append(stmt)

  sections: list[str] = []
  resolved_topic = topic.strip() or focal.topic.strip()
  if resolved_topic and grouped:
    sections.append(
        _DELIBERATION_FRAMING_TEMPLATE.format(
            group_count=len(grouped), topic=resolved_topic
        )
    )
  sections.append(format_proposal_bundle(focal))

  if grouped:
    evidence_lines = ["### Participant Feedback Statements", ""]
    for group_key in sorted(grouped):
      group_items = grouped[group_key]
      evidence_lines.append(
          f"#### Group {group_key} ({len(group_items)} statements)"
      )
      evidence_lines.append("")
      for stmt in group_items:
        clean = " ".join(stmt.text.split())
        evidence_lines.append(f'- **[{stmt.speaker_id}]:** "{clean}"')
      evidence_lines.append("")
    sections.append("\n".join(evidence_lines).rstrip())
  return "\n\n".join(sections)


def build_refinement_task_block(
    proposal: ProposalSpec,
    *,
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
    has_corpus: bool = True,
) -> str:
  """Builds the core refinement task block placed around feedback context."""
  corpus_clause = " against the local reference corpus" if has_corpus else ""
  return _REFINEMENT_TASK_TEMPLATE.format(
      min_variations=min_variations,
      max_variations=max_variations,
      title=proposal.title,
      proposal_id=proposal.proposal_id,
      statement=proposal.statement,
      corpus_clause=corpus_clause,
  )


def build_workflow_block(
    proposal: ProposalSpec,
    *,
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
    has_corpus: bool = True,
) -> str:
  """Builds the post-context verification workflow and JSON format block."""
  header = _WORKFLOW_CORPUS_HEADER if has_corpus else _WORKFLOW_NO_CORPUS_HEADER
  if has_corpus:
    feasibility_step = (
        "Research each new component in the reference corpus for authority,"
        " funding limits, operational constraints, or conflicting facts, and"
        " record what you checked and found."
    )
  else:
    feasibility_step = (
        "Assess each new component's operational, legal, and funding"
        " feasibility against the supplied context and record what you checked"
        " and found."
    )
  body = _WORKFLOW_BODY_TEMPLATE.format(
      min_variations=min_variations,
      max_variations=max_variations,
      feasibility_step=feasibility_step,
  )
  output_format = _OUTPUT_FORMAT_TEMPLATE.format(
      proposal_id=proposal.proposal_id
  )
  return f"{header}{body}\n\n{output_format}"


def build_refinement_prompt(
    proposal: ProposalSpec,
    feedback_context: str,
    *,
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
    additional_context: str | None = None,
    has_corpus: bool = True,
) -> str:
  """Assembles an instruction-sandwich refinement prompt.

  Places the refinement directives before and after the participant feedback
  dossier, injects optional ``additional_context`` in an
  ``<additionalContext>`` block, and appends the draft-slate verification
  workflow and JSON output format.

  Args:
    proposal: Focal actionable proposal to refine.
    feedback_context: Participant feedback or dialogue context dossier.
    min_variations: Minimum number of variations requested.
    max_variations: Maximum number of variations requested.
    additional_context: Optional domain or output-language guidance.
    has_corpus: Whether local reference corpus tools are attached.

  Returns:
    Complete prompt string ready for an agent session.
  """
  task_block = build_refinement_task_block(
      proposal,
      min_variations=min_variations,
      max_variations=max_variations,
      has_corpus=has_corpus,
  )
  workflow_block = build_workflow_block(
      proposal,
      min_variations=min_variations,
      max_variations=max_variations,
      has_corpus=has_corpus,
  )
  parts = [task_block]
  context_block = _format_additional_context(additional_context)
  if context_block:
    parts.append(context_block)
  if feedback_context.strip():
    parts.append(feedback_context.strip())
  else:
    parts.append(format_proposal_bundle(proposal))
  parts.append(task_block)
  parts.append(workflow_block)
  return "\n\n".join(parts)


def collect_stacked_candidates(
    slate: candidate_drafts.CandidateDraftSlate,
    filtered_variations: Sequence[Mapping[str, Any]],
    *,
    max_items: int = DEFAULT_MAX_STACKED_FOR_DECOMPOSITION,
) -> list[dict[str, Any]]:
  """Collects over-stacked drafts and filtered variations for decomposition.

  Args:
    slate: Main session's candidate draft slate.
    filtered_variations: Returned variations dropped by the hard addition
      filter.
    max_items: Maximum number of stacked candidates to forward.

  Returns:
    Deduplicated list of dicts containing ``candidate_id``, ``title``,
    ``statement``, ``target_friction_resolved``, and ``new_components``.
  """
  _validate_positive_int("max_items", max_items)
  collected: list[dict[str, Any]] = []
  seen: set[str] = set()

  for draft in slate.stacking_failures():
    key = candidate_drafts.normalize_text(draft.statement)
    if key and key not in seen:
      seen.add(key)
      collected.append({
          "candidate_id": draft.candidate_id,
          "title": draft.title,
          "statement": draft.statement,
          "target_friction_resolved": draft.target_friction_resolved,
          "new_components": draft.new_components,
      })

  for variation in filtered_variations:
    statement = str(variation.get("description", ""))
    key = candidate_drafts.normalize_text(statement)
    if key and key not in seen:
      seen.add(key)
      cid = str(
          variation.get("candidate_id") or variation.get("variation_id") or ""
      )
      collected.append({
          "candidate_id": cid,
          "title": str(variation.get("title", "")),
          "statement": statement,
          "target_friction_resolved": str(
              variation.get("target_friction_resolved", "")
          ),
          "new_components": "",
      })
  return collected[:max_items]


def build_decomposition_prompt(
    proposal: ProposalSpec,
    stacked: Sequence[Mapping[str, Any]],
    *,
    max_out: int,
    has_corpus: bool = True,
    additional_context: str | None = None,
) -> str:
  """Builds the prompt for a follow-up session splitting over-stacked drafts."""
  _validate_positive_int("max_out", max_out)
  items = "\n\n".join(
      f"**{item['candidate_id']}: {item['title']}**\n"
      f"Statement: {item['statement']}\n"
      f"Participant concern targeted: {item['target_friction_resolved']}\n"
      "Declared new components:"
      f" {item.get('new_components') or '(not declared)'}"
      for item in stacked
  )
  example_id = str(stacked[0]["candidate_id"]) if stacked else "c01"
  if has_corpus:
    tool_guidance = (
        "Use `list_corpus_files`, `search_corpus`, and `read_corpus_file` to"
        " check each split candidate's feasibility, and"
        " `write_candidate_draft` / `view_candidate_drafts` to record it.\n"
    )
  else:
    tool_guidance = (
        "Use `write_candidate_draft` and `view_candidate_drafts` to record and"
        " verify each split candidate.\n"
    )
  raw_context = _format_additional_context(additional_context)
  context_block = f"\n\n{raw_context}" if raw_context else ""
  return _DECOMPOSITION_PROMPT_TEMPLATE.format(
      title=proposal.title,
      proposal_id=proposal.proposal_id,
      context_block=context_block,
      statement=proposal.statement,
      stacked_items=items,
      tool_guidance=tool_guidance,
      example_id=example_id,
      max_out=max_out,
  )


def _import_antigravity() -> Any:
  """Lazily imports the optional `google.antigravity` SDK."""
  try:
    return importlib.import_module("google.antigravity")
  except ImportError as exc:
    raise ImportError(
        "The optional 'google-antigravity' SDK is not installed in this"
        " environment. Install 'google-antigravity' to run live agentic"
        " refinement sessions, or pass an explicit session_executor."
    ) from exc


def _validate_save_dir(save_dir: pathlib.Path | str | None) -> pathlib.Path:
  """Validates that `save_dir` is an explicit persistent directory path."""
  if save_dir is None:
    raise ValueError("A persistent caller-owned save_dir is required.")
  raw = str(save_dir).strip()
  if not raw or raw == ".":
    raise ValueError(
        "save_dir must be a non-empty persistent directory path (not '' or"
        " '.')."
    )
  return pathlib.Path(raw)


def build_agent_config(request: SessionRequest) -> Any:
  """Builds an explicit, capability-restricted Antigravity LocalAgentConfig.

  Always binds ``CapabilitiesConfig(enabled_tools=[BuiltinTools.FINISH],
  enable_subagents=False)`` and a persistent caller-owned ``save_dir`` so
  built-in filesystem/shell tools remain disabled and session SQLite traces are
  retained for auditability.

  Args:
    request: SessionRequest containing model, schema, tools, and save_dir.

  Returns:
    Configured ``google.antigravity.LocalAgentConfig`` instance.

  Raises:
    ImportError: If ``google-antigravity`` is not installed.
    ValueError: If ``request.save_dir`` is empty or relative current dir.
  """
  validated_save_dir = _validate_save_dir(request.save_dir)
  antigravity = _import_antigravity()
  optional_kwargs: dict[str, Any] = {}
  if request.api_key is not None:
    optional_kwargs["api_key"] = request.api_key
  return antigravity.LocalAgentConfig(
      model=request.model_name,
      capabilities=antigravity.CapabilitiesConfig(
          enabled_tools=[antigravity.BuiltinTools.FINISH],
          enable_subagents=False,
      ),
      tools=list(request.tools),
      response_schema=request.response_schema,
      system_instructions=antigravity.CustomSystemInstructions(
          text=request.system_instructions
      ),
      save_dir=str(validated_save_dir),
      **optional_kwargs,
  )


async def run_antigravity_session(request: SessionRequest) -> SessionResult:
  """Executes one fresh Antigravity SDK Agent session for `request`."""
  antigravity = _import_antigravity()
  config = build_agent_config(request)
  async with antigravity.Agent(config) as agent:
    chat_response = await agent.chat(request.prompt)
    structured = await chat_response.structured_output()
    text = await chat_response.text()
    return SessionResult(structured_output=structured, raw_text=text or "")


async def _simplify_variations(
    model: base_model.BaseModel,
    variations: Sequence[Mapping[str, Any]],
    *,
    max_fkgl: float = proposal_simplifier.DEFAULT_MAX_FKGL,
    max_sentences: int = proposal_simplifier.DEFAULT_MAX_SENTENCES,
    additional_context: str | None = None,
) -> list[dict[str, Any]]:
  """Runs optional readability-asserted simplification on final variations."""
  if not variations:
    return []
  statements = [str(v.get("description", "")) for v in variations]
  results = await proposal_simplifier.simplify_proposals(
      model,
      statements,
      max_fkgl=max_fkgl,
      max_sentences=max_sentences,
      additional_context=additional_context,
  )
  enriched: list[dict[str, Any]] = []
  for variation, res in zip(variations, results):
    updated = dict(variation)
    updated["pre_simplified_description"] = res.pre_simplified_statement
    updated["description"] = res.post_simplified_statement
    updated["pre_fkgl"] = res.pre_score.fkgl
    updated["post_fkgl"] = res.post_score.fkgl
    updated["readability_compliant"] = res.compliant
    updated["simplification_attempts"] = res.attempts
    enriched.append(updated)
  return enriched


def _create_session_dir(
    sessions_root: pathlib.Path,
    model_name: str,
    label: str,
    sequence_num: int,
) -> pathlib.Path:
  """Creates a unique persistent directory for one agent session attempt."""
  stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
      "%Y%m%dT%H%M%S%fZ"
  )
  model_slug = _SLUG_SANITIZE_RE.sub("-", model_name).strip("-") or "model"
  session_dir = (
      sessions_root / f"{model_slug}__{label}{sequence_num:02d}__{stamp}"
  )
  session_dir.mkdir(parents=True, exist_ok=False)
  return session_dir


async def refine_proposal(
    proposal: ProposalSpec,
    feedback_context: str,
    *,
    save_dir: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    corpus_dir: pathlib.Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
    require_verified_drafts: bool = True,
    decompose_stacked: bool = True,
    simplifier_model: base_model.BaseModel | None = None,
    simplifier_max_fkgl: float = proposal_simplifier.DEFAULT_MAX_FKGL,
    simplifier_max_sentences: int = proposal_simplifier.DEFAULT_MAX_SENTENCES,
    session_executor: SessionExecutor | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
  """Refines a single actionable proposal via gated agentic sessions.

  Each retry attempt creates a fresh persistent subdirectory under `save_dir`,
  a fresh `CandidateDraftSlate`, and a fresh agent session bounded by
  `asyncio.wait_for(..., timeout=timeout_seconds)`.

  Args:
    proposal: Focal actionable proposal specification.
    feedback_context: Participant feedback or dialogue context dossier.
    save_dir: Caller-owned persistent root directory for SQLite session traces
      and draft-slate JSON files.
    output_dir: Optional directory where ``{proposal_id}_variations.json`` is
      written.
    corpus_dir: Optional local reference corpus directory exposed via CL 9
      corpus tools.
    model_name: Model name passed to the agent session.
    min_variations: Minimum accepted variations required (>= 1).
    max_variations: Maximum variations returned (>= min_variations).
    max_attempts: Maximum outer fresh-agent attempts (>= 1).
    timeout_seconds: Hard wall-clock timeout per session attempt (> 0).
    retry_delay_seconds: Delay between failed outer attempts (>= 0).
    additional_context: Optional domain or output-language instructions.
    require_verified_drafts: When True, drops variations that do not match a
      verified draft in the session slate.
    decompose_stacked: When True, runs a follow-up decomposition session to
      split drafts refused or filtered for exceeding the two-addition limit.
    simplifier_model: Optional BaseModel for post-generation readability
      simplification via ``proposal_simplifier``.
    simplifier_max_fkgl: Maximum FKGL threshold for optional simplification.
    simplifier_max_sentences: Maximum sentence count for optional
      simplification.
    session_executor: Optional async callable executing one SessionRequest
      (defaults to ``run_antigravity_session``).
    api_key: Optional API key forwarded to ``LocalAgentConfig``.

  Returns:
    Result dictionary containing ``proposal_id``, ``proposal_title``, ``model``,
    ``generated_at``, ``num_variations``, ``variations``, ``grounding_audit``,
    ``attempts``, and ``session_dirs``.

  Raises:
    ImportError: If no ``session_executor`` is provided and
      ``google-antigravity`` is not installed.
    RuntimeError: If all outer attempts fail to produce ``min_variations``
      compliant variations.
    ValueError: If numeric arguments or ``save_dir`` are invalid.
  """
  _validate_positive_int("min_variations", min_variations)
  _validate_positive_int("max_variations", max_variations)
  if max_variations < min_variations:
    raise ValueError(
        f"max_variations ({max_variations}) must be >= min_variations"
        f" ({min_variations})."
    )
  _validate_positive_int("max_attempts", max_attempts)
  _validate_positive_float("timeout_seconds", timeout_seconds)
  _validate_non_negative_float("retry_delay_seconds", retry_delay_seconds)
  validated_save_dir = _validate_save_dir(save_dir)

  executor = (
      session_executor
      if session_executor is not None
      else run_antigravity_session
  )
  # Fail fast before creating directories if the default SDK is unavailable.
  if session_executor is None:
    _import_antigravity()

  has_corpus = corpus_dir is not None
  full_prompt = build_refinement_prompt(
      proposal,
      feedback_context,
      min_variations=min_variations,
      max_variations=max_variations,
      additional_context=additional_context,
      has_corpus=has_corpus,
  )

  sessions_root = validated_save_dir / proposal.proposal_id
  sessions_root.mkdir(parents=True, exist_ok=True)
  (sessions_root / "refinement_prompt.md").write_text(full_prompt)

  session_dirs: list[str] = []

  async def _execute_session(
      prompt: str,
      *,
      min_v: int,
      max_v: int,
      label: str,
      attempt: int,
  ) -> tuple[SessionResult, candidate_drafts.CandidateDraftSlate, pathlib.Path]:
    session_dir = _create_session_dir(
        sessions_root, model_name, label, len(session_dirs) + 1
    )
    session_dirs.append(str(session_dir))
    slate_path = session_dir / "candidate_drafts.json"
    access_log = candidate_drafts.CorpusAccessLog(
        root=corpus_dir.resolve() if corpus_dir is not None else None
    )
    slate = candidate_drafts.CandidateDraftSlate(
        access_log=access_log,
        slate_path=slate_path,
        seed_statement=proposal.statement,
    )
    tools: list[Callable[..., str]] = []
    if corpus_dir is not None:
      tools.extend(
          corpus_tools.make_corpus_tools(corpus_dir, access_log=access_log)
      )
    tools.extend(candidate_drafts.make_draft_slate_tools(slate))
    request = SessionRequest(
        proposal=proposal,
        prompt=prompt,
        system_instructions=AGENTIC_SYSTEM_INSTRUCTIONS,
        model_name=model_name,
        min_variations=min_v,
        max_variations=max_v,
        response_schema=build_response_schema(
            min_variations=min_v, max_variations=max_v
        ),
        save_dir=session_dir,
        tools=tuple(tools),
        slate=slate,
        label=label,
        attempt=attempt,
        api_key=api_key,
    )
    result = await asyncio.wait_for(executor(request), timeout=timeout_seconds)
    return result, slate, slate_path

  async def _run_decomposition(
      stacked: Sequence[Mapping[str, Any]], room: int, attempt: int
  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decomp_prompt = build_decomposition_prompt(
        proposal,
        stacked,
        max_out=room,
        has_corpus=has_corpus,
        additional_context=additional_context,
    )
    info: dict[str, Any] = {
        "sources": [str(item["candidate_id"]) for item in stacked]
    }
    try:
      result, decomp_slate, _ = await _execute_session(
          decomp_prompt,
          min_v=1,
          max_v=room,
          label="decompose",
          attempt=attempt,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      info["error"] = f"{type(exc).__name__}: {exc}"
      logging.warning(
          "[%s] Decomposition session failed (keeping main output): %s",
          proposal.proposal_id,
          exc,
      )
      return [], info

    payload, _ = extract_payload(result.structured_output, result.raw_text)
    splits, _ = validate_variation_payload(
        payload, min_variations=1, max_variations=room
    )
    decomp_audit = candidate_drafts.audit_final_against_slate(
        splits, decomp_slate
    )
    verified_splits = [
        variation
        for variation, row in zip(splits, decomp_audit["variations"])
        if row["is_verified"]
    ]
    kept_splits, dropped_splits = candidate_drafts.apply_hard_filter(
        verified_splits, proposal.statement, decomp_slate
    )
    tagged_splits: list[dict[str, Any]] = []
    for idx, variation in enumerate(kept_splits, start=1):
      tagged = dict(variation)
      base_id = str(tagged.get("variation_id", "")).strip()
      tagged["variation_id"] = f"d{idx:02d}_{base_id}".rstrip("_")
      tagged["decomposed"] = True
      tagged_splits.append(tagged)

    info.update({
        "n_returned": len(splits),
        "n_verified": decomp_audit["n_verified_matches"],
        "n_hard_filter_dropped": len(dropped_splits),
        "n_kept": len(tagged_splits),
    })
    logging.info(
        "[%s] Decomposition split %d stacked candidate(s) into %d verified"
        " within-limit variation(s).",
        proposal.proposal_id,
        len(stacked),
        len(tagged_splits),
    )
    return tagged_splits, info

  accepted_variations: list[dict[str, Any]] = []
  final_audit: dict[str, Any] = {}
  last_error: Exception | None = None
  attempts_used = 0

  for attempt in range(1, max_attempts + 1):
    attempts_used = attempt
    logging.info(
        "[%s] Starting agentic refinement attempt %d/%d (model=%s,"
        " timeout=%.1fs).",
        proposal.proposal_id,
        attempt,
        max_attempts,
        model_name,
        timeout_seconds,
    )
    try:
      session_result, session_slate, _ = await _execute_session(
          full_prompt,
          min_v=min_variations,
          max_v=max_variations,
          label="session",
          attempt=attempt,
      )
    except asyncio.TimeoutError:
      last_error = TimeoutError(
          f"Agent session timed out after {timeout_seconds:.1f}s for"
          f" '{proposal.proposal_id}' on attempt {attempt}/{max_attempts}."
      )
      logging.warning("%s", last_error)
      if attempt < max_attempts and retry_delay_seconds > 0:
        await asyncio.sleep(retry_delay_seconds)
      continue
    except Exception as exc:  # pylint: disable=broad-exception-caught
      last_error = exc
      kind = "transient" if is_transient_session_error(exc) else "session"
      logging.warning(
          "[%s] Attempt %d/%d failed with %s error: %s",
          proposal.proposal_id,
          attempt,
          max_attempts,
          kind,
          exc,
      )
      if attempt < max_attempts and retry_delay_seconds > 0:
        await asyncio.sleep(retry_delay_seconds)
      continue

    payload, source = extract_payload(
        session_result.structured_output, session_result.raw_text
    )
    valid_items, schema_errors = validate_variation_payload(
        payload, min_variations=min_variations, max_variations=max_variations
    )
    if not valid_items:
      last_error = ValueError(
          f"Invalid variation payload for '{proposal.proposal_id}'"
          f" (source={source}): {'; '.join(schema_errors)}"
      )
      logging.warning(
          "[%s] Attempt %d/%d produced invalid payload: %s",
          proposal.proposal_id,
          attempt,
          max_attempts,
          last_error,
      )
      if attempt < max_attempts and retry_delay_seconds > 0:
        await asyncio.sleep(retry_delay_seconds)
      continue

    audit = candidate_drafts.audit_final_against_slate(
        valid_items, session_slate
    )
    audit["schema_warnings"] = schema_errors

    surviving = valid_items
    if require_verified_drafts and audit["n_unverified"]:
      surviving = [
          variation
          for variation, row in zip(valid_items, audit["variations"])
          if row["is_verified"]
      ]

    kept, hard_dropped = candidate_drafts.apply_hard_filter(
        surviving, proposal.statement, session_slate
    )
    audit["hard_filter_dropped"] = [
        {"variation_id": v.get("variation_id"), "problems": problems}
        for v, problems in hard_dropped
    ]

    stacked = collect_stacked_candidates(
        session_slate, [v for v, _ in hard_dropped]
    )
    room = max_variations - len(kept)
    audit["decomposition"] = {"n_stacked": len(stacked), "room": room}
    if decompose_stacked and stacked and room > 0:
      splits, decomp_info = await _run_decomposition(stacked, room, attempt)
      audit["decomposition"].update(decomp_info)
      seen_descriptions = {
          candidate_drafts.normalize_text(str(v.get("description", "")))
          for v in kept
      }
      added_count = 0
      for split_item in splits:
        norm = candidate_drafts.normalize_text(
            str(split_item.get("description", ""))
        )
        if (
            norm
            and norm not in seen_descriptions
            and len(kept) < max_variations
        ):
          seen_descriptions.add(norm)
          kept.append(split_item)
          added_count += 1
      audit["decomposition"]["n_added"] = added_count

    if len(kept) >= min_variations:
      accepted_variations = kept[:max_variations]
      final_audit = audit
      break

    last_error = ValueError(
        f"Expected at least {min_variations} verified, within-limit variations"
        f" for '{proposal.proposal_id}', got {len(kept)} (source={source})."
    )
    logging.warning(
        "[%s] Attempt %d/%d fell short (%d/%d variations); retrying with fresh"
        " session.",
        proposal.proposal_id,
        attempt,
        max_attempts,
        len(kept),
        min_variations,
    )
    if attempt < max_attempts and retry_delay_seconds > 0:
      await asyncio.sleep(retry_delay_seconds)

  if not accepted_variations:
    raise last_error or RuntimeError(
        f"Agentic refinement failed for '{proposal.proposal_id}' after"
        f" {max_attempts} attempts."
    )

  if simplifier_model is not None:
    accepted_variations = await _simplify_variations(
        simplifier_model,
        accepted_variations,
        max_fkgl=simplifier_max_fkgl,
        max_sentences=simplifier_max_sentences,
        additional_context=additional_context,
    )

  result_payload: dict[str, Any] = {
      "proposal_id": proposal.proposal_id,
      "proposal_title": proposal.title,
      "original_statement": proposal.statement,
      "model": model_name,
      "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "num_variations": len(accepted_variations),
      "variations": accepted_variations,
      "grounding_audit": final_audit,
      "require_verified_drafts": require_verified_drafts,
      "attempts": attempts_used,
      "session_dirs": session_dirs,
  }

  if output_dir is not None:
    out_root = pathlib.Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{proposal.proposal_id}_variations.json"
    out_path.write_text(json.dumps(result_payload, indent=2))
    logging.info(
        "[%s] Wrote %d refined variation(s) to %s.",
        proposal.proposal_id,
        len(accepted_variations),
        out_path,
    )

  return result_payload


async def refine_proposals(
    proposals: Sequence[ProposalSpec],
    feedback_by_id: Mapping[str, str],
    *,
    save_dir: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    corpus_dir: pathlib.Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    min_variations: int = DEFAULT_MIN_VARIATIONS,
    max_variations: int = DEFAULT_MAX_VARIATIONS,
    max_concurrent_sessions: int = DEFAULT_MAX_CONCURRENT_SESSIONS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
    require_verified_drafts: bool = True,
    decompose_stacked: bool = True,
    simplifier_model: base_model.BaseModel | None = None,
    session_executor: SessionExecutor | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
  """Refines multiple proposals concurrently bounded by an asyncio Semaphore."""
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_sessions
  )
  if not proposals:
    return []

  semaphore = asyncio.Semaphore(validated_concurrency)

  async def _run_one(proposal: ProposalSpec) -> dict[str, Any]:
    async with semaphore:
      return await refine_proposal(
          proposal,
          feedback_by_id.get(proposal.proposal_id, ""),
          save_dir=save_dir,
          output_dir=output_dir,
          corpus_dir=corpus_dir,
          model_name=model_name,
          min_variations=min_variations,
          max_variations=max_variations,
          max_attempts=max_attempts,
          timeout_seconds=timeout_seconds,
          retry_delay_seconds=retry_delay_seconds,
          additional_context=additional_context,
          require_verified_drafts=require_verified_drafts,
          decompose_stacked=decompose_stacked,
          simplifier_model=simplifier_model,
          session_executor=session_executor,
          api_key=api_key,
      )

  results = list(await asyncio.gather(*(_run_one(p) for p in proposals)))
  if output_dir is not None:
    out_root = pathlib.Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_path = out_root / "refined_variations.json"
    combined_path.write_text(json.dumps(results, indent=2))
    logging.info(
        "Wrote combined refined variations (%d proposals) to %s.",
        len(results),
        combined_path,
    )
  return results


def _parse_proposal_record(record: Mapping[str, Any], idx: int) -> ProposalSpec:
  """Converts a raw JSON/CSV mapping into a validated ProposalSpec."""
  raw_id = (
      record.get("proposal_id") or record.get("slug") or record.get("id") or ""
  )
  proposal_id = str(raw_id).strip()
  statement = str(
      record.get("statement")
      or record.get("proposition")
      or record.get("text")
      or ""
  ).strip()
  title = str(record.get("title") or proposal_id).strip()
  topic = str(record.get("topic") or "").strip()

  if not proposal_id:
    raise ValueError(
        f"Proposal record at index {idx} is missing 'proposal_id' (or 'slug')."
    )
  if not statement:
    raise ValueError(
        f"Proposal '{proposal_id}' is missing 'statement' (or 'proposition')."
    )

  companions: list[CompanionProposal] = []
  raw_companions = record.get("companion_proposals") or ()
  if isinstance(raw_companions, Sequence) and not isinstance(
      raw_companions, (str, bytes)
  ):
    for comp in raw_companions:
      if not isinstance(comp, Mapping):
        continue
      comp_id = str(
          comp.get("proposal_id") or comp.get("slug") or comp.get("id") or ""
      ).strip()
      comp_stmt = str(
          comp.get("statement")
          or comp.get("proposition")
          or comp.get("text")
          or ""
      ).strip()
      comp_title = str(comp.get("title") or comp_id).strip()
      if comp_id and comp_stmt:
        companions.append(
            CompanionProposal(
                proposal_id=comp_id, title=comp_title, statement=comp_stmt
            )
        )

  return ProposalSpec(
      proposal_id=proposal_id,
      title=title,
      statement=statement,
      topic=topic,
      companion_proposals=tuple(companions),
  )


def load_proposals(input_path: pathlib.Path) -> list[ProposalSpec]:
  """Loads actionable proposals from a `.json` or `.csv` file.

  Args:
    input_path: Path to a JSON file (list of proposal dicts or
      ``{"proposals": [...]}``) or a CSV file readable by pandas.

  Returns:
    Ordered list of validated ProposalSpec objects.

  Raises:
    FileNotFoundError: If `input_path` does not exist.
    ValueError: If the file format is unsupported or records are malformed.
  """
  path = pathlib.Path(input_path)
  if not path.is_file():
    raise FileNotFoundError(f"Input proposals file not found: {path}")

  suffix = path.suffix.lower()
  if suffix == ".json":
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and isinstance(raw.get("proposals"), list):
      records = raw["proposals"]
    elif isinstance(raw, list):
      records = raw
    else:
      raise ValueError(
          f"JSON proposals file must be a list or contain 'proposals': {path}"
      )
    return [_parse_proposal_record(rec, idx) for idx, rec in enumerate(records)]

  if suffix == ".csv":
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    records = df.to_dict(orient="records")
    return [_parse_proposal_record(rec, idx) for idx, rec in enumerate(records)]

  raise ValueError(
      f"Unsupported proposals file extension '{suffix}'; expected .json or"
      " .csv."
  )


def load_feedback_dossier(
    feedback_path: pathlib.Path | None, proposal_id: str
) -> str:
  """Loads a participant feedback dossier for `proposal_id` if available."""
  if feedback_path is None:
    return ""
  path = pathlib.Path(feedback_path)
  if path.is_file():
    return path.read_text()
  if not path.is_dir():
    raise FileNotFoundError(f"Feedback path not found: {path}")

  candidates = (
      path / f"prompt_{proposal_id}.md",
      path / f"{proposal_id}.md",
      path / f"{proposal_id}.txt",
      path / f"{proposal_id}.json",
  )
  for candidate in candidates:
    if candidate.is_file():
      return candidate.read_text()
  logging.warning(
      "No feedback dossier found for '%s' in %s; proceeding with focal proposal"
      " only.",
      proposal_id,
      path,
  )
  return ""


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for agentic proposal refinement."""
  parser = argparse.ArgumentParser(
      description=(
          "Run agentic actionable proposal refinement with draft-slate"
          " verification gating and optional readability simplification."
      )
  )
  parser.add_argument(
      "--input_proposals_json",
      "--input_proposals",
      dest="input_proposals_path",
      type=pathlib.Path,
      required=True,
      help="Path to input proposals file (.json or .csv).",
  )
  parser.add_argument(
      "--feedback_dir",
      type=pathlib.Path,
      default=None,
      help=(
          "Optional directory containing per-proposal feedback dossiers"
          " (prompt_<id>.md, <id>.md, or <id>.txt) or a single feedback file."
      ),
  )
  parser.add_argument(
      "--corpus_dir",
      type=pathlib.Path,
      default=None,
      help=(
          "Optional root directory of a local reference corpus exposed to the"
          " refinement agent via bounded read-only corpus tools."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=pathlib.Path,
      required=True,
      help="Directory to write per-proposal and combined refined JSON outputs.",
  )
  parser.add_argument(
      "--save_dir",
      type=pathlib.Path,
      required=True,
      help=(
          "Persistent caller-owned root directory for Antigravity SQLite"
          " session databases and draft-slate JSON logs."
      ),
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help=(
          "Model name for agentic refinement sessions. Default:"
          f" {DEFAULT_MODEL_NAME}."
      ),
  )
  parser.add_argument(
      "--simplifier_model_name",
      type=str,
      default=None,
      help=(
          "Optional BaseModel name for post-generation readability"
          " simplification via proposal_simplifier."
      ),
  )
  parser.add_argument(
      "--gemini_api_key",
      type=str,
      default=None,
      help=(
          "Optional Gemini API key passed to the agent session and optional"
          " simplifier model."
      ),
  )
  parser.add_argument(
      "--min_variations",
      type=_positive_int_arg,
      default=DEFAULT_MIN_VARIATIONS,
      help=(
          "Minimum verified variations per proposal. Default:"
          f" {DEFAULT_MIN_VARIATIONS}."
      ),
  )
  parser.add_argument(
      "--max_variations",
      type=_positive_int_arg,
      default=DEFAULT_MAX_VARIATIONS,
      help=(
          "Maximum variations returned per proposal. Default:"
          f" {DEFAULT_MAX_VARIATIONS}."
      ),
  )
  parser.add_argument(
      "--max_concurrent_sessions",
      type=runner_utils.max_concurrent_calls_arg,
      default=DEFAULT_MAX_CONCURRENT_SESSIONS,
      help=(
          "Maximum concurrent proposal refinement sessions. Default:"
          f" {DEFAULT_MAX_CONCURRENT_SESSIONS}."
      ),
  )
  parser.add_argument(
      "--max_attempts",
      type=_positive_int_arg,
      default=DEFAULT_MAX_ATTEMPTS,
      help=(
          "Maximum fresh-agent outer retry attempts per proposal. Default:"
          f" {DEFAULT_MAX_ATTEMPTS}."
      ),
  )
  parser.add_argument(
      "--timeout_seconds",
      type=_positive_float_arg,
      default=DEFAULT_TIMEOUT_SECONDS,
      help=(
          "Hard wall-clock ceiling in seconds per agent session attempt."
          f" Default: {DEFAULT_TIMEOUT_SECONDS}."
      ),
  )
  parser.add_argument(
      "--require_verified_drafts",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Require every returned variation to match a verified draft slate.",
  )
  parser.add_argument(
      "--decompose_stacked",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Run a second-pass decomposition session for over-stacked drafts that"
          " fail the two-addition limit."
      ),
  )
  runner_utils.add_additional_context_args(parser)
  return parser.parse_args(argv)


async def run_cli(
    args: argparse.Namespace,
    *,
    session_executor: SessionExecutor | None = None,
    simplifier_model: base_model.BaseModel | None = None,
) -> list[dict[str, Any]]:
  """Executes agentic refinement from parsed CLI arguments."""
  proposals = load_proposals(args.input_proposals_path)
  additional_context = runner_utils.get_additional_context(args)
  feedback_by_id = {
      proposal.proposal_id: load_feedback_dossier(
          args.feedback_dir, proposal.proposal_id
      )
      for proposal in proposals
  }

  active_simplifier = simplifier_model
  if active_simplifier is None and args.simplifier_model_name:
    active_simplifier = model_factory.get_model(
        model_name=args.simplifier_model_name,
        gemini_api_key=args.gemini_api_key,
    )

  return await refine_proposals(
      proposals,
      feedback_by_id,
      save_dir=args.save_dir,
      output_dir=args.output_dir,
      corpus_dir=args.corpus_dir,
      model_name=args.model_name,
      min_variations=args.min_variations,
      max_variations=args.max_variations,
      max_concurrent_sessions=args.max_concurrent_sessions,
      max_attempts=args.max_attempts,
      timeout_seconds=args.timeout_seconds,
      additional_context=additional_context,
      require_verified_drafts=args.require_verified_drafts,
      decompose_stacked=args.decompose_stacked,
      simplifier_model=active_simplifier,
      session_executor=session_executor,
      api_key=args.gemini_api_key,
  )


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for agentic proposal refinement."""
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
  )
  args = get_args(argv)
  asyncio.run(run_cli(args))


if __name__ == "__main__":
  main()
