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

"""Post-hoc directive compliance, atomicity, and simplifier-drift autorater.

Evaluates refined actionable proposals independently of any agent runtime or
session database using a polymorphic ``base_model.BaseModel`` rubric combined
with deterministic sentence-count and component-count checks from
``candidate_drafts``.

Each evaluated item is audited along three dimensions:
1. **Seed anchoring & core-purpose preservation:** Whether the refined proposal
   keeps the original seed proposal's core purpose, remains complete and
   self-contained, avoids unrelated additions, and does not substitute a
   companion proposal's mechanism.
2. **Atomicity & non-stacking:** Whether the refinement stays within the
   configured ceiling of primary new mechanisms (default ``<= 2``) and net added
   sentences (default ``<= 2``) rather than bundling unrelated ideas.
3. **Optional simplifier fidelity / drift audit:** When both pre-simplified and
   post-simplified statements are provided, whether simplification preserved all
   substantive commitments, numbers, and conditions without distorting meaning.
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
from typing import Any

import pandas as pd

from src import runner_utils
from src.models import base_model
from src.models import model_factory
from src.proposition_refinement.agentic import candidate_drafts

DEFAULT_MODEL_NAME: str = "gemini-3.8-flash"
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_MAX_CONCURRENT_CALLS: int = 16
DEFAULT_MAX_ATTEMPTS: int = 3
DEFAULT_RETRY_DELAY_SECONDS: float = 1.0

COMPONENT_KINDS: tuple[str, ...] = (
    "parameter_change",
    "new_mechanism",
    "safeguard_or_oversight",
    "disclosure_or_qualification",
    "other",
)
COMPONENT_RELATIONS: tuple[str, ...] = (
    "directly_addresses_target_friction",
    "supports_same_mechanism",
    "unrelated_problem",
)
DRIFT_LEVELS: tuple[str, ...] = (
    "none",
    "wording_only",
    "substantive_change",
    "not_applicable",
)

_COMPONENT_KINDS_SET: frozenset[str] = frozenset(COMPONENT_KINDS)
_COMPONENT_RELATIONS_SET: frozenset[str] = frozenset(COMPONENT_RELATIONS)
_DRIFT_LEVELS_SET: frozenset[str] = frozenset(DRIFT_LEVELS)

DIRECTIVE_SYSTEM_PROMPT: str = (
    "You are a strict, literal post-hoc auditor evaluating refined actionable"
    " proposals against drafting directives. Judge only the supplied texts."
    " Return pure valid JSON matching the requested schema."
)

DIRECTIVE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "added_components": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": list(COMPONENT_KINDS)},
                    "relation": {
                        "type": "string",
                        "enum": list(COMPONENT_RELATIONS),
                    },
                    "borrowed_from_companion": {"type": "boolean"},
                },
                "required": [
                    "text",
                    "kind",
                    "relation",
                    "borrowed_from_companion",
                ],
            },
        },
        "n_primary_new_components": {"type": "integer"},
        "preserves_core_purpose": {"type": "boolean"},
        "is_atomic": {"type": "boolean"},
        "has_unrelated_addition": {"type": "boolean"},
        "companion_substitution": {"type": "boolean"},
        "self_contained": {"type": "boolean"},
        "simplifier_drift": {"type": "string", "enum": list(DRIFT_LEVELS)},
        "drift_note": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "added_components",
        "n_primary_new_components",
        "preserves_core_purpose",
        "is_atomic",
        "has_unrelated_addition",
        "companion_substitution",
        "self_contained",
        "simplifier_drift",
        "drift_note",
        "rationale",
    ],
}

_ADDITIONAL_CONTEXT_BLOCK: str = (
    "<additionalContext>\n{additional_context}\n</additionalContext>\n\n"
)

_DIRECTIVE_PROMPT_TEMPLATE: str = (
    "{context_block}You are auditing a refined actionable proposal for"
    " compliance with drafting directives. Be strict, literal, and consistent."
    " Judge the EVALUATED CANDIDATE STATEMENT against the ORIGINAL SEED"
    " PROPOSAL.\n\n"
    "## Drafting Directives\n"
    "1. **Seed Anchoring & Core-Purpose Preservation:** Build on the original"
    " seed proposal rather than writing from scratch. Keep its core purpose"
    " (what it does and what its resources or funding are for) intact, keep the"
    " statement complete and self-contained, avoid solving unrelated problems,"
    " and do not replace the focal proposal with a companion proposal.\n"
    "2. **Atomicity & Non-Stacking:** Limit additions to at most"
    " {max_new_components} primary new mechanism(s), safeguard(s), or"
    " parameter change(s) that are cohesive in execution, and at most"
    " {max_new_sentences} net additional sentence(s) beyond the seed proposal."
    " Distinct, unrelated ideas should be split into separate proposals rather"
    " than bundled together.\n"
    "3. **Simplifier Fidelity:** {simplifier_directive}\n\n"
    "## Original Seed Proposal{title_suffix}\n"
    "ORIGINAL SEED STATEMENT: {seed_proposal}\n"
    "(Deterministic sentence count: {seed_sentence_count})\n"
    "{companions_block}\n"
    "## Candidate Refinement Under Audit\n"
    "{metadata_block}"
    "EVALUATED CANDIDATE STATEMENT: {evaluated_statement}\n"
    "(Deterministic sentence count: {candidate_sentence_count}; net added"
    " sentences: {added_sentences})\n"
    "{simplifier_block}\n"
    "## Rating Rubric\n"
    "1. `added_components`: List every separable change in the evaluated"
    " candidate statement relative to the seed proposal that a reviewer could"
    " accept or reject independently. Exclude verbatim reused seed text, pure"
    " rewording, and rationale clauses that add no obligation. Merge"
    " inseparable implementation details of one mechanism into a single entry."
    " For each entry provide:\n"
    "   - `kind`: `parameter_change` (alters a number, rate, term, or scope of"
    " an existing seed element), `new_mechanism` (new program, process,"
    " requirement, or funding source), `safeguard_or_oversight` (audit,"
    " board, sunset, cap, lockbox, or reporting rule),"
    " `disclosure_or_qualification` (explicitly discloses a required approval,"
    " legal limit, delay, or trade-off without adding a new obligation), or"
    " `other`.\n"
    "   - `relation`: `directly_addresses_target_friction`,"
    " `supports_same_mechanism`, or `unrelated_problem`.\n"
    "   - `borrowed_from_companion`: true if the component substantially"
    " adopts a companion proposal's mechanism.\n"
    "2. `n_primary_new_components`: Count of `added_components` whose `kind`"
    " is NOT `disclosure_or_qualification`.\n"
    "3. `preserves_core_purpose`: true if the evaluated candidate keeps the"
    " original seed proposal's essential core purpose and intended use of"
    " resources intact; false if it abandons, inverts, or hollows out the"
    " original purpose.\n"
    "4. `is_atomic`: true if `n_primary_new_components` <= 1, or =="
    " {max_new_components} when the additions are closely related and cohesive"
    " in execution; false if it stacks more than {max_new_components} primary"
    " additions or bundles disparate ideas reviewers would evaluate"
    " separately.\n"
    "5. `has_unrelated_addition`: true if any component has `relation` =="
    " `unrelated_problem`.\n"
    "6. `companion_substitution`: true only if the focal proposal's core"
    " mechanism is abandoned or replaced by a companion proposal's mechanism.\n"
    "7. `self_contained`: true if the evaluated candidate stands alone as a"
    " complete actionable proposal without requiring the reader to consult the"
    " seed.\n"
    "8. `simplifier_drift` and `drift_note`: {drift_rubric}\n"
    "9. `rationale`: One concise sentence summarizing the audit verdict."
)

_SIMPLIFIER_DIRECTIVE_ACTIVE: str = (
    "Compare the PRE-simplified and POST-simplified statements to verify that"
    " plain-language editing preserved all substantive commitments, numbers,"
    " funding rules, and conditions without distorting meaning."
)
_SIMPLIFIER_DIRECTIVE_INACTIVE: str = (
    "No pre/post simplification pair was supplied for this item."
)

_DRIFT_RUBRIC_ACTIVE: str = (
    "Compare PRE-simplified vs POST-simplified statements and rate as `none`"
    " (identical or trivial punctuation change), `wording_only` (same"
    " commitments, numbers, and conditions in simpler phrasing), or"
    " `substantive_change` (POST drops, adds, or alters a commitment, number,"
    " safeguard, funding rule, or condition). Explain any drift in"
    " `drift_note` (empty string if `none`)."
)
_DRIFT_RUBRIC_INACTIVE: str = (
    "Set `simplifier_drift` to `not_applicable` and `drift_note` to an empty"
    " string."
)


@dataclasses.dataclass(frozen=True, slots=True)
class CompanionContext:
  """A companion actionable proposal deliberated alongside the focal item."""

  proposal_id: str
  title: str
  statement: str


@dataclasses.dataclass(frozen=True, slots=True)
class DirectiveAuditItem:
  """Input record for post-hoc directive, atomicity, and drift autorating."""

  item_id: str
  seed_proposal: str
  candidate_proposal: str
  proposal_title: str = ""
  target_friction: str = ""
  refinement_archetype: str = ""
  companion_proposals: tuple[CompanionContext, ...] = ()
  pre_simplified_statement: str | None = None
  post_simplified_statement: str | None = None
  declared_new_components: str = ""

  @property
  def evaluated_statement(self) -> str:
    """Returns the final participant-facing statement to audit."""
    if (
        self.post_simplified_statement
        and self.post_simplified_statement.strip()
    ):
      return self.post_simplified_statement.strip()
    return (self.candidate_proposal or "").strip()

  @property
  def has_simplifier_pair(self) -> bool:
    """Returns True when a pre-simplified statement is present."""
    return bool(
        self.pre_simplified_statement and self.pre_simplified_statement.strip()
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AddedComponent:
  """One separable amendment identified in a refined actionable proposal."""

  text: str
  kind: str
  relation: str
  borrowed_from_companion: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class DirectiveVerdict:
  """Structured post-hoc directive, atomicity, and simplifier-drift verdict."""

  item_id: str
  passed: bool
  failure_reasons: tuple[str, ...]
  preserves_core_purpose: bool
  is_atomic: bool
  n_primary_new_components: int
  has_unrelated_addition: bool
  companion_substitution: bool
  self_contained: bool
  added_components: tuple[AddedComponent, ...]
  seed_sentence_count: int
  candidate_sentence_count: int
  added_sentences: int
  exceeds_sentence_limit: bool
  declared_component_count: int
  exceeds_declared_component_limit: bool
  simplifier_drift: str
  drift_note: str
  rationale: str
  attempts: int
  error: str | None = None

  def to_dict(self) -> dict[str, Any]:
    """Serializes the verdict into plain JSON-compatible primitives."""
    return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DirectiveAuditConfig:
  """Immutable configuration for running a file-based directive audit."""

  input_path: pathlib.Path
  output_path: pathlib.Path
  temperature: float = DEFAULT_TEMPERATURE
  max_new_components: int = candidate_drafts.MAX_NEW_COMPONENTS
  max_new_sentences: int = candidate_drafts.MAX_NEW_SENTENCES
  max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS
  max_attempts: int = DEFAULT_MAX_ATTEMPTS
  retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS
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


def compute_deterministic_metrics(
    item: DirectiveAuditItem,
    *,
    max_new_sentences: int = candidate_drafts.MAX_NEW_SENTENCES,
    max_new_components: int = candidate_drafts.MAX_NEW_COMPONENTS,
) -> dict[str, Any]:
  """Computes deterministic sentence and declared-component metrics.

  Delegates sentence counting, net sentence delta, and component-line parsing
  to ``candidate_drafts`` so pre-submission gating and post-hoc autorating share
  identical deterministic boundaries.

  Args:
    item: Candidate proposal record under audit.
    max_new_sentences: Maximum permitted net added sentences over the seed.
    max_new_components: Maximum permitted declared new components.

  Returns:
    Dictionary containing sentence counts, net added sentences, declared
    component count, and limit-violation booleans.
  """
  _validate_positive_int("max_new_sentences", max_new_sentences)
  _validate_positive_int("max_new_components", max_new_components)
  evaluated = item.evaluated_statement
  seed_sentence_count = candidate_drafts.count_sentences(item.seed_proposal)
  candidate_sentence_count = candidate_drafts.count_sentences(evaluated)
  added_sentences = candidate_drafts.added_sentence_count(
      evaluated, item.seed_proposal
  )
  declared = candidate_drafts.parse_component_lines(
      item.declared_new_components
  )
  return {
      "seed_sentence_count": seed_sentence_count,
      "candidate_sentence_count": candidate_sentence_count,
      "added_sentences": added_sentences,
      "exceeds_sentence_limit": added_sentences > max_new_sentences,
      "declared_component_count": len(declared),
      "exceeds_declared_component_limit": len(declared) > max_new_components,
  }


def build_directive_prompt(
    item: DirectiveAuditItem,
    *,
    max_new_components: int = candidate_drafts.MAX_NEW_COMPONENTS,
    max_new_sentences: int = candidate_drafts.MAX_NEW_SENTENCES,
    additional_context: str | None = None,
) -> str:
  """Builds the Check-A directive compliance and drift rubric prompt.

  Args:
    item: Candidate proposal record under audit.
    max_new_components: Maximum allowed primary new mechanisms/components.
    max_new_sentences: Maximum allowed net added sentences over the seed.
    additional_context: Optional domain or language guidance wrapped in an
      ``<additionalContext>`` block.

  Returns:
    Complete prompt text ready for ``BaseModel.generate_content``.
  """
  metrics = compute_deterministic_metrics(
      item,
      max_new_sentences=max_new_sentences,
      max_new_components=max_new_components,
  )
  context_block = ""
  if additional_context and additional_context.strip():
    context_block = _ADDITIONAL_CONTEXT_BLOCK.format(
        additional_context=additional_context.strip()
    )

  title_suffix = (
      f": {item.proposal_title.strip()}" if item.proposal_title.strip() else ""
  )

  companions_block = ""
  if item.companion_proposals:
    comp_lines = [
        "",
        "## Companion Proposals Deliberated Alongside the Focal Proposal",
    ]
    for comp in item.companion_proposals:
      comp_lines.append(
          f"- {comp.title} (`{comp.proposal_id}`): {comp.statement}"
      )
    companions_block = "\n".join(comp_lines) + "\n"

  metadata_lines: list[str] = []
  if item.target_friction.strip():
    metadata_lines.append(
        f"Target friction addressed: {item.target_friction.strip()}"
    )
  if item.refinement_archetype.strip():
    metadata_lines.append(
        f"Refinement archetype: {item.refinement_archetype.strip()}"
    )
  metadata_block = ("\n".join(metadata_lines) + "\n") if metadata_lines else ""

  if item.has_simplifier_pair:
    pre_text = (item.pre_simplified_statement or "").strip()
    simplifier_block = (
        f"PRE-SIMPLIFIED STATEMENT: {pre_text}\n"
        f"POST-SIMPLIFIED STATEMENT: {item.evaluated_statement}\n"
    )
    simplifier_directive = _SIMPLIFIER_DIRECTIVE_ACTIVE
    drift_rubric = _DRIFT_RUBRIC_ACTIVE
  else:
    simplifier_block = ""
    simplifier_directive = _SIMPLIFIER_DIRECTIVE_INACTIVE
    drift_rubric = _DRIFT_RUBRIC_INACTIVE

  return _DIRECTIVE_PROMPT_TEMPLATE.format(
      context_block=context_block,
      max_new_components=max_new_components,
      max_new_sentences=max_new_sentences,
      simplifier_directive=simplifier_directive,
      title_suffix=title_suffix,
      seed_proposal=item.seed_proposal.strip(),
      seed_sentence_count=metrics["seed_sentence_count"],
      companions_block=companions_block,
      metadata_block=metadata_block,
      evaluated_statement=item.evaluated_statement,
      candidate_sentence_count=metrics["candidate_sentence_count"],
      added_sentences=metrics["added_sentences"],
      simplifier_block=simplifier_block,
      drift_rubric=drift_rubric,
  )


def _require_bool(mapping: Mapping[str, Any], field: str) -> bool:
  """Extracts a strict boolean field from a parsed JSON object."""
  value = mapping.get(field)
  if not isinstance(value, bool):
    raise ValueError(
        f"Field '{field}' must be a boolean, got {type(value).__name__}."
    )
  return value


def _require_str(mapping: Mapping[str, Any], field: str) -> str:
  """Extracts a string field from a parsed JSON object."""
  value = mapping.get(field)
  if not isinstance(value, str):
    raise ValueError(
        f"Field '{field}' must be a string, got {type(value).__name__}."
    )
  return value.strip()


def parse_directive_response(
    raw_text: str, *, expect_simplifier_check: bool
) -> dict[str, Any]:
  """Parses and validates a JSON verdict response from the directive judge.

  Args:
    raw_text: Raw model output text (may include markdown code fences).
    expect_simplifier_check: Whether a pre/post simplification pair was
      supplied.

  Returns:
    Validated dictionary of rubric fields with normalized ``AddedComponent``
    tuples.

  Raises:
    ValueError: If `raw_text` is not valid JSON or violates required schema
      fields, types, or enum values.
  """
  cleaned = _clean_json_text(raw_text)
  if not cleaned:
    raise ValueError("Empty response from directive autorater model.")
  try:
    decoded = json.loads(cleaned)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Malformed JSON from directive autorater: {exc}") from exc

  if not isinstance(decoded, Mapping):
    raise ValueError("Directive autorater response must be a JSON object.")

  raw_components = decoded.get("added_components")
  if not isinstance(raw_components, list):
    raise ValueError("Field 'added_components' must be a JSON array.")

  components: list[AddedComponent] = []
  for idx, entry in enumerate(raw_components):
    if not isinstance(entry, Mapping):
      raise ValueError(f"added_components[{idx}] must be a JSON object.")
    comp_text = _require_str(entry, "text")
    comp_kind = _require_str(entry, "kind")
    if comp_kind not in _COMPONENT_KINDS_SET:
      raise ValueError(
          f"added_components[{idx}].kind '{comp_kind}' is not in"
          f" {COMPONENT_KINDS}."
      )
    comp_relation = _require_str(entry, "relation")
    if comp_relation not in _COMPONENT_RELATIONS_SET:
      raise ValueError(
          f"added_components[{idx}].relation '{comp_relation}' is not in"
          f" {COMPONENT_RELATIONS}."
      )
    borrowed = _require_bool(entry, "borrowed_from_companion")
    components.append(
        AddedComponent(
            text=comp_text,
            kind=comp_kind,
            relation=comp_relation,
            borrowed_from_companion=borrowed,
        )
    )

  n_primary = decoded.get("n_primary_new_components")
  if isinstance(n_primary, bool) or not isinstance(n_primary, int):
    raise ValueError(
        "Field 'n_primary_new_components' must be an integer, got"
        f" {type(n_primary).__name__}."
    )
  if n_primary < 0:
    raise ValueError(
        f"Field 'n_primary_new_components' must be >= 0, got {n_primary}."
    )

  drift = _require_str(decoded, "simplifier_drift")
  if drift not in _DRIFT_LEVELS_SET:
    raise ValueError(
        f"Field 'simplifier_drift' '{drift}' is not in {DRIFT_LEVELS}."
    )
  drift_note = _require_str(decoded, "drift_note")
  if not expect_simplifier_check:
    drift = "not_applicable"
    drift_note = ""

  return {
      "added_components": tuple(components),
      "n_primary_new_components": n_primary,
      "preserves_core_purpose": _require_bool(
          decoded, "preserves_core_purpose"
      ),
      "is_atomic": _require_bool(decoded, "is_atomic"),
      "has_unrelated_addition": _require_bool(
          decoded, "has_unrelated_addition"
      ),
      "companion_substitution": _require_bool(
          decoded, "companion_substitution"
      ),
      "self_contained": _require_bool(decoded, "self_contained"),
      "simplifier_drift": drift,
      "drift_note": drift_note,
      "rationale": _require_str(decoded, "rationale"),
  }


def build_verdict(
    item: DirectiveAuditItem,
    parsed: Mapping[str, Any],
    *,
    attempts: int,
    max_new_components: int = candidate_drafts.MAX_NEW_COMPONENTS,
    max_new_sentences: int = candidate_drafts.MAX_NEW_SENTENCES,
) -> DirectiveVerdict:
  """Combines deterministic metrics and parsed rubric output into a verdict.

  Args:
    item: Audited input item.
    parsed: Output of ``parse_directive_response``.
    attempts: Number of model attempts executed.
    max_new_components: Hard ceiling on primary new components.
    max_new_sentences: Hard ceiling on net added sentences.

  Returns:
    Immutable ``DirectiveVerdict`` summarizing pass/fail status and reasons.
  """
  metrics = compute_deterministic_metrics(
      item,
      max_new_sentences=max_new_sentences,
      max_new_components=max_new_components,
  )
  n_primary = int(parsed["n_primary_new_components"])
  preserves_core_purpose = bool(parsed["preserves_core_purpose"])
  is_atomic = bool(parsed["is_atomic"]) and n_primary <= max_new_components
  has_unrelated_addition = bool(parsed["has_unrelated_addition"])
  companion_substitution = bool(parsed["companion_substitution"])
  self_contained = bool(parsed["self_contained"])
  simplifier_drift = str(parsed["simplifier_drift"])

  reasons: list[str] = []
  if not preserves_core_purpose:
    reasons.append("does_not_preserve_core_purpose")
  if companion_substitution:
    reasons.append("companion_substitution")
  if not self_contained:
    reasons.append("not_self_contained")
  if has_unrelated_addition:
    reasons.append("has_unrelated_addition")
  if n_primary > max_new_components:
    reasons.append(
        f"too_many_new_components ({n_primary} > {max_new_components})"
    )
  elif not is_atomic:
    reasons.append("not_atomic")
  if metrics["exceeds_sentence_limit"]:
    reasons.append(
        f"exceeds_sentence_limit ({metrics['added_sentences']} >"
        f" {max_new_sentences})"
    )
  if metrics["exceeds_declared_component_limit"]:
    reasons.append(
        "exceeds_declared_component_limit"
        f" ({metrics['declared_component_count']} > {max_new_components})"
    )
  if simplifier_drift == "substantive_change":
    reasons.append("substantive_simplifier_drift")

  return DirectiveVerdict(
      item_id=item.item_id,
      passed=not reasons,
      failure_reasons=tuple(reasons),
      preserves_core_purpose=preserves_core_purpose,
      is_atomic=is_atomic,
      n_primary_new_components=n_primary,
      has_unrelated_addition=has_unrelated_addition,
      companion_substitution=companion_substitution,
      self_contained=self_contained,
      added_components=tuple(parsed["added_components"]),
      seed_sentence_count=metrics["seed_sentence_count"],
      candidate_sentence_count=metrics["candidate_sentence_count"],
      added_sentences=metrics["added_sentences"],
      exceeds_sentence_limit=metrics["exceeds_sentence_limit"],
      declared_component_count=metrics["declared_component_count"],
      exceeds_declared_component_limit=metrics[
          "exceeds_declared_component_limit"
      ],
      simplifier_drift=simplifier_drift,
      drift_note=str(parsed["drift_note"]),
      rationale=str(parsed["rationale"]),
      attempts=attempts,
      error=None,
  )


def _error_verdict(
    item: DirectiveAuditItem,
    *,
    error_message: str,
    attempts: int,
    max_new_components: int,
    max_new_sentences: int,
) -> DirectiveVerdict:
  """Builds a failing verdict when LLM evaluation exhausts retries."""
  metrics = compute_deterministic_metrics(
      item,
      max_new_sentences=max_new_sentences,
      max_new_components=max_new_components,
  )
  return DirectiveVerdict(
      item_id=item.item_id,
      passed=False,
      failure_reasons=(f"autorater_error: {error_message}",),
      preserves_core_purpose=False,
      is_atomic=False,
      n_primary_new_components=0,
      has_unrelated_addition=False,
      companion_substitution=False,
      self_contained=False,
      added_components=(),
      seed_sentence_count=metrics["seed_sentence_count"],
      candidate_sentence_count=metrics["candidate_sentence_count"],
      added_sentences=metrics["added_sentences"],
      exceeds_sentence_limit=metrics["exceeds_sentence_limit"],
      declared_component_count=metrics["declared_component_count"],
      exceeds_declared_component_limit=metrics[
          "exceeds_declared_component_limit"
      ],
      simplifier_drift="not_applicable",
      drift_note="",
      rationale="",
      attempts=attempts,
      error=error_message,
  )


async def evaluate_directive_item(
    model: base_model.BaseModel,
    item: DirectiveAuditItem,
    *,
    max_new_components: int = candidate_drafts.MAX_NEW_COMPONENTS,
    max_new_sentences: int = candidate_drafts.MAX_NEW_SENTENCES,
    temperature: float = DEFAULT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
) -> DirectiveVerdict:
  """Evaluates one refined actionable proposal against directive rubrics.

  Retries model errors and malformed JSON responses up to ``max_attempts``.
  When all attempts fail, returns a failing ``DirectiveVerdict`` with ``error``
  populated rather than aborting the surrounding batch.

  Args:
    model: Polymorphic ``BaseModel`` instance.
    item: Input proposal record to audit.
    max_new_components: Maximum allowed primary new components (>= 1).
    max_new_sentences: Maximum allowed net added sentences (>= 1).
    temperature: Sampling temperature (>= 0).
    max_attempts: Maximum attempts before recording an error verdict (>= 1).
    retry_delay_seconds: Sleep delay between failed attempts (>= 0).
    additional_context: Optional domain or output-language context.

  Returns:
    Immutable ``DirectiveVerdict`` for `item`.
  """
  _validate_positive_int("max_new_components", max_new_components)
  _validate_positive_int("max_new_sentences", max_new_sentences)
  _validate_positive_int("max_attempts", max_attempts)
  _validate_non_negative_float("temperature", temperature)
  _validate_non_negative_float("retry_delay_seconds", retry_delay_seconds)

  if not (item.seed_proposal or "").strip():
    raise ValueError(f"Item '{item.item_id}' has an empty seed_proposal.")
  if not item.evaluated_statement:
    raise ValueError(f"Item '{item.item_id}' has an empty candidate_proposal.")

  prompt = build_directive_prompt(
      item,
      max_new_components=max_new_components,
      max_new_sentences=max_new_sentences,
      additional_context=additional_context,
  )
  last_error = "unknown error"

  for attempt in range(1, max_attempts + 1):
    try:
      response = await model.generate_content(
          prompt=prompt,
          run_name=f"directive_audit_{item.item_id}_attempt_{attempt}",
          temperature=temperature,
          system_prompt=DIRECTIVE_SYSTEM_PROMPT,
          response_mime_type="application/json",
          response_schema=DIRECTIVE_RESPONSE_SCHEMA,
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
      parsed = parse_directive_response(
          raw_text, expect_simplifier_check=item.has_simplifier_pair
      )
      return build_verdict(
          item,
          parsed,
          attempts=attempt,
          max_new_components=max_new_components,
          max_new_sentences=max_new_sentences,
      )
    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
      last_error = str(exc)
      logging.warning(
          "[%s] Directive audit attempt %d/%d failed: %s",
          item.item_id,
          attempt,
          max_attempts,
          last_error,
      )
      if attempt < max_attempts and retry_delay_seconds > 0:
        await asyncio.sleep(retry_delay_seconds)

  return _error_verdict(
      item,
      error_message=last_error,
      attempts=max_attempts,
      max_new_components=max_new_components,
      max_new_sentences=max_new_sentences,
  )


async def evaluate_directive_batch(
    model: base_model.BaseModel,
    items: Sequence[DirectiveAuditItem],
    *,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
    max_new_components: int = candidate_drafts.MAX_NEW_COMPONENTS,
    max_new_sentences: int = candidate_drafts.MAX_NEW_SENTENCES,
    temperature: float = DEFAULT_TEMPERATURE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    additional_context: str | None = None,
) -> list[DirectiveVerdict]:
  """Evaluates a batch of items concurrently using an ``asyncio.Semaphore``.

  Args:
    model: Polymorphic ``BaseModel`` instance.
    items: Ordered sequence of ``DirectiveAuditItem`` records.
    max_concurrent_calls: Concurrency cap (>= 1).
    max_new_components: Maximum allowed primary new components (>= 1).
    max_new_sentences: Maximum allowed net added sentences (>= 1).
    temperature: Sampling temperature (>= 0).
    max_attempts: Maximum attempts per item (>= 1).
    retry_delay_seconds: Sleep delay between failed attempts (>= 0).
    additional_context: Optional domain or output-language context.

  Returns:
    List of ``DirectiveVerdict`` objects aligned with `items`.
  """
  validated_concurrency = base_model.validate_max_concurrent_calls(
      max_concurrent_calls
  )
  _validate_positive_int("max_new_components", max_new_components)
  _validate_positive_int("max_new_sentences", max_new_sentences)
  _validate_positive_int("max_attempts", max_attempts)
  _validate_non_negative_float("temperature", temperature)
  _validate_non_negative_float("retry_delay_seconds", retry_delay_seconds)

  if not items:
    return []

  semaphore = asyncio.Semaphore(validated_concurrency)

  async def _evaluate_one(item: DirectiveAuditItem) -> DirectiveVerdict:
    async with semaphore:
      return await evaluate_directive_item(
          model,
          item,
          max_new_components=max_new_components,
          max_new_sentences=max_new_sentences,
          temperature=temperature,
          max_attempts=max_attempts,
          retry_delay_seconds=retry_delay_seconds,
          additional_context=additional_context,
      )

  return list(await asyncio.gather(*(_evaluate_one(item) for item in items)))


def summarize_verdicts(verdicts: Sequence[DirectiveVerdict]) -> dict[str, Any]:
  """Computes aggregate pass/fail and defect counts across verdicts."""
  total = len(verdicts)
  passed = sum(v.passed for v in verdicts)
  errors = sum(v.error is not None for v in verdicts)
  return {
      "total": total,
      "passed": passed,
      "failed": total - passed,
      "errors": errors,
      "non_atomic": sum(v.error is None and not v.is_atomic for v in verdicts),
      "core_purpose_failed": sum(
          v.error is None and not v.preserves_core_purpose for v in verdicts
      ),
      "sentence_limit_exceeded": sum(
          v.exceeds_sentence_limit for v in verdicts
      ),
      "substantive_simplifier_drift": sum(
          v.simplifier_drift == "substantive_change" for v in verdicts
      ),
  }


def _parse_companions(raw: Any) -> tuple[CompanionContext, ...]:
  """Parses an optional sequence of companion proposal mappings."""
  if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
    return ()
  companions: list[CompanionContext] = []
  for entry in raw:
    if not isinstance(entry, Mapping):
      continue
    proposal_id = str(
        entry.get("proposal_id") or entry.get("slug") or entry.get("id") or ""
    ).strip()
    statement = str(
        entry.get("statement")
        or entry.get("proposition")
        or entry.get("text")
        or ""
    ).strip()
    title = str(entry.get("title") or proposal_id).strip()
    if proposal_id and statement:
      companions.append(
          CompanionContext(
              proposal_id=proposal_id, title=title, statement=statement
          )
      )
  return tuple(companions)


def _optional_text(mapping: Mapping[str, Any], *keys: str) -> str | None:
  """Returns the first non-empty string found under `keys`, or None."""
  for key in keys:
    value = mapping.get(key)
    if value is None:
      continue
    if isinstance(value, float) and math.isnan(value):
      continue
    text = str(value).strip()
    if text:
      return text
  return None


def _build_audit_item_from_mapping(
    mapping: Mapping[str, Any],
    *,
    item_id: str,
    seed_proposal: str,
    proposal_title: str,
    companions: tuple[CompanionContext, ...],
) -> DirectiveAuditItem:
  """Constructs a validated DirectiveAuditItem from a variation mapping."""
  pre_stmt = _optional_text(
      mapping, "pre_simplified_statement", "pre_simplified_description"
  )
  post_stmt = _optional_text(
      mapping, "post_simplified_statement", "post_simplified_description"
  )
  candidate = (
      _optional_text(mapping, "candidate_proposal", "description", "statement")
      or post_stmt
  )
  if not candidate:
    raise ValueError(
        f"Record '{item_id}' is missing 'candidate_proposal' (or"
        " 'description')."
    )
  return DirectiveAuditItem(
      item_id=item_id,
      seed_proposal=seed_proposal,
      candidate_proposal=candidate,
      proposal_title=proposal_title,
      target_friction=_optional_text(
          mapping, "target_friction", "target_friction_resolved"
      )
      or "",
      refinement_archetype=_optional_text(mapping, "refinement_archetype")
      or "",
      companion_proposals=companions,
      pre_simplified_statement=pre_stmt,
      post_simplified_statement=post_stmt,
      declared_new_components=_optional_text(
          mapping, "declared_new_components", "new_components"
      )
      or "",
  )


def _expand_refiner_record(
    record: Mapping[str, Any],
) -> list[DirectiveAuditItem]:
  """Expands one nested refiner output record into flat audit items."""
  proposal_id = str(record.get("proposal_id") or "").strip()
  proposal_title = str(
      record.get("proposal_title") or record.get("title") or proposal_id
  ).strip()
  seed = _optional_text(
      record, "original_statement", "seed_proposal", "statement"
  )
  if not seed:
    raise ValueError(
        f"Proposal '{proposal_id or '<unknown>'}' is missing"
        " 'original_statement' or 'seed_proposal'."
    )
  companions = _parse_companions(record.get("companion_proposals"))
  variations = record.get("variations")
  if not isinstance(variations, list):
    raise ValueError(
        f"Proposal '{proposal_id or '<unknown>'}' field 'variations' must be a"
        " JSON list."
    )
  items: list[DirectiveAuditItem] = []
  for idx, var in enumerate(variations):
    if not isinstance(var, Mapping):
      continue
    var_id = str(var.get("variation_id") or f"v{idx + 1:02d}").strip()
    item_id = f"{proposal_id}_{var_id}" if proposal_id else var_id
    items.append(
        _build_audit_item_from_mapping(
            var,
            item_id=item_id,
            seed_proposal=seed,
            proposal_title=proposal_title,
            companions=companions,
        )
    )
  return items


def _parse_flat_record(
    record: Mapping[str, Any], idx: int
) -> DirectiveAuditItem:
  """Converts one flat JSON/CSV mapping into a validated DirectiveAuditItem."""
  item_id = (
      _optional_text(record, "item_id", "variation_id", "id")
      or f"item_{idx + 1}"
  )
  seed = _optional_text(
      record, "seed_proposal", "original_statement", "seed_statement"
  )
  if not seed:
    raise ValueError(
        f"Record '{item_id}' is missing 'seed_proposal' (or"
        " 'original_statement')."
    )
  return _build_audit_item_from_mapping(
      record,
      item_id=item_id,
      seed_proposal=seed,
      proposal_title=_optional_text(record, "proposal_title", "title") or "",
      companions=_parse_companions(record.get("companion_proposals")),
  )


def load_audit_items(input_path: pathlib.Path) -> list[DirectiveAuditItem]:
  """Loads directive audit items from a `.json` or `.csv` file.

  Supports flat item lists, ``{"items": [...]}``, single or batched outputs
  from ``refiner.py`` (containing ``variations`` arrays), and CSV tables.

  Args:
    input_path: Path to a `.json` or `.csv` file.

  Returns:
    Ordered list of validated ``DirectiveAuditItem`` objects.

  Raises:
    FileNotFoundError: If `input_path` does not exist.
    ValueError: If the extension is unsupported or records are malformed.
  """
  path = pathlib.Path(input_path)
  if not path.is_file():
    raise FileNotFoundError(f"Input audit file not found: {path}")

  suffix = path.suffix.lower()
  if suffix == ".json":
    raw = json.loads(path.read_text())
    if isinstance(raw, Mapping) and "variations" in raw:
      return _expand_refiner_record(raw)
    if isinstance(raw, Mapping) and isinstance(raw.get("items"), list):
      records = raw["items"]
    elif isinstance(raw, list):
      records = raw
    else:
      raise ValueError(
          "JSON input must be a list, contain 'items', or contain 'variations':"
          f" {path}"
      )
    items: list[DirectiveAuditItem] = []
    for idx, rec in enumerate(records):
      if not isinstance(rec, Mapping):
        raise ValueError(f"JSON record at index {idx} must be an object.")
      if "variations" in rec:
        items.extend(_expand_refiner_record(rec))
      else:
        items.append(_parse_flat_record(rec, idx))
    return items

  if suffix == ".csv":
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    records = df.to_dict(orient="records")
    return [_parse_flat_record(rec, idx) for idx, rec in enumerate(records)]

  raise ValueError(
      f"Unsupported input file extension '{suffix}'; expected .json or .csv."
  )


def write_verdicts(
    verdicts: Sequence[DirectiveVerdict], output_path: pathlib.Path
) -> None:
  """Writes directive verdicts and summary statistics to `.json` or `.csv`."""
  path = pathlib.Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  suffix = path.suffix.lower()
  if suffix == ".json":
    payload = {
        "summary": summarize_verdicts(verdicts),
        "verdicts": [v.to_dict() for v in verdicts],
    }
    path.write_text(json.dumps(payload, indent=2))
  elif suffix == ".csv":
    rows: list[dict[str, Any]] = []
    for verdict in verdicts:
      row = verdict.to_dict()
      row["failure_reasons"] = json.dumps(row["failure_reasons"])
      row["added_components"] = json.dumps(row["added_components"])
      rows.append(row)
    columns = [field.name for field in dataclasses.fields(DirectiveVerdict)]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
  else:
    raise ValueError(
        f"Unsupported output file extension '{suffix}'; expected .json or .csv."
    )
  logging.info("Wrote %d directive verdict(s) to %s.", len(verdicts), path)


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the CLI argument parser for the directive & atomicity autorater."""
  parser = argparse.ArgumentParser(
      description=(
          "Run post-hoc directive compliance, atomicity, and simplifier-drift"
          " autorating over refined actionable proposals."
      )
  )
  parser.add_argument(
      "--input_path",
      type=pathlib.Path,
      required=True,
      help=(
          "Path to input refined proposals file (.json or .csv, including"
          " refiner.py output JSON)."
      ),
  )
  parser.add_argument(
      "--output_path",
      type=pathlib.Path,
      required=True,
      help="Destination path for verdict output (.json or .csv).",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help=f"Model name for the judge LLM. Default: {DEFAULT_MODEL_NAME}.",
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
      "--max_new_components",
      type=_positive_int_arg,
      default=candidate_drafts.MAX_NEW_COMPONENTS,
      help=(
          "Maximum permitted primary new mechanisms/components. Default:"
          f" {candidate_drafts.MAX_NEW_COMPONENTS}."
      ),
  )
  parser.add_argument(
      "--max_new_sentences",
      type=_positive_int_arg,
      default=candidate_drafts.MAX_NEW_SENTENCES,
      help=(
          "Maximum permitted net added sentences beyond the seed proposal."
          f" Default: {candidate_drafts.MAX_NEW_SENTENCES}."
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
      help=(
          "Maximum retry attempts per item on model or parse errors. Default:"
          f" {DEFAULT_MAX_ATTEMPTS}."
      ),
  )
  runner_utils.add_additional_context_args(parser)
  return parser


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments for the directive & atomicity autorater."""
  return build_arg_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> DirectiveAuditConfig:
  """Constructs an immutable ``DirectiveAuditConfig`` from parsed CLI args."""
  return DirectiveAuditConfig(
      input_path=args.input_path,
      output_path=args.output_path,
      temperature=args.temperature,
      max_new_components=args.max_new_components,
      max_new_sentences=args.max_new_sentences,
      max_concurrent_calls=args.max_concurrent_calls,
      max_attempts=args.max_attempts,
      additional_context=runner_utils.get_additional_context(args),
  )


async def run_directive_audit(
    config: DirectiveAuditConfig,
    *,
    model: base_model.BaseModel,
) -> list[DirectiveVerdict]:
  """Loads audit items, evaluates them with `model`, and writes verdicts."""
  items = load_audit_items(config.input_path)
  verdicts = await evaluate_directive_batch(
      model,
      items,
      max_concurrent_calls=config.max_concurrent_calls,
      max_new_components=config.max_new_components,
      max_new_sentences=config.max_new_sentences,
      temperature=config.temperature,
      max_attempts=config.max_attempts,
      retry_delay_seconds=config.retry_delay_seconds,
      additional_context=config.additional_context,
  )
  write_verdicts(verdicts, config.output_path)
  return verdicts


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for post-hoc directive and atomicity autorating."""
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
  asyncio.run(run_directive_audit(config, model=model))


if __name__ == "__main__":
  main()
