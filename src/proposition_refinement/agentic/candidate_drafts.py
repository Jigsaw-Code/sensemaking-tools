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

"""Draft-slate tooling that gates verification of candidate proposals.

An agentic refinement session writes each candidate actionable proposal into a
persisted draft slate, researches its feasibility against a local reference
corpus, and revises, verifies, or rejects each candidate before emitting a
final answer.

Marking a candidate ``verified`` is mechanically gated. It requires:
- A declared list of new components added beyond the original proposal, at most
  ``MAX_NEW_COMPONENTS``.
- At most ``MAX_NEW_SENTENCES`` more sentences than the original proposal
  statement (a net sentence count).
- An explicit statement confirming that the original proposal's core purpose is
  preserved.
- A feasibility assessment explaining what was researched for each new
  component or change and what was found.

After the session, ``audit_final_against_slate`` checks that every returned
variation matches a verified draft, ``hard_filter_problems`` and
``apply_hard_filter`` enforce the same addition ceilings as a fallback filter,
and ``CandidateDraftSlate.stacking_failures`` collects candidates refused for
too many additions so they can be decomposed in a follow-up pass.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import pathlib
import re
from typing import Any

DRAFT_STATUSES: tuple[str, ...] = ("draft", "verified", "rejected")
MAX_NEW_COMPONENTS: int = 2
MAX_NEW_SENTENCES: int = 2
STACKING_PROBLEM_TAG: str = "[too-many-additions]"

_STATEMENT_MATCH_RATIO: float = 0.9
_SENTENCE_SPLIT_RE = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"')]))\s+(?=[A-Z0-9\"'(])"
)
_WHITESPACE_RE = re.compile(r"\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
_EMPTY_COMPONENT_VALUES: frozenset[str] = frozenset({"none", "n/a"})


@dataclasses.dataclass
class CandidateDraft:
  """One candidate actionable proposal statement and its verification state."""

  candidate_id: str
  title: str
  statement: str
  target_friction_resolved: str
  feasibility_assessment: str = ""
  new_components: str = ""
  seed_purpose_preserved: str = ""
  status: str = "draft"
  revisions: int = 1


def count_sentences(text: str) -> int:
  """Counts sentences in `text` using terminal punctuation boundaries."""
  stripped = text.strip()
  if not stripped:
    return 0
  return sum(1 for _ in _SENTENCE_SPLIT_RE.finditer(stripped)) + 1


def normalize_text(text: str) -> str:
  """Collapses whitespace and lowercases text for fuzzy comparison."""
  return _WHITESPACE_RE.sub(" ", text.strip().lower())


def added_sentence_count(statement: str, seed: str) -> int:
  """Returns net added sentences in `statement` over `seed`, floored at 0."""
  return max(0, count_sentences(statement) - count_sentences(seed))


def parse_component_lines(text: str) -> list[str]:
  """Parses one-component-per-line text, ignoring bullets and placeholders."""
  components: list[str] = []
  for raw_line in text.splitlines():
    cleaned = _LIST_PREFIX_RE.sub("", raw_line).strip()
    if cleaned and cleaned.lower() not in _EMPTY_COMPONENT_VALUES:
      components.append(cleaned)
  return components


class CorpusAccessLog:
  """Records which reference corpus files an agent accessed during a session."""

  def __init__(self, root: pathlib.Path | None = None) -> None:
    """Initializes the access log.

    Args:
      root: Optional corpus root directory kept for provenance.
    """
    self.files: set[str] = set()
    self.root: pathlib.Path | None = root

  def record(self, rel_path: str) -> None:
    """Adds a corpus-relative file path to the log."""
    normalized = rel_path.strip().lstrip("/")
    if normalized:
      self.files.add(normalized)


class CandidateDraftSlate:
  """Persisted slate of candidate proposal drafts with verification gating."""

  def __init__(
      self,
      access_log: CorpusAccessLog | None = None,
      slate_path: pathlib.Path | None = None,
      seed_statement: str = "",
  ) -> None:
    """Initializes the candidate draft slate.

    Args:
      access_log: Shared log of corpus files accessed during the session. When
        None, an empty CorpusAccessLog is created.
      slate_path: Optional JSON path where the slate and revision history are
        persisted after every write.
      seed_statement: Original actionable proposal statement used for net
        sentence-addition checks. If empty, the sentence-addition check is
        skipped.
    """
    self.access_log = (
        access_log if access_log is not None else CorpusAccessLog()
    )
    self.slate_path = slate_path
    self.seed_statement = seed_statement
    self._seed_sentence_count = count_sentences(seed_statement)
    self.drafts: dict[str, CandidateDraft] = {}
    self.history: list[dict[str, Any]] = []

  def _persist(self) -> None:
    """Writes the current slate state and revision history to JSON."""
    if self.slate_path is None:
      return
    try:
      self.slate_path.parent.mkdir(parents=True, exist_ok=True)
      payload = {
          "seed_statement": self.seed_statement,
          "drafts": [dataclasses.asdict(d) for d in self.drafts.values()],
          "accessed_corpus_files": sorted(self.access_log.files),
          "history": self.history,
      }
      self.slate_path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
      logging.warning(
          "Failed to persist candidate draft slate to %s: %s",
          self.slate_path,
          exc,
      )

  def summary(self) -> str:
    """Returns a one-paragraph status summary of the slate."""
    by_status: dict[str, list[str]] = {status: [] for status in DRAFT_STATUSES}
    for draft in self.drafts.values():
      by_status[draft.status].append(draft.candidate_id)
    return (
        f"Slate: {len(by_status['verified'])} verified"
        f" {by_status['verified']}, {len(by_status['draft'])} unverified drafts"
        f" {by_status['draft']}, {len(by_status['rejected'])} rejected"
        f" {by_status['rejected']}."
    )

  def verification_problems(
      self,
      statement: str,
      target_friction_resolved: str,
      feasibility_assessment: str,
      new_components: str,
      seed_purpose_preserved: str,
  ) -> tuple[list[str], dict[str, Any]]:
    """Returns every reason a candidate cannot be marked verified.

    Args:
      statement: Candidate actionable proposal statement.
      target_friction_resolved: Participant concern addressed.
      feasibility_assessment: Explanation of the feasibility checks performed.
      new_components: One new component per line beyond the original proposal.
      seed_purpose_preserved: Confirmation beginning with ``yes``.

    Returns:
      Tuple of ``(problems, details)``. Stacking-limit violations are prefixed
      with ``STACKING_PROBLEM_TAG``.
    """
    problems: list[str] = []
    details: dict[str, Any] = {}
    if not feasibility_assessment.strip():
      problems.append(
          "a feasibility_assessment is required to verify: for each new"
          " component or change, explain what you researched and what you found"
      )
    if not target_friction_resolved.strip():
      problems.append(
          "target_friction_resolved must name the participant concern this"
          " candidate addresses"
      )

    components = parse_component_lines(new_components)
    details["new_components"] = components
    if not components:
      problems.append(
          "new_components must list each new component this candidate adds"
          " beyond the original proposal, one per line"
      )
    elif len(components) > MAX_NEW_COMPONENTS:
      problems.append(
          f"{STACKING_PROBLEM_TAG} {len(components)} new components declared;"
          f" the hard limit is {MAX_NEW_COMPONENTS}. Split this into separate"
          " candidates with at most two new components each"
      )

    if self.seed_statement:
      added = max(0, count_sentences(statement) - self._seed_sentence_count)
      details["added_sentences"] = added
      if added > MAX_NEW_SENTENCES:
        problems.append(
            f"{STACKING_PROBLEM_TAG} the statement has {added} more sentences"
            f" than the original proposal (limit {MAX_NEW_SENTENCES}). Keep the"
            " original sentences and add at most two, or split the ideas into"
            " separate candidates"
        )

    if not seed_purpose_preserved.strip().lower().startswith("yes"):
      problems.append(
          "seed_purpose_preserved must begin with 'yes' and explain how the"
          " original proposal's core purpose (what it does and what its"
          " resources or funding are for) is kept; if it is not kept, revise or"
          " reject the candidate"
      )
    return problems, details

  def write(
      self,
      candidate_id: str,
      title: str,
      statement: str,
      target_friction_resolved: str,
      feasibility_assessment: str = "",
      status: str = "draft",
      new_components: str = "",
      seed_purpose_preserved: str = "",
  ) -> str:
    """Creates or revises a candidate draft, enforcing the verification gate.

    Args:
      candidate_id: Stable identifier (for example ``c01``); reuse to revise.
      title: Short candidate title.
      statement: Complete actionable proposal statement exactly as returned.
      target_friction_resolved: Participant concern this candidate addresses.
      feasibility_assessment: Required for ``verified`` and ``rejected``.
      status: One of ``draft``, ``verified``, or ``rejected``.
      new_components: New components beyond the original proposal, one per line.
      seed_purpose_preserved: ``yes: ...`` confirming core purpose is preserved.

    Returns:
      Confirmation or detailed gate-failure message followed by slate status.
    """
    cid = candidate_id.strip()
    normalized_status = status.strip().lower()
    if not cid:
      return "ERROR: candidate_id is required."
    if normalized_status not in DRAFT_STATUSES:
      return f"ERROR: status must be one of {DRAFT_STATUSES}; got '{status}'."
    if not statement.strip():
      return "ERROR: statement is required (write the full proposal statement)."

    problems: list[str] = []
    details: dict[str, Any] = {}
    if normalized_status == "rejected" and not feasibility_assessment.strip():
      problems.append(
          "a feasibility_assessment is required to mark a candidate rejected"
      )
    if normalized_status == "verified":
      problems, details = self.verification_problems(
          statement,
          target_friction_resolved,
          feasibility_assessment,
          new_components,
          seed_purpose_preserved,
      )
    applied_status = normalized_status if not problems else "draft"

    prior = self.drafts.get(cid)
    draft = CandidateDraft(
        candidate_id=cid,
        title=title.strip(),
        statement=statement.strip(),
        target_friction_resolved=target_friction_resolved.strip(),
        feasibility_assessment=feasibility_assessment.strip(),
        new_components=new_components.strip(),
        seed_purpose_preserved=seed_purpose_preserved.strip(),
        status=applied_status,
        revisions=(prior.revisions + 1) if prior else 1,
    )
    self.drafts[cid] = draft
    self.history.append({
        "candidate_id": cid,
        "requested_status": normalized_status,
        "applied_status": applied_status,
        "statement": draft.statement,
        "accessed_corpus_files_so_far": sorted(self.access_log.files),
        "gate_problems": problems,
        "gate_details": details,
    })
    self._persist()

    if problems:
      stacked = any(p.startswith(STACKING_PROBLEM_TAG) for p in problems)
      if stacked:
        advice = (
            " Split it: keep this candidate to its single most important change"
            " and draft the other ideas as separate new candidates."
        )
      else:
        advice = (
            " Fix the problems and try again. If it cannot pass every check,"
            " reject it and draft a new candidate from scratch in its place."
        )
      return (
          f"NOT VERIFIED: {cid} saved as 'draft' because "
          + "; ".join(problems)
          + "."
          + advice
          + " "
          + self.summary()
      )
    verb = "Revised" if prior else "Saved"
    return f"{verb} {cid} as '{applied_status}'. " + self.summary()

  def stacking_failures(self) -> list[CandidateDraft]:
    """Returns unverified drafts whose latest check failed for stacking."""
    latest_verify: dict[str, dict[str, Any]] = {}
    for entry in self.history:
      if entry["requested_status"] == "verified":
        latest_verify[entry["candidate_id"]] = entry

    failed: list[CandidateDraft] = []
    for cid, entry in latest_verify.items():
      draft = self.drafts.get(cid)
      if draft is None or draft.status == "verified":
        continue
      if any(
          p.startswith(STACKING_PROBLEM_TAG) for p in entry["gate_problems"]
      ):
        failed.append(draft)
    return failed

  def view(self) -> str:
    """Returns the full draft slate formatted as markdown."""
    if not self.drafts:
      return (
          "The draft slate is empty. Use write_candidate_draft to add"
          " candidates."
      )
    blocks = [self.summary(), ""]
    for draft in self.drafts.values():
      blocks.append(
          f"## {draft.candidate_id} [{draft.status}] {draft.title}"
          f" (revision {draft.revisions})\n"
          f"Statement: {draft.statement}\n"
          f"Addresses: {draft.target_friction_resolved}\n"
          f"New components: {draft.new_components or '(none declared)'}\n"
          "Core purpose kept:"
          f" {draft.seed_purpose_preserved or '(not stated)'}\n"
          "Feasibility research:"
          f" {draft.feasibility_assessment or '(none yet)'}"
      )
    return "\n\n".join(blocks)


def make_draft_slate_tools(
    slate: CandidateDraftSlate,
) -> list[Callable[..., str]]:
  """Creates agent-facing tool callables bound to `slate`.

  Args:
    slate: CandidateDraftSlate instance that the returned tools read and mutate.

  Returns:
    Two-element list ``[write_candidate_draft, view_candidate_drafts]``.
  """

  def write_candidate_draft(
      candidate_id: str,
      title: str,
      statement: str,
      target_friction_resolved: str,
      feasibility_assessment: str = "",
      status: str = "draft",
      new_components: str = "",
      seed_purpose_preserved: str = "",
  ) -> str:
    """Writes or revises one candidate actionable proposal in the draft slate.

    Call once per candidate with status 'draft' when first writing it. Call
    again with the same candidate_id to revise it, mark it 'verified' once it
    passes every check, or mark it 'rejected' if it cannot pass.

    Marking 'verified' is refused unless ALL of these hold:
    (1) new_components lists each new component added beyond the original
        proposal, one per line, and there are at most two;
    (2) the statement has at most two more sentences than the original
        proposal;
    (3) seed_purpose_preserved begins with 'yes' and says how the original
        proposal's core purpose is kept;
    (4) feasibility_assessment explains, for each new component or change,
        what was researched to check feasibility and what was found.

    Args:
      candidate_id: Stable id such as 'c01'. Reuse it to revise a candidate.
      title: Short title.
      statement: Complete, self-contained actionable proposal statement,
        exactly as it should appear in the final output.
      target_friction_resolved: Participant concern about this proposal that
        the candidate addresses.
      feasibility_assessment: Required for 'verified' or 'rejected'. For
        'verified', the checks performed for each change and what was found;
        for 'rejected', why it failed.
      status: 'draft', 'verified', or 'rejected'.
      new_components: New components beyond the original proposal, one per line
        (a changed number or rate counts as one component).
      seed_purpose_preserved: 'yes: ...' followed by how the original
        proposal's core purpose is kept.

    Returns:
      Confirmation or gate refusal reasons, followed by slate status.
    """
    return slate.write(
        candidate_id=candidate_id,
        title=title,
        statement=statement,
        target_friction_resolved=target_friction_resolved,
        feasibility_assessment=feasibility_assessment,
        status=status,
        new_components=new_components,
        seed_purpose_preserved=seed_purpose_preserved,
    )

  def view_candidate_drafts() -> str:
    """Shows every candidate in the draft slate with status and checks."""
    return slate.view()

  return [write_candidate_draft, view_candidate_drafts]


def _match_draft(
    variation: Mapping[str, Any], slate: CandidateDraftSlate
) -> tuple[str | None, float]:
  """Finds the best matching draft in `slate` for a returned variation."""
  desc = normalize_text(variation.get("description") or "")
  cid = (variation.get("candidate_id") or "").strip()
  best_id: str | None = None
  best_ratio = 0.0
  candidate_ids = [cid] if cid in slate.drafts else list(slate.drafts)
  for key in candidate_ids:
    ratio = difflib.SequenceMatcher(
        None, desc, normalize_text(slate.drafts[key].statement)
    ).ratio()
    if ratio > best_ratio:
      best_id = key
      best_ratio = ratio
  return best_id, best_ratio


def audit_final_against_slate(
    variations: Sequence[Mapping[str, Any]], slate: CandidateDraftSlate
) -> dict[str, Any]:
  """Checks that each returned variation matches a verified draft in `slate`.

  A variation matches when its ``candidate_id`` names a verified draft whose
  statement equals the variation description (case/whitespace-insensitive) or
  reaches at least 90% sequence similarity. Falls back to the best-matching
  draft across the slate when ``candidate_id`` is missing or unknown.

  Args:
    variations: Final variation mappings returned by an agent session.
    slate: Session draft slate to audit against.

  Returns:
    Dictionary containing per-variation match rows and aggregate counts.
  """
  verified = {
      cid: draft
      for cid, draft in slate.drafts.items()
      if draft.status == "verified"
  }
  rows: list[dict[str, Any]] = []
  verified_count = 0
  for variation in variations:
    cid = (variation.get("candidate_id") or "").strip()
    best_id, best_ratio = _match_draft(variation, slate)
    matched = best_id is not None and best_ratio >= _STATEMENT_MATCH_RATIO
    status = slate.drafts[best_id].status if best_id else None
    is_verified = bool(matched and best_id in verified)
    if is_verified:
      verified_count += 1
    rows.append({
        "variation_id": variation.get("variation_id"),
        "candidate_id": cid or None,
        "matched_draft": best_id if matched else None,
        "match_ratio": round(best_ratio, 3),
        "draft_status": status if matched else None,
        "is_verified": is_verified,
    })

  return {
      "n_variations": len(rows),
      "n_verified_matches": verified_count,
      "n_unverified": len(rows) - verified_count,
      "n_drafts": len(slate.drafts),
      "n_drafts_verified": len(verified),
      "n_drafts_rejected": sum(
          draft.status == "rejected" for draft in slate.drafts.values()
      ),
      "n_accessed_corpus_files": len(slate.access_log.files),
      "variations": rows,
  }


def hard_filter_problems(
    variation: Mapping[str, Any],
    seed_statement: str,
    slate: CandidateDraftSlate | None = None,
) -> list[str]:
  """Returns reasons a returned variation breaks hard addition limits.

  Args:
    variation: Returned variation mapping containing a ``description`` string.
    seed_statement: Original actionable proposal statement.
    slate: Optional session slate used to inspect the matched draft's declared
      ``new_components``.

  Returns:
    List of human-readable problem descriptions (empty when compliant).
  """
  problems: list[str] = []
  added = added_sentence_count(
      variation.get("description") or "", seed_statement
  )
  if added > MAX_NEW_SENTENCES:
    problems.append(
        f"{added} more sentences than the original (limit {MAX_NEW_SENTENCES})"
    )
  if slate is not None:
    best_id, ratio = _match_draft(variation, slate)
    if best_id is not None and ratio >= _STATEMENT_MATCH_RATIO:
      components = parse_component_lines(slate.drafts[best_id].new_components)
      if len(components) > MAX_NEW_COMPONENTS:
        problems.append(
            f"{len(components)} declared new components"
            f" (limit {MAX_NEW_COMPONENTS})"
        )
  return problems


def apply_hard_filter(
    variations: Sequence[Mapping[str, Any]],
    seed_statement: str,
    slate: CandidateDraftSlate | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[str]]]]:
  """Splits variations into those within hard addition limits and those over.

  Does not mutate input variation mappings.

  Args:
    variations: Returned variation mappings.
    seed_statement: Original actionable proposal statement.
    slate: Optional session slate for declared component checks.

  Returns:
    Tuple ``(kept_variations, dropped_entries)`` where each dropped entry is a
    ``(variation_dict, problems)`` pair.
  """
  kept: list[dict[str, Any]] = []
  dropped: list[tuple[dict[str, Any], list[str]]] = []
  for variation in variations:
    item = dict(variation)
    problems = hard_filter_problems(item, seed_statement, slate)
    if problems:
      dropped.append((item, problems))
    else:
      kept.append(item)
  return kept, dropped
