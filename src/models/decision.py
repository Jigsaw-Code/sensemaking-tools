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

"""Judgment operations that a System One model can answer.

Generative stages stay on the existing model. These helpers cover the stages
that already pick a label, a probability, or a winner: assignment, scoring,
approval votes, dedup, and autorater verdicts.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Iterator, assert_never

from src.models.systemone import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    FailureKind,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    SystemOneClient,
    SystemOneError,
)

ENDPOINT_TYPE: Final = "systemone"


@dataclass(frozen=True, slots=True)
class _NoOverride:
  """Sentinel: no test client is installed."""


_NO_OVERRIDE = _NoOverride()
_override: SystemOneClient | None | _NoOverride = _NO_OVERRIDE
_cached: SystemOneClient | None = None

RUBRIC: Final[dict[str, str]] = {
    "4": "The response performs well on all criteria.",
    "3": "The response performs well on most criteria.",
    "2": "The response performs well on some criteria.",
    "1": "The response is somewhat aligned with the criteria.",
    "0": "The response falls short on all criteria.",
}

_EQUIVALENCE_STATE: Final = (
    "Two propositions are effectively equivalent when a survey participant"
    " would find them repetitive: synonyms, or the same claim at a different"
    " level of specificity. Distinct claims are not equivalent."
)


@dataclass(frozen=True, slots=True)
class AttributeSpec:
  """One moderation or bridging attribute to score as a probability."""

  name: str
  label: str
  definition: str
  guidance: str = ""


def decision_enabled() -> bool:
  """Returns whether judgment stages should call System One."""
  match _override:
    case _NoOverride():
      return os.getenv("DECISION_ENDPOINT_TYPE") == ENDPOINT_TYPE
    case None:
      return False
    case SystemOneClient():
      return True
    case unreachable:
      assert_never(unreachable)


def decision_client() -> SystemOneClient | None:
  """Returns the configured client, or None when the decision path is off."""
  global _cached
  match _override:
    case _NoOverride():
      if os.getenv("DECISION_ENDPOINT_TYPE") != ENDPOINT_TYPE:
        return None
      if _cached is None:
        _cached = client_from_env()
      return _cached
    case None:
      return None
    case SystemOneClient() as installed:
      return installed
    case unreachable:
      assert_never(unreachable)


def client_from_env() -> SystemOneClient:
  """Builds a client from DECISION_MODEL and OLLAMA_HOST."""
  model = os.getenv("DECISION_MODEL")
  if not model:
    raise SystemOneError(
        FailureKind.CONFIG,
        "DECISION_MODEL must be set when DECISION_ENDPOINT_TYPE is"
        " 'systemone'.",
    )
  host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
  raw_limit = os.getenv("DECISION_MAX_CONCURRENT", "1")
  try:
    limit = int(raw_limit)
  except ValueError as exc:
    raise SystemOneError(
        FailureKind.CONFIG,
        f"DECISION_MAX_CONCURRENT must be an integer, got {raw_limit!r}.",
    ) from exc
  return SystemOneClient(model=model, base_url=host, max_concurrent=limit)


@contextmanager
def use_decision_client(
    client: SystemOneClient | None,
) -> Iterator[SystemOneClient | None]:
  """Installs a client for the duration of a test or a nested call."""
  global _override
  previous = _override
  _override = client
  try:
    yield client
  finally:
    _override = previous


def threshold() -> float:
  """Returns the probability cutoff for labels and equivalence."""
  raw = os.getenv("DECISION_THRESHOLD", "0.5")
  try:
    value = float(raw)
  except ValueError as exc:
    raise SystemOneError(
        FailureKind.CONFIG,
        f"DECISION_THRESHOLD must be a float, got {raw!r}.",
    ) from exc
  if not 0.0 <= value <= 1.0:
    raise SystemOneError(
        FailureKind.CONFIG,
        f"DECISION_THRESHOLD must be between 0 and 1, got {value}.",
    )
  return value


async def assign_labels(
    client: SystemOneClient,
    text: str,
    labels: list[str],
    *,
    kind: str,
    cutoff: float | None = None,
) -> list[str]:
  """Assigns every label whose yes-probability meets the cutoff.

  A text with no label above the cutoff gets the single highest label, so the
  caller still receives an assignment. ``Other`` is dropped when any specific
  label was selected.
  """
  ordered = list(dict.fromkeys(labels))
  if not ordered:
    return []
  cutoff = threshold() if cutoff is None else cutoff
  questions = {
      f"q{index}": NoulQuestion(
          instructions=f"Does this text belong under the {kind} '{label}'?",
          true=f"The text contains a claim that belongs under '{label}'.",
          false=f"The text does not belong under '{label}'.",
      )
      for index, label in enumerate(ordered)
  }
  answers = await client.ask(text, questions)
  probabilities = {
      label: _noul(answers[f"q{index}"], label)
      for index, label in enumerate(ordered)
  }
  return _select_labels(ordered, probabilities, cutoff)


async def score_attributes(
    client: SystemOneClient,
    text: str,
    attributes: list[AttributeSpec],
) -> dict[str, float]:
  """Scores each attribute as the probability that the text exhibits it."""
  if not attributes:
    return {}
  questions = {}
  for index, attribute in enumerate(attributes):
    true = attribute.definition
    if attribute.guidance:
      true = f"{attribute.definition} {attribute.guidance}"
    questions[f"a{index}"] = NoulQuestion(
        instructions=(
            f"Does this text exhibit {attribute.label}? {attribute.definition}"
        ),
        true=true,
        false=f"The text does not exhibit {attribute.label}.",
    )
  answers = await client.ask(text, questions)
  return {
      attribute.name: _noul(answers[f"a{index}"], attribute.name)
      for index, attribute in enumerate(attributes)
  }


async def equivalence_sets(
    client: SystemOneClient,
    items: dict[str, str],
    cutoff: float | None = None,
) -> list[list[str]]:
  """Clusters ids whose texts are effectively equivalent.

  Pairs are judged independently, then joined with union-find. Sets of one
  are omitted, matching the generative dedup prompt.
  """
  ids = list(items)
  if len(ids) < 2:
    return []
  cutoff = threshold() if cutoff is None else cutoff
  pairs = [
      (left, right)
      for index, left in enumerate(ids)
      for right in ids[index + 1 :]
  ]
  questions: dict[str, NoulQuestion] = {}
  pair_by_key: dict[str, tuple[str, str]] = {}
  for index, (left, right) in enumerate(pairs):
    key = f"p{index}"
    pair_by_key[key] = (left, right)
    questions[key] = NoulQuestion(
        instructions=(
            "Are these two propositions effectively equivalent in meaning"
            " for a survey participant?\n"
            f"A ({left}): {items[left]}\n"
            f"B ({right}): {items[right]}"
        ),
        true=(
            "A survey participant would find these repetitive: synonyms, or"
            " the same claim at a different level of specificity."
        ),
        false="These are distinct claims.",
    )
  answers = await client.ask(_EQUIVALENCE_STATE, questions)
  linked = [
      pair_by_key[key]
      for key, answer in answers.items()
      if _noul(answer, key) >= cutoff
  ]
  return _clusters(ids, linked)


async def choose_one(
    client: SystemOneClient,
    state: str,
    options: list[tuple[str, str]],
    instructions: str,
) -> str:
  """Returns the id of the winning option.

  Choice accepts at most 26 options, so larger sets are reduced by a
  tournament. Option ids may contain spaces; wire keys do not.
  """
  if not options:
    raise SystemOneError(FailureKind.CONFIG, "choose_one requires an option.")
  if len(options) == 1:
    return options[0][0]
  if len(options) <= 26:
    return await _choose_pack(client, state, options, instructions)
  winners: list[tuple[str, str]] = []
  for start in range(0, len(options), 26):
    pack = options[start : start + 26]
    if len(pack) == 1:
      winners.append(pack[0])
      continue
    winner_id = await _choose_pack(client, state, pack, instructions)
    winners.append(next(item for item in pack if item[0] == winner_id))
  return await choose_one(client, state, winners, instructions)


async def approval_votes(
    client: SystemOneClient,
    participant: str,
    statements: list[str],
    options: list[str],
) -> dict[str, str]:
  """Predicts one scale label per statement for a participant."""
  if len(options) < 2:
    raise SystemOneError(
        FailureKind.CONFIG,
        "approval votes need at least two scale options.",
    )
  questions = {}
  statement_by_key = {}
  for index, statement in enumerate(statements):
    key = f"s{index}"
    statement_by_key[key] = statement
    questions[key] = ChoiceQuestion(
        instructions=(
            "How would this participant vote on the following statement?\n"
            f"{statement}"
        ),
        criteria={
            f"o{opt_index}": option for opt_index, option in enumerate(options)
        },
    )
  answers = await client.ask(participant, questions)
  votes: dict[str, str] = {}
  for key, statement in statement_by_key.items():
    votes[statement] = _choice_label(answers[key], options, key)
  return votes


async def rubric_level(
    client: SystemOneClient,
    state: str,
    instructions: str | None = None,
) -> int:
  """Picks the 0-4 autorater level. Clef does not write the explanation."""
  question = ChoiceQuestion(
      instructions=instructions
      or (
          "Which rubric level best describes the evaluation target in the"
          " state? Choose 4 only when the target performs well on all"
          " criteria."
      ),
      criteria=dict(RUBRIC),
  )
  answers = await client.ask(state, {"rubric": question})
  choice = _choice_key(answers["rubric"], "rubric")
  try:
    return int(choice)
  except ValueError as exc:
    raise SystemOneError(
        FailureKind.PARSE,
        f"rubric choice {choice!r} is not a level.",
    ) from exc


def _select_labels(
    labels: list[str],
    probabilities: dict[str, float],
    cutoff: float,
) -> list[str]:
  above = [label for label in labels if probabilities[label] >= cutoff]
  specific = [label for label in above if label != "Other"]
  if specific:
    return specific
  if above:
    return above
  return [
      max(
          labels, key=lambda label: (probabilities[label], -labels.index(label))
      )
  ]


def _choice_key(answer: Answer, name: str) -> str:
  match answer:
    case ChoiceAnswer(choice=choice):
      return choice
    case NoulAnswer() | ScoreAnswer():
      raise SystemOneError(
          FailureKind.PARSE,
          f"question {name} did not return a choice.",
      )
    case unreachable:
      assert_never(unreachable)


def _choice_label(answer: Answer, options: list[str], name: str) -> str:
  choice = _choice_key(answer, name)
  try:
    opt_index = int(choice.removeprefix("o"))
    return options[opt_index]
  except (ValueError, IndexError) as exc:
    raise SystemOneError(
        FailureKind.PARSE,
        f"approval choice {choice!r} is not a scale option.",
    ) from exc


def _noul(answer: Answer, name: str) -> float:
  match answer:
    case NoulAnswer(noul=noul):
      return noul
    case ChoiceAnswer() | ScoreAnswer():
      raise SystemOneError(
          FailureKind.PARSE,
          f"question {name} did not return a noul answer.",
      )
    case unreachable:
      assert_never(unreachable)


async def _choose_pack(
    client: SystemOneClient,
    state: str,
    options: list[tuple[str, str]],
    instructions: str,
) -> str:
  criteria = {
      f"k{index}": description for index, (_, description) in enumerate(options)
  }
  answers = await client.ask(
      state,
      {"winner": ChoiceQuestion(instructions=instructions, criteria=criteria)},
  )
  choice = _choice_key(answers["winner"], "winner")
  try:
    index = int(choice.removeprefix("k"))
  except ValueError as exc:
    raise SystemOneError(
        FailureKind.PARSE,
        f"choice {choice!r} is not an option key.",
    ) from exc
  try:
    return options[index][0]
  except IndexError as exc:
    raise SystemOneError(
        FailureKind.PARSE,
        f"choice {answer.choice!r} is outside the option list.",
    ) from exc


def _clusters(
    ids: list[str],
    links: list[tuple[str, str]],
) -> list[list[str]]:
  parent = {item: item for item in ids}

  def find(item: str) -> str:
    while parent[item] != item:
      parent[item] = parent[parent[item]]
      item = parent[item]
    return item

  for left, right in links:
    root_left = find(left)
    root_right = find(right)
    if root_left != root_right:
      parent[root_right] = root_left

  groups: dict[str, list[str]] = {}
  for item in ids:
    groups.setdefault(find(item), []).append(item)
  return [sorted(group) for group in groups.values() if len(group) > 1]
