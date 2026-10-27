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

"""Describes participants from what they say about themselves.

Surveys often collect few demographics, yet responses carry context such as
"I run a bakery" or "my elderly mother relies on paratransit". For each
configured attribute, an extraction model marks the participant's own words
as evidence. When it marks nothing, the attribute gets no evidence and the
label "none" without a further question. Marked evidence is only a
candidate, as extraction models sometimes mark unrelated words, so the
decision model, when one is configured, checks it: for an attribute with
labels it picks a label or "none", and for one without labels it answers
whether the evidence states what the attribute asks, and rejected evidence
is dropped. Without a decision model, evidence is unchecked.
"""

from __future__ import annotations

import asyncio  # noqa: ANYIO_OK
import dataclasses
import json
from typing import Final, Mapping, Sequence

from src.models import extraction
from src.models import model_factory
from src.models import systemone
from src.quote_extraction import quote_extraction_lib

NONE_LABEL: Final = "none"


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute:
  """One thing to learn about each participant.

  Attributes:
    name: Column prefix for the outputs; letters, digits and underscores.
    instructions: What the extraction model should mark as evidence.
    labels: Label keys (no spaces) and descriptions for the decision model.
      Empty means evidence only, which the decision model checks with a
      yes/no question when one is configured.
  """

  name: str
  instructions: str
  labels: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, slots=True)
class AttributeResult:
  """What was learned about one participant for one attribute.

  Attributes:
    evidence: The participant's own words, or None when they said nothing.
    not_stated: The extraction model's probability that the response says
      nothing for the attribute, when it reports one.
    label: A label key, NONE_LABEL, or None when no labels are asked.
    label_probability: The decision model's probability for the label.
  """

  evidence: str | None
  not_stated: float | None
  label: str | None
  label_probability: float | None


DEFAULT_ATTRIBUTES: Final = (
    Attribute(
        name="role",
        instructions=(
            "Words where the participant says who they are or what role they"
            " speak from, such as running a business, raising children or"
            " caring for a family member, working shifts, or working from"
            " home. Do not mark people they only mention."
        ),
        labels={
            "business_owner": "Owns or runs a business.",
            "parent_or_caregiver": (
                "Is a parent, or cares for a family member."
            ),
            "shift_worker": "Works shifts or nonstandard hours.",
            "remote_worker": "Works from home.",
        },
    ),
)


def load_attributes(path: str) -> tuple[Attribute, ...]:
  """Reads attributes from a JSON file.

  The file holds a list of objects with "name", "instructions" and optional
  "labels" (an object of label key to description).

  Args:
    path: Path to the JSON file.

  Returns:
    The attributes, in file order.

  Raises:
    ValueError: If the file does not describe valid attributes.
  """
  with open(path, encoding="utf-8") as handle:
    raw = json.load(handle)
  if not isinstance(raw, list) or not raw:
    raise ValueError(f"{path} must hold a non-empty list of attributes.")
  attributes = []
  for item in raw:
    if not isinstance(item, dict):
      raise ValueError(f"{path}: each attribute must be an object.")
    name = item.get("name")
    instructions = item.get("instructions")
    labels = item.get("labels", {})
    if not isinstance(name, str) or not name.isidentifier():
      raise ValueError(f"{path}: attribute name {name!r} is not a valid name.")
    if not isinstance(instructions, str) or not instructions.strip():
      raise ValueError(f"{path}: attribute {name} needs instructions.")
    if not isinstance(labels, dict) or any(
        not isinstance(k, str) or not k or any(c.isspace() for c in k)
        for k in labels
    ):
      raise ValueError(
          f"{path}: labels of {name} must map keys without spaces to"
          " descriptions."
      )
    if NONE_LABEL in labels:
      raise ValueError(f"{path}: label {NONE_LABEL!r} is reserved.")
    attributes.append(Attribute(name, instructions, dict(labels)))
  if len({a.name for a in attributes}) != len(attributes):
    raise ValueError(f"{path}: attribute names must be unique.")
  return tuple(attributes)


async def describe_participant(
    extractor: extraction.Extractor,
    decision_client: systemone.SystemOneClient | None,
    text: str,
    attributes: Sequence[Attribute],
    additional_context: str | None = None,
) -> dict[str, AttributeResult]:
  """Finds evidence for each attribute in one response and labels it.

  Args:
    extractor: Marks evidence spans; one request per response.
    decision_client: Picks labels, and checks the evidence of attributes
      without labels; None leaves labels unset and evidence unchecked.
    text: The participant's response; <response> segments are joined.
    attributes: What to learn.
    additional_context: Optional survey context for both models.

  Returns:
    Results keyed by attribute name.

  Raises:
    SystemOneError: If a model request fails.
  """
  text = quote_extraction_lib.join_response_text(text)
  context = (
      f"Survey context: {additional_context.strip()}\n"
      if additional_context and additional_context.strip()
      else ""
  )
  extracted = await extractor.extract(
      text,
      {
          attribute.name: extraction.Field(
              instructions=f"{context}{attribute.instructions}", many=False
          )
          for attribute in attributes
      },
  )
  results = {}
  for attribute in attributes:
    found = extracted[attribute.name]
    evidence = found.spans[0].text if found.spans else None
    if (
        decision_client is not None
        and evidence is not None
        and not attribute.labels
        and not await _confirms(
            decision_client, text, evidence, attribute, context
        )
    ):
      evidence = None
    label, probability = None, None
    if decision_client is not None and attribute.labels:
      if evidence is None:
        label = NONE_LABEL
      else:
        label, probability = await _label(
            decision_client, text, evidence, attribute, context
        )
    results[attribute.name] = AttributeResult(
        evidence=evidence,
        not_stated=found.not_stated,
        label=label,
        label_probability=probability,
    )
  return results


async def describe_participants(
    extractor: extraction.Extractor,
    decision_client: systemone.SystemOneClient | None,
    texts: Sequence[str],
    attributes: Sequence[Attribute] = DEFAULT_ATTRIBUTES,
    additional_context: str | None = None,
) -> list[dict[str, AttributeResult]]:
  """Describes every participant; see describe_participant.

  Args:
    extractor: Marks evidence spans.
    decision_client: Picks labels; None leaves labels unset.
    texts: One response per participant.
    attributes: What to learn.
    additional_context: Optional survey context.

  Returns:
    One result mapping per response, in input order.
  """
  return list(
      await asyncio.gather(*(
          describe_participant(
              extractor, decision_client, text, attributes, additional_context
          )
          for text in texts
      ))
  )


def result_columns(
    results: Sequence[Mapping[str, AttributeResult]],
    attributes: Sequence[Attribute],
) -> dict[str, list[object]]:
  """Flattens results into output columns.

  Args:
    results: One result mapping per participant.
    attributes: The attributes that were asked.

  Returns:
    Columns <name>_evidence, <name>_not_stated, <name>_label and
    <name>_label_probability for each attribute.
  """
  columns: dict[str, list[object]] = {}
  for attribute in attributes:
    for suffix in ("evidence", "not_stated", "label", "label_probability"):
      columns[f"{attribute.name}_{suffix}"] = [
          getattr(result[attribute.name], suffix) for result in results
      ]
  return columns


async def _confirms(
    client: systemone.SystemOneClient,
    text: str,
    evidence: str,
    attribute: Attribute,
    context: str,
) -> bool:
  """Asks the decision model whether evidence states what is asked."""
  question = systemone.NoulQuestion(
      instructions=(
          f"{context}Does the evidence state the following about the"
          f" participant themselves? {attribute.instructions}"
      ),
      true="The evidence states this about the participant.",
      false=(
          "The evidence does not state this about the participant; it only"
          " shares words with it or is about something else."
      ),
  )
  answers = await client.ask(
      f"Participant response:\n{text}\n\nEvidence: {evidence}",
      {"evidence": question},
  )
  answer = answers["evidence"]
  if not isinstance(answer, systemone.NoulAnswer):
    raise systemone.SystemOneError(
        systemone.FailureKind.PARSE,
        "evidence question did not return a yes/no answer.",
    )
  return answer.noul >= model_factory.get_decision_threshold()


async def _label(
    client: systemone.SystemOneClient,
    text: str,
    evidence: str,
    attribute: Attribute,
    context: str,
) -> tuple[str, float]:
  """Asks the decision model which label, or none, the evidence supports."""
  question = systemone.ChoiceQuestion(
      instructions=(
          f"{context}Based on the evidence the participant gives about"
          f" themselves, which best describes them for '{attribute.name}'?"
      ),
      criteria={
          **attribute.labels,
          NONE_LABEL: "None of these describes the participant.",
      },
  )
  answers = await client.ask(
      f"Participant response:\n{text}\n\nEvidence: {evidence}",
      {"label": question},
  )
  answer = answers["label"]
  if not isinstance(answer, systemone.ChoiceAnswer):
    raise systemone.SystemOneError(
        systemone.FailureKind.PARSE, "label question did not return a choice."
    )
  return answer.choice, answer.probabilities.get(answer.choice, 0.0)
