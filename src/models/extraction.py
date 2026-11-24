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

"""Marks exact spans of text through interchangeable extraction models.

An extractor takes a text and named fields described in plain language, and
returns for each field the spans of the text that match it. Spans are exact
substrings: every backend checks ``text[start:end] == span.text`` before
returning, so callers can rely on quotes and evidence being verbatim.

Two backends are provided:

* SystemOneExtractor asks extract questions through a System One decision
  endpoint (Inception Mercury Decide answers them).
* ClairicExtractor calls Cloudflare's Clairic extraction model on Workers AI.
"""

from __future__ import annotations

import asyncio  # noqa: ANYIO_OK
import dataclasses
import json
from typing import Final, Mapping, Protocol

import pydantic

from src.models import systemone

CLAIRIC_MODEL: Final = "@cf/cloudflare/clairic"


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
  """One thing to mark in the text.

  Attributes:
    instructions: Plain-language description of what to mark.
    many: Whether several spans may be marked. A single-span field returns at
      most one span.
  """

  instructions: str
  many: bool = True


class Extraction(pydantic.BaseModel):
  """The spans marked for one field, in text order.

  Attributes:
    spans: Exact substrings of the text.
    not_stated: Probability that the text says nothing for the field, when the
      backend reports one.
  """

  model_config = pydantic.ConfigDict(frozen=True)

  spans: tuple[systemone.Span, ...]
  not_stated: float | None = None


class Extractor(Protocol):
  """Marks spans of a text for named fields."""

  model: str

  async def extract(
      self, text: str, fields: Mapping[str, Field]
  ) -> dict[str, Extraction]:
    """Returns the spans marked for each field, keyed like fields."""
    ...


def check_spans(text: str, name: str, spans: tuple[systemone.Span, ...]) -> None:
  """Raises unless every span is the exact text at its offsets.

  Offsets count Unicode code points, as Python string indexing does.

  Args:
    text: The text the spans index into.
    name: Field name, for the error message.
    spans: Spans to check.

  Raises:
    SystemOneError: With kind PARSE if a span is not at its offsets.
  """
  for span in spans:
    if not 0 <= span.start <= span.end <= len(text) or (
        text[span.start : span.end] != span.text
    ):
      raise systemone.SystemOneError(
          systemone.FailureKind.PARSE,
          f"field {name}: span {span.start}-{span.end} {span.text!r} is not"
          " the text at those offsets.",
      )


class SystemOneExtractor:
  """Asks extract questions through a System One decision endpoint."""

  def __init__(self, client: systemone.SystemOneClient):
    """Initializes the extractor.

    Args:
      client: A client whose host answers extract questions.
    """
    self._client = client
    self.model = client.model

  async def extract(
      self, text: str, fields: Mapping[str, Field]
  ) -> dict[str, Extraction]:
    """Asks one extract question per field and returns the marked spans.

    One question per field keeps fields independent, so two fields may mark
    the same words. A single-span field keeps its most confident span.

    Args:
      text: The text to mark.
      fields: Fields to mark, keyed by name.

    Returns:
      The spans marked for each field.

    Raises:
      SystemOneError: If the host rejects or cannot answer the request.
    """
    answers = await self._client.ask(
        text,
        {
            name: systemone.ExtractQuestion(instructions=field.instructions)
            for name, field in fields.items()
        },
    )
    results = {}
    for name, field in fields.items():
      answer = answers[name]
      if not isinstance(answer, systemone.ExtractAnswer):
        raise systemone.SystemOneError(
            systemone.FailureKind.PARSE,
            f"question {name} did not return an extract answer.",
        )
      spans = answer.spans
      if not field.many and len(spans) > 1:
        spans = (max(spans, key=lambda span: span.confidence),)
      results[name] = Extraction(spans=spans, not_stated=answer.not_stated)
    return results


class _ClairicAnswer(pydantic.BaseModel):
  """Wire shape of one Clairic text or list answer."""

  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  value: str | list[str] | None = None
  span: list[int] | None = None
  spans: list[list[int] | None] | None = None
  confidence: float | None = None
  p_null: float | None = None


class ClairicExtractor:
  """Marks spans with Cloudflare's Clairic extraction model on Workers AI.

  A field that allows several spans is asked as a list of text, and a
  single-span field as text. Answers that come back without offsets are
  dropped, as they cannot be checked against the text. A field missing from
  the response, or offsets that are not a [start, end] pair, are parse
  errors.
  """

  def __init__(
      self,
      endpoint: str,
      api_key: str | None = None,
      transport: systemone.Transport | None = None,
      max_concurrent: int = 1,
      retry_delay_seconds: float = 1.0,
  ):
    """Initializes the extractor.

    Args:
      endpoint: The Workers AI run URL for the Clairic model.
      api_key: Cloudflare API token with Workers AI access.
      transport: Optional byte transport. Tests inject this.
      max_concurrent: How many requests may be in flight.
      retry_delay_seconds: Backoff unit between attempts after a transient
        failure.

    Raises:
      SystemOneError: If max_concurrent is below 1.
    """
    if max_concurrent < 1:
      raise systemone.SystemOneError(
          systemone.FailureKind.CONFIG,
          f"max_concurrent must be >= 1, got {max_concurrent}.",
      )
    self.endpoint = endpoint
    self.model = CLAIRIC_MODEL
    self._transport = transport or systemone.make_transport(api_key)
    self._max_concurrent = max_concurrent
    self._retry_delay_seconds = retry_delay_seconds
    self._limiter: asyncio.Semaphore | None = None

  async def extract(
      self, text: str, fields: Mapping[str, Field]
  ) -> dict[str, Extraction]:
    """Sends every field in one request and returns the marked spans.

    Args:
      text: The text to mark.
      fields: Fields to mark, keyed by name.

    Returns:
      The spans marked for each field.

    Raises:
      SystemOneError: If the request fails, the response cannot be parsed, or
        a span is not at its offsets.
    """
    body = json.dumps({
        "state": text,
        "fields": {
            name: (
                {
                    "type": "list",
                    "items": "text",
                    "instructions": field.instructions,
                }
                if field.many
                else {"type": "text", "instructions": field.instructions}
            )
            for name, field in fields.items()
        },
    }).encode("utf-8")
    if self._limiter is None:
      self._limiter = asyncio.Semaphore(self._max_concurrent)
    raw = await systemone.post_with_retry(
        self._transport,
        self.endpoint,
        body,
        self._limiter,
        self._retry_delay_seconds,
    )
    answers = _parse_clairic(raw)
    results = {}
    for name in fields:
      if name not in answers:
        raise systemone.SystemOneError(
            systemone.FailureKind.PARSE,
            f"response did not answer field {name}.",
        )
      answer = answers[name]
      spans = _clairic_spans(name, answer)
      check_spans(text, name, spans)
      results[name] = Extraction(spans=spans, not_stated=answer.p_null)
    return results


def _parse_clairic(raw: bytes) -> dict[str, _ClairicAnswer]:
  """Parses a Clairic response, with or without the Workers AI envelope."""
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise systemone.SystemOneError(
        systemone.FailureKind.PARSE, "response was not JSON."
    ) from exc
  if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
    payload = payload["result"]
  answers = payload.get("answers") if isinstance(payload, dict) else None
  if not isinstance(answers, dict):
    raise systemone.SystemOneError(
        systemone.FailureKind.PARSE, "response did not contain answers."
    )
  try:
    return {
        name: _ClairicAnswer.model_validate(answer)
        for name, answer in answers.items()
    }
  except pydantic.ValidationError as exc:
    raise systemone.SystemOneError(
        systemone.FailureKind.PARSE, "response did not match Clairic."
    ) from exc


def _clairic_spans(
    name: str, answer: _ClairicAnswer
) -> tuple[systemone.Span, ...]:
  """Turns a Clairic text or list answer into spans in text order.

  Args:
    name: The field name, for error messages.
    answer: The parsed answer for the field.

  Returns:
    The answer's spans that have offsets, sorted by start.

  Raises:
    SystemOneError: If offsets are not a [start, end] pair.
  """
  confidence = answer.confidence if answer.confidence is not None else 0.0
  if isinstance(answer.value, list):
    pairs = [
        (value, offsets)
        for value, offsets in zip(answer.value, answer.spans or [])
        if offsets is not None
    ]
  elif answer.value is not None and answer.span is not None:
    pairs = [(answer.value, answer.span)]
  else:
    pairs = []
  for _, offsets in pairs:
    if len(offsets) != 2:
      raise systemone.SystemOneError(
          systemone.FailureKind.PARSE,
          f"field {name} has offsets {offsets}, not [start, end].",
      )
  spans = [
      systemone.Span(
          text=value, start=offsets[0], end=offsets[1], confidence=confidence
      )
      for value, offsets in pairs
  ]
  return tuple(sorted(spans, key=lambda span: span.start))
