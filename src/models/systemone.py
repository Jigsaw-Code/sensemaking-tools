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

"""Client for Ollama System One decision models such as Clef.

Clef does not implement chat or completion. Judgments go to
``POST /v1/systemone`` and come back as choice, noul, or score answers.
"""

from __future__ import annotations

import asyncio  # noqa: ANYIO_OK
import dataclasses
import enum
import json
import urllib.error
import urllib.request
from typing import Annotated, Callable, Final, Literal, assert_never

import pydantic

MAX_QUESTIONS: Final = 64
MAX_CHOICE_OPTIONS: Final = 26
MAX_BODY_BYTES: Final = 64 * 1024
# Leave headroom under the server's 64 KiB text limit while packing.
PACK_BODY_BYTES: Final = 60 * 1024
DEFAULT_TIMEOUT_SECONDS: Final = 600


class FailureKind(enum.Enum):
  """Why a System One call could not be completed."""

  HTTP = "http"
  PAYLOAD = "payload"
  PARSE = "parse"
  CONFIG = "config"


@dataclasses.dataclass(frozen=True, slots=True)
class SystemOneError(Exception):
  """A decision request failed before a usable answer was available."""

  kind: FailureKind
  message: str
  status: int | None = None

  def __str__(self) -> str:
    if self.status is None:
      return self.message
    return f"{self.status}: {self.message}"


@dataclasses.dataclass(frozen=True, slots=True)
class ChoiceQuestion:
  """Pick one of 2-26 named options."""

  instructions: str
  criteria: dict[str, str]


@dataclasses.dataclass(frozen=True, slots=True)
class NoulQuestion:
  """Probability that a yes/no condition holds."""

  instructions: str
  true: str
  false: str


@dataclasses.dataclass(frozen=True, slots=True)
class ScoreQuestion:
  """Probability-weighted position on an ordered rubric."""

  instructions: str
  criteria: tuple[str, ...]


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


class _ChoiceAnswer(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["choice"]
  choice: str
  probabilities: dict[str, float]
  confidence: float


class _NoulAnswer(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["noul"]
  noul: float


class _ScoreAnswer(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["score"]
  score: float
  probabilities: dict[str, float]
  confidence: float


class ChoiceAnswer(pydantic.BaseModel):
  """A parsed choice answer."""

  model_config = pydantic.ConfigDict(frozen=True)

  choice: str
  probabilities: dict[str, float]
  confidence: float


class NoulAnswer(pydantic.BaseModel):
  """A parsed yes/no probability."""

  model_config = pydantic.ConfigDict(frozen=True)

  noul: float


class ScoreAnswer(pydantic.BaseModel):
  """A parsed ordered-rubric score."""

  model_config = pydantic.ConfigDict(frozen=True)

  score: float
  probabilities: dict[str, float]
  confidence: float


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer

_RawAnswer = Annotated[
    _ChoiceAnswer | _NoulAnswer | _ScoreAnswer,
    pydantic.Field(discriminator="type"),
]


class _Usage(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  input_tokens: int = 0
  output_tokens: int = 0


class _Response(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  model: str
  answers: dict[str, _RawAnswer]
  usage: _Usage = _Usage()


Transport = Callable[[str, bytes], bytes]


def _require_text(value: str, what: str) -> str:
  if not value or value.isspace():
    raise SystemOneError(
        FailureKind.CONFIG,
        f"{what} must be non-empty.",
    )
  return value


def _dump_question(question: Question) -> dict[str, object]:  # noqa: DICT_OK
  """Serializes one question. The dict is the HTTP boundary, not a return."""
  match question:
    case ChoiceQuestion(instructions=instructions, criteria=criteria):
      if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
        raise SystemOneError(
            FailureKind.CONFIG,
            f"choice questions need 2-26 options, got {len(criteria)}.",
        )
      for key in criteria:
        if not key or key.isspace() or any(ch.isspace() for ch in key):
          raise SystemOneError(
              FailureKind.CONFIG,
              f"choice key {key!r} must be non-blank and contain no spaces.",
          )
      return {
          "type": "choice",
          "instructions": _require_text(instructions, "instructions"),
          "criteria": criteria,
      }
    case NoulQuestion(instructions=instructions, true=true, false=false):
      return {
          "type": "noul",
          "instructions": _require_text(instructions, "instructions"),
          "criteria": {"true": true, "false": false},
      }
    case ScoreQuestion(instructions=instructions, criteria=criteria):
      if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
        raise SystemOneError(
            FailureKind.CONFIG,
            f"score questions need 2-26 levels, got {len(criteria)}.",
        )
      return {
          "type": "score",
          "instructions": _require_text(instructions, "instructions"),
          "criteria": list(criteria),
      }
    case unreachable:
      assert_never(unreachable)


def _parse_answer(raw: _ChoiceAnswer | _NoulAnswer | _ScoreAnswer) -> Answer:
  match raw:
    case _ChoiceAnswer(
        choice=choice, probabilities=probabilities, confidence=confidence
    ):
      return ChoiceAnswer(
          choice=choice,
          probabilities=probabilities,
          confidence=confidence,
      )
    case _NoulAnswer(noul=noul):
      return NoulAnswer(noul=noul)
    case _ScoreAnswer(
        score=score, probabilities=probabilities, confidence=confidence
    ):
      return ScoreAnswer(
          score=score,
          probabilities=probabilities,
          confidence=confidence,
      )
    case unreachable:
      assert_never(unreachable)


def _urllib_transport(url: str, body: bytes) -> bytes:
  request = urllib.request.Request(
      url,
      data=body,
      headers={"Content-Type": "application/json"},
      method="POST",
  )
  try:
    with urllib.request.urlopen(
        request, timeout=DEFAULT_TIMEOUT_SECONDS
    ) as response:
      return response.read()
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")
    raise SystemOneError(
        FailureKind.HTTP,
        detail or str(exc.reason),
        status=exc.code,
    ) from exc
  except urllib.error.URLError as exc:
    raise SystemOneError(FailureKind.HTTP, str(exc.reason)) from exc


class SystemOneClient:
  """Calls one local System One model."""

  def __init__(
      self,
      model: str,
      base_url: str = "http://127.0.0.1:11434",
      transport: Transport | None = None,
      max_concurrent: int = 1,
      keep_alive: str = "30m",
  ):
    """Initializes the client.

    Args:
      model: Local model name, for example ``clef``.
      base_url: Ollama host, without a path.
      transport: Optional byte transport. Tests inject this. Production uses
        urllib against ``/v1/systemone``.
      max_concurrent: How many requests may be in flight. A local Clef runner
        has one slot; the default is 1.
      keep_alive: How long Ollama should keep the model loaded.
    """
    if max_concurrent < 1:
      raise SystemOneError(
          FailureKind.CONFIG,
          f"max_concurrent must be >= 1, got {max_concurrent}.",
      )
    self.model = _require_text(model, "model")
    self.base_url = base_url.rstrip("/")
    self._transport = transport or _urllib_transport
    self._max_concurrent = max_concurrent
    self._keep_alive = keep_alive
    self._limiter: asyncio.Semaphore | None = None

  @property
  def endpoint(self) -> str:
    return f"{self.base_url}/v1/systemone"

  async def ask(
      self,
      state: str,
      questions: dict[str, Question],
  ) -> dict[str, Answer]:
    """Asks every question about the same state.

    Questions are packed into requests of at most 64, and under 60 KiB, because
    that is what ``/v1/systemone`` accepts. Answers are keyed by the caller's
    question ids.
    """
    _require_text(state, "state")
    if not questions:
      raise SystemOneError(
          FailureKind.CONFIG, "at least one question is required."
      )
    for key in questions:
      if not key or key.isspace() or any(ch.isspace() for ch in key):
        raise SystemOneError(
            FailureKind.CONFIG,
            f"question id {key!r} must be non-blank and contain no spaces.",
        )

    dumped = {
        key: _dump_question(question) for key, question in questions.items()
    }
    answers: dict[str, Answer] = {}
    for pack in _packs(state, self.model, self._keep_alive, dumped):
      body = json.dumps(pack).encode("utf-8")
      if len(body) > MAX_BODY_BYTES:
        raise SystemOneError(
            FailureKind.PAYLOAD,
            f"decision request is {len(body)} bytes; the limit is"
            f" {MAX_BODY_BYTES}.",
        )
      raw = await self._post(body)
      parsed = _parse_response(raw)
      for key, answer in parsed.items():
        answers[key] = answer
    missing = [key for key in questions if key not in answers]
    if missing:
      raise SystemOneError(
          FailureKind.PARSE,
          f"response omitted questions: {missing}",
      )
    return answers

  async def _post(self, body: bytes) -> bytes:
    if self._limiter is None:
      self._limiter = asyncio.Semaphore(self._max_concurrent)
    async with self._limiter:
      return await asyncio.to_thread(self._transport, self.endpoint, body)


def _packs(
    state: str,
    model: str,
    keep_alive: str,
    questions: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
  """Splits questions so each request fits the server limits."""
  items = list(questions.items())
  packs: list[list[tuple[str, dict[str, object]]]] = []
  current: list[tuple[str, dict[str, object]]] = []
  for item in items:
    trial = current + [item]
    if (
        len(trial) > MAX_QUESTIONS
        or _nbytes(state, model, keep_alive, trial) > PACK_BODY_BYTES
    ):
      if not current:
        raise SystemOneError(
            FailureKind.PAYLOAD,
            "a single question does not fit in a System One request.",
        )
      packs.append(current)
      current = [item]
      if _nbytes(state, model, keep_alive, current) > MAX_BODY_BYTES:
        raise SystemOneError(
            FailureKind.PAYLOAD,
            "a single question does not fit in a System One request.",
        )
    else:
      current = trial
  if current:
    packs.append(current)
  return [_body(state, model, keep_alive, pack) for pack in packs]


def _body(
    state: str,
    model: str,
    keep_alive: str,
    pack: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
  return {
      "model": model,
      "state": state,
      "questions": {key: question for key, question in pack},
      "keep_alive": keep_alive,
  }


def _nbytes(
    state: str,
    model: str,
    keep_alive: str,
    pack: list[tuple[str, dict[str, object]]],
) -> int:
  return len(json.dumps(_body(state, model, keep_alive, pack)).encode("utf-8"))


def _parse_response(raw: bytes) -> dict[str, Answer]:
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise SystemOneError(FailureKind.PARSE, "response was not JSON.") from exc
  if not isinstance(payload, dict):
    raise SystemOneError(FailureKind.PARSE, "response was not a JSON object.")
  if "error" in payload and "answers" not in payload:
    message = payload["error"]
    raise SystemOneError(FailureKind.HTTP, str(message))
  try:
    parsed = _Response.model_validate(payload)
  except pydantic.ValidationError as exc:
    raise SystemOneError(
        FailureKind.PARSE, "response did not match System One."
    ) from exc
  return {key: _parse_answer(answer) for key, answer in parsed.answers.items()}
