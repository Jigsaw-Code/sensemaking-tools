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

"""Client for System One decision models.

System One decision models do not implement chat or completion. A request
sends one state and a set of questions; the answers come back as choice, noul,
or score answers. Hosts share this request shape and differ in path and auth:
Ollama serves ``POST /v1/systemone`` locally, and hosted services take a
bearer key at their own endpoint. Tested with Clef on Ollama and Workers AI,
TypeSafe Jev, and Inception Mercury Decide.
"""

from __future__ import annotations

import asyncio  # noqa: ANYIO_OK
import dataclasses
import enum
import functools
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Callable, Final, Literal, Mapping, assert_never

import pydantic

# Questions per request that Ollama accepts. Other hosts may allow more or
# fewer, so the client takes its own limit.
MAX_QUESTIONS: Final = 64
MAX_CHOICE_OPTIONS: Final = 26
MAX_BODY_BYTES: Final = 64 * 1024
# Leave headroom under the server's 64 KiB text limit while packing.
PACK_BODY_BYTES: Final = 60 * 1024
DEFAULT_TIMEOUT_SECONDS: Final = 600
# Attempts per request when the host fails transiently (429, 5xx, no answer).
MAX_ATTEMPTS: Final = 3
# Ollama's port, assumed for a bare host as the Ollama CLI does.
OLLAMA_DEFAULT_PORT: Final = 11434
SYSTEMONE_PATH: Final = "/v1/systemone"
# A local Ollama server, used when no host is configured.
DEFAULT_ENDPOINT: Final = (
    f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}{SYSTEMONE_PATH}"
)


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
    """Returns the message, prefixed with the HTTP status when known."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class ExtractQuestion:
  """Marks exact spans of the state that match a described field.

  The field is named after the question id. Not every host supports
  extraction; Inception Mercury Decide does.
  """

  instructions: str


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion | ExtractQuestion


class _ChoiceAnswer(pydantic.BaseModel):
  """Wire shape of a choice answer."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["choice"]
  choice: str
  probabilities: dict[str, float]
  confidence: float


class _NoulAnswer(pydantic.BaseModel):
  """Wire shape of a noul answer."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["noul"]
  noul: float


class _ScoreAnswer(pydantic.BaseModel):
  """Wire shape of a score answer."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["score"]
  score: float
  probabilities: dict[str, float]
  confidence: float


class _Span(pydantic.BaseModel):
  """Wire shape of one extracted span."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  text: str
  start: int
  end: int
  confidence: float


class _ExtractAnswer(pydantic.BaseModel):
  """Wire shape of an extract answer."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  type: Literal["extract"]
  spans: dict[str, list[_Span]]
  not_stated: dict[str, float] = {}


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


class Span(pydantic.BaseModel):
  """An exact substring of the state: ``state[start:end] == text``."""

  model_config = pydantic.ConfigDict(frozen=True)

  text: str
  start: int
  end: int
  confidence: float


class ExtractAnswer(pydantic.BaseModel):
  """Spans marked for one extract question, in state order."""

  model_config = pydantic.ConfigDict(frozen=True)

  spans: tuple[Span, ...]
  not_stated: float | None


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer | ExtractAnswer

_RawAnswer = Annotated[
    _ChoiceAnswer | _NoulAnswer | _ScoreAnswer | _ExtractAnswer,
    pydantic.Field(discriminator="type"),
]


class _Usage(pydantic.BaseModel):
  """Wire shape of token usage."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  input_tokens: int = 0
  output_tokens: int = 0


class _Response(pydantic.BaseModel):
  """Wire shape of a System One response."""
  model_config = pydantic.ConfigDict(frozen=True, extra="ignore")

  model: str
  answers: dict[str, _RawAnswer]
  usage: _Usage = _Usage()


Transport = Callable[[str, bytes], bytes]


def _require_text(value: str, what: str) -> str:
  """Returns value, or raises a config error when it is blank."""
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
    case ExtractQuestion(instructions=instructions):
      return {
          "type": "extract",
          "instructions": _require_text(instructions, "instructions"),
      }
    case unreachable:
      assert_never(unreachable)


def _parse_answer(
    raw: _ChoiceAnswer | _NoulAnswer | _ScoreAnswer | _ExtractAnswer,
    key: str,
) -> Answer:
  """Converts a wire answer into its public answer type.

  An extract answer's field is named after its question id; a single field
  under another name is accepted too.
  """
  match raw:
    case _ExtractAnswer(spans=spans, not_stated=not_stated):
      field = key if key in spans or len(spans) != 1 else next(iter(spans))
      return ExtractAnswer(
          spans=tuple(
              Span(
                  text=span.text,
                  start=span.start,
                  end=span.end,
                  confidence=span.confidence,
              )
              for span in sorted(spans.get(field, []), key=lambda s: s.start)
          ),
          not_stated=not_stated.get(field),
      )
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


def _urllib_transport(
    url: str, body: bytes, api_key: str | None = None
) -> bytes:
  """Posts one JSON body and returns the raw response.

  Args:
    url: Endpoint URL.
    body: Encoded JSON request.
    api_key: Optional bearer key.

  Returns:
    The response body.

  Raises:
    SystemOneError: On an HTTP error status, or with no status when the
      connection fails or drops before a response arrives.
  """
  headers = {"Content-Type": "application/json"}
  if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
  request = urllib.request.Request(
      url,
      data=body,
      headers=headers,
      method="POST",
  )
  try:
    with urllib.request.urlopen(
        request, timeout=DEFAULT_TIMEOUT_SECONDS
    ) as response:
      return response.read()
  except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")
    if exc.code == 404:
      detail = (
          f"{url} does not serve System One decisions ({detail.strip()})."
          " Ollama serves them from version 0.35.1; for another host, set"
          " DECISION_MODEL_HOST to its full endpoint URL."
      )
    raise SystemOneError(
        FailureKind.HTTP,
        detail or str(exc.reason),
        status=exc.code,
    ) from exc
  except urllib.error.URLError as exc:
    raise SystemOneError(FailureKind.HTTP, str(exc.reason)) from exc
  except (http.client.HTTPException, ConnectionError) as exc:
    # urlopen lets a connection that drops before or during the response
    # escape unwrapped (for example http.client.RemoteDisconnected).
    raise SystemOneError(
        FailureKind.HTTP, f"connection to {url} failed: {exc!r}"
    ) from exc


def endpoint_url(host: str) -> str:
  """Turns a decision model host into the URL that requests are posted to.

  A bare ``host`` or ``host:port`` is read as the Ollama CLI reads
  OLLAMA_HOST: http, and port 11434 when none is given. With an explicit
  scheme the scheme's own default port applies. A URL without a path gets
  ``/v1/systemone``; a URL with a path is used as given, for hosts that serve
  decisions elsewhere.

  Args:
    host: A host, ``host:port``, base URL, or full endpoint URL.

  Returns:
    The endpoint URL, without a trailing slash.

  Raises:
    SystemOneError: If host is blank or not a valid host.
  """
  host = _require_text(host, "host").strip()
  if "://" in host:
    url = host
  else:
    url = f"http://{host}"
  try:
    parts = urllib.parse.urlsplit(url)
    port = parts.port
  except ValueError as exc:
    raise SystemOneError(
        FailureKind.CONFIG, f"host {host!r} is not a valid host."
    ) from exc
  if not parts.hostname:
    raise SystemOneError(
        FailureKind.CONFIG, f"host {host!r} is not a valid host."
    )
  if "://" not in host and port is None:
    parts = parts._replace(netloc=f"{parts.netloc}:{OLLAMA_DEFAULT_PORT}")
  if parts.path in ("", "/"):
    parts = parts._replace(path=SYSTEMONE_PATH)
  return parts.geturl().rstrip("/")


class SystemOneClient:
  """Calls one System One decision model."""

  def __init__(
      self,
      model: str,
      endpoint: str = DEFAULT_ENDPOINT,
      transport: Transport | None = None,
      max_concurrent: int = 1,
      keep_alive: str | None = None,
      api_key: str | None = None,
      max_questions: int = MAX_QUESTIONS,
      retry_delay_seconds: float = 1.0,
  ):
    """Initializes the client.

    Args:
      model: Model name, for example ``clef``.
      endpoint: URL that requests are posted to; see endpoint_url().
      transport: Optional byte transport. Tests inject this. Production uses
        urllib.
      max_concurrent: How many requests may be in flight. A local runner has
        one slot; the default is 1.
      keep_alive: How long Ollama should keep the model loaded. Only Ollama
        accepts this field, so it is sent only when set.
      api_key: Optional bearer key for hosted endpoints.
      max_questions: Most questions the host accepts in one request.
      retry_delay_seconds: Backoff unit between attempts after a transient
        failure; the nth retry waits n units.
    """
    if max_concurrent < 1:
      raise SystemOneError(
          FailureKind.CONFIG,
          f"max_concurrent must be >= 1, got {max_concurrent}.",
      )
    if max_questions < 1:
      raise SystemOneError(
          FailureKind.CONFIG,
          f"max_questions must be >= 1, got {max_questions}.",
      )
    self.model = _require_text(model, "model")
    self._endpoint = _require_text(endpoint, "endpoint").rstrip("/")
    self._transport = transport or make_transport(api_key)
    self._max_concurrent = max_concurrent
    self._keep_alive = keep_alive
    self._max_questions = max_questions
    self._retry_delay_seconds = retry_delay_seconds
    self._limiter: asyncio.Semaphore | None = None

  @property
  def endpoint(self) -> str:
    """The URL that requests are posted to."""
    return self._endpoint

  async def ask(
      self,
      state: str,
      questions: Mapping[str, Question],
  ) -> dict[str, Answer]:
    """Asks every question about the same state.

    Questions are packed into requests of at most max_questions, and under
    60 KiB. Answers are keyed by the caller's question ids.
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
    for pack in _packs(
        state, self.model, self._keep_alive, dumped, self._max_questions
    ):
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
        if isinstance(answer, ExtractAnswer):
          _check_spans(state, key, answer)
        answers[key] = answer
    missing = [key for key in questions if key not in answers]
    if missing:
      raise SystemOneError(
          FailureKind.PARSE,
          f"response omitted questions: {missing}",
      )
    return answers

  async def _post(self, body: bytes) -> bytes:
    """Posts one request, retrying transient failures.

    A rate limit, a server error, or a dropped connection is retried with
    backoff; any other error is raised at once.
    """
    if self._limiter is None:
      self._limiter = asyncio.Semaphore(self._max_concurrent)
    return await post_with_retry(
        self._transport,
        self.endpoint,
        body,
        self._limiter,
        self._retry_delay_seconds,
    )


def make_transport(api_key: str | None = None) -> Transport:
  """Returns the production urllib transport, with an optional bearer key.

  Args:
    api_key: Optional bearer key sent with every request.

  Returns:
    A transport that posts JSON and raises SystemOneError on failure.
  """
  return functools.partial(_urllib_transport, api_key=api_key)


async def post_with_retry(
    transport: Transport,
    url: str,
    body: bytes,
    limiter: asyncio.Semaphore,
    retry_delay_seconds: float,
) -> bytes:
  """Posts one request through transport, retrying transient failures.

  A rate limit, a server error, or a dropped connection is retried with
  backoff, up to MAX_ATTEMPTS attempts; any other error is raised at once.

  Args:
    transport: Byte transport to call.
    url: Endpoint URL.
    body: Encoded request.
    limiter: Bounds how many requests are in flight.
    retry_delay_seconds: Backoff unit; the nth retry waits n units.

  Returns:
    The response body.

  Raises:
    SystemOneError: If the last attempt fails, or on a non-transient error.
  """
  for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
      async with limiter:
        return await asyncio.to_thread(transport, url, body)
    except SystemOneError as exc:
      transient = exc.kind is FailureKind.HTTP and (
          exc.status is None or exc.status == 429 or exc.status >= 500
      )
      if not transient or attempt == MAX_ATTEMPTS:
        raise
      await asyncio.sleep(retry_delay_seconds * attempt)
  raise AssertionError("unreachable")


def _packs(
    state: str,
    model: str,
    keep_alive: str | None,
    questions: dict[str, dict[str, object]],
    max_questions: int,
) -> list[dict[str, object]]:
  """Splits questions so each request fits the server limits."""
  items = list(questions.items())
  packs: list[list[tuple[str, dict[str, object]]]] = []
  current: list[tuple[str, dict[str, object]]] = []
  for item in items:
    trial = current + [item]
    if (
        len(trial) > max_questions
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
    keep_alive: str | None,
    pack: list[tuple[str, dict[str, object]]],
) -> dict[str, object]:
  """Builds one request body; keep_alive is included only when set."""
  body: dict[str, object] = {
      "model": model,
      "state": state,
      "questions": {key: question for key, question in pack},
  }
  if keep_alive is not None:
    body["keep_alive"] = keep_alive
  return body


def _nbytes(
    state: str,
    model: str,
    keep_alive: str | None,
    pack: list[tuple[str, dict[str, object]]],
) -> int:
  """Returns the encoded size of the request body for a pack."""
  return len(json.dumps(_body(state, model, keep_alive, pack)).encode("utf-8"))


def _parse_response(raw: bytes) -> dict[str, Answer]:
  """Parses a response body into answers keyed by question id."""
  try:
    payload = json.loads(raw)
  except json.JSONDecodeError as exc:
    raise SystemOneError(FailureKind.PARSE, "response was not JSON.") from exc
  if not isinstance(payload, dict):
    raise SystemOneError(FailureKind.PARSE, "response was not a JSON object.")
  # Workers AI wraps the answers in a result envelope.
  if "answers" not in payload and isinstance(payload.get("result"), dict):
    payload = payload["result"]
  if "error" in payload and "answers" not in payload:
    message = payload["error"]
    raise SystemOneError(FailureKind.HTTP, str(message))
  try:
    parsed = _Response.model_validate(payload)
  except pydantic.ValidationError as exc:
    raise SystemOneError(
        FailureKind.PARSE, "response did not match System One."
    ) from exc
  return {
      key: _parse_answer(answer, key) for key, answer in parsed.answers.items()
  }


def _check_spans(state: str, key: str, answer: ExtractAnswer) -> None:
  """Raises unless every span is the exact text at its offsets in state.

  Offsets count Unicode code points, as Python string indexing does.
  """
  for span in answer.spans:
    if not 0 <= span.start <= span.end <= len(state) or (
        state[span.start : span.end] != span.text
    ):
      raise SystemOneError(
          FailureKind.PARSE,
          f"question {key}: span {span.start}-{span.end} {span.text!r} is"
          " not the text at those offsets.",
      )
