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

"""Shared retry wrapper and helper for `BaseModel.generate_content` calls.

`GenaiModel.process_prompts_concurrently` includes built-in backoff and retry
handling, whereas single-call `BaseModel.generate_content` implementations
execute one API request and surface errors directly in the returned dict. This
module centralizes bounded retry logic for single-call consumers via:

  - `RetryingModel`: a `BaseModel` decorator that adds bounded retry handling to
    `generate_content` while delegating all other `BaseModel` methods.
  - `generate_with_retry`: an async helper adding optional semaphore throttling
    and response-parser retry (e.g., JSON validation) around `generate_content`.
"""

import asyncio
from collections.abc import Callable
import contextlib
import logging
import math
from typing import Any, TypeVar, overload

import pandas as pd

from src.models import base_model
from src.models import genai_model

DEFAULT_MAX_RETRIES = 3

T = TypeVar("T")


def _validate_retry_config(
    max_retries: int, retry_delay_seconds: float
) -> None:
  """Validates retry count and delay parameters."""
  # Check bool explicitly because bool subclasses int in Python.
  if isinstance(max_retries, bool) or not isinstance(max_retries, int):
    raise TypeError(
        f"max_retries must be an integer, got {type(max_retries).__name__}."
    )
  if max_retries < 1:
    raise ValueError(f"max_retries must be >= 1, got {max_retries}.")
  if isinstance(retry_delay_seconds, bool) or not isinstance(
      retry_delay_seconds, (int, float)
  ):
    raise TypeError(
        "retry_delay_seconds must be a non-negative number, got"
        f" {type(retry_delay_seconds).__name__}."
    )
  if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
    raise ValueError(
        "retry_delay_seconds must be finite and >= 0, got"
        f" {retry_delay_seconds}."
    )


def _response_failure_reason(
    response: Any, *, require_non_empty_text: bool
) -> str | None:
  """Returns an error description if a model response should be retried."""
  if not isinstance(response, dict):
    return f"Unexpected non-dict response: {response!r}"
  if response.get("error"):
    return f"Model error: {response['error']}"
  if (
      require_non_empty_text
      and not response.get("function_name")
      and not str(response.get("text") or "").strip()
  ):
    return "Empty response text."
  return None


class RetryingModel(base_model.BaseModel):
  """`BaseModel` decorator that retries failed `generate_content` calls."""

  def __init__(
      self,
      model: base_model.BaseModel,
      *,
      max_retries: int = DEFAULT_MAX_RETRIES,
      retry_delay_seconds: float = 0.0,
      require_non_empty_text: bool = True,
  ):
    """Initializes the retrying wrapper around an underlying `BaseModel`.

    Args:
      model: Underlying `BaseModel` instance to decorate.
      max_retries: Maximum number of attempts per `generate_content` call.
      retry_delay_seconds: Optional sleep duration between failed attempts.
      require_non_empty_text: When True, responses with blank text (and no
        `function_name`) are treated as retryable failures.
    """
    _validate_retry_config(max_retries, retry_delay_seconds)
    super().__init__(
        strict_array_bounds=model.strict_array_bounds(),
        max_constrained_enum_chars=model.max_constrained_enum_chars(),
    )
    self._model = model
    self._max_retries = max_retries
    self._retry_delay_seconds = float(retry_delay_seconds)
    self._require_non_empty_text = require_non_empty_text

  @property
  def wrapped_model(self) -> base_model.BaseModel:
    """Returns the decorated `BaseModel` instance."""
    return self._model

  def strict_array_bounds(self) -> bool:
    return self._model.strict_array_bounds()

  def max_constrained_enum_chars(self) -> int | None:
    return self._model.max_constrained_enum_chars()

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: Any | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    """Calls the wrapped model's `generate_content` with bounded retries."""
    last_response: dict[str, Any] = {"error": "No response received."}
    for attempt in range(1, self._max_retries + 1):
      attempt_run_name = (
          run_name if attempt == 1 else f"{run_name}_try{attempt}"
      )
      try:
        response = await self._model.generate_content(
            prompt=prompt,
            run_name=attempt_run_name,
            temperature=temperature,
            system_prompt=system_prompt,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            **kwargs,
        )
      except genai_model.ModelStalledError:
        raise
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning(
            "generate_content %s attempt %d/%d raised an error: %s",
            run_name,
            attempt,
            self._max_retries,
            exc,
        )
        last_response = {"error": exc}
      else:
        failure = _response_failure_reason(
            response, require_non_empty_text=self._require_non_empty_text
        )
        if failure is None:
          return response
        logging.warning(
            "generate_content %s attempt %d/%d failed: %s",
            run_name,
            attempt,
            self._max_retries,
            failure,
        )
        last_response = (
            response
            if isinstance(response, dict) and response.get("error")
            else {"error": failure}
        )

      if attempt < self._max_retries and self._retry_delay_seconds > 0:
        await asyncio.sleep(self._retry_delay_seconds)

    return last_response

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: Callable[[str, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs: Any,
  ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    return await self._model.process_prompts_concurrently(
        prompts=prompts,
        response_parser=response_parser,
        max_concurrent_calls=max_concurrent_calls,
        retry_attempts=retry_attempts,
        **kwargs,
    )

  def estimate_token_count_needed(
      self,
      prompt: str,
      run_name: str = "",
      temperature: float = 0.0,
  ) -> int:
    return self._model.estimate_token_count_needed(
        prompt=prompt, run_name=run_name, temperature=temperature
    )

  def log_stats_summary(
      self,
      final_stats: list[dict[str, Any]],
      stage_name: str,
      wall_delay: float,
      duration: float,
  ) -> None:
    self._model.log_stats_summary(
        final_stats=final_stats,
        stage_name=stage_name,
        wall_delay=wall_delay,
        duration=duration,
    )


@overload
async def generate_with_retry(
    model: base_model.BaseModel,
    *,
    prompt: str,
    run_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
    semaphore: asyncio.Semaphore | None = None,
    parser: None = None,
    allow_empty_text: bool = False,
    **kwargs: Any,
) -> str:
  ...


@overload
async def generate_with_retry(
    model: base_model.BaseModel,
    *,
    prompt: str,
    run_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
    semaphore: asyncio.Semaphore | None = None,
    parser: Callable[[str], T],
    allow_empty_text: bool = False,
    **kwargs: Any,
) -> T:
  ...


async def generate_with_retry(
    model: base_model.BaseModel,
    *,
    prompt: str,
    run_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = 0.0,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any | None = None,
    semaphore: asyncio.Semaphore | None = None,
    parser: Callable[[str], T] | None = None,
    allow_empty_text: bool = False,
    **kwargs: Any,
) -> T | str:
  """Calls `model.generate_content` with retry, throttling, and optional parsing.

  Retries on raised exceptions (except `ModelStalledError`), error dicts, blank
  responses (unless `allow_empty_text=True`), and `ValueError` raised by
  `parser`.

  Args:
    model: Initialized `BaseModel` instance.
    prompt: Prompt text sent to the model.
    run_name: Label used in logging and telemetry.
    max_retries: Maximum number of attempts (must be >= 1).
    retry_delay_seconds: Optional sleep duration between failed attempts.
    temperature: Sampling temperature passed to `generate_content`.
    system_prompt: Optional system instruction string.
    response_mime_type: Optional response MIME type (e.g., `"application/json"`).
    response_schema: Optional structured output schema.
    semaphore: Optional semaphore bounding concurrent model calls.
    parser: Optional callable parsing response text; `ValueError` triggers retry.
    allow_empty_text: If True, allows blank text responses without retrying.
    **kwargs: Extra keyword arguments forwarded to `model.generate_content`.

  Returns:
    Parsed output from `parser(text)` when `parser` is provided, otherwise the
    raw response text string.

  Raises:
    ModelStalledError: Propagated immediately without retrying.
    RuntimeError: If all `max_retries` attempts fail.
    TypeError: If `max_retries` or `retry_delay_seconds` has an invalid type.
    ValueError: If `max_retries < 1` or `retry_delay_seconds` is negative/NaN.
  """
  _validate_retry_config(max_retries, retry_delay_seconds)
  last_error: str | Exception = "No attempts executed."

  for attempt in range(1, max_retries + 1):
    attempt_run_name = run_name if attempt == 1 else f"{run_name}_try{attempt}"
    try:
      async with contextlib.AsyncExitStack() as stack:
        if semaphore is not None:
          await stack.enter_async_context(semaphore)
        response = await model.generate_content(
            prompt=prompt,
            run_name=attempt_run_name,
            temperature=temperature,
            system_prompt=system_prompt,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            **kwargs,
        )
    except genai_model.ModelStalledError:
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      last_error = exc
      logging.warning(
          "Call %s attempt %d/%d raised an error: %s",
          run_name,
          attempt,
          max_retries,
          exc,
      )
    else:
      failure = _response_failure_reason(
          response, require_non_empty_text=not allow_empty_text
      )
      if failure is not None:
        last_error = failure
        logging.warning(
            "Call %s attempt %d/%d failed: %s",
            run_name,
            attempt,
            max_retries,
            failure,
        )
      else:
        raw_text = str(response.get("text") or "")
        if parser is None:
          return raw_text
        try:
          return parser(raw_text)
        except ValueError as exc:
          last_error = exc
          logging.warning(
              "Call %s attempt %d/%d failed parsing: %s",
              run_name,
              attempt,
              max_retries,
              exc,
          )

    if attempt < max_retries and retry_delay_seconds > 0:
      await asyncio.sleep(retry_delay_seconds)

  raise RuntimeError(
      f"All {max_retries} attempt(s) failed for {run_name}: {last_error}"
  )
