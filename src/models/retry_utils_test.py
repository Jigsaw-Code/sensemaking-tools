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

"""Tests for src/models/retry_utils.py."""

import asyncio
from collections.abc import Sequence
import json
from typing import Any
import unittest

from src.models import base_model
from src.models import genai_model
from src.models import retry_utils


class _ScriptedModel(base_model.BaseModel):
  """Minimal BaseModel replaying scripted responses or exceptions."""

  def __init__(
      self,
      outcomes: Sequence[dict[str, Any] | Exception],
      *,
      delay_seconds: float = 0.0,
      strict_array_bounds: bool = False,
      max_constrained_enum_chars: int | None = None,
  ):
    super().__init__(
        strict_array_bounds=strict_array_bounds,
        max_constrained_enum_chars=max_constrained_enum_chars,
    )
    self._outcomes = list(outcomes)
    self._delay_seconds = delay_seconds
    self.calls: list[dict[str, Any]] = []
    self._active_calls = 0
    self.peak_concurrent_calls = 0

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
    self._active_calls += 1
    self.peak_concurrent_calls = max(
        self.peak_concurrent_calls, self._active_calls
    )
    try:
      if self._delay_seconds > 0:
        await asyncio.sleep(self._delay_seconds)
      self.calls.append({
          "prompt": prompt,
          "run_name": run_name,
          "temperature": temperature,
          "system_prompt": system_prompt,
          "response_mime_type": response_mime_type,
          "response_schema": response_schema,
          **kwargs,
      })
      outcome = self._outcomes[len(self.calls) - 1]
      if isinstance(outcome, Exception):
        raise outcome
      return outcome
    finally:
      self._active_calls -= 1

  async def process_prompts_concurrently(
      self, prompts, response_parser, **kwargs
  ):
    raise NotImplementedError


class RetryingModelTest(unittest.IsolatedAsyncioTestCase):
  """Tests for the RetryingModel BaseModel decorator."""

  async def test_retries_exception_error_dict_and_empty_text(self):
    underlying = _ScriptedModel([
        RuntimeError("transient socket reset"),
        {"error": "rate limit exceeded"},
        {"text": "   "},
        {"text": "ok"},
    ])
    model = retry_utils.RetryingModel(underlying, max_retries=4)

    response = await model.generate_content(prompt="hi", run_name="unit")

    self.assertEqual(response, {"text": "ok"})
    self.assertEqual(len(underlying.calls), 4)
    self.assertEqual(
        [call["run_name"] for call in underlying.calls],
        ["unit", "unit_try2", "unit_try3", "unit_try4"],
    )

  async def test_returns_final_error_dict_when_exhausted(self):
    underlying = _ScriptedModel([
        {"error": "first"},
        {"error": "second"},
    ])
    model = retry_utils.RetryingModel(underlying, max_retries=2)

    response = await model.generate_content(prompt="hi", run_name="unit")

    self.assertEqual(response, {"error": "second"})
    self.assertEqual(len(underlying.calls), 2)

  async def test_propagates_model_stalled_error_without_retrying(self):
    underlying = _ScriptedModel([
        genai_model.ModelStalledError("stalled"),
        {"text": "unreachable"},
    ])
    model = retry_utils.RetryingModel(underlying, max_retries=3)

    with self.assertRaises(genai_model.ModelStalledError):
      await model.generate_content(prompt="hi", run_name="unit")
    self.assertEqual(len(underlying.calls), 1)

  async def test_preserves_function_call_response_with_empty_text(self):
    function_response = {
        "function_name": "lookup",
        "function_args": {"id": 1},
        "text": "",
        "error": None,
    }
    underlying = _ScriptedModel([function_response])
    model = retry_utils.RetryingModel(underlying, max_retries=2)

    response = await model.generate_content(prompt="hi", run_name="unit")

    self.assertEqual(response, function_response)
    self.assertEqual(len(underlying.calls), 1)

  def test_delegates_schema_constraints_and_validates_config(self):
    underlying = _ScriptedModel(
        [], strict_array_bounds=True, max_constrained_enum_chars=512
    )
    model = retry_utils.RetryingModel(underlying, max_retries=2)
    self.assertIs(model.wrapped_model, underlying)
    self.assertTrue(model.strict_array_bounds())
    self.assertEqual(model.max_constrained_enum_chars(), 512)

    with self.assertRaises(ValueError):
      retry_utils.RetryingModel(underlying, max_retries=0)
    with self.assertRaises(TypeError):
      retry_utils.RetryingModel(underlying, max_retries=True)
    with self.assertRaises(ValueError):
      retry_utils.RetryingModel(underlying, retry_delay_seconds=-0.5)


class GenerateWithRetryTest(unittest.IsolatedAsyncioTestCase):
  """Tests for generate_with_retry helper."""

  async def test_retries_errors_and_parser_value_errors(self):
    underlying = _ScriptedModel([
        {"error": "503"},
        {"text": "not valid json"},
        {"text": '{"ok": true}'},
    ])

    result = await retry_utils.generate_with_retry(
        underlying,
        prompt="return json",
        run_name="parse_test",
        max_retries=3,
        parser=json.loads,
    )

    self.assertEqual(result, {"ok": True})
    self.assertEqual(len(underlying.calls), 3)

  async def test_raises_runtime_error_when_exhausted(self):
    underlying = _ScriptedModel([
        {"text": "bad"},
        {"text": "still bad"},
    ])

    with self.assertRaisesRegex(RuntimeError, "All 2 attempt"):
      await retry_utils.generate_with_retry(
          underlying,
          prompt="return json",
          run_name="parse_fail",
          max_retries=2,
          parser=json.loads,
      )

  async def test_does_not_retry_model_stalled_error(self):
    underlying = _ScriptedModel([
        genai_model.ModelStalledError("outage"),
        {"text": "unreachable"},
    ])

    with self.assertRaises(genai_model.ModelStalledError):
      await retry_utils.generate_with_retry(
          underlying, prompt="hi", run_name="stall", max_retries=3
      )
    self.assertEqual(len(underlying.calls), 1)

  async def test_semaphore_bounds_concurrency(self):
    underlying = _ScriptedModel(
        [{"text": f"ok_{i}"} for i in range(6)],
        delay_seconds=0.02,
    )
    sem = asyncio.Semaphore(2)

    results = await asyncio.gather(*(
        retry_utils.generate_with_retry(
            underlying,
            prompt=f"p{i}",
            run_name=f"run_{i}",
            semaphore=sem,
        )
        for i in range(6)
    ))

    self.assertEqual(len(results), 6)
    self.assertLessEqual(underlying.peak_concurrent_calls, 2)


if __name__ == "__main__":
  unittest.main()
