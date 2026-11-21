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

"""Reusable fake BaseModel for unit tests."""

import collections
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.models import base_model


class FakeBaseModel(base_model.BaseModel):
  """Configurable BaseModel test double with call recording.

  Pass `responses` to replay scripted `generate_content` return values or
  exceptions in FIFO order while recording call metadata in `self.calls`.
  Leave `responses=None` when subclassing and overriding only the methods a
  test exercises; unconfigured abstract methods raise `NotImplementedError`.
  """

  def __init__(
      self,
      responses: Sequence[dict[str, Any] | Exception] | None = None,
      *,
      strict_array_bounds: bool | None = None,
      max_constrained_enum_chars: int | None = None,
  ):
    """Initializes FakeBaseModel.

    Args:
      responses: Optional sequence of response dicts or Exception instances to
        replay from `generate_content`. If None, calling `generate_content`
        raises `NotImplementedError`.
      strict_array_bounds: Forwarded to `base_model.BaseModel.__init__`.
      max_constrained_enum_chars: Forwarded to `base_model.BaseModel.__init__`.
    """
    super().__init__(
        strict_array_bounds=strict_array_bounds,
        max_constrained_enum_chars=max_constrained_enum_chars,
    )
    self._responses: collections.deque[dict[str, Any] | Exception] | None = (
        None if responses is None else collections.deque(responses)
    )
    self.calls: list[dict[str, Any]] = []

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: dict[str, Any] | None = None,
      **kwargs: Any,
  ) -> dict[str, Any]:
    """Records call metadata and replays the next scripted response.

    Args:
      prompt: Prompt text sent to the model.
      run_name: Run name for logging.
      temperature: Sampling temperature.
      system_prompt: Optional system prompt.
      response_mime_type: Optional response MIME type.
      response_schema: Optional structured response schema.
      **kwargs: Ignored extra arguments.

    Returns:
      The next scripted response dictionary.

    Raises:
      NotImplementedError: If `responses` was not configured.
      AssertionError: If scripted responses have been exhausted.
      Exception: Any scripted exception item popped from `responses`.
    """
    del kwargs
    if self._responses is None:
      raise NotImplementedError(
          "FakeBaseModel.generate_content was not configured."
      )
    self.calls.append({
        "prompt": prompt,
        "run_name": run_name,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "response_mime_type": response_mime_type,
        "response_schema": response_schema,
    })
    if not self._responses:
      raise AssertionError("FakeBaseModel received more calls than scripted.")
    next_item = self._responses.popleft()
    if isinstance(next_item, Exception):
      raise next_item
    return next_item

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: Callable[[str, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs: Any,
  ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Raises NotImplementedError unless overridden by a subclass."""
    del prompts, response_parser, max_concurrent_calls, retry_attempts, kwargs
    raise NotImplementedError(
        "FakeBaseModel.process_prompts_concurrently was not configured."
    )
