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

"""Abstract base class for model implementations."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
import pandas as pd


class BaseModel(ABC):
  """Abstract base class for LLM models."""

  @abstractmethod
  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: dict[str, Any] | None = None,
      **kwargs
  ) -> dict[str, Any]:
    """Calls the model with the given prompt.

    Args:
        prompt: The prompt to send to the model.
        run_name: The name of the run for logging purposes.
        temperature: The temperature to use.
        system_prompt: Optional system prompt.
        response_mime_type: Optional response mime type (e.g., 'application/json').
        response_schema: Optional response schema (e.g., Pydantic model or dict).
        **kwargs: Additional model-specific arguments.

    Returns:
        A dictionary containing the response and metadata.
        Expected keys: 'text', 'error', and token counts if available.
    """
    pass

  @abstractmethod
  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: Callable[[str, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs
  ) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Orchestrates processing of multiple prompts concurrently.

    Args:
        prompts: A list of dicts, each containing at least a 'prompt' key.
        response_parser: A callable to parse the response for each job.
        max_concurrent_calls: Maximum number of concurrent calls.
        retry_attempts: Maximum number of retries per job.
        **kwargs: Additional model-specific arguments.

    Returns:
        A tuple containing:
        - llm_response: DataFrame with results.
        - llm_response_stats: DataFrame with statistics.
        - wall_delay: Total wall-clock delay.
        - duration: Total wall-clock duration.
    """
    pass

  def log_stats_summary(
      self,
      final_stats: list[dict],
      stage_name: str,
      wall_delay: float,
      duration: float,
  ):
    """Logs a summary of the processing stats. Default implementation does nothing."""
    pass
