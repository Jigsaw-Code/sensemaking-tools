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

"""OpenAI API compatible model implementation."""

import asyncio
import logging
import os
from typing import Any, Callable, Tuple
from openai import AsyncOpenAI
from pydantic import BaseModel as PydanticBaseModel
import pandas as pd
from src.models.base_model import BaseModel
import random
import time
import tqdm.asyncio

# Timeout in seconds for API calls. Default Gemini timeout is 10 minutes.
TIMEOUT_SECONDS = 601


DEFAULT_MAX_CONCURRENT_CALLS = 20


class OpenAICompatibleModel(BaseModel):
  """Wrapper for OpenAI API compatible endpoints."""

  def __init__(
      self,
      model_name: str,
      endpoint_url: str,
      api_key: str | None = None,
      max_llm_retries: int | None = None,
      stats_log_file: str | None = None,
      **kwargs,
  ):
    """Initializes the OpenAICompatibleModel.

    Args:
        model_name: The name of the model to use.
        endpoint_url: The base URL for the OpenAI API compatible endpoint.
        api_key: Optional API key.
        max_llm_retries: Optional maximum number of retries for API calls.
        stats_log_file: Optional path to a file where stats will be logged.
    """
    self.model = model_name
    self.endpoint_url = endpoint_url
    self.api_key = (
        api_key or os.getenv("OPENAI_API_ENDPOINT_KEY") or "dummy_key"
    )
    self.max_llm_retries = max_llm_retries
    self.stats_log_file = stats_log_file
    self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.endpoint_url)
    logging.info(
        f"Initialized OpenAICompatibleModel with endpoint: {endpoint_url}"
    )

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: Any | None = None,
      **kwargs,
  ) -> dict[str, Any]:
    """Calls the OpenAI API compatible endpoint with the given prompt."""
    if not prompt:
      raise ValueError("Prompt must be present to call model.")

    messages = []
    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    extra_args = {}
    if response_mime_type == "application/json" or response_schema:
      if response_schema:
        # Handle Pydantic models
        if isinstance(response_schema, type) and issubclass(
            response_schema, PydanticBaseModel
        ):
          schema = response_schema.model_json_schema()
          name = response_schema.__name__
        else:
          schema = response_schema
          name = "custom_schema"

        format_style = os.getenv(
            "OPENAI_RESPONSE_FORMAT_STYLE", "openai_json_schema"
        )
        if format_style == "llama_cpp_old":
          extra_args["response_format"] = {
              "type": "json_object",
              "schema": schema,
          }
        elif format_style == "openai_json_schema":
          extra_args["response_format"] = {
              "type": "json_schema",
              "json_schema": {"name": name, "strict": True, "schema": schema},
          }
        else:
          # Default to just json_object without schema
          extra_args["response_format"] = {"type": "json_object"}
      else:
        extra_args["response_format"] = {"type": "json_object"}

    try:
      response = await asyncio.wait_for(
          self.client.chat.completions.create(
              model=self.model,
              messages=messages,
              temperature=temperature,
              **extra_args,
              **kwargs,
          ),
          timeout=TIMEOUT_SECONDS,
      )

      logging.debug(f"OpenAI response: {response}")
      choice = response.choices[0]
      text = choice.message.content if choice.message.content else ""

      # Extract token counts if available
      usage = response.usage
      total_token_count = usage.total_tokens if usage else 0
      prompt_token_count = usage.prompt_tokens if usage else 0
      candidates_token_count = usage.completion_tokens if usage else 0

      return {
          "text": text,
          "total_token_count": total_token_count,
          "prompt_token_count": prompt_token_count,
          "candidates_token_count": candidates_token_count,
          "tool_use_prompt_token_count": (
              0
          ),  # Not supported directly or differently
          "thoughts_token_count": 0,  # Not supported directly or differently
          "error": None,
      }
    except Exception as e:
      logging.error(f"Error calling OpenAI endpoint: {e}")
      return {"error": str(e)}

  async def _api_worker_with_retry(
      self,
      worker_id: int,
      queue: asyncio.Queue,
      results_list: list,
      stats_list: list,
      stop_event: asyncio.Event,
      response_parser: Callable[[str, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      pbar: Any = None,
  ):
    """Consumes jobs from the queue, calls the OpenAI API with retry logic."""
    max_concurrent_calls = max_concurrent_calls or DEFAULT_MAX_CONCURRENT_CALLS
    initial_jitter = random.uniform(0, 1)
    await asyncio.sleep(initial_jitter)

    while not stop_event.is_set():
      try:
        job = await asyncio.wait_for(queue.get(), timeout=1.0)
      except asyncio.TimeoutError:
        continue

      if job is None:
        break

      prompt = job.get("prompt")
      stats = job.setdefault("stats", {})
      retry_attempts = job.get("retry_attempts", 3)
      temperature = job.get("temperature", 0.0)

      log_prefix = f"[Worker-{worker_id}]"

      stats["api_calls_made"] = 0
      stats["is_success"] = False
      stats["is_complete_failure"] = False

      failed_tries = []
      attempt = 0

      while attempt < retry_attempts:
        if stop_event.is_set():
          break

        try:
          logging.debug(f"{log_prefix} Processing job (Attempt {attempt + 1})")
          stats["api_calls_made"] += 1

          resp = await self.generate_content(
              prompt=prompt,
              run_name=f"job_{job.get('job_id')}",
              temperature=temperature,
              system_prompt=job.get("system_prompt"),
              response_mime_type=job.get("response_mime_type"),
              response_schema=job.get("response_schema"),
          )

          if resp.get("error"):
            raise Exception(resp["error"])

          try:
            result = response_parser(resp, job)
          except Exception as e:
            raise Exception(f"Response parsing failed: {e}")

          # Success
          result_data = {
              "result": result,
              "temperature": temperature,
              "total_token_used": resp.get("total_token_count", 0),
              "prompt_token_count": resp.get("prompt_token_count", 0),
              "candidates_token_count": resp.get("candidates_token_count", 0),
              "failed_tries": pd.DataFrame(failed_tries),
          }
          result_data = {**job, **result_data}
          results_list.append(result_data)

          stats["is_success"] = True
          stats_list.append(stats)
          break

        except Exception as e:
          logging.debug(f"{log_prefix} Error: {e}")
          failed_tries.append({
              "attempt_index": attempt,
              "error_message": str(e),
          })
          attempt += 1
          await asyncio.sleep(1.0)  # Simple sleep for retry

      if not stats["is_success"]:
        stats["is_complete_failure"] = True
        stats_list.append(stats)
        results_list.append({
            **job,
            "result": {"error": f"Failed after {retry_attempts} attempts"},
            "failed_tries": pd.DataFrame(failed_tries),
        })

      if pbar is not None:
        pbar.update(1)
      queue.task_done()

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: Callable[[str, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = 3,
      **kwargs,
  ) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Orchestrates processing of multiple prompts concurrently."""
    max_concurrent_calls = max_concurrent_calls or DEFAULT_MAX_CONCURRENT_CALLS
    stage_start_time = time.time()
    queue = asyncio.Queue()
    final_results = []
    final_stats = []
    stop_event = asyncio.Event()

    pbar = tqdm.asyncio.tqdm(total=len(prompts), desc="Processing prompts")

    workers = [
        asyncio.create_task(
            self._api_worker_with_retry(
                i,
                queue,
                final_results,
                final_stats,
                stop_event,
                response_parser,
                max_concurrent_calls,
                pbar,
            )
        )
        for i in range(max_concurrent_calls)
    ]

    for i, prompt_data in enumerate(prompts):
      job = prompt_data.copy()
      job["job_id"] = i
      job["retry_attempts"] = retry_attempts
      await queue.put(job)

    for _ in range(max_concurrent_calls):
      await queue.put(None)

    await asyncio.gather(*workers)
    pbar.close()

    duration = time.time() - stage_start_time

    llm_response = pd.DataFrame(final_results)
    llm_response_stats = pd.DataFrame(final_stats)

    # Ensure responses are sorted in the same order as the prompts.
    if not llm_response.empty and "job_id" in llm_response.columns:
      llm_response = llm_response.sort_values(by="job_id").reset_index(
          drop=True
      )

    return llm_response, llm_response_stats, 0.0, duration

  def log_stats_summary(
      self,
      final_stats: list[dict],
      stage_name: str,
      wall_delay: float,
      duration: float,
  ):
    """Logs a summary of the processing stats to the stats log file."""
    if not self.stats_log_file or not final_stats:
      return

    import logging

    total_calls = len(final_stats)
    total_api_calls = sum(s.get("api_calls_made", 0) for s in final_stats)
    total_succeeded = sum(1 for s in final_stats if s.get("is_success", False))
    total_failed = total_calls - total_succeeded
    total_max_retries = sum(
        1 for s in final_stats if s.get("is_complete_failure", False)
    )

    logging.info(f"Stats for {stage_name}:")
    logging.info(f"  Total calls: {total_calls}")
    logging.info(f"  Total API calls: {total_api_calls}")
    logging.info(f"  Total succeeded: {total_succeeded}")
    logging.info(f"  Total failed: {total_failed}")
    logging.info(f"  Total max retries reached: {total_max_retries}")

    with open(self.stats_log_file, "a") as f:
      f.write(f"Stats for {stage_name}:\n")
      f.write(f"  Total calls: {total_calls}\n")
      f.write(f"  Total API calls: {total_api_calls}\n")
      f.write(f"  Total succeeded: {total_succeeded}\n")
      f.write(f"  Total failed: {total_failed}\n")
      f.write(f"  Total max retries reached: {total_max_retries}\n")
      f.write(f"  Duration: {duration:.2f}s\n\n")
