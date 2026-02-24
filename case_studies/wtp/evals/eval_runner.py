# Copyright 2025 Google LLC
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

"""Shared utility for running AI evaluations using GenaiModel."""

import asyncio
import logging
import json
import random
from typing import TypedDict, Any
import pandas as pd
from case_studies.wtp.models import genai_model

# The maximum number of times an evaluation call should be retried.
MAX_EVAL_RETRIES = 6
# How long in seconds to wait between evaluation calls.
RETRY_DELAY_SEC = 5
# Maximum number of concurrent evaluation calls.
MAX_CONCURRENT_EVALS = 100


class EvalJob(TypedDict, total=False):
  """A TypedDict for representing an evaluation job."""

  job_id: int
  prompt: str
  retry_attempts: int
  delay_between_calls_seconds: int
  metadata: dict[str, Any]


class EvalRunner:
  """A wrapper to run evaluations concurrently with retries using GenaiModel."""

  def __init__(self, model: genai_model.GenaiModel):
    self.model = model

  async def _eval_worker_with_retry(
      self,
      worker_id: int,
      queue: asyncio.Queue,
      results_list: list,
      stop_event: asyncio.Event,
  ):
    """
    Consumes evaluation jobs from the queue, runs them with retry logic,
    and appends results to a shared list.
    """
    logging.info(f"[EvalWorker-{worker_id}] Started.")
    while not stop_event.is_set():
      try:
        job: EvalJob = await asyncio.wait_for(queue.get(), timeout=1.0)
      except asyncio.TimeoutError:
        continue

      if job is None:
        break

      job_id, prompt, retry_attempts, delay, metadata = (
          job.get("job_id"),
          job.get("prompt"),
          job.get("retry_attempts"),
          job.get("delay_between_calls_seconds"),
          job.get("metadata", {}),
      )
      log_prefix = f"[EvalWorker-{worker_id}] Job-{job_id}"
      failed_tries = []

      for attempt in range(retry_attempts):
        if stop_event.is_set():
          logging.info(f"{log_prefix} Stop event received, terminating.")
          break
        try:
          logging.info(f"{log_prefix} (Attempt {attempt + 1})...")

          # Call GenaiModel
          response = await self.model.call_gemini(
              prompt=prompt,
              run_name=f"eval_job_{job_id}",
              response_mime_type="application/json",
          )

          result_text = response.get("text", "")
          score = 0.0
          explanation = ""
          try:
            # The model is instructed to return JSON.
            json_result = json.loads(result_text)
            score = float(json_result.get("score", 0))
            explanation = json_result.get("explanation", "")
          except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse JSON response: {result_text}. Error: {e}"
            )

          results_list.append({
              "job_id": job_id,
              "score": score,
              "explanation": explanation,
              "failed_tries": pd.DataFrame(failed_tries),
              "metadata": metadata,
          })
          logging.info(f"✅ {log_prefix} Successfully processed.")
          await asyncio.sleep(delay)
          break
        except Exception as e:
          error_msg = (
              f"❌ {log_prefix} Error on attempt {attempt + 1}: {repr(e)}"
          )
          logging.error(error_msg)
          failed_tries.append(
              {"attempt_index": attempt, "error_message": str(e)}
          )
          if attempt < retry_attempts - 1:
            backoff_delay = (2**attempt) + random.uniform(0, 1)
            logging.info(f"   Retrying in {backoff_delay:.2f} seconds...")
            await asyncio.sleep(backoff_delay)
          else:
            logging.error(
                f"Failed to process job {job_id} after {retry_attempts}"
                " attempts. Returning neutral score."
            )
            results_list.append({
                "job_id": job_id,
                "score": 2.0,  # Neutral
                "explanation": "Failed to evaluate.",
                "error": str(e),
                "failed_tries": pd.DataFrame(failed_tries),
                "metadata": metadata,
            })
      queue.task_done()
    logging.info(f"[EvalWorker-{worker_id}] Finished.")

  async def process_evals_concurrently(
      self,
      eval_jobs: list[dict[str, Any]],
      max_concurrent_calls: int = MAX_CONCURRENT_EVALS,
      retry_attempts: int = MAX_EVAL_RETRIES,
      delay_between_calls_seconds: int = RETRY_DELAY_SEC,
  ) -> pd.DataFrame:
    """
    Orchestrates running evaluation tasks concurrently using a queue and workers.
    """
    queue: asyncio.Queue = asyncio.Queue()
    final_results: list[dict] = []
    stop_event = asyncio.Event()

    workers = [
        asyncio.create_task(
            self._eval_worker_with_retry(i, queue, final_results, stop_event)
        )
        for i in range(max_concurrent_calls)
    ]

    for i, job_data in enumerate(eval_jobs):
      if stop_event.is_set():
        logging.info("Stopping eval job submission.")
        break

      job: EvalJob = {
          "job_id": i,
          "prompt": job_data["prompt"],
          "metadata": job_data.get("metadata", {}),
          "retry_attempts": retry_attempts,
          "delay_between_calls_seconds": delay_between_calls_seconds,
      }
      await queue.put(job)

    # Add None to the queue to signal each worker to stop.
    for _ in range(max_concurrent_calls):
      await queue.put(None)

    try:
      await asyncio.gather(*workers)
    except KeyboardInterrupt:
      logging.info("\nKeyboardInterrupt received. Stopping workers...")
      stop_event.set()
      await asyncio.gather(*workers, return_exceptions=True)
      logging.info("Workers stopped.")

    return pd.DataFrame(final_results)


def create_eval_prompt(
    metric_name: str,
    criteria: dict[str, str],
    input_variables: list[str],
    rating_rubric: dict[str, str],
    row_data: dict[str, Any],
) -> str:
  """Creates a prompt for evaluation."""
  criteria_str = json.dumps(criteria, indent=2)
  rubric_str = json.dumps(rating_rubric, indent=2)

  input_context = ""
  for var in input_variables:
    val = row_data.get(var, "N/A")
    input_context += f"{var}: {val}\n"

  # Also try to grab 'response' if it exists in row_data but not in input_variables
  # In run_evals.py, 'response' was explicitly added.
  response = row_data.get("response", "N/A")

  return f"""You are an expert evaluator.

Task: Evaluate the 'response' based on the provided inputs and criteria.

Criteria:
{criteria_str}

Rating Rubric:
{rubric_str}

Input Context:
{input_context}

Response to Evaluate:
{response}

Provide your evaluation in JSON format with two fields:
- "score": A number indicating the score based on the rubric (e.g. 1-4).
- "explanation": A brief explanation of why this score was given.
Do not include markdown code blocks.
"""
