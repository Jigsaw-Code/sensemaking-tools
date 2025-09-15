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

"""
This module provides a wrapper around the Google Generative AI API.
"""

import asyncio
import logging
import random
from typing import Any, Callable, Tuple, Dict, List, Optional, TypedDict
from google import genai
import pandas as pd


class Job(TypedDict, total=False):
  """A TypedDict for representing a job to be processed by the LLM."""

  allocations: Optional[Any]
  delay_between_calls_seconds: int
  initial_retry_delay: int
  job_id: int
  opinion: Optional[str]
  opinion_num: Optional[int]
  prompt: str
  response_mime_type: Optional[str]
  response_schema: Optional[Dict[str, Any]]
  retry_attempts: int
  stats: Optional[Dict[str, Any]]
  system_prompt: Optional[str]
  topic: Optional[str]


# The maximum number of times an LLM call should be retried.
MAX_LLM_RETRIES = 6
# How long in seconds to wait between LLM calls. This is needed due to per
# minute limits Vertex AI imposes.
RETRY_DELAY_SEC = 60
# How long in seconds to wait before first LLM calls.
INITIAL_RETRY_DELAY = 60
# Maximum number of concurrent API calls. By default Genai limits to 10.
MAX_CONCURRENT_CALLS = 10


class GenaiModel:
  """A wrapper around the Google Generative AI API."""

  def __init__(
      self,
      api_key: str,
      model_name: str,
      safety_filters_on: bool = False,
  ):
    """Initializes the GenaiModel.

    Args:
      api_key: The Google Generative AI API key.
      model_name: The name of the model to use.
      safety_filters_on: Whether to enable safety filters. Defaults to False.
    """
    self.client = genai.Client(api_key=api_key)
    self.model = model_name
    self.safety_settings = (
        [
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        if safety_filters_on
        else [
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            genai.types.SafetySetting(
                category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=genai.types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
    )

  async def _api_worker_with_retry(
      self,
      worker_id: int,
      queue: asyncio.Queue,
      results_list: list,
      stats_list: list,
      stop_event: asyncio.Event,
      response_parser: Callable[[str, Dict[str, Any]], Any],
  ):
    """
    Consumes jobs from the queue, calls the Gemini API with retry logic,
    and appends results to shared lists.
    """
    logging.info(f"[Worker-{worker_id}] Started.")
    while not stop_event.is_set():
      try:
        # Use a timeout to periodically check the stop_event
        job: Job = await asyncio.wait_for(queue.get(), timeout=1.0)
      except asyncio.TimeoutError:
        continue  # No job in queue, check stop_event and loop again

      # The 'None' sentinel means the producer is done
      if job is None:
        break

      job_id = job.get("job_id")
      opinion_num = job.get("opinion_num")
      topic = job.get("topic")
      prompt = job.get("prompt")
      opinion = job.get("opinion")
      allocations = job.get("allocations")
      stats = job.get("stats")
      combined_tokens = stats.get("combined_tokens") if stats else None
      retry_attempts = job.get("retry_attempts")
      initial_retry_delay = job.get("initial_retry_delay")
      delay_between_calls_seconds = job.get("delay_between_calls_seconds")
      system_prompt = job.get("system_prompt")
      response_mime_type = job.get("response_mime_type")
      response_schema = job.get("response_schema")

      # Prepare logging prefix
      log_prefix = f"[Worker-{worker_id}]"

      if opinion_num is not None:
        log_prefix = f"[O#{opinion_num} {log_prefix[1:-1]}]"
      if opinion is not None:
        log_prefix += f" Processing opinion '{opinion[:20]}'"
      elif topic is not None:
        log_prefix += f" Processing topic '{topic}'"
      else:
        log_prefix += f" Processing job"
      # This list tracks failures for this job, to be included in the final
      # results for debugging. It is not part of the retry logic itself.
      failed_tries = []
      # The main retry loop. This will continue until the job succeeds or is
      # stopped, at which point the loop will `break`.
      for attempt in range(retry_attempts):
        resp = None  # Initialize resp for this attempt
        if stop_event.is_set():
          logging.info(f"{log_prefix} Stop event received, terminating.")
          break

        try:
          logging.info(f"{log_prefix} (Attempt {attempt + 1})...")

          # Make the actual API call
          resp = await self._call_gemini(
              prompt=prompt,
              run_name=opinion,
              system_prompt=system_prompt,
              response_mime_type=response_mime_type,
              response_schema=response_schema,
          )

          if resp.get("error"):
            error_message = f"API Error: {resp['error']}"
            if resp.get("finish_message"):
              error_message += f" - {resp['finish_message']}"
            if resp.get("token_count"):
              error_message += f" (Tokens: {resp['token_count']})"
            raise Exception(error_message)

          try:
            result = response_parser(resp["text"], job)
          except Exception as e:
            raise Exception(f"Response parsing failed: {e}")

          # --- Success Path ---
          result_data = {
              "result": result,
              "propositions": result,  # For backward compatibility
              "allocations": allocations,
              "token_used": resp["input_token_count"],
              "failed_tries": pd.DataFrame(failed_tries),
          }
          # Merge the original job data into the result
          result_data = {**job, **result_data}
          results_list.append(result_data)

          if stats is not None:
            stats_list.append(stats)

          logging.info(f"✅ {log_prefix} Successfully processed.")

          # Add a delay after a successful call to respect rate limits.
          await asyncio.sleep(delay_between_calls_seconds)

          # Break the retry loop on success
          break

        except Exception as e:
          error_msg = f"❌ {log_prefix} Error on opinion '{opinion[:20]}',"
          f" input_token: {combined_tokens}, attempt"
          f" {attempt + 1}: {repr(e)}"
          if combined_tokens is not None:
            error_msg = f"{error_msg}, input_token: {combined_tokens}"
          logging.error(error_msg)

          failed_tries.append({
              "attempt_index": attempt,
              "error_message": str(e),
              "raw_response": resp.get("text", "") if resp else "",
              "prompt": prompt,
          })

          if attempt < retry_attempts - 1:
            # Exponential backoff with a bit of randomness (jitter)
            delay = (initial_retry_delay**attempt) + random.uniform(0, 1)
            logging.info(f"   Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
          else:
            logging.error(
                f"Failed to process opinion '{opinion[:20]}' after"
                f" {retry_attempts} attempts."
            )

      queue.task_done()

    logging.info(f"[Worker-{worker_id}] Finished.")

  async def process_prompts_concurrently(
      self,
      prompts: List[Dict[str, Any]],
      response_parser: Callable[[str, Dict[str, Any]], Any],
      max_concurrent_calls: int = MAX_CONCURRENT_CALLS,
      retry_attempts: int = MAX_LLM_RETRIES,
      initial_retry_delay: int = INITIAL_RETRY_DELAY,
      delay_between_calls_seconds: int = RETRY_DELAY_SEC,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the process of generating prompts and processing them
    using a queue and concurrent workers.
    """
    # Queue to hold all the jobs
    queue: asyncio.Queue = asyncio.Queue()
    # Lists to aggregate results from all workers
    final_results: List[Dict] = []
    final_stats: List[Dict] = []
    stop_event = asyncio.Event()

    # Create and start the worker tasks
    workers: List[asyncio.Task] = [
        asyncio.create_task(
            self._api_worker_with_retry(
                i,
                queue,
                final_results,
                final_stats,
                stop_event,
                response_parser,
            )
        )
        for i in range(max_concurrent_calls)
    ]

    for i, prompt_data in enumerate(prompts):
      if stop_event.is_set():
        logging.info("Stopping generation process.")
        break

      job: Job = prompt_data.copy()
      job["job_id"] = i  # Add a unique identifier
      job["opinion_num"] = i + 1
      job["retry_attempts"] = retry_attempts
      job["initial_retry_delay"] = initial_retry_delay
      job["delay_between_calls_seconds"] = delay_between_calls_seconds

      await queue.put(job)

    # --- Signal workers to stop once the queue is empty ---
    for _ in range(max_concurrent_calls):
      await queue.put(None)

    # --- Wait for all workers to finish their tasks ---
    try:
      await asyncio.gather(*workers)
    except KeyboardInterrupt:
      logging.info("\nKeyboardInterrupt received. Stopping workers...")
      stop_event.set()
      # Wait for workers to finish gracefully
      await asyncio.gather(*workers, return_exceptions=True)
      logging.info("Workers stopped.")

    # --- Create final DataFrames from the aggregated results ---
    llm_response = pd.DataFrame(final_results)
    llm_response_stats = pd.DataFrame(final_stats)

    self._log_retry_summary(llm_response)

    return llm_response, llm_response_stats

  def _log_retry_summary(self, results_df: pd.DataFrame):
    """Logs a summary of how many retries each job required."""
    if "failed_tries" not in results_df.columns:
      return

    retry_counts = results_df["failed_tries"].apply(
        lambda df: len(df) if isinstance(df, pd.DataFrame) else 0
    )

    if retry_counts.sum() == 0:
      logging.info("All jobs succeeded on the first attempt.")
      return

    summary = retry_counts.value_counts().sort_index()
    logging.info("\n--- Job Retry Summary ---")
    for num_retries, count in summary.items():
      if num_retries == 0:
        logging.info(
            f"Jobs with 0 retries (succeeded on first attempt): {count}"
        )
      else:
        logging.info(f"Jobs with {num_retries} retries: {count}")
    logging.info("-----------------------\n")

  async def _call_gemini(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: Optional[str] = None,
      response_mime_type: Optional[str] = None,
      response_schema: Optional[Dict[str, Any]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Calls the Gemini model with the given prompt.

    Args:
      prompt: The prompt to send to the model.
      run_name: The topic or opinion name for logging purposes.
      temperature: The temperature to use for the model.
      system_prompt: The system prompt to use for the model.
      response_mime_type: The response mime type to use for the model.
      response_schema: The response schema to use for the model.

    Returns:
      A dictionary containing the model's response and token count,
      or None if an error occurred.
    """
    if not prompt:
      raise ValueError("Prompt must be present to call Gemini.")

    try:
      response = await self.client.aio.models.generate_content(
          model=self.model,
          contents=prompt,
          config=genai.types.GenerateContentConfig(
              system_instruction=system_prompt,
              temperature=temperature,
              safety_settings=self.safety_settings,
              response_mime_type=response_mime_type,
              response_schema=response_schema,
              automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                  maximum_remote_calls=MAX_CONCURRENT_CALLS
              ),
          ),
      )
      if not response.candidates:
        logging.error("The response from the API contained no candidates.")
        logging.error("This might be due to a problem with the prompt itself.")
        return {"error": response.prompt_feedback}

      candidate = response.candidates[0]

      if candidate.finish_reason.name != "STOP":
        logging.error(
            "The model stopped generating for a reason: '%s' for: %s",
            candidate.finish_reason.name,
            run_name,
        )
        logging.error(f"Safety Ratings: {candidate.safety_ratings}")
        return {
            "error": candidate.finish_reason.name,
            "finish_message": candidate.finish_message,
            "token_count": candidate.token_count,
        }

      return {
          "text": candidate.content.parts[0].text,
          "input_token_count": response.usage_metadata.total_token_count,
          "error": None,
      }
    except Exception as e:
      logging.error(
          "An unexpected error occurred during content generation: %s", repr(e)
      )
      return {"error": e}

  def calculate_token_count_needed(
      self,
      prompt: str,
      run_name: str = "",
      temperature: float = 0.0,
  ) -> int:
    """Calculates the number of tokens needed for a given prompt.

    Args:
      prompt: The prompt to calculate the token count for.

    Returns:
      The number of tokens needed for the prompt.
    """
    token_count = self.client.models.count_tokens(
        model=self.model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature,
            safety_settings=self.safety_settings,
            automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_CONCURRENT_CALLS
            ),
        ),
    ).total_tokens
    logging.info(
        f"Token count for prompt of the run '{run_name}': {token_count}"
    )
    return token_count
