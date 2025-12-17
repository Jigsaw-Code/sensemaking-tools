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
import os
import time
from typing import Any, Callable, Tuple, Dict, List, Optional, TypedDict
from google import genai
from google.api_core import exceptions as google_api_core_exceptions
from google.genai import errors as google_genai_errors
from google.protobuf import duration_pb2, json_format
from google.api_core import exceptions as google_exceptions
import pandas as pd


class GenaiModelError(Exception):
  """Base exception for errors in the GenaiModel."""

  pass


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
  stats: Dict[str, Any]
  system_prompt: Optional[str]
  topic: Optional[str]
  thinking_budget: Optional[int]
  temperature: Optional[float]


# The maximum number of times an LLM call should be retried.
MAX_LLM_RETRIES = 4
# How long in seconds to wait between LLM calls. This is needed due to per
# minute limits Vertex AI imposes.
RETRY_DELAY_SEC = 1
# How long in seconds to wait before first LLM calls.
INITIAL_RETRY_DELAY = 0.5
# Maximum number of concurrent API calls.
MAX_CONCURRENT_CALLS = 100

COMPLETED_BATCH_JOB_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
})


class GenaiModel:
  """A wrapper around the Google Generative AI API."""

  def __init__(
      self,
      model_name: str,
      api_key: str | None = None,
      safety_filters_on: bool = False,
  ):
    """Initializes the GenaiModel.

    Args:
      model_name: The name of the model to use.
      api_key: The Google Generative AI API key. If not provided, the
        GOOGLE_API_KEY environment variable will be used.
      safety_filters_on: Whether to enable safety filters. Defaults to False.
    """
    if not api_key:
      api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
      raise ValueError(
          "Google API key not provided and GOOGLE_API_KEY environment variable"
          " is not set."
      )

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
    # Event to signal a global pause for all workers, e.g., for quota limits.
    # It's set by default, meaning workers can proceed.
    self._quota_available_event = asyncio.Event()
    self._quota_available_event.set()
    # Lock to ensure only one worker handles a quota error at a time.
    self._quota_availability_lock = asyncio.Lock()

    # --- Service Availability Backoff ---
    # Event to signal a global pause for all workers for service availability.
    # It's set by default, meaning workers can proceed.
    self._system_available_event = asyncio.Event()
    self._system_available_event.set()
    # Lock to ensure only one worker handles an availability error at a time.
    self._system_availability_lock = asyncio.Lock()
    # Constants for the exponential backoff delay
    self._initial_backoff_delay = 2
    self._max_backoff_delay = 64
    self._backoff_delay = self._initial_backoff_delay

  def _parse_duration(self, duration_str: str) -> int:
    """Parses a duration string (e.g., '18s') into seconds."""
    duration_proto = duration_pb2.Duration()
    json_format.Parse(f'"{duration_str}"', duration_proto)
    return duration_proto.seconds

  async def _handle_quota_pause(self, delay: int):
    """Sleeps for a specified duration and then resumes all workers."""
    logging.info(f"   Global pause for {delay} seconds...")
    await asyncio.sleep(delay)
    logging.info("   Resuming all workers.")
    self._quota_available_event.set()

  async def _handle_availability_pause(self, delay: int):
    """
    Pauses all workers for the current backoff duration and then resumes them.
    """
    logging.info(f"   Global availability pause for {delay} seconds...")
    await asyncio.sleep(delay)
    logging.info("   Resuming all workers after availability pause.")
    self._system_available_event.set()

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
      stats = job["stats"]
      combined_tokens = stats.get("combined_tokens")
      retry_attempts = job.get("retry_attempts")
      initial_retry_delay = job.get("initial_retry_delay")
      delay_between_calls_seconds = job.get("delay_between_calls_seconds")
      system_prompt = job.get("system_prompt")
      response_mime_type = job.get("response_mime_type")
      response_schema = job.get("response_schema")
      thinking_budget = job.get("thinking_budget")
      temperature = job.get("temperature", 0.0)

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

      # Initialize failure tracking stats
      stats["non_quota_failures"] = 0
      stats["is_complete_failure"] = False

      # This list tracks failures for this job, to be included in the final
      # results for debugging. It is not part of the retry logic itself.
      failed_tries = []
      # The main retry loop. This will continue until the job succeeds or is
      # stopped, at which point the loop will `break`.
      attempt = 0
      while attempt < retry_attempts:
        # Wait here if a global pause is in effect for quota.
        await self._quota_available_event.wait()
        # Wait here if a global pause is in effect for availability.
        await self._system_available_event.wait()

        resp = None  # Initialize resp for this attempt
        if stop_event.is_set():
          logging.info(f"{log_prefix} Stop event received, terminating.")
          break

        API_ERROR = "API Error"

        try:
          logging.info(f"{log_prefix} (Attempt {attempt + 1})...")

          # Make the actual API call
          resp = await self._call_gemini(
              prompt=prompt,
              run_name=opinion,
              system_prompt=system_prompt,
              response_mime_type=response_mime_type,
              response_schema=response_schema,
              thinking_budget=thinking_budget,
              temperature=temperature,
          )

          if resp.get("error"):
            # Raise the error to be handled by the common exception block
            error = resp["error"]
            if isinstance(error, BaseException):
              raise error
            raise GenaiModelError(error)

          try:
            job["current_attempt"] = attempt
            result = response_parser(resp, job)
          except Exception as e:
            raise Exception(f"Response parsing failed: {e}")

          # --- Success Path ---
          result_data = {
              "result": result,
              "propositions": result,  # For backward compatibility
              "allocations": allocations,
              "total_token_used": resp["total_token_count"],
              "prompt_token_count": resp["prompt_token_count"],
              "candidates_token_count": resp["candidates_token_count"],
              "tool_use_prompt_token_count": resp[
                  "tool_use_prompt_token_count"
              ],
              "thoughts_token_count": resp["thoughts_token_count"],
              "failed_tries": pd.DataFrame(failed_tries),
          }
          # Merge the original job data into the result
          result_data = {**job, **result_data}
          results_list.append(result_data)

          stats["total_token_used"] = resp["total_token_count"]
          stats["prompt_token_count"] = resp["prompt_token_count"]
          stats["candidates_token_count"] = resp["candidates_token_count"]
          stats_list.append(stats)

          # On success, reset the availability backoff delay
          self._backoff_delay = self._initial_backoff_delay

          logging.info(f"✅ {log_prefix} Successfully processed.")

          # Add a delay after a successful call to respect rate limits.
          await asyncio.sleep(delay_between_calls_seconds)

          # Break the retry loop on success
          break

        # Universal Error Handling
        except Exception as e:

          # --- Quota Error Handling ---
          if (
              isinstance(e, google_genai_errors.ClientError)
              and hasattr(e, "response")
              and e.response is not None
              and e.response.status == 429
          ):
            async with self._quota_availability_lock:
              if self._quota_available_event.is_set():
                logging.warning(
                    f"{log_prefix} Quota limit hit. Initiating global pause."
                )
                self._quota_available_event.clear()
                logging.info(
                    f"{log_prefix} I am the leader. Pausing all workers."
                )

                # Extract the dictionary from the error message
                error_details = await e.response.json()
                # Find the retryDelay in the details
                for detail in error_details.get("error", {}).get("details", []):
                  if (
                      detail.get("@type")
                      == "type.googleapis.com/google.rpc.RetryInfo"
                  ):
                    retry_delay_str = detail.get("retryDelay", "60s")
                    delay = self._parse_duration(retry_delay_str) + 1
                    break

                asyncio.create_task(self._handle_quota_pause(delay))

            # Do NOT increment attempt counter for quota errors, just restart the loop
            continue

          # --- Availability Error Handling ---
          if isinstance(e, google_exceptions.ServiceUnavailable) or (
              isinstance(e, google_genai_errors.ClientError)
              and hasattr(e, "response")
              and e.response is not None
              and e.response.status == 503
          ):
            async with self._system_availability_lock:
              if self._system_available_event.is_set():
                logging.warning(
                    f"{log_prefix} Service unavailable. Initiating global"
                    " backoff."
                )
                self._system_available_event.clear()
                delay = self._backoff_delay
                asyncio.create_task(self._handle_availability_pause(delay))
                # Increase the backoff for the next potential failure
                self._backoff_delay = min(
                    self._backoff_delay**2, self._max_backoff_delay
                )
            # Do not increment the attempt counter for availability errors
            continue

          # --- Generic Error Handling ---
          error_parts = [f"❌ {log_prefix}"]
          if opinion:
            error_parts.append(f"Error on opinion '{opinion[:20]}'")
          elif topic:
            error_parts.append(f"Error on topic '{topic}'")

          if combined_tokens is not None:
            error_parts.append(f"input_token: {combined_tokens}")

          error_parts.append(f"attempt {attempt + 1}: {repr(e)}")
          error_msg = ", ".join(error_parts)
          logging.error(error_msg)

          # Increment the non-quota failure count in the stats object
          stats["non_quota_failures"] += 1
          if resp and "total_token_count" in resp:
            stats["total_token_used"] = resp.get("total_token_count")
            stats["prompt_token_count"] = resp.get("prompt_token_count")
            stats["candidates_token_count"] = resp.get("candidates_token_count")

          failed_tries.append({
              "attempt_index": attempt,
              "error_message": str(e),
              "raw_response": resp.get("text", "") if resp else "",
              "prompt": prompt,
          })

          attempt += 1
          temperature += 0.02
          if attempt < retry_attempts:
            # Initialize delay to < 1s, with randomness (jitter)
            delay = random.uniform(0, 1)
            logging.info(f"   Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
          else:
            log_identifier = ""
            if opinion:
              log_identifier = f"opinion '{opinion[:20]}'"
            elif topic:
              log_identifier = f"topic '{topic}'"
            else:
              log_identifier = f"job {job_id}"
            logging.error(
                f"Failed to process {log_identifier} after"
                f" {retry_attempts} attempts."
            )
            # Mark this job as a complete failure in the stats
            stats["is_complete_failure"] = True

      # Always append the stats object to the list, regardless of success.
      stats_list.append(stats)
      queue.task_done()

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

    Returns:
        A tuple containing:
        - llm_response: A DataFrame with the successful results.
        - llm_response_stats: A DataFrame with statistics for each job that
          produced a stats object. Note: This may not include entries for jobs
          that fail in a way that prevents a stats object from being returned.
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

      # Ensure a stats object exists for every job
      if "stats" not in job or job["stats"] is None:
        job["stats"] = {}

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
      thinking_budget: Optional[int] = None,
  ) -> Optional[Dict[str, Any]]:
    """Calls the Gemini model with the given prompt.

    Args:
      prompt: The prompt to send to the model.
      run_name: The topic or opinion name for logging purposes.
      temperature: The temperature to use for the model.
      system_prompt: The system prompt to use for the model.
      response_mime_type: The response mime type to use for the model.
      response_schema: The response schema to use for the model.
      thinking_budget: The token budget for the model's thinking process.

    Returns:
      A dictionary containing the model's response and token count,
      or None if an error occurred.
    """
    if not prompt:
      raise ValueError("Prompt must be present to call Gemini.")

    thinking_config = (
        genai.types.ThinkingConfig(thinking_budget=thinking_budget)
        if thinking_budget is not None
        else None
    )

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
              thinking_config=thinking_config,
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

      if (
          candidate.content.parts
          and hasattr(candidate.content.parts[0], "function_call")
          and candidate.content.parts[0].function_call
          and candidate.content.parts[0].function_call.name
      ):
        function_call = candidate.content.parts[0].function_call
        return {
            "function_name": function_call.name,
            "function_args": json_format.MessageToDict(function_call.args),
            "text": "",  # Ensure text field exists to avoid key errors
            "total_token_count": response.usage_metadata.total_token_count,
            "prompt_token_count": response.usage_metadata.prompt_token_count,
            "candidates_token_count": (
                response.usage_metadata.candidates_token_count
            ),
            "tool_use_prompt_token_count": (
                response.usage_metadata.tool_use_prompt_token_count
            ),
            "thoughts_token_count": (
                response.usage_metadata.thoughts_token_count
            ),
            "error": None,
        }

      if candidate.finish_reason and candidate.finish_reason.name != "STOP":
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
          "text": (
              candidate.content.parts[0].text if candidate.content.parts else ""
          ),
          "total_token_count": response.usage_metadata.total_token_count,
          "prompt_token_count": response.usage_metadata.prompt_token_count,
          "candidates_token_count": (
              response.usage_metadata.candidates_token_count
          ),
          "tool_use_prompt_token_count": (
              response.usage_metadata.tool_use_prompt_token_count
          ),
          "thoughts_token_count": response.usage_metadata.thoughts_token_count,
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

  def _parse_batch_responses(
      self, batch_job: Any, num_expected_prompts: int
  ) -> List[Optional[Dict[str, Any]]]:
    """Parses the inlined responses from a completed batch job."""
    results = []
    if batch_job.dest and batch_job.dest.inlined_responses:
      for inline_response in batch_job.dest.inlined_responses:
        if inline_response.response and hasattr(
            inline_response.response, "text"
        ):
          results.append({"text": inline_response.response.text, "error": None})
        elif inline_response.error:
          results.append({"error": str(inline_response.error)})
        else:
          results.append({"error": "Unknown response format"})
    else:
      return [
          {"error": "No inline results found."}
          for _ in range(num_expected_prompts)
      ]

    if len(results) != num_expected_prompts:
      logging.warning("Mismatch between number of prompts and results.")

    return results

  async def start_prompts_batch(self, prompts: List[str]) -> str:
    """Starts a batch job and returns the job name."""
    if not prompts:
      return ""

    inline_requests = [
        {"contents": [{"parts": [{"text": p}], "role": "user"}]}
        for p in prompts
    ]

    loop = asyncio.get_running_loop()
    model_for_batch = f"models/{self.model}"

    inline_batch_job = await loop.run_in_executor(
        None,
        lambda: self.client.batches.create(
            model=model_for_batch,
            src=inline_requests,
        ),
    )
    logging.info(f"Created batch job: {inline_batch_job.name}")
    return inline_batch_job.name

  async def get_batch_job(self, job_name: str):
    """Gets a batch job by name."""
    try:
      loop = asyncio.get_running_loop()
      return await loop.run_in_executor(
          None, lambda: self.client.batches.get(name=job_name)
      )
    except google_api_core_exceptions.NotFound:
      return None

  async def poll_batch_job(
      self,
      job_name: str,
      num_prompts: int,
      polling_interval_seconds: int = 30,
  ) -> List[Optional[Dict[str, Any]]]:
    """Polls a batch job until it is complete and returns the results."""

    start_time = time.time()

    while True:
      batch_job = await self.get_batch_job(job_name)

      if not batch_job:
        logging.error(
            f"Batch job {job_name} not found or disappeared during polling."
        )
        return [
            {"error": "Job not found or disappeared"}
            for _ in range(num_prompts)
        ]

      if batch_job.state.name in COMPLETED_BATCH_JOB_STATES:
        break

      logging.info(
          f"Polling for job {job_name}. Current state: {batch_job.state.name}"
      )
      await asyncio.sleep(polling_interval_seconds)

    end_time = time.time()
    duration = end_time - start_time
    logging.info(
        f"Batch job {job_name} finished polling in {duration:.2f} seconds."
    )
    logging.info(f"Job {job_name} finished with state: {batch_job.state.name}")

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
      error_message = f"Batch job failed with state {batch_job.state.name}"
      if batch_job.error:
        error_message += f": {batch_job.error}"
      return [{"error": error_message} for _ in range(num_prompts)]

    return self._parse_batch_responses(batch_job, num_prompts)
