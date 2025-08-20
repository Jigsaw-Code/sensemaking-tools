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
from typing import Any, Callable, Tuple, Dict, List, Optional
from google import genai
import pandas as pd

# The maximum number of times an LLM call should be retried.
MAX_LLM_RETRIES = 4
# How long in seconds to wait between LLM calls. This is needed due to per
# minute limits Vertex AI imposes.
RETRY_DELAY_SEC = 60
# How long in seconds to wait before first LLM calls.
INITIAL_RETRY_DELAY = 60
# Maximum number of concurrent API calls. By default Genai limits to 10.
MAX_CONCURRENT_CALLS = 5


class GenaiModel:
  """A wrapper around the Google Generative AI API."""

  def __init__(
      self,
      api_key: str,
      model_name: str,
      embedding_model_name: str,
      safety_filters_on: bool = False,
  ):
    """Initializes the GenaiModel.

    Args:
      api_key: The Google Generative AI API key.
      model_name: The name of the model to use.
      embedding_model_name: The name of the embedding model to use.
    """
    self.client = genai.Client(api_key=api_key)
    self.model = model_name
    self.embedding_model = embedding_model_name
    self.safety_settings = (
        [
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
        if safety_filters_on
        else []
    )

  async def _api_worker_with_retry(
      self,
      worker_id: int,
      queue: asyncio.Queue,
      results_list: list,
      stats_list: list,
      stop_event: asyncio.Event,
      response_parser: Callable[[str], pd.DataFrame],
  ):
    """
    Consumes jobs from the queue, calls the Gemini API with retry logic,
    and appends results to shared lists.
    """
    logging.info(f"[Worker-{worker_id}] Started.")
    while not stop_event.is_set():
      try:
        # Use a timeout to periodically check the stop_event
        job = await asyncio.wait_for(queue.get(), timeout=1.0)
      except asyncio.TimeoutError:
        continue  # No job in queue, check stop_event and loop again

      # The 'None' sentinel means the producer is done
      if job is None:
        break

      topic_num = job["topic_num"]
      topic = job["topic"]
      prompt = job["prompt"]
      allocations = job["allocations"]
      stats = job["stats"]
      combined_tokens = stats["combined_tokens"]
      retry_attempts = job["retry_attempts"]
      initial_retry_delay = job["initial_retry_delay"]
      delay_between_calls_seconds = job["delay_between_calls_seconds"]
      system_prompt = job["system_prompt"]
      response_mime_type = job["response_mime_type"]
      response_schema = job["response_schema"]

      # Retry logic

      for attempt in range(retry_attempts):
        if stop_event.is_set():
          logging.info(
              f"[T#{topic_num} Worker-{worker_id}] Stop event received,"
              " terminating."
          )
          break

        try:
          logging.info(
              f"[T#{topic_num} Worker-{worker_id}] Processing topic '{topic}'"
              f" (Attempt {attempt + 1})..."
          )

          # Make the actual API call
          resp = await self._call_gemini(
              prompt=prompt,
              topic=topic,
              system_prompt=system_prompt,
              response_mime_type=response_mime_type,
              response_schema=response_schema,
          )
          if not resp:
            raise Exception("Error happened in the _call_gemini method")
          error_reason = resp["error"]
          if error_reason:
            finish_message = resp.get("finish_message")
            token_count = resp.get("token_count")
            exception_str = f"Error reason: {error_reason!r}"
            if finish_message and token_count:
              exception_str += (
                  f", token_count: {token_count}, message: {finish_message}"
              )
            raise Exception(exception_str)
          if not resp.get("text"):
            raise Exception("Empty response from API")

          extracted_statements_df = response_parser(resp["text"])

          # On success, process the result and add it to the list
          result_data = {
              "topic": topic,
              "propositions": extracted_statements_df,
              "allocations": allocations,
              "token_used": resp["input_token_count"],
          }
          results_list.append(result_data)
          stats_list.append(stats)

          logging.info(
              f"✅ [T#{topic_num} Worker-{worker_id}] Successfully processed"
              f" topic '{topic}'. Tokens used: {resp['input_token_count']}"
          )

          # Add a delay after a successful call
          await asyncio.sleep(delay_between_calls_seconds)

          # Break the retry loop on success
          break

        except Exception as e:
          logging.error(
              f"❌ [T#{topic_num} Worker-{worker_id}] Error on topic '{topic}',"
              f" input_token: {combined_tokens}, attempt {attempt + 1}:"
              f" {repr(e)}"
          )
          if attempt < retry_attempts - 1:
            logging.info(f"Retrying in {initial_retry_delay:.2f} seconds...")
            await asyncio.sleep(initial_retry_delay)
          else:
            logging.error(
                f"Failed to process topic '{topic}' after {retry_attempts}"
                " attempts."
            )

      queue.task_done()

    logging.info(f"[Worker-{worker_id}] Finished.")

  async def process_prompts_concurrently(
      self,
      prompts: List[Dict[str, Any]],
      response_parser: Callable[[str], pd.DataFrame],
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

      job = {
          "topic_num": i + 1,
          "topic": prompt_data["topic"],
          "prompt": prompt_data["prompt"],
          "allocations": prompt_data["allocations"],
          "stats": prompt_data["stats"],
          "retry_attempts": retry_attempts,
          "initial_retry_delay": initial_retry_delay,
          "delay_between_calls_seconds": delay_between_calls_seconds,
          "system_prompt": prompt_data["system_prompt"],
          "response_mime_type": prompt_data["response_mime_type"],
          "response_schema": prompt_data["response_schema"],
      }
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

    return llm_response, llm_response_stats

  async def _call_gemini(
      self,
      prompt: str,
      topic: str,
      temperature: float = 0.0,
      system_prompt: Optional[str] = None,
      response_mime_type: Optional[str] = None,
      response_schema: Optional[Dict[str, Any]] = None,
  ) -> Optional[Dict[str, Any]]:
    """Calls the Gemini model with the given prompt.

    Args:
      prompt: The prompt to send to the model.
      topic: The topic for logging purposes.
      temperature: The temperature to use for the model.

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
          ),
      )
      if not response.candidates:
        logging.error("The response from the API contained no candidates.")
        logging.error("This might be due to a problem with the prompt itself.")
        return {"error": response.prompt_feedback}

      candidate = response.candidates[0]

      if candidate.finish_reason.name != "STOP":
        logging.error(
            "The model stopped generating for a reason: '%s' for topic: %s",
            candidate.finish_reason.name,
            topic,
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

  def call_gemini_embedding(
      self, texts: List[str]
  ) -> Optional[List[List[float]]]:
    """Calls the Gemini embedding model with the given texts.

    Args:
      texts: A list of strings to embed.

    Returns:
      A list of embeddings, or None if an error occurred.
    """
    try:
      response = self.client.embed_content(
          model=self.embedding_model, content=texts
      )
      return response["embeddings"][0]
    except Exception as e:
      logging.error(
          "An unexpected error occurred during embedding generation: %s", e
      )
      return None

  def calculate_token_count_needed(
      self,
      prompt: str,
      temperature: float = 0.0,
  ) -> int:
    """Calculates the number of tokens needed for a given prompt.

    Args:
      prompt: The prompt to calculate the token count for.

    Returns:
      The number of tokens needed for the prompt.
    """
    return self.client.models.count_tokens(
        model=self.model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=temperature, safety_settings=self.safety_settings
        ),
    ).total_tokens
