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

import asyncio
import json
import logging
from typing import Any, List, Callable, Type
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    GenerationResponse,
)
import google.auth
from google.auth.transport.requests import Request as AuthRequest
from requests.adapters import HTTPAdapter
import requests
from pydantic import ValidationError, TypeAdapter

from .model import Model as BaseModelClass, SchemaType, ListSchemaType
from .model_util import MAX_LLM_RETRIES, DEFAULT_VERTEX_PARALLELISM, RETRY_DELAY_SEC, MAX_RETRIES


class TokenLimitExceededError(Exception):
  """Custom exception for when the token limit is exceeded."""

  pass


class VertexModel(BaseModelClass):

  def __init__(
      self,
      project: str,
      location: str,
      model_name: str,
  ):
    # Initialize Vertex AI SDK
    creds = custom_pool_creds()  # Enables high concurrency

    vertexai.init(project=project, location=location, credentials=creds)

    self.llm = GenerativeModel(
        model_name=model_name,
        generation_config={
            # Param docs: http://cloud/vertex-ai/generative-ai/docs/model-reference/inference#generationconfig
            "temperature": 0,
            "top_p": 0,
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: (
                HarmBlockThreshold.BLOCK_NONE
            ),
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
                HarmBlockThreshold.BLOCK_NONE
            ),
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
                HarmBlockThreshold.BLOCK_NONE
            ),
            HarmCategory.HARM_CATEGORY_HARASSMENT: (
                HarmBlockThreshold.BLOCK_NONE
            ),
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
                HarmBlockThreshold.BLOCK_NONE
            ),
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: (
                HarmBlockThreshold.BLOCK_NONE
            ),
        },
    )

  async def generate_text(self, prompt: str) -> str:
    return await self._call_llm_with_retry(prompt)

  async def generate_data(
      self, prompt: str, schema: Type[SchemaType] | Type[ListSchemaType]
  ) -> SchemaType | ListSchemaType:
    response_text = await self._call_llm_with_retry(prompt)

    # Drop markdown code block delimiters if present
    if response_text.startswith("```json"):
      response_text = response_text[7:]  # Remove ```json
    if response_text.endswith("```"):
      response_text = response_text[:-3]  # Remove ```

    try:
      # Use TypeAdapter for robust parsing of schema
      # The schema argument itself is the type hint, e.g. Topic or List[Topic]
      adapter = TypeAdapter(schema)
      return adapter.validate_json(response_text)

    except ValidationError as e:  # Catches Pydantic validation errors
      logging.debug(
          f"Model response failed Pydantic validation for schema {schema}:"
          f" {response_text}.\nError: {e}"
      )
      raise ValueError(f"Model response failed Pydantic validation") from e
    except (
        json.JSONDecodeError
    ) as e:  # Catches errors if response_text is not valid JSON
      logging.debug(f"Model returned invalid JSON: {response_text}.")
      raise ValueError(f"Model returned invalid JSON") from e
    except Exception as e:  # Catches any other unexpected errors during parsing
      logging.error(
          f"Failed to parse or validate model response against schema {schema}:"
          f" {response_text}.\nError: {e}"
      )
      raise ValueError(
          f"Failed to parse or validate model response: {response_text}."
      ) from e

  async def _call_llm_with_retry(self, prompt: str) -> str:

    async def call_llm_inner() -> GenerationResponse:
      return await self.llm.generate_content_async(
          prompt
      )  # TODO: pass schema for constraint decoding

    def validate_response(response: GenerationResponse | None) -> bool:
      if not response:
        logging.error("Failed to get a model response.")
        return False

      if (
          not response.candidates
          or not response.candidates[0].content.parts
          or not response.candidates[0].content.parts[0].text
      ):
        logging.error(f"Model returned an incomplete response: {response}")
        return False

      logging.debug(
          "✓ Completed LLM call (input:"
          f" {response.usage_metadata.prompt_token_count} tokens, output:"
          f" {response.usage_metadata.candidates_token_count} tokens)"
      )
      return True

    result = await _retry_call(
        call_llm_inner,
        validate_response,
        MAX_LLM_RETRIES,
        "Failed to get a valid model response.",
        RETRY_DELAY_SEC,
    )
    return result.text


async def _retry_call(
    func: Callable[..., Any],  # The async function to call
    validator: Callable[[Any], bool],  # A function to validate the response
    max_retries: int,
    error_message: str,
    retry_delay_sec: float,
    func_args: List[Any] | None = None,
    validator_args: List[Any] | None = None,
):
  func_args = func_args or []
  validator_args = validator_args or []
  backoff_growth_rate = 2.5  # Controls how quickly delay increases b/w retries

  for attempt in range(1, max_retries + 1):
    try:
      response = await func(*func_args)

      if validator(response, *validator_args):
        return response

      logging.error(f"Attempt {attempt} failed. Invalid response: {response}")
    except Exception as error:
      if "exceeds the maximum number of tokens allowed" in str(error):
        logging.warning("Input token limit exceeded. Not retrying.")
        raise TokenLimitExceededError(error) from error
      logging.error(f"Attempt {attempt} failed: {error}")

    # Exponential backoff calculation
    delay = retry_delay_sec * (backoff_growth_rate ** (attempt - 1))
    logging.info(f"Retrying in {delay} seconds (attempt {attempt})")
    await asyncio.sleep(delay)

  raise Exception(f"Failed after {max_retries} attempts: {error_message}")


async def run_tasks_in_parallel(
    items: List[Any],
    func: Callable,  # func should be an async function: async def func(item, *args, **kwargs)
    limit: int = DEFAULT_VERTEX_PARALLELISM,
    *args: Any,
    **kwargs: Any,
) -> List[Any]:
  semaphore = asyncio.Semaphore(limit)  # Controls concurrency

  async def limited_task(item: Any) -> Any:
    async with semaphore:
      return await func(item, *args, **kwargs)

  if items:
    logging.info(
        f"Running up to {limit} tasks in parallel for a total of"
        f" {len(items)} tasks..."
    )

  tasks = [limited_task(item) for item in items]
  results = await asyncio.gather(
      *tasks, return_exceptions=False
  )  # Set return_exceptions=True to debug individual task failures
  return results


def custom_pool_creds():
  # By default, GCP uses connection pool of size 10, we need to override it to support higher concurrency.
  # Create a session that holds up to 100 connections using a custom adaptor
  session = requests.Session()
  adapter = HTTPAdapter(
      pool_connections=DEFAULT_VERTEX_PARALLELISM,  # number of connection pools to cache
      pool_maxsize=DEFAULT_VERTEX_PARALLELISM,  # max connections per pool
      max_retries=MAX_RETRIES,
      pool_block=True,  # block when no free connections
  )
  session.mount("https://", adapter)
  # Build a google‑auth request using that session
  auth_request = AuthRequest(session=session)
  # Retrieve app-default creds and refresh them with the new settings
  creds, _ = google.auth.default()
  creds.refresh(auth_request)  # primes the pool
  return creds
