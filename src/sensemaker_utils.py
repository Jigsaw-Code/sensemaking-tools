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

"""Common utils for Sensemaker."""

import asyncio
import html
import logging
import random
import re
from typing import List, Callable, TypeVar, Any, Coroutine, Optional

RETRY_DELAY_SEC = 10


_ALLOWED_TAG_REGEX = re.compile(r"(</?(?:question|response|quote)>)")


def get_prompt(
    instructions: str,
    data: List[str],
    additional_context: Optional[str] = None,
) -> str:
  """
  Combines the data and instructions into a prompt to send to an LLM.
  Each data item is wrapped in <statement> tags.

  Args:
      instructions: What the model should do.
      data: The data that the model should consider (list of strings, typically statements).
      additional_context: Additional context to include in the prompt.
  Returns:
      The instructions and the data as a text.
  """
  cleaned_data = []
  for item in data:
    # We need to escape '<' and '>' characters that are not part of an allowed tag.
    # We do this by splitting the string by the allowed tags, and then escaping the parts in between.
    parts = _ALLOWED_TAG_REGEX.split(item)
    escaped_parts = []
    for i, part in enumerate(parts):
      if i % 2 == 1:  # This is a tag, so we leave it as is.
        escaped_parts.append(part)
      else:  # This is not a tag, so we escape it.
        escaped_parts.append(html.escape(part))
    cleaned_data.append("".join(escaped_parts))

  wrapped_data = [f"<item>{item}</item>" for item in cleaned_data]

  return f"""
{f'''<additionalContext>
  {additional_context}
</additionalContext>''' if additional_context else ""}

<instructions>
  {instructions}
</instructions>

<data>
{'''
'''.join(wrapped_data)}
</data>"""


# A generic type variable used to ensure that the type of an argument is the
# same as the type of a return value in functions like `retry_call` and
# `execute_concurrently`.
T = TypeVar("T")


async def execute_concurrently(
    callbacks: List[Callable[[], Coroutine[Any, Any, T]]],
) -> List[T]:
  """
  Executes a list of awaitable tasks (callbacks that return coroutines) concurrently.

  Args:
      callbacks: A list of functions, where each function returns a coroutine.
                 Example: [functools.partial(async_func1), functools.partial(async_func2)]
  Returns:
      A list containing the resolved values of the coroutines,
      in the same order as the input callbacks.
  """
  # Create coroutine objects by calling the functions in the callbacks list
  coroutines_to_run: List[Coroutine[Any, Any, T]] = [cb() for cb in callbacks]

  # NOTE: If at least one callback fails, the entire batch fails.
  # Retries should happen within the individual callbacks if needed.
  results: List[T] = await asyncio.gather(
      *coroutines_to_run, return_exceptions=False
  )
  return results


async def retry_call(
    func: Callable[..., Coroutine[Any, Any, T]],
    is_valid: Callable[[T, ...], bool],
    max_retries: int,
    error_msg: str,
    retry_delay_sec: int = RETRY_DELAY_SEC,
    func_args: Optional[List[Any]] = None,
    is_valid_args: Optional[List[Any]] = None,
) -> T:
  """
  Reruns an async function multiple times if validation fails or an exception occurs.

  Args:
      func: The async function to attempt.
      is_valid: A function that checks if the response from func is valid.
      max_retries: The maximum number of times to retry func.
      error_msg: The error message to raise if all retries fail.
      retry_delay_sec: How long to wait in seconds between calls.
      func_args: Arguments for func.
      is_valid_args: Arguments for is_valid (after the response argument).
  Returns:
      The valid response from func.
  Raises:
      Exception if all retries fail.
  """
  func_args = func_args or []
  is_valid_args = is_valid_args or []
  current_delay_sec = float(retry_delay_sec)

  for attempt in range(1, max_retries + 1):
    try:
      response = await func(*func_args)
      if is_valid(response, *is_valid_args):
        return response
      logging.warning(
          f"Attempt {attempt} failed validation. Invalid response: {response}"
      )
    except Exception as e:
      error_str = str(e).lower()
      if (
          "too many requests" in error_str
          or "429" in error_str  # HTTP code for too many requests
          or "resource_exhausted" in error_str
          or "service is currently unavailable" in error_str
          or "internal error" in error_str
          or "deadline exceeded" in error_str
      ):
        logging.warning(
            f"Attempt {attempt} failed with potentially transient error: {e}"
        )
      else:
        logging.error(
            f"Attempt {attempt} failed with error: {e}", exc_info=True
        )

    if attempt >= max_retries:
      break

    logging.info(
        f"Retrying in {current_delay_sec} seconds (attempt"
        f" {attempt} of {max_retries})..."
    )
    await asyncio.sleep(current_delay_sec)
    # Exponential backoff with jitter
    current_delay_sec = min(current_delay_sec * 1.5, 60.0)  # Cap at 60 seconds
    current_delay_sec += random.uniform(0, 0.1 * current_delay_sec)

  raise Exception(f"Failed after {max_retries} attempts: {error_msg}")
