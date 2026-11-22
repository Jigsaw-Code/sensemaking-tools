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

"""Module to summarize room transcripts using an LLM."""

import argparse
import asyncio
from collections import abc
import json
import logging
import os
import re
from typing import Any

from src.models import base_model
from src.models import model_factory

DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
DEFAULT_MAX_CONCURRENT_CALLS = 5

_ROOM_HEADER_SPLIT_RE = re.compile(r"(===== Room:.*?=====)")

PROMPT_TEMPLATE = (
    "You are analyzing a room's transcript. The transcript begins with a header"
    " that lists the room's topic and members, followed by the transcript"
    " itself.\nYour task is to convert the transcript into a per-user"
    " first-person statement that approximates the views they expressed in"
    " their room.\nFor participants who did not provide a view or say anything"
    ' meaningful, you must output exactly "Also present" as their'
    " statement.\n\nReturn a JSON object where the keys are the participant"
    " names (extracted from the members list or transcript) and the values are"
    " their first-person statements.\n\nTranscript:\n{transcript}\n"
)


def parse_transcript(
    file_path: str | os.PathLike[str],
) -> list[dict[str, str]]:
  """Parses a transcript file into room sections.

  Args:
    file_path: Path to the transcript file.

  Returns:
    A list of dictionaries, each containing a 'header' and 'content' for a room.
  """
  with open(file_path, "r") as f:
    content = f.read()

  # Split based on "===== Room:" markers.
  parts = _ROOM_HEADER_SPLIT_RE.split(content)

  rooms = []
  # parts[0] holds any preamble before the first room header.
  for i in range(1, len(parts), 2):
    header = parts[i]
    text = parts[i + 1] if i + 1 < len(parts) else ""
    room_content = header + "\n" + text.strip()
    rooms.append({"header": header, "content": room_content})
  return rooms


def response_parser(
    resp: dict[str, Any], job: dict[str, Any]
) -> dict[str, str]:
  """Parses the JSON response from the language model.

  Args:
    resp: The response dictionary from the model.
    job: The job dictionary.

  Returns:
    The parsed dictionary of user statements.

  Raises:
    ValueError: If the API returns an error or JSON parsing fails.
  """
  del job
  if resp.get("error"):
    raise ValueError(f"API Error: {resp['error']}")
  text = resp.get("text", "").strip()
  try:
    return json.loads(text)
  except json.JSONDecodeError as e:
    logging.error("Failed to parse JSON: %s. Raw text: %r", e, text)
    raise ValueError(f"Failed to parse JSON: {e}") from e


def load_room_summaries(
    path: str | os.PathLike[str],
    skip_speaker_prefixes: abc.Sequence[str] = ("Also present",),
) -> list[dict[str, Any]]:
  """Loads room summaries JSON and filters out attendance placeholders.

  Args:
    path: Path to a room summaries JSON file written by `summarize_rooms`.
    skip_speaker_prefixes: Case-insensitive prefixes for speaker labels or
      placeholder statements (such as `"Also present"`) to exclude.

  Returns:
    Normalized list of room dictionaries with `"room_header"` and filtered
    `"summaries"` mappings.

  Raises:
    ValueError: If the JSON root is not a list of room objects.
  """
  with open(path, "r") as f:
    raw_rooms = json.load(f)
  if not isinstance(raw_rooms, list):
    raise ValueError("Room summaries JSON must contain a top-level list.")

  lowered_prefixes = tuple(
      prefix.strip().lower()
      for prefix in skip_speaker_prefixes
      if prefix.strip()
  )
  normalized_rooms: list[dict[str, Any]] = []
  for entry in raw_rooms:
    if not isinstance(entry, dict):
      continue
    header = str(entry.get("room_header", "")).strip()
    raw_summaries = entry.get("summaries", {})
    filtered: dict[str, str] = {}
    if isinstance(raw_summaries, dict):
      for speaker, statement in raw_summaries.items():
        if not isinstance(statement, str):
          continue
        clean_speaker = str(speaker).strip()
        clean_statement = statement.strip()
        if not clean_speaker or not clean_statement:
          continue
        if lowered_prefixes and (
            clean_speaker.lower().startswith(lowered_prefixes)
            or clean_statement.lower().startswith(lowered_prefixes)
        ):
          continue
        filtered[clean_speaker] = clean_statement
    normalized_rooms.append({"room_header": header, "summaries": filtered})
  return normalized_rooms


async def summarize_rooms(
    transcript_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    model: base_model.BaseModel,
    max_concurrent_calls: int = DEFAULT_MAX_CONCURRENT_CALLS,
) -> list[dict[str, Any]]:
  """Reads transcripts, queries the model for summaries, and writes JSON output.

  Args:
    transcript_path: Path to the input transcript.
    output_path: Path to write the JSON results.
    model: Initialized `BaseModel` instance.
    max_concurrent_calls: Maximum concurrent model requests.

  Returns:
    A list of dictionaries containing `"room_header"` and `"summaries"`.
  """
  base_model.validate_max_concurrent_calls(max_concurrent_calls)
  rooms = parse_transcript(transcript_path)
  logging.info("Found %d rooms.", len(rooms))

  prompts = [
      {
          "prompt": PROMPT_TEMPLATE.format(transcript=room["content"]),
          "room_header": room["header"],
          "response_mime_type": "application/json",
      }
      for room in rooms
  ]

  logging.info("Sending transcripts to model...")
  llm_response, _, _, _ = await model.process_prompts_concurrently(
      prompts=prompts,
      response_parser=response_parser,
      max_concurrent_calls=max_concurrent_calls,
  )

  results = [
      {
          "room_header": row["room_header"],
          "summaries": row["result"],
      }
      for _, row in llm_response.iterrows()
  ]

  # Ensure parent directory exists when output_path includes a directory prefix.
  output_dir = os.path.dirname(os.fspath(output_path))
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

  logging.info("Saved results to %s", output_path)
  return results


def main() -> None:
  """Main entry point."""
  parser = argparse.ArgumentParser(description="Summarize room transcripts.")
  parser.add_argument(
      "--transcript",
      type=str,
      required=True,
      help="Path to transcript file.",
  )
  parser.add_argument(
      "--output",
      type=str,
      required=True,
      help="Path to output JSON file.",
  )
  parser.add_argument(
      "--model",
      type=str,
      default=DEFAULT_MODEL_NAME,
      help="Name of the model to use.",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  model = model_factory.get_model(args.model)
  asyncio.run(summarize_rooms(args.transcript, args.output, model))


if __name__ == "__main__":
  main()
