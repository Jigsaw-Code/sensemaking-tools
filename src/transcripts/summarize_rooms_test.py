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

"""Tests for summarize_rooms."""

from collections import abc
import json
import os
import pathlib
import tempfile
from typing import Any
import unittest

import pandas as pd

from src.models import base_model
from src.transcripts import summarize_rooms


class _FakeModel(base_model.BaseModel):
  """Minimal fake BaseModel for deterministic room-summarization tests."""

  def __init__(self, responses: list[dict[str, Any]]):
    super().__init__()
    self._responses = list(responses)
    self.received_prompts: list[dict[str, Any]] = []

  async def generate_content(
      self,
      prompt: str,
      run_name: str,
      temperature: float = 0.0,
      system_prompt: str | None = None,
      response_mime_type: str | None = None,
      response_schema: dict[str, Any] | None = None,
      **kwargs,
  ) -> dict[str, Any]:
    del prompt, run_name, temperature, system_prompt
    del response_mime_type, response_schema, kwargs
    return self._responses.pop(0)

  async def process_prompts_concurrently(
      self,
      prompts: list[dict[str, Any]],
      response_parser: abc.Callable[[Any, dict[str, Any]], Any],
      max_concurrent_calls: int | None = None,
      retry_attempts: int | None = None,
      **kwargs,
  ) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    del max_concurrent_calls, retry_attempts, kwargs
    self.received_prompts.extend(prompts)
    rows = []
    for job, resp in zip(prompts, self._responses):
      parsed = response_parser(resp, job)
      rows.append({**job, "result": parsed})
    return pd.DataFrame(rows), pd.DataFrame(), 0.0, 0.0


class SummarizeRoomsTest(unittest.TestCase):
  """Tests for transcript parsing, response parsing, and summary loading."""

  def test_parse_transcript(self):
    transcript_content = (
        "Some initial text\n"
        "===== Room: Topic A =====\n"
        "User1: Hello\n"
        "User2: Hi\n"
        "===== Room: Topic B =====\n"
        "User3: Test"
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      transcript_file = pathlib.Path(tmp_dir) / "transcript.txt"
      transcript_file.write_text(transcript_content)

      rooms = summarize_rooms.parse_transcript(transcript_file)
      self.assertEqual(len(rooms), 2)
      self.assertEqual(rooms[0]["header"], "===== Room: Topic A =====")
      self.assertIn("User1: Hello", rooms[0]["content"])
      self.assertEqual(rooms[1]["header"], "===== Room: Topic B =====")
      self.assertIn("User3: Test", rooms[1]["content"])

  def test_response_parser_success(self):
    resp = {"text": '{"User1": "I said hello"}'}
    result = summarize_rooms.response_parser(resp, {})
    self.assertEqual(result, {"User1": "I said hello"})

  def test_response_parser_error(self):
    resp = {"error": "API Error"}
    with self.assertRaisesRegex(ValueError, "API Error"):
      summarize_rooms.response_parser(resp, {})

  def test_response_parser_invalid_json(self):
    resp = {"text": "This is not JSON"}
    with self.assertRaisesRegex(ValueError, "Failed to parse JSON"):
      summarize_rooms.response_parser(resp, {})

  def test_load_room_summaries_filters_placeholders_and_prefixes(self):
    raw_rooms = [{
        "room_header": "  ===== Room: Topic A =====  ",
        "summaries": {
            "Participant 1": "  I support expanding bus service.  ",
            "Participant 2": "Also present",
            "Participant 3": "also present.",
            "Also present": "Participant 4",
            "Facilitator": "Let us move to the next option.",
            "Participant 5": "   ",
        },
    }]
    with tempfile.TemporaryDirectory() as tmp_dir:
      summary_path = pathlib.Path(tmp_dir) / "summaries.json"
      summary_path.write_text(json.dumps(raw_rooms))

      default_loaded = summarize_rooms.load_room_summaries(summary_path)
      self.assertEqual(len(default_loaded), 1)
      self.assertEqual(
          default_loaded[0]["room_header"], "===== Room: Topic A ====="
      )
      self.assertCountEqual(
          default_loaded[0]["summaries"].keys(),
          ["Participant 1", "Facilitator"],
      )
      self.assertEqual(
          default_loaded[0]["summaries"]["Participant 1"],
          "I support expanding bus service.",
      )

      custom_loaded = summarize_rooms.load_room_summaries(
          summary_path,
          skip_speaker_prefixes=("Also present", "Facilitator"),
      )
      self.assertEqual(
          custom_loaded[0]["summaries"],
          {"Participant 1": "I support expanding bus service."},
      )

  def test_load_room_summaries_rejects_non_list_root(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      summary_path = pathlib.Path(tmp_dir) / "invalid.json"
      summary_path.write_text('{"room_header": "Room 1"}')
      with self.assertRaisesRegex(ValueError, "top-level list"):
        summarize_rooms.load_room_summaries(summary_path)


class SummarizeRoomsAsyncTest(unittest.IsolatedAsyncioTestCase):
  """Async tests for `summarize_rooms`."""

  async def test_summarize_rooms_writes_nested_output(self):
    transcript_content = (
        "===== Room: Topic A =====\n"
        "User1: We need safer crosswalks.\n"
        "===== Room: Topic B =====\n"
        "User2: Focus on drainage first.\n"
    )
    fake_model = _FakeModel([
        {"text": json.dumps({"User1": "I want safer crosswalks."})},
        {"text": json.dumps({"User2": "I want better drainage."})},
    ])
    with tempfile.TemporaryDirectory() as tmp_dir:
      transcript_path = pathlib.Path(tmp_dir) / "transcript.txt"
      transcript_path.write_text(transcript_content)
      output_path = pathlib.Path(tmp_dir) / "nested" / "room_summaries.json"

      results = await summarize_rooms.summarize_rooms(
          transcript_path, output_path, fake_model
      )

      self.assertEqual(len(results), 2)
      self.assertEqual(len(fake_model.received_prompts), 2)
      written = json.loads(output_path.read_text())
      self.assertEqual(written, results)
      self.assertEqual(
          written[0]["summaries"], {"User1": "I want safer crosswalks."}
      )

  async def test_summarize_rooms_handles_bare_output_filename(self):
    transcript_content = "===== Room: Topic A =====\nUser1: Hello.\n"
    fake_model = _FakeModel([
        {"text": json.dumps({"User1": "I said hello."})},
    ])
    with tempfile.TemporaryDirectory() as tmp_dir:
      transcript_path = pathlib.Path(tmp_dir) / "transcript.txt"
      transcript_path.write_text(transcript_content)
      prev_cwd = os.getcwd()
      os.chdir(tmp_dir)
      try:
        await summarize_rooms.summarize_rooms(
            transcript_path, "bare_output.json", fake_model
        )
      finally:
        os.chdir(prev_cwd)

      written = json.loads(
          (pathlib.Path(tmp_dir) / "bare_output.json").read_text()
      )
      self.assertEqual(len(written), 1)
      self.assertEqual(written[0]["summaries"], {"User1": "I said hello."})


if __name__ == "__main__":
  unittest.main()
