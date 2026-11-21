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

"""Unit tests for fake_base_model."""

import asyncio
import unittest

from src.models import fake_base_model


class FakeBaseModelTest(unittest.TestCase):
  """Tests FakeBaseModel defaults, scripted replay, and call recording."""

  def test_unconfigured_methods_raise_not_implemented(self):
    model = fake_base_model.FakeBaseModel()
    with self.assertRaises(NotImplementedError):
      asyncio.run(model.generate_content("hello", "run"))
    with self.assertRaises(NotImplementedError):
      asyncio.run(
          model.process_prompts_concurrently([], lambda text, meta: text)
      )
    self.assertEqual(model.calls, [])

  def test_scripted_responses_exception_replay_and_call_recording(self):
    schema = {"type": "object"}
    model = fake_base_model.FakeBaseModel([
        {"text": "first"},
        RuntimeError("transient error"),
        {"text": "second"},
    ])

    first = asyncio.run(
        model.generate_content(
            "p1",
            "run1",
            temperature=0.2,
            system_prompt="sys",
            response_mime_type="application/json",
            response_schema=schema,
        )
    )
    self.assertEqual(first, {"text": "first"})

    with self.assertRaisesRegex(RuntimeError, "transient error"):
      asyncio.run(model.generate_content("p2", "run2"))

    second = asyncio.run(model.generate_content("p3", "run3"))
    self.assertEqual(second, {"text": "second"})

    self.assertEqual(len(model.calls), 3)
    self.assertEqual(
        model.calls[0],
        {
            "prompt": "p1",
            "run_name": "run1",
            "temperature": 0.2,
            "system_prompt": "sys",
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )
    self.assertEqual(model.calls[1]["prompt"], "p2")
    self.assertEqual(model.calls[2]["prompt"], "p3")

  def test_exhausted_responses_raises_assertion_error(self):
    model = fake_base_model.FakeBaseModel([{"text": "only"}])
    self.assertEqual(
        asyncio.run(model.generate_content("p1", "run1")), {"text": "only"}
    )
    with self.assertRaises(AssertionError):
      asyncio.run(model.generate_content("p2", "run2"))

  def test_forwards_schema_constraint_settings(self):
    default_model = fake_base_model.FakeBaseModel()
    self.assertFalse(default_model.strict_array_bounds())
    self.assertIsNone(default_model.max_constrained_enum_chars())

    overridden = fake_base_model.FakeBaseModel(
        strict_array_bounds=True, max_constrained_enum_chars=512
    )
    self.assertTrue(overridden.strict_array_bounds())
    self.assertEqual(overridden.max_constrained_enum_chars(), 512)


if __name__ == "__main__":
  unittest.main()
