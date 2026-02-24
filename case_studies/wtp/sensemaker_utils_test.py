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

"""Tests for sensemaker_utils."""

from case_studies.wtp import sensemaker_utils


def test_get_prompt_escaping():
  """Tests that get_prompt correctly escapes special characters."""
  instructions = "Test instructions."

  # 1. String with no special characters
  data1 = ["This is a test."]
  prompt1 = sensemaker_utils.get_prompt(instructions, data1)
  assert "<item>This is a test.</item>" in prompt1

  # 2. String with special characters to be escaped
  data2 = ["I <3 you."]
  prompt2 = sensemaker_utils.get_prompt(instructions, data2)
  assert "<item>I &lt;3 you.</item>" in prompt2

  # 3. String with an allowed tag
  data3 = ["This is a <quote>quote</quote>."]
  prompt3 = sensemaker_utils.get_prompt(instructions, data3)
  assert "<item>This is a <quote>quote</quote>.</item>" in prompt3

  # 4. String with both allowed tag and special characters
  data4 = ["This is a <quote>quote with <3</quote>."]
  prompt4 = sensemaker_utils.get_prompt(instructions, data4)
  assert "<item>This is a <quote>quote with &lt;3</quote>.</item>" in prompt4

  # 5. String with an unallowed tag
  data5 = ["This is an <unallowed>tag</unallowed>."]
  prompt5 = sensemaker_utils.get_prompt(instructions, data5)
  assert (
      "<item>This is an &lt;unallowed&gt;tag&lt;/unallowed&gt;.</item>"
      in prompt5
  )

  # 6. String with ampersand
  data6 = ["A&B"]
  prompt6 = sensemaker_utils.get_prompt(instructions, data6)
  assert "<item>A&amp;B</item>" in prompt6

  # 7. Multiple items
  data7 = ["I <3 you.", "Me & you.", "A <quote>quote</quote>."]
  prompt7 = sensemaker_utils.get_prompt(instructions, data7)
  assert "<item>I &lt;3 you.</item>" in prompt7
  assert "<item>Me &amp; you.</item>" in prompt7
  assert "<item>A <quote>quote</quote>.</item>" in prompt7

  # 8. Empty data
  data8 = []
  prompt8 = sensemaker_utils.get_prompt(instructions, data8)
  assert "<data>\n\n</data>" in prompt8


def test_get_prompt_with_additional_context():
  """Tests that get_prompt generates the additional context tag correctly."""
  instructions = "Test instructions."
  data = ["Item 1"]

  # 1. No additional context
  prompt_no_context = sensemaker_utils.get_prompt(
      instructions, data, additional_context=None
  )
  assert "<additionalContext>" not in prompt_no_context

  # 2. Empty additional context
  prompt_empty_context = sensemaker_utils.get_prompt(
      instructions, data, additional_context=""
  )
  assert "<additionalContext>" not in prompt_empty_context

  # 3. With additional context
  prompt_with_context = sensemaker_utils.get_prompt(
      instructions, data, additional_context="Some extra background info."
  )
  assert "<additionalContext>" in prompt_with_context
  assert "Some extra background info." in prompt_with_context
  assert "</additionalContext>" in prompt_with_context
