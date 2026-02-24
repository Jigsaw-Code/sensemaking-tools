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

import json
import os
import unittest
from unittest.mock import patch

from src.models import custom_types
from src import runner_utils


class RunnerUtilsTest(unittest.TestCase):

  def test_filter_large_statements(self):
    statements = [
        custom_types.Statement(id="1", text="small text"),
        custom_types.Statement(id="2", text="large " * 1000),
    ]

    # Set limit to something small.
    # "small text" is 10 chars -> 2 tokens.
    # "large " * 1000 is 6000 chars -> 1500 tokens.
    valid, skipped = runner_utils.filter_large_statements(
        statements, token_limit=10
    )

    self.assertEqual(len(valid), 1)
    self.assertEqual(valid[0].id, "1")
    self.assertEqual(len(skipped), 1)
    self.assertEqual(skipped[0].id, "2")

  def test_generate_and_save_topic_tree(self):
    topic_tree_data = [{
        "topic_name": "Topic 1",
        "opinions": [
            {
                "opinion_text": "Opinion 1.1",
                "representative_texts": ["Quote 1.1"],
            },
            {
                "opinion_text": "Opinion 1.2",
                "representative_texts": ["Quote 1.2"],
            },
        ],
    }]
    output_file_base = "test_topic_tree"

    with patch("builtins.print"):  # suppress output prints
      runner_utils.generate_and_save_topic_tree(
          topic_tree_data, output_file_base
      )

    # Check if the TXT file was created and has the correct content
    txt_file_path = f"{output_file_base}.txt"
    self.assertTrue(os.path.exists(txt_file_path))

    with open(txt_file_path, "r", encoding="utf-8") as f:
      content = f.read()
      expected_content = (
          "1. Topic 1 (2 quotes)\n"
          "  1. Opinion 1.1 (1 quotes)\n"
          "  2. Opinion 1.2 (1 quotes)\n\n"
          "Total number of unique opinions: 2"
      )
      self.assertEqual(content, expected_content)

    # Clean up the created files
    os.remove(txt_file_path)


if __name__ == "__main__":
  unittest.main()
