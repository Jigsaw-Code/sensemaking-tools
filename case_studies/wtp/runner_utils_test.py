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

from case_studies.wtp.models import custom_types
from case_studies.wtp import runner_utils


class RunnerUtilsTest(unittest.TestCase):

  def run_generate_and_save_topic_tree(self, topic_tree_data) -> str:
    """Wrapper for generate_and_save_topic_tree that reads output file and returns str."""

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

    # Clean up the created files
    os.remove(txt_file_path)

    return content

  def test_generate_and_save_topic_tree(self):
    tree_str = self.run_generate_and_save_topic_tree([{
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
    }])
    self.assertEqual(tree_str, (
        "1. Topic 1 (2 quotes)\n"
        "  1. Opinion 1.1 (1 quotes)\n"
        "  2. Opinion 1.2 (1 quotes)\n\n"
        "Total number of unique opinions: 2"
    ))

  def test_generate_and_save_topic_tree_with_duplicate_quotes(self):
    tree_str = self.run_generate_and_save_topic_tree([{
        "topic_name": "Topic 1",
        "opinions": [
            {
                "opinion_text": "Opinion 1.1",
                "representative_texts": ["Quote 1.1"],
            },
            {
                "opinion_text": "Opinion 1.2",
                "representative_texts": ["Quote 1.1", "Quote 1.2"],
            },
        ],
    }])
    self.assertEqual(tree_str, (
        "1. Topic 1 (2 quotes)\n"
        "  1. Opinion 1.2 (2 quotes)\n"
        "  2. Opinion 1.1 (1 quotes)\n\n"
        "Total number of unique opinions: 2"
    ))


if __name__ == "__main__":
  unittest.main()