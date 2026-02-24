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
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from case_studies.wtp.models import custom_types
from case_studies.wtp.sensemaker import Sensemaker


class SensemakerTest(unittest.TestCase):

  def setUp(self):
    """Set up a Sensemaker instance with a mock model."""
    self.mock_genai_llm = MagicMock()
    self.mock_genai_llm.process_prompts_batch = AsyncMock(
        return_value=[{"text": "mock quote"}]
    )

    self.sensemaker = Sensemaker(
        genai_model=self.mock_genai_llm,
    )
    self.sample_statements = [custom_types.Statement(id="s1", text="text1")]
    self.sample_topics = [custom_types.FlatTopic(name="topic1")]

  @patch("case_studies.wtp.sensemaker.categorization")
  @patch("case_studies.wtp.sensemaker.quote_extraction_lib")
  @patch("case_studies.wtp.sensemaker.checkpoint_utils")
  def test_categorize_statements_no_checkpoints(
      self, mock_checkpoint_utils, mock_quote_lib, mock_categorization
  ):
    """Test the full categorization flow when no checkpoints exist.

    Verifies that all processing steps are called and checkpoints are saved.
    """
    # Arrange: Mock functions to have async behavior
    mock_categorization.categorize_topics = AsyncMock(
        return_value=(self.sample_statements, self.sample_topics)
    )
    mock_quote_lib.extract_quotes_from_text = AsyncMock(
        return_value=self.sample_statements
    )
    mock_categorization.categorize_opinions = AsyncMock(
        return_value=self.sample_statements
    )
    mock_categorization.learn_global_opinions = AsyncMock(return_value={})

    # Mock load_checkpoint to simulate no checkpoints being found
    mock_checkpoint_utils.load_checkpoint.return_value = None

    # Act
    async def run_test():
      await self.sensemaker.categorize_statements(
          statements=self.sample_statements,
          output_dir="/fake/dir",
      )

    asyncio.run(run_test())

    # Assert
    # Verify that we attempted to load the topics checkpoint
    mock_checkpoint_utils.load_checkpoint.assert_any_call(
        "statements_with_topics_and_learned_topics", "/fake/dir"
    )

    # Verify that the core processing functions were called
    mock_categorization.categorize_topics.assert_called_once()
    mock_quote_lib.extract_quotes_from_text.assert_called_once()
    mock_categorization.categorize_opinions.assert_called_once()

    # Verify that we saved the checkpoints (topics, quotes, learned_opinions, opinions)
    self.assertEqual(mock_checkpoint_utils.save_checkpoint.call_count, 4)
    mock_checkpoint_utils.save_checkpoint.assert_any_call(
        (self.sample_statements, self.sample_topics),
        "statements_with_topics_and_learned_topics",
        "/fake/dir",
    )

  @patch("case_studies.wtp.sensemaker.categorization")
  @patch("case_studies.wtp.sensemaker.quote_extraction_lib")
  @patch("case_studies.wtp.sensemaker.checkpoint_utils")
  def test_categorize_statements_with_checkpoints(
      self, mock_checkpoint_utils, mock_quote_lib, mock_categorization
  ):
    """Test the categorization flow when a topic checkpoint exists.

    Verifies that the topic processing step is skipped.
    """
    # Arrange: Mock functions to have async behavior
    mock_categorization.categorize_opinions = AsyncMock(
        return_value=self.sample_statements
    )
    mock_categorization.learn_global_opinions = AsyncMock(return_value={})
    mock_quote_lib.extract_quotes_from_text = AsyncMock(
        return_value=self.sample_statements
    )

    # Mock load_checkpoint to return pre-computed data for topics
    # We want topics to be loaded, but quotes and opinions to be None
    # The code calls load_checkpoint for:
    # 1. topics -> return (statements, topics)
    # 2. quotes -> return None (so it extracts quotes)
    # 3. opinions -> return None (so it learns opinions)
    # 4. learned_opinions -> return None
    mock_checkpoint_utils.load_checkpoint.side_effect = [
        (self.sample_statements, self.sample_topics),  # topics loaded
        None,  # quotes not loaded
        None,  # opinions not loaded
        None,  # learned opinions not loaded
    ]

    # Act
    async def run_test():
      await self.sensemaker.categorize_statements(
          statements=self.sample_statements,
          output_dir="/fake/dir",
      )

    asyncio.run(run_test())

    # Assert
    # Verify that we attempted to load the topics checkpoint
    self.assertEqual(mock_checkpoint_utils.load_checkpoint.call_count, 4)
    mock_checkpoint_utils.load_checkpoint.assert_any_call(
        "statements_with_topics_and_learned_topics", "/fake/dir"
    )

    # Verify that topic categorization was NOT called
    mock_categorization.categorize_topics.assert_not_called()

    # Quote extraction and opinion categorization should still be called
    mock_quote_lib.extract_quotes_from_text.assert_called_once()
    mock_categorization.categorize_opinions.assert_called_once()

    # Verify that checkpoints were saved (quotes, learned_opinions, opinions)
    self.assertEqual(mock_checkpoint_utils.save_checkpoint.call_count, 3)


if __name__ == "__main__":
  unittest.main()
