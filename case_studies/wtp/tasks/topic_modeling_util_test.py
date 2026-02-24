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

from typing import List
import unittest
from unittest.mock import AsyncMock, patch

from case_studies.wtp.models.genai_model import GenaiModel
from case_studies.wtp.models import custom_types
from case_studies.wtp.tasks import topic_modeling_util
import pandas as pd


class TopicModelingUtilTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.model = AsyncMock(spec=GenaiModel)
    self.parent_topic = custom_types.FlatTopic(name="Parent Topic")

  async def test_generate_topics_with_chunking_single_chunk(self):
    # Setup
    self.model.call_gemini.return_value = {
        "text": '{"topics": [{"name": "Topic 1"}]}',
        "error": None,
    }

    # Act
    with patch(
        "case_studies.wtp.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["statement"]]
      # process_prompts_concurrently returns tuple (df, stats)
      self.model.process_prompts_concurrently.return_value = (
          pd.DataFrame({
              "result": [
                  custom_types.FlatTopicList(
                      topics=[custom_types.FlatTopic(name="Topic 1")]
                  )
              ]
          }),
          pd.DataFrame(),
      )
      result = await topic_modeling_util.generate_topics_with_chunking(
          self.model,
          "instructions",
          ["statement"],
          custom_types.FlatTopicList,
      )

    # Assert
    self.assertEqual(len(result.topics), 1)
    self.assertEqual(result.topics[0].name, "Topic 1")
    self.model.call_gemini.assert_called_once()
    self.model.process_prompts_concurrently.assert_called_once()

  async def test_generate_topics_with_chunking_multiple_chunks(self):
    # Setup
    with patch(
        "case_studies.wtp.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["s1"], ["s2"]]

      # Mock process_prompts_concurrently response
      results_df = pd.DataFrame({
          "result": [
              [custom_types.FlatTopic(name="Topic 1")],
              [custom_types.FlatTopic(name="Topic 2")],
          ]
      })
      self.model.process_prompts_concurrently.return_value = (
          results_df,
          pd.DataFrame(),
      )

      # Mock call_gemini for merge step
      self.model.call_gemini.return_value = {
          "text": '{"topics": [{"name": "Merged Topic"}]}',
          "error": None,
      }

      # Act
      result = await topic_modeling_util.generate_topics_with_chunking(
          self.model, "instructions", ["s1", "s2"], custom_types.FlatTopicList
      )

      # Assert
      self.assertEqual(len(result.topics), 1)
      self.assertEqual(result.topics[0].name, "Merged Topic")
      self.model.process_prompts_concurrently.assert_called_once()
      self.model.call_gemini.assert_called_once()

  async def test_generate_opinions_with_chunking_single_chunk(self):
    # Setup
    self.model.call_gemini.return_value = {
        "text": (
            '{"name": "Parent Topic", "subtopics": [{"name": "Opinion 1"}]}'
        ),
        "error": None,
    }

    # Act
    with patch(
        "case_studies.wtp.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["statement"]]
      self.model.process_prompts_concurrently.return_value = (
          pd.DataFrame({
              "result": [
                  custom_types.NestedTopic(
                      name="Parent Topic",
                      subtopics=[custom_types.FlatTopic(name="Opinion 1")],
                  )
              ]
          }),
          pd.DataFrame(),
      )
      result = await topic_modeling_util.generate_opinions_with_chunking(
          self.model,
          "instructions",
          ["statement"],
          custom_types.NestedTopic,
          self.parent_topic,
      )

    # Assert
    self.assertEqual(result.name, "Parent Topic")
    self.assertEqual(len(result.subtopics), 1)
    self.assertEqual(result.subtopics[0].name, "Opinion 1")
    self.model.call_gemini.assert_called_once()

  async def test_generate_opinions_with_chunking_multiple_chunks(self):
    with patch(
        "case_studies.wtp.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["s1"], ["s2"]]

      # Mock process_prompts_concurrently response
      results_df = pd.DataFrame({
          "result": [
              custom_types.NestedTopic(
                  name="P", subtopics=[custom_types.FlatTopic(name="O1")]
              ),
              custom_types.NestedTopic(
                  name="P", subtopics=[custom_types.FlatTopic(name="O2")]
              ),
          ]
      })
      self.model.process_prompts_concurrently.return_value = (
          results_df,
          pd.DataFrame(),
      )

      # Mock call_gemini for merge step
      self.model.call_gemini.return_value = {
          "text": (
              '{"name": "Parent Topic", "subtopics": [{"name": "Merged'
              ' Opinion"}]}'
          ),
          "error": None,
      }

      # Act
      result = await topic_modeling_util.generate_opinions_with_chunking(
          self.model,
          "instructions",
          ["s1", "s2"],
          custom_types.NestedTopic,
          self.parent_topic,
      )

      # Assert
      self.assertEqual(result.name, "Parent Topic")
      self.assertEqual(len(result.subtopics), 1)
      self.assertEqual(result.subtopics[0].name, "Merged Opinion")
      self.model.process_prompts_concurrently.assert_called_once()
      self.model.call_gemini.assert_called_once()

  @patch("case_studies.wtp.sensemaker_utils.get_prompt")
  async def test_merge_opinions(self, mock_get_prompt):
    expected_result = custom_types.NestedTopic(
        name=self.parent_topic.name,
        subtopics=[custom_types.NestedTopic(name="Merged Opinion")],
    )
    # Mock call_gemini to return a JSON string as expected by _merge_opinions
    self.model.call_gemini.return_value = {
        "text": (
            '{"name": "Parent Topic", "subtopics": [{"name": "Merged'
            ' Opinion"}]}'
        ),
        "error": None,
    }
    mock_get_prompt.return_value = "prompt"

    partial_results = [
        custom_types.NestedTopic(
            name=self.parent_topic.name,
            subtopics=[custom_types.FlatTopic(name="Opinion 1")],
        ),
        custom_types.NestedTopic(
            name=self.parent_topic.name,
            subtopics=[custom_types.FlatTopic(name="Opinion 2")],
        ),
    ]
    schema_to_expect = custom_types.NestedTopic
    additional_context = "context"

    result = await topic_modeling_util.merge_opinions(
        self.model,
        partial_results,
        schema_to_expect,
        self.parent_topic,
        additional_context,
    )

    self.assertEqual(result, expected_result)

    self.model.call_gemini.assert_called_once_with(
        prompt="prompt",
        run_name="merge_opinions",
        response_schema=schema_to_expect,
        max_concurrent_calls=20,
    )
    mock_get_prompt.assert_called_once()


if __name__ == "__main__":
  unittest.main()
