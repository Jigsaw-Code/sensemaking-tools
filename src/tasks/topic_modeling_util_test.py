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

import logging
from typing import List
import unittest
from unittest.mock import AsyncMock, patch

from src.models.genai_model import GenaiModel
from src.models import custom_types
from src import run_health
from src.tasks import topic_modeling
from src.tasks import topic_modeling_util
import pandas as pd


class TopicModelingUtilTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.model = AsyncMock(spec=GenaiModel)
    self.parent_topic = custom_types.FlatTopic(name="Parent Topic")
    run_health.reset()
    self.addCleanup(run_health.reset)
    # Other test modules disable logging globally at import; re-enable it so
    # assertLogs works regardless of test ordering.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    self.addCleanup(logging.disable, previous_disable)

  async def test_generate_topics_with_chunking_single_chunk(self):
    # Setup
    self.model.generate_content.return_value = {
        "text": '{"topics": [{"name": "Topic 1"}]}',
        "error": None,
    }

    # Act
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
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
          0.0,
          1.0,
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
    self.model.generate_content.assert_called_once()
    self.model.process_prompts_concurrently.assert_called_once()

  async def test_generate_topics_with_chunking_multiple_chunks(self):
    # Setup
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
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
          0.0,
          1.0,
      )

      # Mock call_gemini for merge step
      self.model.generate_content.return_value = {
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
      self.model.generate_content.assert_called_once()

  async def test_generate_topics_with_chunking_forwards_max_concurrent_calls(
      self,
  ):
    self.model.process_prompts_concurrently.return_value = (
        pd.DataFrame({
            "result": [
                custom_types.FlatTopicList(
                    topics=[custom_types.FlatTopic(name="Topic 1")]
                )
            ]
        }),
        pd.DataFrame(),
        0.0,
        1.0,
    )
    self.model.generate_content.return_value = {
        "text": '{"topics": [{"name": "Topic 1"}]}',
        "error": None,
    }

    await topic_modeling_util.generate_topics_with_chunking(
        self.model,
        "instructions",
        ["statement"],
        custom_types.FlatTopicList,
        chunks=[["statement"]],
        max_concurrent_calls=5,
    )

    self.assertEqual(
        self.model.process_prompts_concurrently.call_args.kwargs[
            "max_concurrent_calls"
        ],
        5,
    )

  async def test_generate_opinions_with_chunking_single_chunk(self):
    # Setup
    self.model.generate_content.return_value = {
        "text": (
            '{"name": "Parent Topic", "subtopics": [{"name": "Opinion 1"}]}'
        ),
        "error": None,
    }

    # Act
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
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
          0.0,
          1.0,
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
    self.model.generate_content.assert_called_once()

  async def test_generate_opinions_with_chunking_multiple_chunks(self):
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
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
          0.0,
          1.0,
      )

      # Mock call_gemini for merge step
      self.model.generate_content.return_value = {
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
      self.model.generate_content.assert_called_once()

  @patch("src.sensemaker_utils.get_prompt")
  async def test_merge_opinions(self, mock_get_prompt):
    expected_result = custom_types.NestedTopic(
        name=self.parent_topic.name,
        subtopics=[custom_types.NestedTopic(name="Merged Opinion")],
    )
    # Mock call_gemini to return a JSON string as expected by _merge_opinions
    self.model.generate_content.return_value = {
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

    self.model.generate_content.assert_called_once_with(
        prompt="prompt",
        run_name="merge_opinions",
        response_schema=schema_to_expect,
        max_concurrent_calls=20,
    )
    mock_get_prompt.assert_called_once()

  async def test_merge_topics_filters_placeholders_and_falls_back_when_needed(
      self,
  ):
    partial_results = [
        custom_types.FlatTopic(name="Topic 1"),
        custom_types.FlatTopic(name="Topic 2"),
        custom_types.FlatTopic(name="Roads"),
        custom_types.FlatTopic(name="Topic 1: Fire and Rescue,"),
    ]
    self.model.generate_content.return_value = {
        "text": '{"topics": [{"name": "Topic 1"}, {"name": "Topic 2"}]}',
        "error": None,
    }

    result = await topic_modeling_util._merge_topics(
        self.model,
        partial_results,
        custom_types.FlatTopicList,
    )

    self.assertEqual(
        [t.name for t in result.topics],
        ["Roads", "Fire and Rescue"],
    )

  def test_parse_response_repairs_truncated_json(self):
    truncated_after_item = (
        '{"topics": [{"name": "Public Safety"}, {"name": "Infrastructure"},'
        ' {"name": "Truncated'
    )
    parsed = topic_modeling_util.parse_response(
        truncated_after_item, custom_types.FlatTopicList
    )
    self.assertEqual(
        [t.name for t in parsed.topics], ["Public Safety", "Infrastructure"]
    )

    truncated_first_string = (
        '{"name": "Public Health", "subtopics": [{"name": "Mental Health &'
        " Physical Health & Mental Health & Physical Health"
    )
    parsed_op = topic_modeling_util.parse_response(
        truncated_first_string, custom_types.OpinionResponseSchema
    )
    self.assertEqual(parsed_op.name, "Public Health")
    self.assertEqual(len(parsed_op.subtopics), 1)
    self.assertTrue(
        parsed_op.subtopics[0].name.startswith("Mental Health & Physical Health")
    )

  async def test_generate_topics_reports_failed_chunks(self):
    """Tests that failed topic chunks log a WARNING and record run_health."""
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["s1"], ["s2"], ["s3"]]
      results_df = pd.DataFrame({
          "chunk": ["topic_chunk_0/3", "topic_chunk_1/3", "topic_chunk_2/3"],
          "result": [
              [custom_types.FlatTopic(name="Topic 1")],
              {"error": "Failed after 20 attempts"},
              [custom_types.FlatTopic(name="Topic 3")],
          ],
      })
      self.model.process_prompts_concurrently.return_value = (
          results_df,
          pd.DataFrame(),
          0.0,
          1.0,
      )
      self.model.generate_content.return_value = {
          "text": '{"topics": [{"name": "Merged Topic"}]}',
          "error": None,
      }

      with self.assertLogs(level="WARNING") as logs:
        result = await topic_modeling_util.generate_topics_with_chunking(
            self.model,
            "instructions",
            ["s1", "s2", "s3"],
            custom_types.FlatTopicList,
        )

    self.assertEqual(result.topics[0].name, "Merged Topic")
    self.assertTrue(
        any("failed 1 of 3 chunks" in line for line in logs.output),
        logs.output,
    )
    self.assertEqual(run_health.summary(), {"topic_chunk_failed": 1})
    self.assertIn("topic_chunk_1/3", run_health.details("topic_chunk_failed")[0])

  async def test_generate_topics_no_failures_records_nothing(self):
    """Tests that fully successful chunking records no run_health events."""
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["s1"], ["s2"]]
      self.model.process_prompts_concurrently.return_value = (
          pd.DataFrame({
              "result": [
                  [custom_types.FlatTopic(name="Topic 1")],
                  [custom_types.FlatTopic(name="Topic 2")],
              ]
          }),
          pd.DataFrame(),
          0.0,
          1.0,
      )
      self.model.generate_content.return_value = {
          "text": '{"topics": [{"name": "Merged Topic"}]}',
          "error": None,
      }
      await topic_modeling_util.generate_topics_with_chunking(
          self.model, "instructions", ["s1", "s2"], custom_types.FlatTopicList
      )
    self.assertEqual(run_health.summary(), {})

  async def test_generate_topics_counts_missing_rows_as_failed(self):
    """Tests that chunks with no result row at all are counted as failed."""
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks:
      mock_create_chunks.return_value = [["s1"], ["s2"]]
      self.model.process_prompts_concurrently.return_value = (
          pd.DataFrame({"result": [[custom_types.FlatTopic(name="Topic 1")]]}),
          pd.DataFrame(),
          0.0,
          1.0,
      )
      self.model.generate_content.return_value = {
          "text": '{"topics": [{"name": "Merged Topic"}]}',
          "error": None,
      }
      with self.assertLogs(level="WARNING") as logs:
        await topic_modeling_util.generate_topics_with_chunking(
            self.model, "instructions", ["s1", "s2"], custom_types.FlatTopicList
        )
    self.assertTrue(any("failed 1 of 2 chunks" in l for l in logs.output))
    self.assertEqual(run_health.summary(), {"topic_chunk_failed": 1})

  async def test_generate_opinions_reports_failed_chunks(self):
    """Tests that failed opinion chunks log a WARNING and record run_health."""
    with patch(
        "src.tasks.topic_modeling_util.create_chunks"
    ) as mock_create_chunks, patch(
        "src.tasks.topic_modeling_util.merge_opinions",
        new_callable=AsyncMock,
    ) as mock_merge:
      mock_create_chunks.return_value = [["q1"], ["q2"]]
      ok = custom_types.NestedTopic(
          name="Parent Topic",
          subtopics=[custom_types.FlatTopic(name="Opinion 1")],
      )
      self.model.process_prompts_concurrently.return_value = (
          pd.DataFrame({
              "chunk": ["opinion_chunk_0/2", "opinion_chunk_1/2"],
              "result": [ok, None],
          }),
          pd.DataFrame(),
          0.0,
          1.0,
      )
      mock_merge.return_value = ok
      with self.assertLogs(level="WARNING") as logs:
        await topic_modeling_util.generate_opinions_with_chunking(
            self.model,
            "instructions",
            ["q1", "q2"],
            custom_types.OpinionResponseSchema,
            self.parent_topic,
        )
    self.assertTrue(any("failed 1 of 2 chunks" in l for l in logs.output))
    self.assertEqual(run_health.summary(), {"opinion_chunk_failed": 1})
    # Only the successful chunk is merged.
    self.assertEqual(mock_merge.call_args.args[1], [ok])

  async def test_merge_opinions_error_records_run_health(self):
    """Tests that a failed opinion merge is recorded in run_health."""
    self.model.generate_content.return_value = {"error": "boom"}
    partial = [
        custom_types.NestedTopic(
            name="Parent Topic",
            subtopics=[custom_types.FlatTopic(name="Opinion 1")],
        )
    ]
    result = await topic_modeling_util.merge_opinions(
        self.model, partial, custom_types.NestedTopic, self.parent_topic
    )
    self.assertEqual(result.subtopics, [])
    self.assertEqual(run_health.summary(), {"opinion_merge_failed": 1})

  async def test_learn_topics_failure_records_run_health(self):
    """Tests that learn_topics giving up records topic_learning_failed."""
    self.model.max_llm_retries = 1
    statements = [custom_types.Statement(id="1", text="t", topics=[], quotes=[])]
    with patch(
        "src.tasks.topic_modeling.topic_modeling_util.create_chunks",
        new_callable=AsyncMock,
        return_value=[["t"]],
    ), patch(
        "src.tasks.topic_modeling.sensemaker_utils.retry_call",
        new_callable=AsyncMock,
        return_value=None,
    ):
      result = await topic_modeling.learn_topics(statements, self.model)
    self.assertEqual(result, [])
    self.assertEqual(run_health.summary(), {"topic_learning_failed": 1})


if __name__ == "__main__":
  unittest.main()
