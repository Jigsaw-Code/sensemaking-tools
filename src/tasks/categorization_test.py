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

from src.models import custom_types
from src.tasks import categorization


class CategorizationTest(unittest.TestCase):

  def test_categorize_opinions_uses_quote_ids(self):
    import pandas as pd
    # Setup
    statement = custom_types.Statement(
        id="statement1",
        text="This is a statement about topic A.",
        topics=[custom_types.FlatTopic(name="Topic A")],
        quotes=[
            custom_types.Quote(
                id="statement1-Topic A",
                text="This is a quote about topic A.",
                topic=custom_types.FlatTopic(name="Topic A"),
            )
        ],
    )
    topics_to_process = [custom_types.FlatTopic(name="Topic A")]

    opinions = [custom_types.FlatTopic(name="Opinion 1")]
    nested_topic = custom_types.NestedTopic(name="Topic A", subtopics=opinions)
    topic_map = {"Topic A": nested_topic}

    mock_model = MagicMock()

    # Mock process_prompts_concurrently
    fake_record = custom_types.StatementRecord(
        id="statement1",
        quote_id="statement1-Topic A",
        topics=[custom_types.FlatTopic(name="Opinion 1")],
    )

    # The DF should simulate the prompt job + result
    results_data = [{
        "result": [fake_record],
        "target_opinions": opinions,
        "parent_topic_obj": nested_topic,
        "batch_items": [statement],
        "work_queue_topic_name": "Topic A",
    }]

    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(pd.DataFrame(results_data), {})
    )

    # Mock autorater to pass
    # mock_run_opinion_eval.return_value = {"passed": [fake_record], "failed": []}
    queue = asyncio.Queue()
    stop_event = asyncio.Event()
    autorater_result_entry = {
        "result": {"score": 4, "explanation": "Good"},
        "metadata": {
            "original_record": fake_record,
            "parent_topic_obj": nested_topic,
            "parent_topic_name": "Topic A",
        },
    }
    mock_model.start_concurrent_workers.return_value = (
        queue,
        [],
        [autorater_result_entry],
        [],
        stop_event,
    )

    # Execution
    result = list(
        asyncio.run(
            categorization.categorize_opinions(
                statements_with_topics_and_quotes=[statement],
                topics_to_process=topics_to_process,
                topic_to_opinions_map=topic_map,
                model=mock_model,
            )
        )
    )

    # Assertion
    self.assertEqual(len(result), 1)
    updated_statement = result[0]
    self.assertEqual(len(updated_statement.quotes), 1)
    updated_quote = updated_statement.quotes[0]
    self.assertIsInstance(updated_quote.topic, custom_types.NestedTopic)
    self.assertEqual(updated_quote.topic.name, "Topic A")
    self.assertEqual(len(updated_quote.topic.subtopics), 1)
    self.assertEqual(updated_quote.topic.subtopics[0].name, "Opinion 1")

  def test_categorize_opinions_handles_mismatched_quote_id_with_unique_match(
      self,
  ):
    import pandas as pd
    # Setup
    statement = custom_types.Statement(
        id="statement1",
        text="This is a statement about topic A.",
        topics=[custom_types.FlatTopic(name="Topic A")],
        quotes=[
            custom_types.Quote(
                id="statement1-Topic A",
                text="This is a quote about topic A.",
                topic=custom_types.FlatTopic(name="Topic A"),
            )
        ],
    )
    topics_to_process = [custom_types.FlatTopic(name="Topic A")]

    opinions = [custom_types.FlatTopic(name="Opinion 1")]
    nested_topic = custom_types.NestedTopic(name="Topic A", subtopics=opinions)
    topic_map = {"Topic A": nested_topic}

    mock_model = MagicMock()

    # Mock process_prompts_concurrently with mismatched quote_id
    fake_record = custom_types.StatementRecord(
        id="statement1",
        quote_id="mismatched-id",
        topics=[custom_types.FlatTopic(name="Opinion 1")],
    )

    results_data = [{
        "result": [fake_record],
        "target_opinions": opinions,
        "parent_topic_obj": nested_topic,
        "batch_items": [statement],
        "work_queue_topic_name": "Topic A",
    }]

    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(pd.DataFrame(results_data), {})
    )

    # Mock autorater to pass
    queue = asyncio.Queue()
    stop_event = asyncio.Event()
    autorater_result_entry = {
        "result": {"score": 4, "explanation": "Good"},
        "metadata": {
            "original_record": fake_record,
            "parent_topic_obj": nested_topic,
            "parent_topic_name": "Topic A",
        },
    }
    mock_model.start_concurrent_workers.return_value = (
        queue,
        [],
        [autorater_result_entry],
        [],
        stop_event,
    )

    # Execution
    result = list(
        asyncio.run(
            categorization.categorize_opinions(
                statements_with_topics_and_quotes=[statement],
                topics_to_process=topics_to_process,
                topic_to_opinions_map=topic_map,
                model=mock_model,
            )
        )
    )

    # Assertion
    self.assertEqual(len(result), 1)
    updated_statement = result[0]
    self.assertEqual(len(updated_statement.quotes), 1)
    updated_quote = updated_statement.quotes[0]
    self.assertIsInstance(updated_quote.topic, custom_types.NestedTopic)
    self.assertEqual(updated_quote.topic.name, "Topic A")
    self.assertEqual(len(updated_quote.topic.subtopics), 1)
    self.assertEqual(updated_quote.topic.subtopics[0].name, "Opinion 1")

  def test_create_token_based_batches_respects_max_items(self):
    statements = [
        custom_types.Statement(id=f"{i}", text="text", topics=[])
        for i in range(100)
    ]
    # Max items 10, max tokens very high
    batches = categorization._create_token_based_batches(
        statements, max_tokens=100000, max_items=10
    )
    self.assertEqual(len(batches), 10)
    for batch in batches:
      self.assertEqual(len(batch), 10)

  def test_create_token_based_batches_respects_token_limit(self):
    # Each statement is small, but we force small token limit
    statements = [
        custom_types.Statement(id=f"{i}", text="text", topics=[])
        for i in range(10)
    ]
    # Estimate tokens: "text" -> 1 token + 5 overhead = 6 tokens per item.
    # Set limit to 10 tokens -> 1 item per batch.
    batches = categorization._create_token_based_batches(
        statements, max_tokens=10, max_items=100
    )
    # Should get 10 batches
    self.assertEqual(len(batches), 10)
    for batch in batches:
      self.assertEqual(len(batch), 1)

  def test_learn_global_opinions_adds_other_opinion(self):
    statements_with_topics = [
        custom_types.Statement(
            id="s1",
            text="text",
            quotes=[
                custom_types.Quote(
                    id="q1",
                    text="quote",
                    topic=custom_types.FlatTopic(name="T1"),
                )
            ],
        )
    ]
    topics = [custom_types.FlatTopic(name="T1")]
    mock_model = MagicMock()

    # Mock chunks
    with patch(
        "src.tasks.topic_modeling_util.create_chunks",
        new_callable=AsyncMock,
    ) as mock_chunks:
      mock_chunks.return_value = ["chunk1"]

      # Mock process_prompts_concurrently
      # Return a dataframe with a result that has NO "Other" opinion
      mock_result = custom_types.OpinionResponseSchema(
          name="T1", subtopics=[custom_types.FlatTopic(name="Opinion 1")]
      )

      # We need a proper DataFrame mock or look-alike
      import pandas as pd

      results_df = pd.DataFrame(
          [{"topic_obj": topics[0], "result": mock_result}]
      )

      mock_model.process_prompts_concurrently = AsyncMock(
          return_value=(results_df, {})
      )

      result_map = asyncio.run(
          categorization.learn_global_opinions(
              statements_with_topics, topics, mock_model
          )
      )

      self.assertIn("T1", result_map)
      t1_result = result_map["T1"]
      self.assertEqual(t1_result.name, "T1")
      # Verify "Other" was added
      opinion_names = [op.name for op in t1_result.subtopics]
      self.assertIn("Other", opinion_names)
      self.assertIn("Opinion 1", opinion_names)

  @patch(
      "src.tasks.categorization.asyncio.sleep",
      new_callable=AsyncMock,
  )
  def test_categorize_opinions_retries_and_fails_to_other(self, mock_sleep):
    import pandas as pd
    # Setup
    statement = custom_types.Statement(
        id="statement1",
        text="text",
        topics=[custom_types.FlatTopic(name="T1")],
        quotes=[
            custom_types.Quote(
                id="q1",
                text="quote",
                topic=custom_types.FlatTopic(name="T1"),
            )
        ],
    )
    topics = [custom_types.FlatTopic(name="T1")]
    topic_map = {
        "T1": custom_types.NestedTopic(
            name="T1", subtopics=[custom_types.FlatTopic(name="Op1")]
        )
    }

    mock_model = MagicMock()

    # Mock process_prompts
    # It will be called multiple times: 1st attempt, 2nd, 3rd.
    # We simulate ALWAYS returning a valid record from LLM, but Autorater REJECTS it.
    fake_record = custom_types.StatementRecord(
        id="statement1",
        quote_id="q1",
        topics=[custom_types.FlatTopic(name="Op1")],
    )

    results_data = [{
        "result": [fake_record],
        "target_opinions": topic_map["T1"].subtopics,
        "parent_topic_obj": topics[0],
        "batch_items": [statement],
        "work_queue_topic_name": "T1",
    }]

    # Return same result every time
    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(pd.DataFrame(results_data), {})
    )

    # Mock autorater to FAIL every time
    queue = asyncio.Queue()
    stop_event = asyncio.Event()
    autorater_result_entry_fail = {
        "result": {"score": 2, "explanation": "Bad"},
        "metadata": {
            "original_record": fake_record,
            "parent_topic_obj": topics[0],
            "parent_topic_name": "T1",
        },
    }
    mock_model.start_concurrent_workers.return_value = (
        queue,
        [],
        [autorater_result_entry_fail],
        [],
        stop_event,
    )

    # Execution
    # MAX_AUTORATER_RETRIES is 3.
    # So it should try 3 times, then default to "Other"

    # We need to verify that we default to "Other" eventually
    result = list(
        asyncio.run(
            categorization.categorize_opinions(
                statements_with_topics_and_quotes=[statement],
                topics_to_process=topics,
                topic_to_opinions_map=topic_map,
                model=mock_model,
            )
        )
    )

    # Assertion
    self.assertEqual(len(result), 1)
    updated_st = result[0]
    self.assertTrue(updated_st.quotes)
    self.assertEqual(updated_st.quotes[0].topic.name, "T1")
    # It failed 3 times, should have been assigned to "Other"
    self.assertEqual(len(updated_st.quotes[0].topic.subtopics), 1)
    self.assertEqual(updated_st.quotes[0].topic.subtopics[0].name, "Other")

    # Verify calls -> Should be called 3 times (initial + 2 retries? Or 3 full attempts?)
    # categorization loop runs until queue empty or MAX_LLM_RETRIES.
    # If autorater always fails, it stays in queue.
    # Logic:
    # 1. Attempt 1: Fail Autorater -> count=1 -> needs_retry.
    # 2. Attempt 2: Fail Autorater -> count=2 -> needs_retry.
    # 3. Attempt 3: Fail Autorater -> count=3 -> LIMIT HIT -> assigned to Other -> NOT needs_retry.
    # Queue is empty. Loop breaks.

    # So process_prompts_concurrently should be called 3 times.
    self.assertEqual(mock_model.process_prompts_concurrently.call_count, 3)

  @patch(
      "src.tasks.categorization.asyncio.sleep",
      new_callable=AsyncMock,
  )
  def test_categorize_opinions_retries_and_succeeds(self, mock_sleep):
    import pandas as pd

    statement = custom_types.Statement(
        id="statement1",
        text="text",
        topics=[custom_types.FlatTopic(name="T1")],
        quotes=[
            custom_types.Quote(
                id="q1", text="quote", topic=custom_types.FlatTopic(name="T1")
            )
        ],
    )
    topics = [custom_types.FlatTopic(name="T1")]
    topic_map = {
        "T1": custom_types.NestedTopic(
            name="T1", subtopics=[custom_types.FlatTopic(name="Op1")]
        )
    }
    mock_model = MagicMock()

    fake_record = custom_types.StatementRecord(
        id="statement1",
        quote_id="q1",
        topics=[custom_types.FlatTopic(name="Op1")],
    )
    results_data = [{
        "result": [fake_record],
        "target_opinions": topic_map["T1"].subtopics,
        "parent_topic_obj": topics[0],
        "batch_items": [statement],
        "work_queue_topic_name": "T1",
    }]

    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(pd.DataFrame(results_data), {})
    )

    # Side effect for autorater: Fail twice, then Pass
    queue = asyncio.Queue()
    stop_event = asyncio.Event()

    autorater_result_entry_fail = {
        "result": {"score": 2, "explanation": "Bad"},
        "metadata": {
            "original_record": fake_record,
            "parent_topic_obj": topics[0],
            "parent_topic_name": "T1",
        },
    }
    autorater_result_entry_pass = {
        "result": {"score": 4, "explanation": "Good"},
        "metadata": {
            "original_record": fake_record,
            "parent_topic_obj": topics[0],
            "parent_topic_name": "T1",
        },
    }

    mock_model.start_concurrent_workers.side_effect = [
        (queue, [], [autorater_result_entry_fail], [], stop_event),
        (queue, [], [autorater_result_entry_fail], [], stop_event),
        (queue, [], [autorater_result_entry_pass], [], stop_event),
    ]

    result = list(
        asyncio.run(
            categorization.categorize_opinions(
                statements_with_topics_and_quotes=[statement],
                topics_to_process=topics,
                topic_to_opinions_map=topic_map,
                model=mock_model,
            )
        )
    )

    self.assertEqual(len(result), 1)
    updated_st = result[0]
    self.assertEqual(updated_st.quotes[0].topic.subtopics[0].name, "Op1")

    # Should have called LLM 3 times
    self.assertEqual(mock_model.process_prompts_concurrently.call_count, 3)


if __name__ == "__main__":
  unittest.main()
