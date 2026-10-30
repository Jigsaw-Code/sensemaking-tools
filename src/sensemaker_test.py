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
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import custom_types
from src.sensemaker import Sensemaker


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

  @patch("src.sensemaker.categorization")
  @patch("src.sensemaker.quote_extraction_lib")
  @patch("src.sensemaker.checkpoint_utils")
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

    # Verify that we saved the checkpoints (topics, quotes, learned_opinions,
    # statements_with_opinions_pass1, learned_opinions_pass1, opinions)
    self.assertEqual(mock_checkpoint_utils.save_checkpoint.call_count, 6)
    mock_checkpoint_utils.save_checkpoint.assert_any_call(
        (self.sample_statements, self.sample_topics),
        "statements_with_topics_and_learned_topics",
        "/fake/dir",
    )

  @patch("src.sensemaker.categorization")
  @patch("src.sensemaker.quote_extraction_lib")
  @patch("src.sensemaker.checkpoint_utils")
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
    # 4. opinions pass 1 -> return None
    # 5. learned opinions -> return None
    mock_checkpoint_utils.load_checkpoint.side_effect = [
        (self.sample_statements, self.sample_topics),  # topics loaded
        None,  # quotes not loaded
        None,  # opinions not loaded
        None,  # pass-1 opinions not loaded
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
    self.assertEqual(mock_checkpoint_utils.load_checkpoint.call_count, 5)
    mock_checkpoint_utils.load_checkpoint.assert_any_call(
        "statements_with_topics_and_learned_topics", "/fake/dir"
    )

    # Verify that topic categorization was NOT called
    mock_categorization.categorize_topics.assert_not_called()

    # Quote extraction and opinion categorization should still be called
    mock_quote_lib.extract_quotes_from_text.assert_called_once()
    mock_categorization.categorize_opinions.assert_called_once()

    # Verify that checkpoints were saved (quotes, learned_opinions, the two
    # pass-1 checkpoints, opinions)
    self.assertEqual(mock_checkpoint_utils.save_checkpoint.call_count, 5)

  @patch("src.sensemaker.categorization")
  @patch("src.sensemaker.quote_extraction_lib")
  @patch("src.sensemaker.checkpoint_utils")
  def test_categorize_statements_forwards_max_concurrent_calls(
      self, mock_checkpoint_utils, mock_quote_lib, mock_categorization
  ):
    """Test that max_concurrent_calls is forwarded to every LLM stage."""
    mock_categorization.categorize_topics = AsyncMock(
        return_value=(self.sample_statements, self.sample_topics)
    )
    mock_quote_lib.extract_quotes_from_text = AsyncMock(
        return_value=self.sample_statements
    )
    mock_categorization.learn_global_opinions = AsyncMock(return_value={})
    mock_categorization.categorize_opinions = AsyncMock(
        return_value=self.sample_statements
    )
    mock_checkpoint_utils.load_checkpoint.return_value = None

    asyncio.run(
        self.sensemaker.categorize_statements(
            statements=self.sample_statements,
            output_dir="/fake/dir",
            max_concurrent_calls=7,
        )
    )

    for stage in (
        mock_categorization.categorize_topics,
        mock_quote_lib.extract_quotes_from_text,
        mock_categorization.learn_global_opinions,
        mock_categorization.categorize_opinions,
    ):
      with self.subTest(stage=stage._mock_name):
        stage.assert_called_once()
        self.assertEqual(stage.call_args.kwargs["max_concurrent_calls"], 7)

  @patch("src.sensemaker.categorization")
  @patch("src.sensemaker.quote_extraction_lib")
  @patch("src.sensemaker.checkpoint_utils")
  def test_categorize_statements_recategorize_other(
      self, mock_checkpoint_utils, mock_quote_lib, mock_categorization
  ):
    """Test secondary pass when recategorize_other is enabled."""
    mock_categorization.categorize_topics = AsyncMock(
        return_value=(self.sample_statements, self.sample_topics)
    )
    mock_quote_lib.extract_quotes_from_text = AsyncMock(
        return_value=self.sample_statements
    )
    mock_categorization.learn_global_opinions = AsyncMock(return_value={})
    mock_categorization.categorize_opinions = AsyncMock(
        return_value=self.sample_statements
    )
    recat_statements = [custom_types.Statement(id="s1", text="recat")]
    recat_opinions = {
        "topic1": custom_types.NestedTopic(name="topic1", subtopics=[])
    }
    mock_categorization.recategorize_other_quotes = AsyncMock(
        return_value=(recat_statements, recat_opinions)
    )
    mock_checkpoint_utils.load_checkpoint.return_value = None

    result = asyncio.run(
        self.sensemaker.categorize_statements(
            statements=self.sample_statements,
            output_dir="/fake/dir",
            recategorize_other=True,
            offer_other_opinion=False,
            offer_other_opinion_on_recategorize=True,
            dedup_new_opinions=True,
            recategorize_other_skip_autoraters=False,
            opinion_matching="supports",
        )
    )

    self.assertEqual(result, recat_statements)
    mock_categorization.learn_global_opinions.assert_called_once()
    self.assertFalse(
        mock_categorization.learn_global_opinions.call_args.kwargs[
            "offer_other_opinion"
        ]
    )
    mock_categorization.categorize_opinions.assert_called_once()
    self.assertFalse(
        mock_categorization.categorize_opinions.call_args.kwargs[
            "offer_other_opinion"
        ]
    )
    self.assertEqual(
        mock_categorization.categorize_opinions.call_args.kwargs[
            "opinion_matching"
        ],
        "supports",
    )
    mock_categorization.recategorize_other_quotes.assert_called_once()
    recat_kwargs = (
        mock_categorization.recategorize_other_quotes.call_args.kwargs
    )
    self.assertTrue(recat_kwargs["offer_other_opinion"])
    self.assertTrue(recat_kwargs["dedup_new_opinions"])
    self.assertTrue(recat_kwargs["run_autoraters"])
    self.assertTrue(recat_kwargs["recategorize_all"])
    self.assertEqual(recat_kwargs["opinion_matching"], "supports")


class _FakeCheckpoints:
  """Dict-backed stand-in for src.checkpoint_utils."""

  def __init__(self, store=None):
    self.store = dict(store or {})
    self.saves = []  # step names, in order

  def load_checkpoint(self, step_name, output_dir):
    del output_dir
    return self.store.get(step_name)

  def save_checkpoint(self, data, step_name, output_dir):
    del output_dir
    self.saves.append(step_name)
    self.store[step_name] = data

  def checkpoint_exists(self, step_name, output_dir):
    del output_dir
    return step_name in self.store


def _quoted_statement(sid, opinion=None):
  topic = (
      custom_types.NestedTopic(
          name="T1", subtopics=[custom_types.FlatTopic(name=opinion)]
      )
      if opinion
      else custom_types.FlatTopic(name="T1")
  )
  return custom_types.Statement(
      id=sid,
      text="t",
      topics=[custom_types.FlatTopic(name="T1")],
      quotes=[custom_types.Quote(id=f"{sid}-T1", text="q", topic=topic)],
  )


class Pass1CheckpointTest(unittest.TestCase):
  """Pass-1 checkpoint preservation and Step 5b resume."""

  def setUp(self):
    # genai_model_test disables logging at import time; assertLogs needs it.
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    self.addCleanup(logging.disable, previous_disable)
    self.sensemaker = Sensemaker(genai_model=MagicMock())
    self.topics = [custom_types.FlatTopic(name="T1")]
    self.input = [_quoted_statement("s1")]
    self.pass1 = [_quoted_statement("s1", "Other")]
    self.pass1_map = {
        "T1": custom_types.NestedTopic(
            name="T1", subtopics=[custom_types.FlatTopic(name="A")]
        )
    }
    self.final = [_quoted_statement("s1", "B")]
    self.final_map = {
        "T1": custom_types.NestedTopic(
            name="T1",
            subtopics=[
                custom_types.FlatTopic(name="A"),
                custom_types.FlatTopic(name="B"),
            ],
        )
    }

  def _run(self, ckpt, mock_cat, **kwargs):
    with patch("src.sensemaker.checkpoint_utils", ckpt):
      return asyncio.run(
          self.sensemaker.categorize_statements(
              statements=self.input,
              topics=self.topics,
              output_dir="/fake",
              **kwargs,
          )
      )

  def _mock_categorization(self, mock_cat, ckpt=None):
    mock_cat.learn_global_opinions = AsyncMock(return_value=self.pass1_map)
    mock_cat.categorize_opinions = AsyncMock(return_value=self.pass1)

    async def _recat(**kwargs):
      if ckpt is not None:
        # Pass-1 checkpoints must already be on disk when 5b starts.
        self.assertIn("statements_with_opinions_pass1", ckpt.store)
        self.assertIn("learned_opinions_pass1", ckpt.store)
      return self.final, self.final_map

    mock_cat.recategorize_other_quotes = AsyncMock(side_effect=_recat)

  @patch("src.sensemaker.categorization")
  def test_full_run_preserves_pass1_checkpoints(self, mock_cat):
    ckpt = _FakeCheckpoints()
    self._mock_categorization(mock_cat, ckpt)
    out = self._run(ckpt, mock_cat, recategorize_other=True)
    self.assertEqual(out, self.final)
    self.assertEqual(ckpt.store["statements_with_opinions_pass1"], self.pass1)
    self.assertEqual(ckpt.store["learned_opinions_pass1"], self.pass1_map)
    self.assertEqual(ckpt.store["statements_with_opinions"], self.final)
    self.assertEqual(ckpt.store["learned_opinions"], self.final_map)
    self.assertEqual(ckpt.saves.count("statements_with_opinions_pass1"), 1)
    self.assertEqual(ckpt.saves.count("learned_opinions_pass1"), 1)
    self.assertLess(
        ckpt.saves.index("learned_opinions_pass1"),
        ckpt.saves.index("statements_with_opinions"),
    )

  @patch("src.sensemaker.categorization")
  def test_resume_runs_5b_from_pass1(self, mock_cat):
    ckpt = _FakeCheckpoints({
        "statements_with_opinions_pass1": self.pass1,
        "learned_opinions_pass1": self.pass1_map,
    })
    self._mock_categorization(mock_cat)
    with self.assertLogs(level="INFO") as logs:
      out = self._run(ckpt, mock_cat, recategorize_other=True)
    self.assertEqual(out, self.final)
    mock_cat.learn_global_opinions.assert_not_called()
    mock_cat.categorize_opinions.assert_not_called()
    kwargs = mock_cat.recategorize_other_quotes.call_args.kwargs
    self.assertEqual(kwargs["statements"], self.pass1)
    self.assertEqual(kwargs["topic_to_opinions_map"], self.pass1_map)
    self.assertEqual(ckpt.store["statements_with_opinions"], self.final)
    self.assertEqual(ckpt.store["learned_opinions"], self.final_map)
    # Pass-1 checkpoints are untouched.
    self.assertNotIn("statements_with_opinions_pass1", ckpt.saves)
    self.assertNotIn("learned_opinions_pass1", ckpt.saves)
    self.assertEqual(ckpt.store["statements_with_opinions_pass1"], self.pass1)
    self.assertTrue(any("RESUME from pass-1" in m for m in logs.output))

  @patch("src.sensemaker.categorization")
  def test_resume_without_5b_uses_pass1_as_final(self, mock_cat):
    ckpt = _FakeCheckpoints({
        "statements_with_opinions_pass1": self.pass1,
        "learned_opinions_pass1": self.pass1_map,
    })
    self._mock_categorization(mock_cat)
    out = self._run(ckpt, mock_cat)
    self.assertEqual(out, self.pass1)
    mock_cat.recategorize_other_quotes.assert_not_called()
    self.assertEqual(ckpt.store["statements_with_opinions"], self.pass1)

  @patch("src.sensemaker.categorization")
  def test_resume_missing_pass1_map_raises(self, mock_cat):
    ckpt = _FakeCheckpoints({"statements_with_opinions_pass1": self.pass1})
    self._mock_categorization(mock_cat)
    with self.assertRaises(RuntimeError):
      self._run(ckpt, mock_cat, recategorize_other=True)

  @patch("src.sensemaker.categorization")
  def test_legacy_final_only_skips_5b_with_warning(self, mock_cat):
    ckpt = _FakeCheckpoints({"statements_with_opinions": self.final})
    self._mock_categorization(mock_cat)
    with self.assertLogs(level="WARNING") as logs:
      out = self._run(ckpt, mock_cat, recategorize_other=True)
    self.assertEqual(out, self.final)
    mock_cat.categorize_opinions.assert_not_called()
    mock_cat.recategorize_other_quotes.assert_not_called()
    self.assertTrue(any("SKIPPED" in m for m in logs.output))
    self.assertEqual(ckpt.saves, [])

  @patch("src.sensemaker.categorization")
  def test_final_and_pass1_present_skips_without_warning(self, mock_cat):
    ckpt = _FakeCheckpoints({
        "statements_with_opinions": self.final,
        "statements_with_opinions_pass1": self.pass1,
        "learned_opinions_pass1": self.pass1_map,
    })
    self._mock_categorization(mock_cat)
    with self.assertLogs(level="INFO") as logs:
      out = self._run(ckpt, mock_cat, recategorize_other=True)
    self.assertEqual(out, self.final)
    mock_cat.recategorize_other_quotes.assert_not_called()
    self.assertFalse(any("WARNING" in m for m in logs.output))
    self.assertTrue(any("already completed" in m for m in logs.output))


if __name__ == "__main__":
  unittest.main()
