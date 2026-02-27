import asyncio
import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.propositions import proposition_generator, prompts_util


class WorldModelBuilderTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_dir = self.temp_dir.name
    self.r1_file = os.path.join(self.output_dir, "r1.csv")
    self.r2_file = os.path.join(self.output_dir, "r2.csv")

    # Create dummy R1 data
    self.r1_data = {
        "participant_id": [1, 2, 3, 4, 5, 6],
        "topic": [
            "Topic A",
            "Topic A",
            "Topic B",
            "Topic B",
            "Topic C",
            "Other",
        ],
        "opinion": [
            "Opinion A1",
            "Opinion A2",
            "Opinion B1",
            "Other",
            "Opinion C1",
            "Some other opinion",
        ],
        "quote": [
            "text a1",
            "text a2",
            "text b1",
            "text b2",
            "text c1",
            "text other",
        ],
    }
    self.df_r1 = pd.DataFrame(self.r1_data)

    # Create dummy R2 data that mimics the structure of pilot_r2_processed.csv
    self.r2_data = {
        "participant_id": [1, 2, 3],
        "ranking_1_topic": ["Topic A", "Topic A", "Topic A"],
        "ranking_1_q_1": ["Opinion A1", "Opinion A1", "Opinion A1"],
        "ranking_1_a_1": [1, 2, 1],
        "ranking_1_q_2": ["Opinion A2", "Opinion A2", "Opinion A2"],
        "ranking_1_a_2": [2, 1, 2],
        "ranking_1_q_3": ["Opinion A3", "Opinion A3", "Opinion A3"],
        "ranking_1_a_3": [3, 3, 3],
        "ranking_1_q_4": ["Free text", "Free text", "Free text"],
        "ranking_1_a_4": ["reason 1", "reason 2", "reason 3"],
        "ranking_2_topic": ["Topic B", "Topic B", "Topic B"],
        "ranking_2_q_1": ["Opinion B1", "Opinion B1", "Opinion B1"],
        "ranking_2_a_1": [1, 1, 1],
        "ranking_2_q_2": ["Opinion B2", "Opinion B2", "Opinion B2"],
        "ranking_2_a_2": [2, 2, 2],
        "ranking_2_q_3": ["Opinion B3", "Opinion B3", "Opinion B3"],
        "ranking_2_a_3": [3, 3, 3],
        "ranking_2_q_4": ["Free text", "Free text", "Free text"],
        "ranking_2_a_4": ["reason b1", "reason b2", "reason b3"],
        "question_1_topic": ["Topic A", "Topic A", "Topic A"],
        "question_1_opinion": ["Opinion A1", "Opinion A1", "Opinion A1"],
        "question_1": [
            "Some question text for opinion A1",
            "Some question text for opinion A1",
            "Some question text for opinion A1",
        ],
        "answer_1": ["agree", "disagree", "agree"],
        "answer_1_agrees": [True, False, True],
        "question_2_topic": ["Topic B", "Topic B", "Topic B"],
        "question_2_opinion": ["Opinion B1", "Opinion B1", "Opinion B1"],
        "question_2": [
            "Some question text for opinion B1",
            "Some question text for opinion B1",
            "Some question text for opinion B1",
        ],
        "answer_2": ["agree", "agree", "agree"],
        "answer_2_agrees": [True, True, True],
    }
    self.df_r2 = pd.DataFrame(self.r2_data)
    self.mock_r1_df_content = pd.DataFrame({
        "participant_id": [1],
        "topic": ["Topic A"],
        "opinion": ["Opinion A1"],
        "quote": ["text a1"],
    })

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_get_r2_data_by_opinion_found(self):
    """Tests that data is correctly extracted when the opinion exists."""
    result_df = asyncio.run(
        proposition_generator._get_r2_data_by_opinion(self.df_r2, "Opinion B1")
    )
    self.assertFalse(result_df.empty)
    self.assertIn("question_2", result_df.columns)
    self.assertIn("answer_2_agrees", result_df.columns)
    self.assertEqual(len(result_df), 3)

  def test_get_r2_data_by_opinion_not_found(self):
    """Tests that an empty DataFrame is returned for a non-existent opinion."""
    result_df = asyncio.run(
        proposition_generator._get_r2_data_by_opinion(self.df_r2, "Opinion Z")
    )
    self.assertTrue(result_df.empty)

  def test_get_r2_data_by_functions_without_agrees_column(self):
    """Tests that the _get_r2_data_by functions run without the _agrees column."""
    df_r2_no_agrees = self.df_r2.drop(
        columns=["answer_1_agrees", "answer_2_agrees"]
    )

    result_df_opinion = asyncio.run(
        proposition_generator._get_r2_data_by_opinion(
            df_r2_no_agrees, "Opinion B1"
        )
    )
    self.assertFalse(result_df_opinion.empty)
    self.assertNotIn("answer_2_agrees", result_df_opinion.columns)

  def test_get_r2_data_by_opinion_filters_empty_answers(self):
    """Tests that rows with all empty answers are filtered out for an opinion."""
    df_r2_with_empty = self.df_r2.copy()
    new_row_data = {
        "participant_id": [4],
        "ranking_1_topic": ["Topic A"],
        "ranking_1_q_1": ["Opinion A1"],
        "ranking_1_a_1": [1],
        "ranking_1_q_2": ["Opinion A2"],
        "ranking_1_a_2": [2],
        "ranking_1_q_3": ["Opinion A3"],
        "ranking_1_a_3": [3],
        "ranking_1_q_4": ["Free text"],
        "ranking_1_a_4": ["reason 4"],
        "ranking_2_topic": ["Topic B"],
        "ranking_2_q_1": ["Opinion B1"],
        "ranking_2_a_1": [1],
        "ranking_2_q_2": ["Opinion B2"],
        "ranking_2_a_2": [2],
        "ranking_2_q_3": ["Opinion B3"],
        "ranking_2_a_3": [3],
        "ranking_2_q_4": ["Free text"],
        "ranking_2_a_4": ["reason b4"],
        "question_1_topic": ["Topic A"],
        "question_1_opinion": ["Opinion A1"],
        "question_1": ["Some question text for opinion A1"],
        "answer_1": [None],
        "question_2_topic": ["Topic B"],
        "question_2_opinion": ["Opinion B1"],
        "question_2": ["Some question text for opinion B1"],
        "answer_2": ["agree"],
    }
    new_row = pd.DataFrame(new_row_data)
    df_r2_with_empty = pd.concat([df_r2_with_empty, new_row], ignore_index=True)

    # For Opinion A1, the new row (rid=4) has answer_1 as None, so it should be filtered out.
    result_df_A1 = asyncio.run(
        proposition_generator._get_r2_data_by_opinion(
            df_r2_with_empty, "Opinion A1"
        )
    )
    self.assertEqual(len(result_df_A1), 3)
    self.assertNotIn(4, result_df_A1["participant_id"].values)

    # For Opinion B1, the new row (rid=4) has a valid answer_2, so it should be included.
    result_df_B1 = asyncio.run(
        proposition_generator._get_r2_data_by_opinion(
            df_r2_with_empty, "Opinion B1"
        )
    )
    self.assertEqual(len(result_df_B1), 4)
    self.assertIn(4, result_df_B1["participant_id"].values)

  def test_analyze_and_allocate_by_opinion_weighted(self):
    """Tests weighted allocation by opinion."""
    result_df = asyncio.run(
        proposition_generator.analyze_and_allocate_by_opinion(
            self.df_r1, self.df_r2, "topic", "opinion", 10, False
        )
    )
    self.assertEqual(result_df["allocations"].sum(), 10)

  def test_analyze_and_allocate_by_opinion_uniform(self):
    """Tests uniform allocation by opinion."""
    result_df = asyncio.run(
        proposition_generator.analyze_and_allocate_by_opinion(
            self.df_r1, self.df_r2, "topic", "opinion", 5, True
        )
    )
    self.assertTrue((result_df["allocations"] == 5).all())
    self.assertEqual(len(result_df), 5)

  def test_analyze_and_allocate_by_opinion_missing_column(self):
    """Tests that a ValueError is raised if the opinion column is missing."""
    with self.assertRaises(ValueError):
      asyncio.run(
          proposition_generator.analyze_and_allocate_by_opinion(
              self.df_r1, self.df_r2, "topic", "non_existent_col", 5, True
          )
      )

  def _setup_main_mocks(
      self,
      mock_genai_model,
      mock_analyze,
      with_reasoning=True,
  ):
    """Helper function to set up common mocks for main tests."""
    mock_analyze.return_value = pd.DataFrame({
        "topic": ["Topic A", "Topic A", "Topic B", "Topic C", "Other"],
        "opinion": [
            "Opinion A1",
            "Opinion A2",
            "Opinion B1",
            "Opinion C1",
            "Some other opinion",
        ],
        "r1_df": [self.mock_r1_df_content] * 5,
        "r2_df": [pd.DataFrame()] * 5,
        "allocations": [5] * 5,
    })

    mock_model_instance = mock.MagicMock()

    async def mock_async_function(*args, **kwargs):
      # args[0] is all_prompts
      # args[1] is the parser function
      parser = args[1]
      if parser == prompts_util.parse_proposition_response_json_reasoning:
        propositions = [
            pd.DataFrame(
                {"proposition": [f"prop {i}"], "reasoning": [f"reason {i}"]}
            )
            for i in range(len(args[0]))
        ]
      else:
        propositions = [
            pd.DataFrame({"proposition": [f"prop {i}"]})
            for i in range(len(args[0]))
        ]

      response_df = pd.DataFrame({
          "topic": mock_analyze.return_value["topic"],
          "opinion": mock_analyze.return_value["opinion"],
          "propositions": propositions,
          "total_token_used": [100] * len(args[0]),
          "prompt_token_count": [90] * len(args[0]),
          "candidates_token_count": [1] * len(args[0]),
          "tool_use_prompt_token_count": [7] * len(args[0]),
          "thoughts_token_count": [2] * len(args[0]),
      })
      stats_df = pd.DataFrame({"combined_tokens": [100 * len(args[0])]})
      return (response_df, stats_df)

    mock_model_instance.process_prompts_concurrently.side_effect = (
        mock_async_function
    )
    mock_model_instance.calculate_token_count_needed.return_value = 100
    mock_genai_model.return_value = mock_model_instance
    return mock_model_instance

  @mock.patch(
      "src.propositions.proposition_generator.analyze_and_allocate_by_opinion"
  )
  @mock.patch("src.propositions.proposition_generator.genai_model.GenaiModel")
  @mock.patch("src.propositions.world_model_util.save_dataframe_to_pickle")
  @mock.patch("src.propositions.world_model_util.save_propositions_as_csv")
  @mock.patch("src.propositions.world_model_util.read_csv_to_dataframe")
  @mock.patch(
      "src.propositions.input_csv_validation.is_r1_df_missing_required_column",
      return_value=None,
  )
  @mock.patch(
      "src.propositions.input_csv_validation.is_r2_df_missing_required_column",
      return_value=None,
  )
  def test_main_split_by_opinion_no_reasoning(
      self,
      mock_is_r2_valid,
      mock_is_r1_valid,
      mock_read_csv,
      mock_save_csv,
      mock_save_pickle,
      mock_genai_model,
      mock_analyze_opinion,
  ):
    """Tests the main function with split_by='opinion' and no reasoning."""
    mock_read_csv.side_effect = [self.df_r1, self.df_r2]
    mock_model_instance = self._setup_main_mocks(
        mock_genai_model,
        mock_analyze_opinion,
        with_reasoning=False,
    )

    test_args = [
        "proposition_generator.py",
        "--r1_input_file",
        self.r1_file,
        "--r2_input_file",
        self.r2_file,
        "--output_dir",
        self.output_dir,
        "--gemini_api_key",
        "test_key",
    ]
    with mock.patch.object(sys, "argv", test_args):
      asyncio.run(proposition_generator.main())

    self.assertEqual(mock_read_csv.call_count, 2)
    mock_analyze_opinion.assert_called_once()
    mock_genai_model.assert_called_once_with(
        api_key="test_key", model_name="gemini-2.5-pro"
    )
    self.assertEqual(
        mock_model_instance.calculate_token_count_needed.call_count, 5
    )
    mock_model_instance.process_prompts_concurrently.assert_called_once()
    # Check that the correct parser was passed
    parser_arg = mock_model_instance.process_prompts_concurrently.call_args[0][
        1
    ]
    self.assertEqual(parser_arg, prompts_util.parse_proposition_response_json)
    mock_save_pickle.assert_called_once()
    mock_save_csv.assert_called_with(
        df=mock.ANY,
        file_path=mock.ANY,
        reasoning=False,
        has_eval_data=False,
    )
    final_df = mock_save_pickle.call_args[0][0]
    self.assertNotIn("reasoning", final_df["propositions"].iloc[0].columns)
    # Check that "Opinion A1" was added to the propositions for that row
    opinion_a1_props = final_df[final_df["opinion"] == "Opinion A1"][
        "propositions"
    ].iloc[0]
    self.assertIn("Opinion A1", opinion_a1_props["proposition"].values)

  @mock.patch(
      "src.propositions.proposition_generator.analyze_and_allocate_by_opinion"
  )
  @mock.patch("src.propositions.proposition_generator.genai_model.GenaiModel")
  @mock.patch("src.propositions.world_model_util.save_dataframe_to_pickle")
  @mock.patch("src.propositions.world_model_util.save_propositions_as_csv")
  @mock.patch("src.propositions.world_model_util.read_csv_to_dataframe")
  @mock.patch(
      "src.propositions.input_csv_validation.is_r1_df_missing_required_column",
      return_value=None,
  )
  @mock.patch(
      "src.propositions.input_csv_validation.is_r2_df_missing_required_column",
      return_value=None,
  )
  def test_main_no_r2_file(
      self,
      mock_is_r2_valid,
      mock_is_r1_valid,
      mock_read_csv,
      mock_save_csv,
      mock_save_pickle,
      mock_genai_model,
      mock_analyze_opinion,
  ):
    """Tests the main function when no r2 file is provided."""
    mock_read_csv.return_value = self.df_r1
    self._setup_main_mocks(mock_genai_model, mock_analyze_opinion)

    test_args = [
        "proposition_generator.py",
        "--r1_input_file",
        self.r1_file,
        "--output_dir",
        self.output_dir,
        "--gemini_api_key",
        "test_key",
        "--reasoning",
    ]
    with mock.patch.object(sys, "argv", test_args):
      asyncio.run(proposition_generator.main())

    mock_read_csv.assert_called_once_with(self.r1_file)
    mock_is_r1_valid.assert_called_once()
    mock_is_r2_valid.assert_not_called()
    mock_analyze_opinion.assert_called_once()
    mock_save_pickle.assert_called_once()

  @mock.patch(
      "src.propositions.proposition_generator.analyze_and_allocate_by_opinion"
  )
  @mock.patch("src.propositions.proposition_generator.genai_model.GenaiModel")
  @mock.patch("src.propositions.world_model_util.save_dataframe_to_pickle")
  @mock.patch("src.propositions.world_model_util.save_propositions_as_csv")
  @mock.patch("src.propositions.world_model_util.read_csv_to_dataframe")
  @mock.patch(
      "src.propositions.input_csv_validation.is_r1_df_missing_required_column",
      return_value=None,
  )
  @mock.patch(
      "src.propositions.input_csv_validation.is_r2_df_missing_required_column",
      return_value=None,
  )
  def test_main_optimization_prop_count_1(
      self,
      mock_is_r2_valid,
      mock_is_r1_valid,
      mock_read_csv,
      mock_save_csv,
      mock_save_pickle,
      mock_genai_model,
      mock_analyze_opinion,
  ):
    """Tests the optimization when prop_count=1 and include_opinion=True."""
    mock_read_csv.return_value = self.df_r1
    mock_model_instance = self._setup_main_mocks(
        mock_genai_model, mock_analyze_opinion
    )

    test_args = [
        "proposition_generator.py",
        "--r1_input_file",
        self.r1_file,
        "--output_dir",
        self.output_dir,
        "--gemini_api_key",
        "test_key",
        "--prop_count=1",
        "--include_opinion",  # Default is True, but being explicit
    ]
    with mock.patch.object(sys, "argv", test_args):
      asyncio.run(proposition_generator.main())

    # Verify model was NOT called
    mock_model_instance.process_prompts_concurrently.assert_not_called()

    # Verify results still saved (programmatically generated)
    mock_save_pickle.assert_called_once()
    final_df = mock_save_pickle.call_args[0][0]

    # Verify original opinion is in the output
    opinion_val = "Opinion A1"
    opinion_props = final_df[final_df["opinion"] == opinion_val][
        "propositions"
    ].iloc[0]
    self.assertIn(opinion_val, opinion_props["proposition"].values)

  @mock.patch(
      "src.propositions.proposition_generator.analyze_and_allocate_by_opinion"
  )
  @mock.patch("src.propositions.proposition_generator.genai_model.GenaiModel")
  @mock.patch("src.propositions.world_model_util.save_dataframe_to_pickle")
  @mock.patch("src.propositions.world_model_util.save_propositions_as_csv")
  @mock.patch("src.propositions.world_model_util.read_csv_to_dataframe")
  @mock.patch(
      "src.propositions.input_csv_validation.is_r1_df_missing_required_column",
      return_value=None,
  )
  @mock.patch(
      "src.propositions.input_csv_validation.is_r2_df_missing_required_column",
      return_value=None,
  )
  def test_main_include_opinion_false(
      self,
      mock_is_r2_valid,
      mock_is_r1_valid,
      mock_read_csv,
      mock_save_csv,
      mock_save_pickle,
      mock_genai_model,
      mock_analyze_opinion,
  ):
    """Tests that opinion is NOT enforced when include_opinion=False."""
    mock_read_csv.return_value = self.df_r1
    mock_model_instance = self._setup_main_mocks(
        mock_genai_model, mock_analyze_opinion
    )

    # Setup the mock to return propositions that DO NOT include the original opinion
    # _setup_main_mocks puts "prop 0" .. into results, which won't match "Opinion A1" etc.
    # So we don't need to change the mock return value, just assert on it.

    test_args = [
        "proposition_generator.py",
        "--r1_input_file",
        self.r1_file,
        "--output_dir",
        self.output_dir,
        "--gemini_api_key",
        "test_key",
        "--no-include_opinion",
    ]
    with mock.patch.object(sys, "argv", test_args):
      asyncio.run(proposition_generator.main())

    # Verify model WAS called
    mock_model_instance.process_prompts_concurrently.assert_called_once()

    mock_save_pickle.assert_called_once()
    final_df = mock_save_pickle.call_args[0][0]

    # Verify original opinion (Opinion A1) was NOT added to the propositions
    opinion_val = "Opinion A1"
    opinion_props = final_df[final_df["opinion"] == opinion_val][
        "propositions"
    ].iloc[0]
    self.assertNotIn(opinion_val, opinion_props["proposition"].values)


if __name__ == "__main__":
  unittest.main()
