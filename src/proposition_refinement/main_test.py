import unittest
import pandas as pd
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import argparse
import pickle
import contextlib

# This is a bit of a hack to allow the test to run from the command line
# and to allow the imports to work correctly.
import sys
import os

sys.path.append(os.getcwd())

from src.proposition_refinement import main as proposition_refinement_main


class PropositionRefinementMainTest(unittest.TestCase):

  def setUp(self):
    """Set up common test data and mocks."""
    self.maxDiff = None
    # Mock for command-line arguments
    self.mock_args = argparse.Namespace(
        propositions_per_opinion=2,
        final_propositions_per_topic=2,
        deduplication_method="rank_filling",
        run_pav_selection=False,
        num_nuanced_propositions=2,
        jury_size=1.0,
        verbose=False,
        output_pkl="test_output.pkl",
        additional_context=None,
    )

    # Sample DataFrame mimicking the structure of world_model['world_model']
    self.sample_by_opinion_df = pd.DataFrame({
        "topic": ["T1", "T1", "T2"],
        "opinion": ["O1", "O2", "O3"],
        "r1_df_length": [10, 5, 8],
        "propositions": [
            pd.DataFrame({"proposition": ["T1O1_P1", "T1O1_P2", "T1O1_P3"]}),
            pd.DataFrame({"proposition": ["T1O2_P1", "T1O2_P2"]}),
            pd.DataFrame({"proposition": ["T2O3_P1", "T2O3_P2", "T2O3_P3"]}),
        ],
        "r1_df": [
            pd.DataFrame({"participant_id": ["r1", "r2"]}),
            pd.DataFrame({"participant_id": ["r3"]}),
            pd.DataFrame({"participant_id": ["r4", "r5"]}),
        ],
        "r2_df": [
            pd.DataFrame({"participant_id": ["r1"], "data": ["d1"]}),
            pd.DataFrame({"participant_id": ["r3"], "data": ["d3"]}),
            pd.DataFrame({"participant_id": ["r4"], "data": ["d4"]}),
        ],
    })

    self.sample_world_model = {
        "world_model": self.sample_by_opinion_df.copy(),
        "simulated_jury_stats": [],
    }

    # Patch file-saving operations to avoid creating artifacts during tests.
    self.pickle_dump_patcher = patch(
        "src.proposition_refinement.main.pickle.dump"
    )
    self.open_patcher = patch("builtins.open")
    self.mock_pickle_dump = self.pickle_dump_patcher.start()
    self.mock_open = self.open_patcher.start()

  def tearDown(self):
    """Stop all patches."""
    self.pickle_dump_patcher.stop()
    self.open_patcher.stop()

  def test_reconstitute_participant_data(self):
    """Tests that participant data from R1 and R2 is merged correctly."""
    jury_pool = proposition_refinement_main.reconstitute_participant_data(
        self.sample_by_opinion_df, self.mock_args
    )
    self.assertIsInstance(jury_pool, pd.DataFrame)
    self.assertEqual(len(jury_pool), 5)  # 5 unique rids
    self.assertIn("data", jury_pool.columns)
    # Check that the outer merge worked by finding a row that only existed in R1
    self.assertTrue(jury_pool[jury_pool["participant_id"] == "r2"]["data"].isnull().all())

  @patch(
      "src.proposition_refinement.main.simulated_jury.run_simulated_jury",
      new_callable=AsyncMock,
  )
  @patch("src.proposition_refinement.main.schulze.get_schulze_ranking")
  def test_run_jury_by_opinion(self, mock_schulze, mock_run_jury):
    """Tests the opinion-level simulated jury stage."""
    # Mock external calls
    mock_run_jury.return_value = (
        pd.DataFrame(
            {"result": [{"ranking": ["P1", "P2"]}, {"ranking": ["P2", "P1"]}]}
        ),
        {},
    )
    mock_schulze.return_value = {"top_propositions": ["P2", "P1"]}

    # Run the stage
    result_wm = asyncio.run(
        proposition_refinement_main.run_jury_by_opinion(
            self.sample_world_model, self.mock_args, pd.DataFrame(), MagicMock()
        )
    )

    # Assertions - expecting only two calls here, since one opinion only has
    # 2 propositions, and we only take the top 2 propositions per opinion.
    self.assertEqual(mock_run_jury.call_count, 2)
    self.assertEqual(mock_schulze.call_count, 2)
    by_opinion_df = result_wm["world_model"]
    self.assertIn("opinion_level_schulze_ranking", by_opinion_df.columns)
    self.assertIn("opinion_level_simulation_results", by_opinion_df.columns)
    self.assertEqual(
        by_opinion_df.loc[0, "opinion_level_schulze_ranking"], ["P2", "P1"]
    )

  @patch(
      "src.proposition_refinement.main.simulated_jury.run_simulated_jury",
      new_callable=AsyncMock,
  )
  @patch("src.proposition_refinement.main.schulze.get_schulze_ranking")
  def test_run_jury_by_topic(self, mock_schulze, mock_run_jury):
    """Tests the topic-level aggregation and jury stage."""
    # Setup: Add the output from the previous stage
    self.sample_by_opinion_df["opinion_level_schulze_ranking"] = [
        ["T1O1_P3", "T1O1_P1", "T1O1_P2"],
        ["T1O2_P2", "T1O2_P1"],
        ["T2O3_P1", "T2O3_P2", "T2O3_P3"],
    ]
    world_model = {
        "world_model": self.sample_by_opinion_df,
        "simulated_jury_stats": [],
    }
    world_model["initial_approval_matrix"] = pd.DataFrame(
        {
            "T1O1_P3": [True],
            "T1O1_P1": [True],
            "T1O1_P2": [False],
            "T1O2_P2": [True],
            "T1O2_P1": [False],
            "T2O3_P1": [True],
            "T2O3_P2": [True],
            "T2O3_P3": [False],
        },
        index=["r1"],
    )

    # Mock external calls
    mock_run_jury.return_value = (
        pd.DataFrame({
            "result": [
                {"ranking": ["T1O1_P3", "T1O2_P2"]},
                {"ranking": ["T1O2_P2", "T1O1_P3"]},
            ]
        }),
        {},
    )
    mock_schulze.return_value = {"top_propositions": ["T1O2_P2", "T1O1_P3"]}

    # Run the stage
    result_wm = asyncio.run(
        proposition_refinement_main.run_jury_by_topic(
            world_model, self.mock_args, pd.DataFrame(), MagicMock()
        )
    )

    # Assertions
    self.assertEqual(mock_run_jury.call_count, 2)  # Once for each topic
    self.assertIn("topic_level_results", result_wm)
    topic_df = result_wm["topic_level_results"]
    self.assertEqual(len(topic_df), 2)
    t1_results = topic_df[topic_df["topic"] == "T1"].iloc[0]
    # Check that it aggregated the top N from each opinion
    self.assertEqual(
        mock_run_jury.call_args_list[0].args[1],
        ["T1O1_P3", "T1O1_P1", "T1O2_P2", "T1O2_P1"],
    )
    self.assertIn("full_schulze_ranking", t1_results)

  @patch(
      "src.proposition_refinement.main.deduplication.run_deduplication",
      new_callable=AsyncMock,
  )
  def test_run_deduplication_stage(self, mock_run_dedup):
    """Tests that the deduplication stage is called correctly."""
    # Setup: Add output from the previous stage
    topic_level_df = pd.DataFrame(
        {"topic": ["T1"], "propositions": [pd.DataFrame()]}
    )
    world_model = {"topic_level_results": topic_level_df}

    # Mock the deduplication function to return a modified DataFrame
    mock_run_dedup.return_value = (
        pd.DataFrame({
            "topic": ["T1"],
            "propositions": [pd.DataFrame({"selected": [True]})],
        }),
        {},
    )

    # Run the stage
    result_wm = asyncio.run(
        proposition_refinement_main.run_deduplication_stage(
            world_model, self.mock_args, MagicMock()
        )
    )

    # Assertions
    mock_run_dedup.assert_called_once()
    self.assertIn(
        "selected", result_wm["topic_level_results"]["propositions"][0].columns
    )

  @patch(
      "src.proposition_refinement.main.nuanced_propositions.combine_propositions",
      new_callable=AsyncMock,
  )
  def test_generate_nuanced_propositions(self, mock_combine_props):
    """Tests the nuanced proposition generation stage."""
    # Setup
    topic_level_df = pd.DataFrame({
        "topic": ["T1"],
        "propositions": [
            pd.DataFrame({
                "proposition": ["p1", "p2"],
                "selected": [True, False],
            })
        ],
    })
    world_model = {"topic_level_results": topic_level_df}
    mock_combine_props.return_value = (
        pd.DataFrame({"result": [["nuanced p1"]]}),
        pd.DataFrame(),
        0.0,
        1.0,
    )

    # Run stage
    result_wm = asyncio.run(
        proposition_refinement_main.generate_nuanced_propositions(
            world_model, self.mock_args, MagicMock()
        )
    )

    # Assertions
    mock_combine_props.assert_called_once_with(
        {"T1": ["p1"]}, model=unittest.mock.ANY, additional_context=None
    )
    self.assertIn("nuanced_propositions", result_wm)
    self.assertEqual(
        result_wm["nuanced_propositions"]["result"][0], ["nuanced p1"]
    )

  @patch(
      "src.proposition_refinement.main.nuanced_propositions.combine_propositions",
      new_callable=AsyncMock,
  )
  def test_generate_nuanced_propositions_with_context(self, mock_combine_props):
    """Tests that additional context is passed to combine_propositions."""
    # Setup
    self.mock_args.additional_context = "test context"
    topic_level_df = pd.DataFrame({
        "topic": ["T1"],
        "propositions": [
            pd.DataFrame({
                "proposition": ["p1"],
                "selected": [True],
            })
        ],
    })
    world_model = {"topic_level_results": topic_level_df}
    mock_combine_props.return_value = (pd.DataFrame(), pd.DataFrame(), 0.0, 1.0)

    # Run stage
    asyncio.run(
        proposition_refinement_main.generate_nuanced_propositions(
            world_model, self.mock_args, MagicMock()
        )
    )

    # Assertions
    mock_combine_props.assert_called_once_with(
        {"T1": ["p1"]},
        model=unittest.mock.ANY,
        additional_context="test context",
    )

  @patch(
      "src.proposition_refinement.main.simulated_jury.run_simulated_jury",
      new_callable=AsyncMock,
  )
  @patch("src.proposition_refinement.main.schulze.get_schulze_ranking")
  def test_run_jury_by_opinion_with_pav(self, mock_schulze, mock_run_jury):
    """
    Tests that the PAV selection logic runs correctly within the
    opinion-level jury stage.
    """
    # 1. Setup args and world model for PAV
    self.mock_args.run_pav_selection = True
    world_model = self.sample_world_model
    world_model["initial_approval_matrix"] = pd.DataFrame(
        {
            "T1O1_P1": [True, True, False, False, False],
            "T1O1_P2": [True, False, False, False, False],
            "T1O1_P3": [False, False, True, False, False],
            "T1O2_P1": [False, False, False, True, False],
            "T1O2_P2": [False, False, False, False, True],
            "T2O3_P1": [True, False, False, True, False],
            "T2O3_P2": [False, True, False, False, True],
            "T2O3_P3": [False, False, True, False, False],
        },
        index=["r1", "r2", "r3", "r4", "r5"],
    )

    # 2. Mock external calls to return different rankings for each opinion group
    async def mock_jury_side_effect(*args, **kwargs):
      opinion = kwargs.get("opinion_name")
      if opinion == "O1":
        return (
            pd.DataFrame({"result": [{"ranking": ["T1O1_P1", "T1O1_P2"]}]}),
            {},
        )
      elif opinion == "O2":
        return (
            pd.DataFrame({"result": [{"ranking": ["T1O2_P2", "T1O2_P1"]}]}),
            {},
        )
      elif opinion == "O3":
        return (
            pd.DataFrame({"result": [{"ranking": ["T2O3_P1", "T2O3_P3"]}]}),
            {},
        )
      return (pd.DataFrame(), {})

    mock_run_jury.side_effect = mock_jury_side_effect

    # Mock Schulze to just return the first ranking it gets
    mock_schulze.side_effect = lambda prefs: {"top_propositions": prefs[0]}

    # 3. Run the stage
    result_wm = asyncio.run(
        proposition_refinement_main.run_jury_by_opinion(
            world_model, self.mock_args, pd.DataFrame(), MagicMock()
        )
    )

    # 4. Assertions
    by_opinion_df = result_wm["world_model"]
    self.assertIn("opinion_level_pav_ranking", by_opinion_df.columns)
    # Check that the ranking was produced for the first opinion group
    self.assertIsInstance(
        by_opinion_df.loc[0, "opinion_level_pav_ranking"], list
    )
    self.assertGreater(
        len(by_opinion_df.loc[0, "opinion_level_pav_ranking"]), 0
    )
    # Check that the Schulze ranking is still there
    self.assertIn("opinion_level_schulze_ranking", by_opinion_df.columns)

  @patch(
      "src.proposition_refinement.main.simulated_jury.run_simulated_jury",
      new_callable=AsyncMock,
  )
  @patch("src.proposition_refinement.main.schulze.get_schulze_ranking")
  def test_run_jury_by_opinion_optimizer(self, mock_schulze, mock_run_jury):
    """
    Tests that the jury is skipped when propositions <= propositions_per_opinion.
    """
    # 1. Setup: world model with:
    # - Multi (5 props): Expect Call
    # - Few (propositions_per_opinion=2): Expect Skip
    # - Single (1 prop): Expect Skip
    if "opinion_level_schulze_ranking" in self.sample_by_opinion_df.columns:
      self.sample_by_opinion_df.drop(
          columns=["opinion_level_schulze_ranking"], inplace=True
      )

    self.mock_args.propositions_per_opinion = 2
    propositions_many = pd.DataFrame(
        {"proposition": [f"P{i}" for i in range(5)]}
    )
    propositions_few = pd.DataFrame({"proposition": ["P1", "P2"]})  # == limit
    propositions_single = pd.DataFrame({"proposition": ["P1"]})  # < limit

    # Row 0: Many (Expect Call)
    # Row 1: Few (Expect Skip)
    # Row 2: Single (Expect Skip)
    self.sample_by_opinion_df.at[0, "propositions"] = propositions_many
    self.sample_by_opinion_df.at[1, "propositions"] = propositions_few
    self.sample_by_opinion_df.at[2, "propositions"] = propositions_single

    world_model = {
        "world_model": self.sample_by_opinion_df,
        "simulated_jury_stats": [],
    }

    # 2. Mock external calls
    mock_run_jury.return_value = (
        pd.DataFrame({"result": [{"ranking": ["P1", "P2"]}]}),
        {},
    )
    mock_schulze.return_value = {"top_propositions": ["P1", "P2"]}

    # 3. Run the stage
    result_wm = asyncio.run(
        proposition_refinement_main.run_jury_by_opinion(
            world_model, self.mock_args, pd.DataFrame(), MagicMock()
        )
    )

    # 4. Assertions
    # Should be called 1 time (only for Row 0), NOT 3
    self.assertEqual(mock_run_jury.call_count, 1)

    by_opinion_df = result_wm["world_model"]

    # For Row 1 (Few), ranking should be NaN/Empty because we skipped it
    # AND we are not programmatically filling it anymore
    val_row_1 = by_opinion_df.loc[1, "opinion_level_schulze_ranking"]
    self.assertTrue(
        isinstance(val_row_1, float)
        and pd.isna(val_row_1)
        or val_row_1 is None
        or len(val_row_1) == 0,
        f"Expected empty/NaN ranking for skipped row, got: {val_row_1}",
    )

  @patch(
      "src.proposition_refinement.main.simulated_jury.run_simulated_jury",
      new_callable=AsyncMock,
  )
  @patch("src.proposition_refinement.main.schulze.get_schulze_ranking")
  def test_run_jury_by_topic_with_missing_rankings(
      self, mock_schulze, mock_run_jury
  ):
    """
    Tests that run_jury_by_topic correctly handles missing opinion-level rankings
    by falling back to raw propositions if they are few enough.
    """
    self.mock_args.propositions_per_opinion = 3

    # Setup:
    # - O1: Has ranking (normal case)
    # - O2: Missing ranking, has 2 props (<= 3, expect inclusion)
    # - O3: Missing ranking, has 4 props (> 3, expect exclusion/warning but no crash)

    propositions_o2 = pd.DataFrame({"proposition": ["O2_P1", "O2_P2"]})
    propositions_o3 = pd.DataFrame(
        {"proposition": ["O3_P1", "O3_P2", "O3_P3", "O3_P4"]}
    )

    self.sample_by_opinion_df["opinion_level_schulze_ranking"] = [
        ["O1_P1", "O1_P2"],  # O1
        None,  # O2 (simulate skip)
        None,  # O3 (simulate skip/error)
    ]
    self.sample_by_opinion_df.at[1, "propositions"] = propositions_o2
    # We remove O3 validation from this specific test to ensure we can verify
    # the fallback logic for O2 without reaching the crash condition.
    # self.sample_by_opinion_df.at[2, "propositions"] = propositions_o3

    # Filter out O3 so we only test the success path here
    self.sample_by_opinion_df = self.sample_by_opinion_df[
        self.sample_by_opinion_df["opinion"] != "O3"
    ]

    world_model = {
        "world_model": self.sample_by_opinion_df,
        "simulated_jury_stats": [],
        "initial_approval_matrix": pd.DataFrame(),
    }

    # Mock jury execution for the topic level
    mock_run_jury.return_value = (
        pd.DataFrame({"result": [{"ranking": ["Winner"]}]}),
        {},
    )
    mock_schulze.return_value = {"top_propositions": ["Winner"]}

    # Run
    result_wm = asyncio.run(
        proposition_refinement_main.run_jury_by_topic(
            world_model, self.mock_args, pd.DataFrame(), MagicMock()
        )
    )

    # Assertions
    # We expect run_simulated_jury to be called for the topics.
    # The key check is: did O2's propositions make it into the candidate pool?
    # Inspect the call args to run_simulated_jury

    # There are 2 topics in sample data: T1 (O1, O2), T2 (O3)
    # T1 call: Should include O1_P1, O1_P2 (from ranking) AND O2_P1, O2_P2 (fallback)
    # T2 call: Should probably be skipped or empty if O3 is dropped (since 4 > 3 and no ranking)

    t1_call_args = None
    for call in mock_run_jury.call_args_list:
      if call.kwargs.get("topic_name") == "T1":
        t1_call_args = call.args[1]  # propositions list
        break

    self.assertIsNotNone(t1_call_args)
    self.assertIn("O2_P1", t1_call_args)
    self.assertIn("O2_P2", t1_call_args)

  @patch(
      "src.proposition_refinement.main.simulated_jury.run_simulated_jury",
      new_callable=AsyncMock,
  )
  @patch("src.proposition_refinement.main.schulze.get_schulze_ranking")
  def test_run_jury_by_topic_raises_error_on_missing_ranking_overflow(
      self, mock_schulze, mock_run_jury
  ):
    """
    Tests that run_jury_by_topic raises ValueError when ranking is missing
    and proposition count exceeds the limit.
    """
    self.mock_args.propositions_per_opinion = 3
    propositions_overflow = pd.DataFrame(
        {"proposition": ["P1", "P2", "P3", "P4"]}
    )

    # Setup a single opinion with missing ranking + too many props
    df = pd.DataFrame({
        "topic": ["T1"],
        "opinion": ["O_Bad"],
        "propositions": [propositions_overflow],
        # Ranking is None/Empty
        "opinion_level_schulze_ranking": [None],
    })

    world_model = {
        "world_model": df,
        "simulated_jury_stats": [],
        "initial_approval_matrix": pd.DataFrame(),
    }

    # Run and expect ValueError
    with self.assertRaisesRegex(ValueError, "No valid ranking for opinion"):
      asyncio.run(
          proposition_refinement_main.run_jury_by_topic(
              world_model, self.mock_args, pd.DataFrame(), MagicMock()
          )
      )

  @contextlib.contextmanager
  def mock_pipeline_stages(self):
    """Result of running simulated jury pipeline stages."""
    with patch(
        "src.proposition_refinement.main.run_r2_opinion_ranking"
    ) as mock_stage_1:
      mock_stage_1.return_value = self.sample_world_model
      with patch(
          "src.proposition_refinement.main.run_initial_approval_jury",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.run_jury_by_opinion",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.run_jury_by_topic",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.run_nuanced_ranking_jury",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.rank_nuanced_propositions",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.generate_nuanced_propositions",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.run_nuanced_approval_jury",
          new_callable=AsyncMock,
      ), patch(
          "src.proposition_refinement.main.genai_model.GenaiModel"
      ):
        yield mock_stage_1

  def test_jury_size_selection(self):
    """
    Tests that jury_size argument correctly selects a subset of participants,
    handling both fractional (0.0 < n < 1.0) and integer (n > 1.0) values.
    """
    # Setup valid mock data.
    large_jury_pool = pd.DataFrame({"participant_id": [f"r{i}" for i in range(100)]})

    # Configure default mock arguments.
    self.mock_args.input_pkl = "dummy.pkl"
    self.mock_args.processed_r2_data = None
    self.mock_args.gemini_api_key = "dummy_key"
    self.mock_args.simulated_jury_model_name = "dummy-model"
    self.mock_args.nuanced_propositions_model_name = "dummy-model"
    self.mock_args.additional_context_file = None
    self.mock_args.approval_batch_size = 15

    # We patch get_jury_pool to return our large pool
    # We also MUST patch pickle.load to return valid data, OR rely on get_jury_pool to return what we want
    # main() calls pickle.load BEFORE get_jury_pool.
    with patch(
        "src.proposition_refinement.main.get_jury_pool"
    ) as mock_get_pool, patch(
        "argparse.ArgumentParser.parse_args", return_value=self.mock_args
    ), patch(
        "src.proposition_refinement.main.pickle.load",
        return_value=self.sample_world_model,
    ):
      mock_get_pool.return_value = (large_jury_pool, self.sample_world_model)

      # Verify fractional sampling.
      self.mock_args.jury_size = 0.5
      # Mock pipeline stages to focus on sampling logic.
      with self.mock_pipeline_stages() as mock_stage_1:
        # Run main
        asyncio.run(proposition_refinement_main.main())

        # Verify sampling output size.
        args, _ = mock_stage_1.call_args
        # args[0] is world_model, args[1] is args, args[2] is jury_pool_df
        passed_jury_pool = args[2]
        self.assertEqual(len(passed_jury_pool), 50)

      # Verify integer sampling.
      self.mock_args.jury_size = 10.0
      with self.mock_pipeline_stages() as mock_stage_1:
        asyncio.run(proposition_refinement_main.main())
        args, _ = mock_stage_1.call_args
        passed_jury_pool = args[2]
        self.assertEqual(len(passed_jury_pool), 10)

      # Verify full dataset usage (no sampling).
      self.mock_args.jury_size = 1.0
      with self.mock_pipeline_stages() as mock_stage_1:
        asyncio.run(proposition_refinement_main.main())
        args, _ = mock_stage_1.call_args
        passed_jury_pool = args[2]
        self.assertEqual(len(passed_jury_pool), 100)


if __name__ == "__main__":
  unittest.main()
