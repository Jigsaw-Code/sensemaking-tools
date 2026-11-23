"""Tests for src.simulated_jury.main."""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.models import fake_base_model
from src.simulated_jury import main
from src.simulated_jury import simulated_jury
from src.social_choice import bridging


class ArgParsingAndConfigTest(unittest.TestCase):
  """Tests non-trivial CLI argument parsing and config normalization."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)

  def test_config_from_args_resolves_batch_size_context_and_bridging(self):
    context_file = os.path.join(self.temp_dir.name, "context.txt")
    with open(context_file, "w") as context_fp:
      context_fp.write("  Evaluate as municipal proposals.  \n")

    args = main.build_arg_parser().parse_args([
        "--participants_csv",
        "participants.csv",
        "--statements_csv",
        "statements.csv",
        "--output_csv",
        "output.csv",
        "--batch_size",
        "7",
        "--max_concurrent_calls",
        "42",
        "--additional_context_file",
        context_file,
        "--segment_columns",
        "seg_a,seg_b",
        "--bridging_metric",
        "gic_geomean",
        "--min_segment_size",
        "5",
        "--percent",
    ])
    config = main.config_from_args(args)

    self.assertEqual(config.approval_batch_size, 7)
    self.assertEqual(config.max_concurrent_calls, 42)
    self.assertEqual(
        config.additional_context, "Evaluate as municipal proposals."
    )
    self.assertEqual(
        config.bridging_config,
        bridging.BridgingConfig(
            segment_column=None,
            segment_columns=("seg_a", "seg_b"),
            metric="gic_geomean",
            min_segment_size=5,
        ),
    )
    self.assertTrue(config.percent)


class RunAnalysisTest(unittest.IsolatedAsyncioTestCase):
  """Tests `main.run_analysis` and `main.run_from_csv` with `FakeBaseModel`."""

  def setUp(self):
    super().setUp()
    self.participants_df = pd.DataFrame({"participant_id": ["p1", "p2"]})
    self.statements_df = pd.DataFrame({"statement": ["A", "B"]})
    self.fake_model = fake_base_model.FakeBaseModel()

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_approval(self, mock_run):
    mock_run.return_value = (
        pd.DataFrame([
            {
                "data_row": {"participant_id": "p1"},
                "result": {"A": True, "B": False},
            },
            {
                "data_row": {"participant_id": "p2"},
                "result": {"A": True, "B": True},
            },
        ]),
        {},
    )

    df = await main.run_analysis(
        self.participants_df,
        self.statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(percent=True),
    )

    _, kwargs = mock_run.call_args
    self.assertIs(kwargs["model"], self.fake_model)
    self.assertIsNone(kwargs["max_concurrent_calls"])

    self.assertEqual(df.shape[0], 2)
    self.assertIn("agree_rate", df.columns)
    self.assertNotIn("agree_rate", self.statements_df.columns)

    row_a = df[df["statement"] == "A"].iloc[0]
    row_b = df[df["statement"] == "B"].iloc[0]
    self.assertEqual(row_a["agree_rate"], "100.0%")
    self.assertEqual(row_b["agree_rate"], "50.0%")

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_ranking(self, mock_run):
    statements_df = pd.DataFrame(
        {"statement": ["A", "B"], "topic": ["t1", "t1"]}
    )
    mock_run.return_value = (
        pd.DataFrame([
            {
                "data_row": {"participant_id": "p1"},
                "result": {"ranking": ["A", "B"]},
            },
            {
                "data_row": {"participant_id": "p2"},
                "result": {"ranking": ["A", "B"]},
            },
        ]),
        {},
    )

    df = await main.run_analysis(
        self.participants_df,
        statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(
            voting_mode="ranking", group_by="topic"
        ),
    )

    _, kwargs = mock_run.call_args
    self.assertIsNone(kwargs["max_concurrent_calls"])

    self.assertIn("schulze_rank", df.columns)
    self.assertNotIn("pav_rank", df.columns)
    self.assertNotIn("agree_rate", df.columns)

    row_a = df[df["statement"] == "A"].iloc[0]
    row_b = df[df["statement"] == "B"].iloc[0]
    self.assertEqual(row_a["schulze_rank"], 1.0)
    self.assertEqual(row_b["schulze_rank"], 2.0)

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_schulze_pav(self, mock_run):
    statements_df = pd.DataFrame(
        {"statement": ["A", "B"], "topic": ["t1", "t1"]}
    )

    def mock_run_side_effect(*args, **kwargs):
      del args
      voting_mode = kwargs.get("voting_mode")
      if voting_mode == simulated_jury.VotingMode.APPROVAL:
        return (
            pd.DataFrame([
                {
                    "data_row": {"participant_id": "p1"},
                    "result": {"A": True, "B": False},
                },
                {
                    "data_row": {"participant_id": "p2"},
                    "result": {"A": True, "B": True},
                },
            ]),
            {},
        )
      return (
          pd.DataFrame([
              {
                  "data_row": {"participant_id": "p1"},
                  "result": {"ranking": ["A", "B"]},
              },
              {
                  "data_row": {"participant_id": "p2"},
                  "result": {"ranking": ["A", "B"]},
              },
          ]),
          {},
      )

    mock_run.side_effect = mock_run_side_effect

    df = await main.run_analysis(
        self.participants_df,
        statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(
            voting_mode="schulze_pav", group_by="topic"
        ),
    )

    self.assertIn("schulze_rank", df.columns)
    self.assertIn("pav_rank", df.columns)
    self.assertIn("agree_rate", df.columns)

    row_a = df[df["statement"] == "A"].iloc[0]
    self.assertEqual(row_a["schulze_rank"], 1.0)
    self.assertEqual(row_a["pav_rank"], 1.0)

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_forwards_concurrency_and_context(self, mock_run):
    mock_run.return_value = (
        pd.DataFrame([
            {
                "data_row": {"participant_id": "p1"},
                "result": {"A": True, "B": False},
            },
            {
                "data_row": {"participant_id": "p2"},
                "result": {"A": True, "B": True},
            },
        ]),
        {},
    )

    await main.run_analysis(
        self.participants_df,
        self.statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(
            max_concurrent_calls=42,
            additional_context="Evaluate as municipal proposals.",
        ),
    )

    _, kwargs = mock_run.call_args
    self.assertEqual(kwargs["max_concurrent_calls"], 42)
    self.assertEqual(
        kwargs["additional_context"], "Evaluate as municipal proposals."
    )

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_from_csv_preserves_none_in_participants_csv(
      self, mock_run
  ):
    with tempfile.TemporaryDirectory() as temp_dir:
      participants_csv = os.path.join(temp_dir, "participants.csv")
      statements_csv = os.path.join(temp_dir, "statements.csv")
      output_csv = os.path.join(temp_dir, "output.csv")
      with open(participants_csv, "w") as f:
        f.write("participant_id,survey_text\np1,None\np2,N/A\n")
      self.statements_df.to_csv(statements_csv, index=False)

      mock_run.return_value = (
          pd.DataFrame([
              {
                  "data_row": {"participant_id": "p1"},
                  "result": {"A": True, "B": False},
              },
              {
                  "data_row": {"participant_id": "p2"},
                  "result": {"A": True, "B": True},
              },
          ]),
          {},
      )

      await main.run_from_csv(
          participants_csv,
          statements_csv,
          output_csv,
          model=self.fake_model,
          config=main.SimulatedJuryConfig(),
      )

      _, kwargs = mock_run.call_args
      passed_df = kwargs["participants_df"]
      self.assertEqual(passed_df["survey_text"].tolist(), ["None", "N/A"])
      self.assertTrue(os.path.exists(output_csv))

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_approval_with_segment_column(self, mock_run):
    participants_df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4"],
        "region": ["North", "North", "South", "South"],
    })

    # Statement A: North=100%, South=0% -> overall=0.5, minmax=0.0, rank=2.
    # Statement B: North=50%, South=50% -> overall=0.5, minmax=0.5, rank=1.
    mock_run.return_value = (
        pd.DataFrame([
            {
                "data_row": {"participant_id": "p1"},
                "result": {"A": True, "B": True},
            },
            {
                "data_row": {"participant_id": "p2"},
                "result": {"A": True, "B": False},
            },
            {
                "data_row": {"participant_id": "p3"},
                "result": {"A": False, "B": True},
            },
            {
                "data_row": {"participant_id": "p4"},
                "result": {"A": False, "B": False},
            },
        ]),
        {},
    )

    df = await main.run_analysis(
        participants_df,
        self.statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(
            bridging_config=bridging.BridgingConfig(
                segment_column="region",
                segment_columns=None,
                metric=bridging.DEFAULT_BRIDGING_METRIC,
                min_segment_size=1,
            )
        ),
    )

    row_a = df[df["statement"] == "A"].iloc[0]
    row_b = df[df["statement"] == "B"].iloc[0]
    self.assertAlmostEqual(row_a["minmax_approval"], 0.0)
    self.assertEqual(row_a["bottleneck_segment"], "South")
    self.assertAlmostEqual(row_a["gic_product"], 0.0)
    self.assertAlmostEqual(row_a["gic_geomean"], 0.0)
    self.assertEqual(row_a["bridging_rank"], 2)

    self.assertAlmostEqual(row_b["minmax_approval"], 0.5)
    self.assertEqual(row_b["bottleneck_segment"], "North")
    self.assertAlmostEqual(row_b["gic_product"], 0.25)
    self.assertAlmostEqual(row_b["gic_geomean"], 0.5)
    self.assertEqual(row_b["bridging_rank"], 1)

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_approval_with_overlapping_segments_and_percent(
      self, mock_run
  ):
    participants_df = pd.DataFrame({
        "participant_id": ["p1", "p2", "p3", "p4"],
        "seg_a": [1, 1, 0, 0],
        "seg_b": [1, 0, 1, 0],
    })

    # Statement A: seg_a=1.0 (p1,p2), seg_b=0.5 (p1=1,p3=0) -> minmax=0.5.
    # Statement B: seg_a=0.5 (p1=1,p2=0), seg_b=1.0 (p1,p3) -> minmax=0.5.
    # Overall: A=0.5 (2/4), B=0.75 (3/4) -> B wins tie-break on gic_geomean.
    mock_run.return_value = (
        pd.DataFrame([
            {
                "data_row": {"participant_id": "p1"},
                "result": {"A": True, "B": True},
            },
            {
                "data_row": {"participant_id": "p2"},
                "result": {"A": True, "B": False},
            },
            {
                "data_row": {"participant_id": "p3"},
                "result": {"A": False, "B": True},
            },
            {
                "data_row": {"participant_id": "p4"},
                "result": {"A": False, "B": True},
            },
        ]),
        {},
    )

    df = await main.run_analysis(
        participants_df,
        self.statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(
            bridging_config=bridging.BridgingConfig(
                segment_column=None,
                segment_columns=("seg_a", "seg_b"),
                metric="gic_geomean",
                min_segment_size=1,
            ),
            percent=True,
        ),
    )

    row_a = df[df["statement"] == "A"].iloc[0]
    row_b = df[df["statement"] == "B"].iloc[0]
    self.assertEqual(row_a["minmax_approval"], "50.0%")
    self.assertEqual(row_a["gic_geomean"], "70.7%")
    self.assertAlmostEqual(row_a["gic_product"], 0.5)
    self.assertEqual(row_a["bottleneck_segment"], "seg_b")
    self.assertEqual(row_a["bridging_rank"], 2)

    self.assertEqual(row_b["minmax_approval"], "50.0%")
    self.assertEqual(row_b["gic_geomean"], "70.7%")
    self.assertAlmostEqual(row_b["gic_product"], 0.5)
    self.assertEqual(row_b["bottleneck_segment"], "seg_a")
    self.assertEqual(row_b["bridging_rank"], 1)

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_schulze_pav_with_group_by_and_bridging(
      self, mock_run
  ):
    participants_df = pd.DataFrame({
        "participant_id": ["p1", "p2"],
        "seg_a": [1, 0],
        "seg_b": [0, 1],
    })
    statements_df = pd.DataFrame({
        "statement": ["A", "B", "C", "D"],
        "topic": ["t1", "t1", "t2", "t2"],
    })

    def mock_run_side_effect(*args, **kwargs):
      del args
      voting_mode = kwargs.get("voting_mode")
      if voting_mode == simulated_jury.VotingMode.APPROVAL:
        return (
            pd.DataFrame([
                {
                    "data_row": {"participant_id": "p1"},
                    "result": {"A": True, "B": True, "C": False, "D": True},
                },
                {
                    "data_row": {"participant_id": "p2"},
                    "result": {"A": False, "B": True, "C": True, "D": True},
                },
            ]),
            {},
        )
      topic_name = kwargs.get("topic_name")
      ranking = ["A", "B"] if topic_name == "t1" else ["C", "D"]
      return (
          pd.DataFrame([
              {
                  "data_row": {"participant_id": "p1"},
                  "result": {"ranking": ranking},
              },
              {
                  "data_row": {"participant_id": "p2"},
                  "result": {"ranking": ranking},
              },
          ]),
          {},
      )

    mock_run.side_effect = mock_run_side_effect

    df = await main.run_analysis(
        participants_df,
        statements_df,
        model=self.fake_model,
        config=main.SimulatedJuryConfig(
            voting_mode="schulze_pav",
            group_by="topic",
            bridging_config=bridging.BridgingConfig(
                segment_column=None,
                segment_columns=("seg_a", "seg_b"),
                metric=bridging.DEFAULT_BRIDGING_METRIC,
                min_segment_size=1,
            ),
        ),
    )

    by_stmt = df.set_index("statement")
    # Within t1, B bridges both segments (rank 1) while A has minmax=0 (rank 2).
    self.assertEqual(by_stmt.loc["B", "bridging_rank"], 1)
    self.assertEqual(by_stmt.loc["A", "bridging_rank"], 2)
    # Within t2, D bridges both segments (rank 1) while C has minmax=0 (rank 2).
    self.assertEqual(by_stmt.loc["D", "bridging_rank"], 1)
    self.assertEqual(by_stmt.loc["C", "bridging_rank"], 2)
    self.assertIn("schulze_rank", df.columns)
    self.assertIn("pav_rank", df.columns)

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_bridging_fails_fast_on_missing_segment_column(
      self, mock_run
  ):
    with self.assertRaisesRegex(ValueError, "missing_col"):
      await main.run_analysis(
          self.participants_df,
          self.statements_df,
          model=self.fake_model,
          config=main.SimulatedJuryConfig(
              bridging_config=bridging.BridgingConfig(
                  segment_column="missing_col",
                  segment_columns=None,
                  metric=bridging.DEFAULT_BRIDGING_METRIC,
                  min_segment_size=1,
              )
          ),
      )

    mock_run.assert_not_called()

  @mock.patch.object(
      simulated_jury,
      "run_simulated_jury",
      new_callable=mock.AsyncMock,
  )
  async def test_run_analysis_fails_fast_on_missing_statement_column(
      self, mock_run
  ):
    with self.assertRaisesRegex(ValueError, "statement_column"):
      await main.run_analysis(
          self.participants_df,
          pd.DataFrame({"other_col": ["A", "B"]}),
          model=self.fake_model,
          config=main.SimulatedJuryConfig(),
      )

    mock_run.assert_not_called()


if __name__ == "__main__":
  unittest.main()
