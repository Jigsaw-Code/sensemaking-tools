import asyncio
import unittest
from unittest import mock

import pandas as pd

from src.propositions import run_evals


class RunEvalsTest(unittest.TestCase):

  def setUp(self):
    self.df_r2 = pd.DataFrame({
        "rid": [1, 2, 3, 4],
        "question_1_topic": ["q1a", "q1a", "q1a", "q1a"],
        "answer_1": ["a1a", "a1a", "a1a", "a1a"],
        "question_2_opinion": ["q2b", "q2b", "q2b", "q2b"],
        "answer_2": ["a2b", "a2b", "a2b", "a2b"],
    })
    self.world_model_df = pd.DataFrame({
        "topic": ["Topic A", "Topic B"],
        "opinion": ["Opinion A1", "Opinion B1"],
        "propositions": [
            pd.DataFrame({"proposition": ["prop a1", "prop a2"]}),
            pd.DataFrame({"proposition": ["prop b1", "prop b2"]}),
        ],
    })

  def test_map_agreement_score(self):
    """Tests the _map_agreement_score helper function."""
    self.assertIs(run_evals._map_agreement_score(4.0), True)
    self.assertIs(run_evals._map_agreement_score(3.1), True)
    self.assertIs(run_evals._map_agreement_score(1.0), False)
    self.assertIs(run_evals._map_agreement_score(1.9), False)
    self.assertIsNone(run_evals._map_agreement_score(2.5))
    self.assertIsNone(run_evals._map_agreement_score(2.0))
    self.assertIsNone(run_evals._map_agreement_score(3.0))

  @mock.patch("src.evals.eval_runner.EvalRunner")
  def test_run_agreement_evals_on_r2(self, mock_eval_runner):
    """Tests run_agreement_evals_on_r2."""
    mock_runner_instance = mock.MagicMock()

    async def mock_process_evals(*args, **kwargs):
      return pd.DataFrame({
          "job_id": [0, 1, 2, 3],
          "score": [4.0, 3.5, 1.0, 1.5],
          "explanation": ["Good", "Okay", "Bad", "Poor"],
      })

    mock_runner_instance.process_evals_concurrently.side_effect = (
        mock_process_evals
    )
    mock_eval_runner.return_value = mock_runner_instance

    mock_model = mock.MagicMock()
    result_df = asyncio.run(
        run_evals.run_agreement_evals_on_r2(self.df_r2, "topic", mock_model)
    )

    mock_eval_runner.assert_called_once_with(mock_model)
    self.assertIn("answer_1_agrees", result_df.columns)
    self.assertEqual(
        result_df["answer_1_agrees"].tolist(), [True, True, False, False]
    )

  @mock.patch("src.evals.eval_runner.EvalRunner")
  def test_run_evals_on_propositions(self, mock_eval_runner):
    """Tests run_evals_on_propositions."""
    mock_runner_instance = mock.MagicMock()

    async def mock_process_evals(eval_jobs, *args, **kwargs):
      metadata = eval_jobs[0].get("metadata", {})
      job_ids = list(range(len(eval_jobs)))
      if metadata.get("type") == "topic":
        scores = [
            4.0,
            3.0,
        ]
        return pd.DataFrame({
            "job_id": range(len(eval_jobs)),
            "score": [4.0] * len(eval_jobs),  # Default
            "explanation": [""] * len(eval_jobs),
        }).assign(score=[4.0, 3.0, 2.0, 1.0])
      else:
        return pd.DataFrame({
            "job_id": range(len(eval_jobs)),
            "score": [1.0, 2.0, 3.0, 4.0],
            "explanation": [""] * len(eval_jobs),
        })

    mock_runner_instance.process_evals_concurrently.side_effect = (
        mock_process_evals
    )
    mock_eval_runner.return_value = mock_runner_instance

    mock_model = mock.MagicMock()
    result_df = asyncio.run(
        run_evals.run_evals_on_propositions(self.world_model_df, mock_model)
    )

    self.assertEqual(mock_eval_runner.call_count, 1)
    mock_eval_runner.assert_called_with(mock_model)
    propositions_a = result_df.loc[0, "propositions"]
    self.assertIn("topic_score", propositions_a.columns)
    self.assertIn("opinion_score", propositions_a.columns)
    self.assertEqual(propositions_a["topic_score"].tolist(), [4.0, 3.0])
    self.assertEqual(propositions_a["opinion_score"].tolist(), [1.0, 2.0])

  @mock.patch("src.evals.eval_runner.EvalRunner")
  def test_run_agreement_evals_on_r2_failure(self, mock_eval_runner):
    """Tests that the original DataFrame is returned on evaluation failure."""
    mock_runner_instance = mock.MagicMock()

    async def mock_process_evals(*args, **kwargs):
      return pd.DataFrame()

    mock_runner_instance.process_evals_concurrently.side_effect = (
        mock_process_evals
    )
    mock_eval_runner.return_value = mock_runner_instance

    mock_model = mock.MagicMock()
    result_df = asyncio.run(
        run_evals.run_agreement_evals_on_r2(self.df_r2, "topic", mock_model)
    )

    pd.testing.assert_frame_equal(result_df, self.df_r2)
    self.assertNotIn("answer_1_agrees", result_df.columns)

  @mock.patch("src.evals.eval_runner.EvalRunner")
  def test_run_evals_on_propositions_failure(self, mock_eval_runner):
    """Tests that the original DataFrame is returned on evaluation failure."""
    mock_runner_instance = mock.MagicMock()

    async def mock_process_evals(*args, **kwargs):
      return pd.DataFrame()

    mock_runner_instance.process_evals_concurrently.side_effect = (
        mock_process_evals
    )
    mock_eval_runner.return_value = mock_runner_instance

    mock_model = mock.MagicMock()
    result_df = asyncio.run(
        run_evals.run_evals_on_propositions(self.world_model_df, mock_model)
    )

    pd.testing.assert_frame_equal(result_df, self.world_model_df)
    self.assertNotIn("topic_score", result_df.loc[0, "propositions"].columns)


if __name__ == "__main__":
  unittest.main()
