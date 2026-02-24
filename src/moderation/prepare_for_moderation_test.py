# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import tempfile
import sys
from google.cloud import dlp_v2
from src.moderation.prepare_for_moderation import main
from src.qualtrics.process_qualtrics_output import RESPONSE_TEXT, SURVEY_TEXT, RESPONDENT_ID, DURATION
from src.evals.eval_metrics import INPUT_EVAL_METRICS


class PrepareForModerationTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.input_csv_path = os.path.join(self.temp_dir.name, "input.csv")
    self.input_evals_csv_path = os.path.join(
        self.temp_dir.name, "input_evals.csv"
    )
    self.output_csv_path = os.path.join(self.temp_dir.name, "output.csv")

    # Create dummy input CSV
    pd.DataFrame({
        RESPONDENT_ID: [1, 2],
        # Each survey text is two question answer pairs. The questions and
        # answers are repeated for simplicity.
        SURVEY_TEXT: [
            "<question> Question </question> <response>This is a clean"
            " comment.</response>"
            * 2,
            "<question> Question </question> <response> This is a toxic"
            " comment. </response>"
            * 2,
        ],
        DURATION: [10, 20],
        RESPONSE_TEXT: [
            "GOV Response:\nThis is a clean comment.",
            "GOV Response:\nThis is a toxic comment.",
        ],
    }).to_csv(self.input_csv_path, index=False)

    # Create dummy input evals CSV
    pd.DataFrame({
        "response": [
            "<question> Question </question> <response>This is a clean"
            " comment.</response>"
            * 2,
            "<question> Question </question> <response> This is a toxic"
            " comment. </response>"
            * 2,
        ],
        INPUT_EVAL_METRICS.name + "Pointwise/score": [4, 1],
    }).to_csv(self.input_evals_csv_path, index=False)

  def tearDown(self):
    self.temp_dir.cleanup()

  @patch("src.moderation.prepare_for_moderation.dlp_v2.DlpServiceClient")
  @patch("src.moderation.prepare_for_moderation.init_client")
  @patch("src.moderation.prepare_for_moderation.score_text")
  @patch.dict(os.environ, {"API_KEY": "test_key"})
  def test_main(self, mock_score_text, mock_init_client, mock_dlp_client):
    # Mock Perspective API client and scoring
    mock_client = MagicMock()
    mock_init_client.return_value = mock_client

    # Mock DLP client
    mock_dlp_instance = mock_dlp_client.return_value
    mock_inspect_result = MagicMock()
    mock_inspect_result.result.findings = []
    mock_dlp_instance.inspect_content.return_value = mock_inspect_result

    def score_side_effect(client, text, attributes):
      scores = {}
      for attribute in attributes:
        if "toxic" in text:
          if attribute == "TOXICITY":
            scores[attribute] = 0.8
          elif attribute == "SEVERE_TOXICITY":
            scores[attribute] = 0.7
        else:
          if attribute == "TOXICITY":
            scores[attribute] = 0.1
          elif attribute == "SEVERE_TOXICITY":
            scores[attribute] = 0.05
      return scores

    mock_score_text.side_effect = score_side_effect

    # Mock sys.argv
    test_args = [
        "prepare_for_moderation.py",
        "--input_csv",
        self.input_csv_path,
        "--input_evals_csv",
        self.input_evals_csv_path,
        "--output_csv",
        self.output_csv_path,
        "--data_type",
        "ROUND_1",
        "--api_key",
        "test_key",
    ]
    with patch.object(sys, "argv", test_args):
      main()

    # Check if output file exists
    self.assertTrue(os.path.exists(self.output_csv_path))

    # Check the content of the output file
    df = pd.read_csv(self.output_csv_path)
    self.assertEqual(len(df), 2)
    self.assertIn("Toxicity Score (of the worst response)", df.columns)
    self.assertIn("Severe Toxicity Score (of the worst response)", df.columns)
    self.assertIn("Spam Score", df.columns)
    self.assertIn("Too Fast Score", df.columns)

    # Check toxicity scores
    self.assertAlmostEqual(
        df.loc[df[RESPONSE_TEXT] == "GOV Response:\nThis is a clean comment."][
            "Toxicity Score (of the worst response)"
        ].iloc[0],
        0.1,
    )
    self.assertAlmostEqual(
        df.loc[df[RESPONSE_TEXT] == "GOV Response:\nThis is a toxic comment."][
            "Toxicity Score (of the worst response)"
        ].iloc[0],
        0.8,
    )
    self.assertAlmostEqual(
        df.loc[df[RESPONSE_TEXT] == "GOV Response:\nThis is a clean comment."][
            "Severe Toxicity Score (of the worst response)"
        ].iloc[0],
        0.05,
    )
    self.assertAlmostEqual(
        df.loc[df[RESPONSE_TEXT] == "GOV Response:\nThis is a toxic comment."][
            "Severe Toxicity Score (of the worst response)"
        ].iloc[0],
        0.7,
    )

    # Check merged eval scores
    self.assertAlmostEqual(
        df.loc[df[RESPONSE_TEXT] == "GOV Response:\nThis is a clean comment."][
            "Spam Score"
        ].iloc[0],
        0,
    )
    self.assertAlmostEqual(
        df.loc[df[RESPONSE_TEXT] == "GOV Response:\nThis is a toxic comment."][
            "Spam Score"
        ].iloc[0],
        0.75,
    )

    # Check that score_text was called correctly
    self.assertEqual(mock_score_text.call_count, 4)


if __name__ == "__main__":
  unittest.main()
