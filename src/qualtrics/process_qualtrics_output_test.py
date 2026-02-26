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
import pandas as pd
import os
import csv
import tempfile
from src.qualtrics import process_qualtrics_output


class ProcessQualtricsOutputTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.input_csv_path = os.path.join(self.temp_dir.name, "input.csv")
    self.output_csv_path = os.path.join(self.temp_dir.name, "output.csv")

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_process_csv_round_1(self):
    # Create a dummy Qualtrics CSV file
    header = [
        "Finished",
        "Status",
        "Q1",
        "Q1FU",
        "Q2",
        "Q2FU",
        "Q3",
        "Q3FU",
        "Q1FU_Text",
        "Q2FU_Text",
        "Q3FU_Text",
        "rid",
        "Duration (in seconds)",
    ]
    # This data is largely unused except to get the question text for the fixed
    # questions.
    meta_row1 = [
        "Finished",
        "Status",
        "Q1 text",
        "empty",
        "Q2 text",
        "empty",
        "Q3 text",
        "empty",
        "empty",
        "empty",
        "empty",
        "rid",
        "Duration",
    ]
    # This meta data is unused - just set it to empty strings
    meta_row2 = ["" for i in range(len(meta_row1))]
    data_row1 = [
        "True",
        "IP Address",
        "Ans1",
        "Q1 follow up text",
        "Ans3",
        "Q2 follow up text",
        "Ans5",
        "Q3 follow up text",
        "Ans2",
        "Ans4",
        "Ans6",
        "resp1",
        "123",
    ]

    with open(self.input_csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerow(meta_row1)
      writer.writerow(meta_row2)
      writer.writerow(data_row1)

    process_qualtrics_output.process_csv(
        self.input_csv_path,
        self.output_csv_path,
        process_qualtrics_output.DataType.ROUND_1,
    )

    self.assertTrue(os.path.exists(self.output_csv_path))

    df = pd.read_csv(self.output_csv_path)

    self.assertEqual(len(df), 1)
    self.assertEqual(
        df.iloc[0][process_qualtrics_output.RESPONDENT_ID], "resp1"
    )
    expected_survey_text = "\n\n".join([
        "<question>Q1 text</question>\n<response>Ans1</response>",
        "<question>Q1 follow up text</question>\n<response>Ans2</response>",
        "<question>Q2 text</question>\n<response>Ans3</response>",
        "<question>Q2 follow up text</question>\n<response>Ans4</response>",
        "<question>Q3 text</question>\n<response>Ans5</response>",
        "<question>Q3 follow up text</question>\n<response>Ans6</response>",
    ])
    self.assertEqual(
        df.iloc[0][process_qualtrics_output.SURVEY_TEXT], expected_survey_text
    )

    expected_response_text = "\n\n".join([
        "Response 1: Ans1",
        "Response 2: Ans2",
        "Response 3: Ans3",
        "Response 4: Ans4",
        "Response 5: Ans5",
        "Response 6: Ans6",
    ])
    self.assertEqual(
        df.iloc[0][process_qualtrics_output.RESPONSE_TEXT],
        expected_response_text,
    )

  def test_process_csv_round_1_with_overrides(self):
    # Test overriding the questions
    # Setup custom override
    custom_questions = ["X1", "X2"]
    custom_fu_questions = ["X1FU", "X2FU"]
    custom_fu_texts = ["X1FU_Text", "X2FU_Text"]

    # Configure the module
    process_qualtrics_output.configure_round_1(
        custom_questions, custom_fu_questions, custom_fu_texts
    )

    # Create CSV matching these custom questions
    header = [
        "Finished",
        "Status",
        "rid",
        "Duration (in seconds)",
        "X1",
        "X2",
        "X1FU",
        "X2FU",
        "X1FU_Text",
        "X2FU_Text",
    ]
    # Meta row needs to ensure X1, X2 have text for X1_Text lookups
    # The logic: df[f"{question}_Text"] = fixed_questions[question]
    # So X1_Text will be taken from X1 column of metadata row 1.
    meta_row1 = [
        "Finished",
        "Status",
        "rid",
        "Duration",
        "Override Q1 Text",
        "Override Q2 Text",
        "empty",
        "empty",
        "empty",
        "empty",
    ]
    meta_row2 = ["" for _ in header]
    data_row1 = [
        "True",
        "IP Address",
        "custom_resp",
        "100",
        "A1",
        "A2",
        "What is FU1?",
        "What is FU2?",
        "FU1",
        "FU2",
    ]

    with open(self.input_csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerow(meta_row1)
      writer.writerow(meta_row2)
      writer.writerow(data_row1)

    process_qualtrics_output.process_csv(
        self.input_csv_path,
        self.output_csv_path,
        process_qualtrics_output.DataType.ROUND_1,
    )

    df = pd.read_csv(self.output_csv_path)
    self.assertEqual(len(df), 1)

    # Verify SURVEY_TEXT
    # Should include 2 questions + 2 FUs = 4 items
    # Order: X1, X1FU, X2, X2FU
    expected_survey_text = "\n\n".join([
        "<question>Override Q1 Text</question>\n<response>A1</response>",
        "<question>What is FU1?</question>\n<response>FU1</response>",
        "<question>Override Q2 Text</question>\n<response>A2</response>",
        "<question>What is FU2?</question>\n<response>FU2</response>",
    ])
    self.assertEqual(
        df.iloc[0][process_qualtrics_output.SURVEY_TEXT], expected_survey_text
    )

  def test_process_csv_round_1_multiline_question_text(self):
    # Test that default behavior keeps only first line, and flag=False keeps all.
    # We restrict to just Q1 to avoid empty tags for Q2, Q3, etc.
    original_q = process_qualtrics_output.ROUND_1_QUESTION_RESPONSE_TEXT
    original_fu = process_qualtrics_output.ROUND_1_FOLLOW_UP_QUESTIONS
    original_fu_text = (
        process_qualtrics_output.ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS
    )

    try:
      process_qualtrics_output.configure_round_1(["Q1"], [], [])

      header = [
          "Finished",
          "Status",
          "Q1",
          "rid",
          "Duration (in seconds)",
      ]
      # Meta row with multiline question
      meta_row1 = [
          "Finished",
          "Status",
          "Q1\nLine2\nLine3",
          "rid",
          "Duration",
      ]
      meta_row2 = ["" for _ in header]
      data_row = ["True", "IP", "A1", "r1", "100"]

      with open(self.input_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(meta_row1)
        writer.writerow(meta_row2)
        writer.writerow(data_row)

      # 1. Test Default (one_line_question_text=True)
      process_qualtrics_output.process_csv(
          self.input_csv_path,
          self.output_csv_path,
          process_qualtrics_output.DataType.ROUND_1,
      )
      df = pd.read_csv(self.output_csv_path)
      expected_survey_text_default = (
          "<question>Q1</question>\n<response>A1</response>"
      )
      self.assertEqual(
          df.iloc[0][process_qualtrics_output.SURVEY_TEXT],
          expected_survey_text_default,
      )

      # 2. Test Full Text (one_line_question_text=False)
      process_qualtrics_output.process_csv(
          self.input_csv_path,
          self.output_csv_path,
          process_qualtrics_output.DataType.ROUND_1,
          one_line_question_text=False,
      )
      df_full = pd.read_csv(self.output_csv_path)
      expected_survey_text_full = (
          "<question>Q1\nLine2\nLine3</question>\n<response>A1</response>"
      )
      self.assertEqual(
          df_full.iloc[0][process_qualtrics_output.SURVEY_TEXT],
          expected_survey_text_full,
      )
    finally:
      # Restore configuration
      process_qualtrics_output.configure_round_1(
          original_q, original_fu, original_fu_text
      )

  def test_rid_generation(self):
    # Test 1: rid missing, rdud present (case 2a)
    header = ["Finished", "Status", "rdud", "Duration (in seconds)"]
    # Ensure we have minimal R1 cols if testing ROUND_1 data type
    header += process_qualtrics_output.ROUND_1_QUESTION_RESPONSE_TEXT
    process_qualtrics_output.configure_round_1(["Q1"], ["Q1FU"], ["Q1FU_Text"])

    minimal_cols = ["Q1", "Q1FU", "Q1FU_Text"]
    header = [
        "Finished",
        "Status",
        "rdud",
        "Duration (in seconds)",
    ] + minimal_cols

    meta_row1 = ["F", "S", "rdud_meta", "D", "Q1_T", "", ""]
    meta_row2 = ["" for _ in header]
    data_row = ["True", "IP", "fallback_id", "10", "a", "b", "c"]

    with open(self.input_csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerow(meta_row1)
      writer.writerow(meta_row2)
      writer.writerow(data_row)

    process_qualtrics_output.process_csv(
        self.input_csv_path,
        self.output_csv_path,
        process_qualtrics_output.DataType.ROUND_1,
    )

    df = pd.read_csv(self.output_csv_path)
    self.assertEqual(df.iloc[0]["rid"], "fallback_id")

    # Test 2: rid missing, rdud missing -> sequential generation
    # Reset file
    header = ["Finished", "Status", "Duration (in seconds)"] + minimal_cols
    meta_row1 = ["F", "S", "D", "Q1_T", "", ""]
    meta_row2 = ["" for _ in header]
    data_row = ["True", "IP", "10", "a", "b", "c"]

    with open(self.input_csv_path, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(header)
      writer.writerow(meta_row1)
      writer.writerow(meta_row2)
      writer.writerow(data_row)

    process_qualtrics_output.process_csv(
        self.input_csv_path,
        self.output_csv_path,
        process_qualtrics_output.DataType.ROUND_1,
    )

    df = pd.read_csv(self.output_csv_path)
    self.assertEqual(df.iloc[0]["rid"], 1)  # Should be int 1

  def test_process_csv_round_2(self):
    # Define all columns required for a Round 2 CSV
    header = (
        process_qualtrics_output.COMMON_QUALTRICS_COLS
        + process_qualtrics_output.ROUND_2_QUESTIONS
    )
    meta_row1 = {col: f"{col}_meta" for col in header}
    meta_row2 = {col: "" for col in header}
    data_row1 = {col: "" for col in header}  # Default all to empty

    # Overwrite specific values for the test
    meta_row1.update({
        "question_1": (
            "Topic: T1\n\nOpinion: O1\n\n“Quote 1”\n\nHow would you"
            " respond to this quote?"
        ),
        "ranking_1_q_1": "Rank Q1",
        "ranking_1_q_c": "Rank Comment Q1",
    })
    data_row1.update({
        "Finished": "True",
        "Status": "IP Address",
        "rid": "resp1",
        "Duration (in seconds)": "123",
        "question_1": "Answer 1",
        "ranking_1_q_1": "3",
        "ranking_1_q_c": "Rank Comment A1",
    })

    with open(self.input_csv_path, "w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=header)
      writer.writeheader()
      writer.writerow(meta_row1)
      writer.writerow(meta_row2)
      writer.writerow(data_row1)

    process_qualtrics_output.process_csv(
        self.input_csv_path,
        self.output_csv_path,
        process_qualtrics_output.DataType.ROUND_2,
    )

    self.assertTrue(os.path.exists(self.output_csv_path))

    df = pd.read_csv(self.output_csv_path)
    self.assertEqual(len(df), 1)

    # Check RESPONSE_TEXT formatting
    expected_response_text = (
        "GOV Response:\nAnswer 1\n\nRanking Response:\nRank Comment A1"
    )
    self.assertEqual(
        df.iloc[0][process_qualtrics_output.RESPONSE_TEXT],
        expected_response_text,
    )

    # Check column creation and renaming
    self.assertEqual(df.iloc[0]["answer_1"], "Answer 1")
    self.assertEqual(df.iloc[0]["question_1"], "Quote 1")
    self.assertEqual(df.iloc[0]["question_1_topic"], "T1")
    self.assertEqual(df.iloc[0]["question_1_opinion"], "O1")
    self.assertEqual(str(df.iloc[0]["ranking_1_a_1"]), "3")
    self.assertEqual(df.iloc[0]["ranking_1_q_1"], "Rank Q1")
    self.assertEqual(df.iloc[0]["ranking_1_a_c"], "Rank Comment A1")
    self.assertEqual(df.iloc[0]["ranking_1_q_c"], "Rank Comment Q1")


if __name__ == "__main__":
  unittest.main()
