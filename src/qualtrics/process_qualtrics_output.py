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

"""
Processes a Qualtrics survey export CSV into a clean format.

This script reads a CSV file exported from Qualtrics, which typically includes
an extra header row with question metadata, and processes it into a standard
CSV format. The primary operation is to skip the second row of the CSV, which
contains question IDs or full question text that is not part of the data itself.

Example Usage:
  python3 -m src.qualtrics.process_qualtrics_output \
    --input_csv /path/to/qualtrics_export.csv \
    --output_csv /path/to/processed_data.csv \
    --data_type "ROUND_1" \
    --one_line_question_text \
    --round_1_question_response_text "Q3.1,Q5.1,Q7.1,Q9.3" \
    --round_1_follow_up_questions "Q1FU,Q2FU,Q3FU" \
    --round_1_follow_up_question_response_text "Q4.1,Q6.1,Q8.1"
"""

import argparse
import os
import pandas as pd
import re
import logging
from enum import Enum
from typing import Callable
from itertools import zip_longest

ROUND_1_QUESTION_RESPONSE_TEXT = ["Q1", "Q2", "Q3"]
ROUND_1_FOLLOW_UP_QUESTIONS = ["Q1FU", "Q2FU", "Q3FU"]
ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS = [
    "Q1FU_Text",
    "Q2FU_Text",
    "Q3FU_Text",
]

ROUND_2_OPEN_QUESTIONS = [f"question_{i}" for i in range(1, 28)]
ROUND_2_RANKING_QUESTIONS = (
    [f"ranking_1_q_{j}" for j in range(1, 10)]
    + [f"ranking_2_q_{j}" for j in range(1, 7)]
    + [f"ranking_3_q_{j}" for j in range(1, 6)]
    + [f"ranking_4_q_{j}" for j in range(1, 8)]
)
ROUND_2_RANKING_QUESTIONS_COMMENT = [f"ranking_{i}_q_c" for i in range(1, 5)]
ROUND_2_META_QUESTIONS = [f"meta_question_{i}" for i in range(1, 9)]
ROUND_2_VISUAL_QUESTIONS = ["visual_question_1"]
ROUND_2_QUESTIONS = [
    *ROUND_2_OPEN_QUESTIONS,
    *ROUND_2_RANKING_QUESTIONS,
    *ROUND_2_RANKING_QUESTIONS_COMMENT,
    *ROUND_2_META_QUESTIONS,
    *ROUND_2_VISUAL_QUESTIONS,
]
ROUND_2_RESPONSE_TEXT_COLUMNS = [
    *ROUND_2_OPEN_QUESTIONS,
    *ROUND_2_RANKING_QUESTIONS_COMMENT,
    *ROUND_2_VISUAL_QUESTIONS,
]


# Columns to Keep
SURVEY_TEXT = "survey_text"
RESPONSE_TEXT = "response_text"
RESPONDENT_ID = "rid"
PARTICIPANT_ID = "participant_id"
DURATION = "Duration (in seconds)"

# Columns for filtering out data.
FINISHED_COL = "Finished"
PREVIEW_STATUS_COL = "Status"

# Regardless of DataType all Qualtrics output should have these columns.
COMMON_QUALTRICS_COLS = [
    RESPONDENT_ID,
    FINISHED_COL,
    PREVIEW_STATUS_COL,
    DURATION,
]

START_QUESTION_TAG = "<question>"
END_QUESTION_TAG = "</question>"
START_RESPONSE_TAG = "<response>"
END_RESPONSE_TAG = "</response>"


def _is_text_response(value: any) -> bool:
  """
  Returns if the value is a non-empty, non-numeric string.

  This is useful for differentating between open-ended text responses and
  numeric/ranking responses in the survey data.
  """
  if not isinstance(value, str):
    return False

  try:
    # If it can be converted to a float, it's not a text response.
    float(value)
    return False
  except ValueError:
    # If it can't, it's a text response, as long as it's not empty/whitespace.
    return bool(value.strip())


def process_round_1_data(
    df: pd.DataFrame, input_path: str, one_line_question_text: bool = True
) -> pd.DataFrame:
  # The first metadata row has the fixed questions that aren't created via intelligent ellicitation.
  try:
    fixed_questions = pd.read_csv(input_path, nrows=1).iloc[0]
  except Exception as e:
    logging.warning(f"Could not read fixed questions from metadata: {e}")
    fixed_questions = {}

  for question in ROUND_1_QUESTION_RESPONSE_TEXT:
    if question in fixed_questions:
      text = fixed_questions[question]
      if one_line_question_text and isinstance(text, str):
        text = text.split("\n")[0]
        # Update fixed_questions so get_question_and_response uses the short version
        fixed_questions[question] = text
      df[f"{question}_Text"] = text
    else:
      pass

  # Interleave ROUND_1_QUESTION_RESPONSE_TEXT and ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS
  # e.g. ["Q1", "Q1FU_Text", "Q2", "Q2FU_Text", etc]
  response_cols_raw = [
      val
      for pair in zip_longest(
          ROUND_1_QUESTION_RESPONSE_TEXT,
          ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS,
      )
      for val in pair
  ]
  # Filter out Nones
  response_cols = [val for val in response_cols_raw if val is not None]

  # Returns the i numbered question from the row
  def get_question_and_response(row, i):
    # response_col is Q1, Q1FU, etc
    try:
      response_col = response_cols[i]
    except IndexError:
      return "", ""

    question = ""
    # Check if it is a follow up question (present in the FU list)
    if response_col in ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS:
      # Question is AI generated and unique per row
      # Get the corresponding text column name
      try:
        idx = ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS.index(response_col)
        text_col = ROUND_1_FOLLOW_UP_QUESTIONS[idx]
        question = row.get(text_col, "")
      except (ValueError, IndexError):
        # Fallback to old convention if lookup fails
        question = row.get(response_col + "_Text", "")
    else:
      # Question is fixed, get from fixed_questions row
      question = fixed_questions.get(response_col, "")

    response = row.get(response_col, "")
    return question, response

  # Create the RESPONSE_TEXT column.  This concats all responses
  # but omits questions
  df[RESPONSE_TEXT] = df.apply(
      lambda row: "\n\n".join([
          f"Response {index+1}: {row[question]}"
          for index, question in enumerate(response_cols)
          if question in row  # Ensure column exists before access
      ]),
      axis=1,
  )

  # Create a SURVEY_TEXT column that combines the question
  # and responses in XML, e.g.
  #   <question>what do you think...</question>
  #   <reponse>I think...</response>
  #   <question>why do you think...</question>
  #   etc
  df[SURVEY_TEXT] = df.apply(
      lambda row: "\n\n".join([
          f"{START_QUESTION_TAG}{question}{END_QUESTION_TAG}\n"
          + f"{START_RESPONSE_TAG}{response}{END_RESPONSE_TAG}"
          for question, response in [
              get_question_and_response(row, i)
              for i, _ in enumerate(response_cols)
          ]
      ]),
      axis=1,
  )

  return df


def process_round_2_data(
    df: pd.DataFrame, input_path: str, **kwargs
) -> pd.DataFrame:
  # The first metadata row has the questions that were asked.
  questions = pd.read_csv(input_path, nrows=1).iloc[0]

  # --- Stage 1: initial columns and updates ---
  stage1_updates = {}

  def format_responses(row):
    # Helper method that extracts opinion string and user answer and builds a
    # meaningfull response text structure.
    responses = []
    for col_name in ROUND_2_RESPONSE_TEXT_COLUMNS:
      if _is_text_response(row[col_name]):
        answer_text = row[col_name].strip()
        prefix = ""
        if col_name in ROUND_2_VISUAL_QUESTIONS:
          prefix = "Visual Response"
        elif col_name in ROUND_2_OPEN_QUESTIONS:
          prefix = f"GOV Response"
        elif col_name in ROUND_2_RANKING_QUESTIONS_COMMENT:
          prefix = f"Ranking Response"
        responses.append(f"{prefix}:\n{answer_text}")
    return "\n\n".join(responses)

  # Combine the text from open text columns into one row for easy reading.
  stage1_updates[RESPONSE_TEXT] = df.apply(format_responses, axis=1)

  # The question column really is the response column, and the question is
  # from the first metadata row.
  for open_question in (
      ROUND_2_OPEN_QUESTIONS + ROUND_2_META_QUESTIONS + ROUND_2_VISUAL_QUESTIONS
  ):
    stage1_updates[open_question.replace("question", "answer")] = df[
        open_question
    ]
    stage1_updates[open_question] = questions[open_question]

  for ranking_question in (
      ROUND_2_RANKING_QUESTIONS + ROUND_2_RANKING_QUESTIONS_COMMENT
  ):
    stage1_updates[ranking_question.replace("q", "a")] = df[ranking_question]
    stage1_updates[ranking_question] = questions[ranking_question]

  # Create a new DataFrame for the new/updated columns
  updates_df_s1 = pd.DataFrame(stage1_updates, index=df.index)
  # Identify columns to be overwritten
  cols_to_overwrite_s1 = [
      key for key in stage1_updates.keys() if key in df.columns
  ]
  # Drop the columns that will be overwritten from the original DataFrame
  df = df.drop(columns=cols_to_overwrite_s1)
  # Concatenate the original DataFrame with the new columns
  df = pd.concat([df, updates_df_s1], axis=1)

  # --- Stage 2: parsing questions and further updates ---
  stage2_updates = {}
  # Since the description includes topic, opinion, and the quote, we split them
  # into seperate columns.
  for open_question in ROUND_2_OPEN_QUESTIONS:
    parsed_cols = df[open_question].str.extract(
        r"Topic: (.*?)\s*\n\nOpinion: (.*?)\s*\n\n“(.*?)”\s*\n\nHow would you"
        r" respond to this quote\?",
        expand=True,
    )
    parsed_cols.columns = ["topic", "opinion", "quote"]

    stage2_updates[f"{open_question}_topic"] = parsed_cols["topic"]
    stage2_updates[f"{open_question}_opinion"] = parsed_cols["opinion"]
    stage2_updates[open_question] = parsed_cols["quote"]

  # Create a new DataFrame for the new/updated columns
  updates_df_s2 = pd.DataFrame(stage2_updates, index=df.index)
  # Identify columns to be overwritten
  cols_to_overwrite_s2 = [
      key for key in stage2_updates.keys() if key in df.columns
  ]
  # Drop the columns that will be overwritten from the original DataFrame
  df = df.drop(columns=cols_to_overwrite_s2)
  # Concatenate the original DataFrame with the new columns
  df = pd.concat([df, updates_df_s2], axis=1)

  return df


class DataType(str, Enum):
  ROUND_1 = "ROUND_1"
  ROUND_2 = "ROUND_2"


class ProcessingInfo:

  def __init__(
      self,
      data_type: DataType,
      columns_to_import: list[str],
      processing_func: Callable[..., pd.DataFrame],
  ):
    self.data_type = data_type
    self.columns_to_import = columns_to_import
    self.processing_func = processing_func


_ROUND_1_PROCESSING_INFO = ProcessingInfo(
    DataType.ROUND_1,
    [
        *COMMON_QUALTRICS_COLS,
        *ROUND_1_QUESTION_RESPONSE_TEXT,
        *ROUND_1_FOLLOW_UP_QUESTIONS,
        *ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS,
    ],
    process_round_1_data,
)
_ROUND_2_PROCESSING_INFO = ProcessingInfo(
    DataType.ROUND_2,
    [*COMMON_QUALTRICS_COLS, *ROUND_2_QUESTIONS],
    process_round_2_data,
)

_DATA_TYPE_TO_INFO = {
    DataType.ROUND_1: _ROUND_1_PROCESSING_INFO,
    DataType.ROUND_2: _ROUND_2_PROCESSING_INFO,
}


# Remove unfinished surveys from dataset.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
  cleaned_df = df.loc[
      (df[FINISHED_COL] == True) & (df[PREVIEW_STATUS_COL] != "Survey Preview")
  ].copy()
  if cleaned_df.empty:
    logging.info(
        "Survey contained 0 rows after cleaning. "
        "Either 'Finished' columns don't have rows with `TRUE`. "
        "And/Or Survey Status only contains rows with 'Survey Preview'"
    )
  return cleaned_df


def configure_round_1(
    questions: list[str],
    follow_up_questions: list[str],
    follow_up_question_response_texts: list[str],
):
  global ROUND_1_QUESTION_RESPONSE_TEXT
  global ROUND_1_FOLLOW_UP_QUESTIONS
  global ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS
  global _ROUND_1_PROCESSING_INFO

  ROUND_1_QUESTION_RESPONSE_TEXT = questions
  ROUND_1_FOLLOW_UP_QUESTIONS = follow_up_questions
  ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS = follow_up_question_response_texts

  # Re-create the ProcessingInfo with new columns
  _ROUND_1_PROCESSING_INFO.columns_to_import = [
      *COMMON_QUALTRICS_COLS,
      *ROUND_1_QUESTION_RESPONSE_TEXT,
      *ROUND_1_FOLLOW_UP_QUESTIONS,
      *ROUND_1_FOLLOW_UP_QUESTION_RESPONSE_TEXTS,
  ]


def process_csv(
    input_path: str,
    output_path: str,
    data_type: DataType,
    one_line_question_text: bool = True,
):
  """
  Reads a Qualtrics CSV, processes it, and saves it to a new location.

  Args:
      input_path: The file path for the input Qualtrics CSV.
      output_path: The file path to save the processed CSV to.
  """
  processing_info = _DATA_TYPE_TO_INFO.get(data_type)

  # Check which columns are actually present in the file to handle RID logic
  # and avoid errors if we request columns that aren't there.
  try:
    # Read just the header
    header_df = pd.read_csv(input_path, nrows=0)
    actual_columns = set(header_df.columns)
  except Exception as e:
    logging.info(f"Error reading CSV header: {e}")
    return

  # Determine columns to import based on what's available
  # processing_info.columns_to_import might have 'rid' which might be missing.
  cols_to_use = list(processing_info.columns_to_import)

  # Handle RID / RDUD logic for import
  # If RID is missing, we don't want to fail import.
  # If RDUD is present, we might want it for fallback.

  if RESPONDENT_ID not in actual_columns:
    if RESPONDENT_ID in cols_to_use:
      cols_to_use.remove(RESPONDENT_ID)

  # Ensure we import 'rdud' if it exists, in case we need it for fallback
  rdud_col = "rdud"
  if rdud_col in actual_columns and rdud_col not in cols_to_use:
    cols_to_use.append(rdud_col)

  # Qualtrics exports have two metadata rows beneath the header, which we skip.
  try:
    df = pd.read_csv(
        input_path,
        usecols=lambda c: c in cols_to_use,
        skiprows=[1, 2],
    )
    logging.info(f"Successfully read {len(df)} rows from {input_path}.")
  except Exception as e:
    logging.info(f"Error reading CSV file with special Qualtrics handling: {e}")
    return

  # --- logic for RID / RDUD ---
  # 1) if rid exists and has values for each row then use that.
  # 2) if rid doesn't exist or has missing values:
  #    2a) if rdud exists and has values for each row -> copy to rid
  #    2b) if rdud exists but has missing values -> generate sequential rid
  # 3) if rid exists but some values missing -> fill from rdud or generate new

  # Create 'rid' column if not exists
  if RESPONDENT_ID not in df.columns:
    df[RESPONDENT_ID] = pd.NA

  # Ensure 'rdud' is available (if not, create empty series to simplify logic)
  if rdud_col not in df.columns:
    df[rdud_col] = pd.NA

  # Check if rid is fully populated
  # We consider empty strings or NaN as missing.
  def is_valid(val):
    if pd.isna(val):
      return False
    if isinstance(val, str) and not val.strip():
      return False
    return True

  rid_missing_mask = df[RESPONDENT_ID].isna() | (
      df[RESPONDENT_ID].astype(str).str.strip() == ""
  )

  if rid_missing_mask.any():
    logging.info("Found missing or empty Respondent IDs. Attempting to fill...")

    # Try to fill from rdud
    df[RESPONDENT_ID] = df[RESPONDENT_ID].fillna(df[rdud_col])

    # Now check again if we still have missing values
    rid_missing_mask = df[RESPONDENT_ID].isna() | (
        df[RESPONDENT_ID].astype(str).str.strip() == ""
    )

    if rid_missing_mask.any():
      logging.info(
          "Respondent IDs still missing after checking 'rdud'. Generating"
          " sequential IDs."
      )
      df[RESPONDENT_ID] = range(1, len(df) + 1)

  completed_surveys = clean_data(df)
  processed_surveys = processing_info.processing_func(
      completed_surveys,
      input_path,
      one_line_question_text=one_line_question_text,
  )

  # Ensure output directory exists
  output_dir = os.path.dirname(output_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  processed_surveys = processed_surveys.rename(columns={
      RESPONDENT_ID: PARTICIPANT_ID})

  processed_surveys.to_csv(output_path, index=False)
  logging.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser(
      description="Process a Qualtrics survey export CSV."
  )
  parser.add_argument(
      "--input_csv", required=True, help="Path to the input Qualtrics CSV file."
  )
  parser.add_argument(
      "--output_csv",
      required=True,
      help="Path to save the processed output CSV file.",
  )
  parser.add_argument(
      "--data_type",
      required=True,
      type=DataType,
      help=f"The type of data being processed, either ROUND_1 or ROUND_2.",
  )
  # Optional arguments for overriding Round 1 constants
  parser.add_argument(
      "--round_1_question_response_text",
      help="Comma-separated list of Round 1 questions (e.g. Q1,Q2,Q3)",
  )
  parser.add_argument(
      "--round_1_follow_up_questions",
      help=(
          "Comma-separated list of Round 1 follow-up questions (e.g."
          " Q1FU,Q2FU,Q3FU)"
      ),
  )
  parser.add_argument(
      "--round_1_follow_up_question_response_text",
      help=(
          "Comma-separated list of Round 1 follow-up question texts (e.g."
          " Q1FU_Text,Q2FU_Text,Q3FU_Text)"
      ),
  )
  parser.add_argument(
      "--one_line_question_text",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Whether to include only the first line of the question text."
          " Defaults to True. Use --no-one_line_question_text to include the"
          " full text."
      ),
  )

  args = parser.parse_args()

  if args.data_type == DataType.ROUND_1:
    # strict check: if one is provided, all should be provided for safety
    if (
        args.round_1_question_response_text
        and args.round_1_follow_up_questions
        and args.round_1_follow_up_question_response_text
    ):
      q_list = [
          x.strip() for x in args.round_1_question_response_text.split(",")
      ]

      questions_col_list = [
          x.strip() for x in args.round_1_follow_up_questions.split(",")
      ]
      answers_col_list = [
          x.strip()
          for x in args.round_1_follow_up_question_response_text.split(",")
      ]

      # Pass answers as the "Questions" list (Main list), and questions as "Texts" list.
      configure_round_1(q_list, questions_col_list, answers_col_list)
    elif (
        args.round_1_question_response_text
        or args.round_1_follow_up_questions
        or args.round_1_follow_up_question_response_text
    ):
      logging.info(
          "Warning: Partial override arguments provided. For Round 1 overrides,"
          " please provide --round_1_question_response_text,"
          " --round_1_follow_up_questions, AND"
          " --round_1_follow_up_question_response_text. Using defaults."
      )

  process_csv(
      args.input_csv,
      args.output_csv,
      args.data_type,
      one_line_question_text=args.one_line_question_text,
  )
