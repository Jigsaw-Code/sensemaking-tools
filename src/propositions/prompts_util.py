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

import pandas as pd
import re
import json
import logging
import string
from typing import Dict, List, Tuple
from enum import Enum
from src.models import genai_model


class QuestionType(Enum):
  """The type of question in the survey."""

  FREE_TEXT = 1
  RANKING = 2


# Helpers
def find_prefix_num(row: pd.Series, prefix: str) -> str | None:
  """Helper to find the common number Z in ranking_Z_*_* columns."""
  pattern = re.compile(rf"{prefix}_(\d+)_\w")
  for col in row.index:
    match = pattern.match(str(col))
    if match:
      return match.group(1)
  return None


# Parsers
def parse_proposition_response_json_reasoning(
    response_data: str | dict, _: genai_model.Job
) -> pd.DataFrame:
  """
  Parses a JSON string containing a list of proposition responses into a pandas
  DataFrame.

  Args:
    response_data: The JSON string or response dict to parse.

  Returns:
    A pandas DataFrame with 'proposition' and 'reasoning' columns.
    Returns an empty DataFrame if the input is invalid or empty.
  """
  try:
    if isinstance(response_data, dict):
      response_data = response_data.get("text", "")

    # Load the JSON string into a Python list of dictionaries
    data = json.loads(response_data)

    # Check if the loaded data is a list and is not empty
    if not isinstance(data, list) or not data:
      logging.warning("JSON string did not contain a non-empty list.")
      return pd.DataFrame(columns=["proposition", "reasoning"])

    # Convert the list of dictionaries to a pandas DataFrame
    parsed_df = pd.DataFrame(data)
    if "statement" in parsed_df.columns:
      parsed_df = parsed_df.rename(columns={"statement": "proposition"})

    return parsed_df[["proposition", "reasoning"]]

  except json.JSONDecodeError as e:
    logging.error("Error decoding JSON string: %s", e)
    return pd.DataFrame(columns=["proposition", "reasoning"])
  except Exception as e:
    logging.error("An unexpected error occurred during JSON parsing: %s", e)
    return pd.DataFrame(columns=["proposition", "reasoning"])


def parse_proposition_response_json(
    resp: dict, _: genai_model.Job
) -> pd.DataFrame:
  """
  Parses a JSON string containing a list of proposition responses into a pandas
  DataFrame.

  Args:
    resp: The response dictionary from the model, containing the JSON string
      in the 'text' key.

  Returns:
    A pandas DataFrame with 'proposition', 'r1_rids', and 'r2_rids' columns.
    Returns an empty DataFrame if the input is invalid or empty.
  """
  response_data = resp["text"]
  try:
    # Load the JSON string into a Python list of dictionaries
    data = json.loads(response_data)

    # Check if the loaded data is a list and is not empty
    if not isinstance(data, list) or not data:
      logging.warning("JSON string did not contain a non-empty list.")
      return pd.DataFrame(columns=["proposition"])

    # Convert the list of dictionaries to a pandas DataFrame
    # Ensure only 'proposition' column is included
    parsed_df = pd.DataFrame({"proposition": data})
    return parsed_df[["proposition"]]

  except json.JSONDecodeError as e:
    logging.error("Error decoding JSON string: %s", e)
    return pd.DataFrame(columns=["proposition"])
  except Exception as e:
    logging.error("An unexpected error occurred during JSON parsing: %s", e)
    return pd.DataFrame(columns=["proposition"])


def extract_reusable_strings(
    df: pd.DataFrame, question_type: QuestionType = QuestionType.RANKING
) -> Tuple[str, Dict[str, str]]:
  """
  Extracts unique opinions from the DataFrame, creates a mapping to short IDs,
  and generates an XML prompt header defining these opinions.

  Args:
      df: The DataFrame containing all survey responses.
      question_type: The type of question to extract opinions for.

  Returns:
      A tuple containing:
      - The XML header string (e.g., "<opinions><A>...</A>...</opinions>").
      - The dictionary mapping opinion text to its ID (e.g., {"text": "A"}).
  """
  # Find the ranking set number (e.g., the '1' in 'ranking_1_q_1')
  # We only need to check the first row to determine the column structure.
  if df.empty:
    return "", {}

  prefix = {
      QuestionType.RANKING: "ranking",
      QuestionType.FREE_TEXT: "freetext",
  }[question_type]

  q_cols_to_scan: List[str] = []

  if question_type == QuestionType.FREE_TEXT:
    q_pattern = re.compile(r"question_(\d+)\b")
    all_question_cols = [c for c in df.columns if q_pattern.match(str(c))]

    # Now, filter that list to exclude any column with 'topic' or 'opinion'
    q_cols_to_scan = [
        c
        for c in all_question_cols
        if "topic" not in str(c) and "opinion" not in str(c)
    ]
  elif question_type == QuestionType.RANKING:
    z = find_prefix_num(df.iloc[0], prefix)
    if z:
      q_cols_to_scan = [
          f"{prefix}_{z}_q_{y}"
          for y in range(1, 4)
          if f"{prefix}_{z}_q_{y}" in df.columns
      ]
  else:
    # Return empty for an invalid prefix
    return "", {}

  if not q_cols_to_scan:
    return "", {}

  # Gather unique texts only from the selected columns
  all_unique_texts = (
      pd.concat([df[col] for col in q_cols_to_scan]).dropna().unique()
  )

  if len(all_unique_texts) == 0:
    return "", {}

  # Create the map and header for this specific type
  definition_ids: List[str] = (
      list(string.ascii_uppercase)[: len(all_unique_texts)]
      if question_type != QuestionType.FREE_TEXT
      else [str(i) for i in range(1, len(all_unique_texts) + 1)]
  )

  definitions_map = dict(zip(all_unique_texts, definition_ids))

  multi_opinion = len(definitions_map) > 1

  if multi_opinion:
    header_parts = [f"<definitions type='{prefix}'>\n"]
  else:
    header_parts = [f"<statement>"]
  for text, def_id in sorted(definitions_map.items(), key=lambda item: item[1]):
    if multi_opinion:
      header_parts.append(f"<{def_id}>{str(text).strip()}</{def_id}>\n")
    else:
      header_parts.append(f"{str(text).strip()}")

  if multi_opinion:
    header_parts.append("</definitions>\n")
  else:
    header_parts.append("</statement>\n")
  prompt_header = "".join(header_parts)

  return prompt_header, definitions_map


# R2 methods
def build_free_text_response_prompt(row: pd.Series, opinions_map: dict) -> str:
  """
  Finds all 'question_X' and 'answer_X' columns, sorts them numerically,
  and formats them into a prompt string.

  Args:
      row: A single row from a Pandas DataFrame.
      opinions_map: A dictionary mapping full opinion text to a short ID.

  Returns:
      A formatted string for the prompt, or an empty string if no
      relevant columns are found.
  """
  # Build a dictionary to keep track of question and answers.
  data_dict = {}
  # Regex to find 'question_2', 'answer_2', etc.
  q_pattern = re.compile(r"question_(\d+)\b")
  a_pattern = re.compile(r"answer_(\d+)\b")

  # Collect all question/answer pairs from the row
  for col, value in row.items():
    q_match = q_pattern.match(str(col))
    if q_match:
      num = int(q_match.group(1))
      if num not in data_dict:
        data_dict[num] = {}
      data_dict[num]["question"] = value
      continue
    a_match = a_pattern.match(str(col))
    if a_match:
      num = int(a_match.group(1))
      if num not in data_dict:
        data_dict[num] = {{}}
      data_dict[num]["answer"] = value

  if not data_dict:
    return ""

  # Tracks if any user data gets added to the prompt. If the prompt does not
  # include user data then the response will be blank string at the end.
  has_content = False

  # Sort by the number in the column name (e.g., question_1 before question_10)
  sorted_keys = sorted(data_dict.keys())

  # Tracks whether the data set includes multipple opinions to be included.
  multi_opinion = len(sorted_keys) > 1

  prompt_parts = []
  # Build the prompt string with new sequential tags (question_1, question_2, ...)
  for _, key in enumerate(sorted_keys, 1):
    pair = data_dict[key]
    q_val = pair.get("question", "")
    a_val = pair.get("answer", "")
    # Look up the ID in the provided map; handle cases where it might not be found
    opinion_id = opinions_map.get(q_val, q_val) if opinions_map else q_val

    has_content = True
    if multi_opinion:
      prompt_parts.append(f"<statement_id>{opinion_id}</statement_id>\n")
      prompt_parts.append(f"<answer>{a_val}</answer>\n")
    else:
      prompt_parts.append(f"{a_val}")

  return "".join(prompt_parts) if has_content else ""


def build_ranking_response_prompt(row: pd.Series, opinions_map: dict) -> str:
  """
  Finds 'ranking_Z_q_Y' and 'ranking_Z_a_Y' columns, sorts them by the
  rank value, and formats them into a prompt string.

  Args:
      row: A single row from a Pandas DataFrame.
      opinions_map: A dictionary mapping full opinion text to a short ID.

  Returns:
      A formatted string for the prompt, or an empty string if no
      relevant columns are found.
  """
  z = find_prefix_num(row, "ranking")
  if z is None:
    return ""

  opinions_to_sort = []
  q4_question_text = ""
  q4_answer_text = ""

  # Collect opinions (1-3) and the final question/answer pair (4)
  for y in range(1, 5):
    q_col = f"ranking_{z}_q_{y}"
    a_col = f"ranking_{z}_a_{y}"

    if q_col in row.index and a_col in row.index:
      q_val = row[q_col]
      a_val = row[a_col]

      if pd.isna(q_val) and pd.isna(a_val):
        continue

      if y < 4:
        try:
          # Convert rank to a number for sorting, handle errors
          rank_num = float(a_val)
        except (ValueError, TypeError):
          rank_num = float("inf")  # Put non-numeric ranks at the end
        opinions_to_sort.append({"opinion": q_val, "rank_val": rank_num})
      else:
        q4_question_text = q_val
        q4_answer_text = a_val

  if not opinions_to_sort and not (q4_question_text and q4_answer_text):
    return ""

  # Sort the opinions based on the numeric rank value
  sorted_opinions = sorted(opinions_to_sort, key=lambda item: item["rank_val"])

  prompt_parts = ["<ranking>"]
  for i, item in enumerate(sorted_opinions, 1):
    opinion_text = item["opinion"]
    # Look up the ID in the provided map; handle cases where it might not be found
    opinion_id = (
        opinions_map.get(opinion_text, opinion_text)
        if opinions_map
        else opinion_text
    )
    prompt_parts.append(f"{opinion_id}")
    if i < len(sorted_opinions):
      prompt_parts.append(">")
  prompt_parts.append(f"</ranking>\n")

  if q4_question_text and q4_answer_text:
    prompt_parts.append(
        f"<followup_question>{q4_question_text}</followup_question>\n"
    )
    prompt_parts.append(f"<answer>{q4_answer_text}</answer>\n")
  return "".join(prompt_parts)
