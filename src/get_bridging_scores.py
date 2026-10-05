# Copyright 2026 Google LLC
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
"""
Gets bridging scores from Gemini or Perspective API.
Example Usage:
 python3 -m src.get_bridging_scores \
     --input_csv <INPUT_CSV> \
     --output_csv <OUTPUT_CSV> \
     --gemini_api_key <GEMINI_API_KEY> \
     --gcloud_api_key <GCLOUD_API_KEY> \
     --scorer_type GEMINI \
     --model_name gemini-3.1-flash-lite-preview
"""

import argparse
import collections
import os
import pandas as pd
from src import get_perspective_scores_lib
from src.get_gemini_scores_lib import ContentScorer
AVERAGE_BRIDGING_COLUMN = "AVERAGE_OF_3_BRIDGING"
BRIDGING_ATTRIBUTES = [
    "CURIOSITY_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
]


def get_bridging_scores(
    df: pd.DataFrame,
    text_column: str,
    gemini_api_key: str,
    gcloud_api_key: str,
    scorer_type: str,
    model_name: str,
) -> pd.DataFrame:
  """Score df with bridging attributes using specified scorer.

  Bridging scores depend only on the text being scored, so each unique text
  is scored once and the scores are copied to every row containing that text.
  This avoids redundant API calls when the same text appears in many rows
  (e.g. categorization output, which has one row per quote/topic/opinion).

  Any bridging score columns already present in df are replaced entirely
  (with a warning) so that old and new scores are never mixed.

  Args:
    df: DataFrame containing the texts to score.
    text_column: Name of the column in df containing the texts.
    gemini_api_key: API key for Gemini, used when scorer_type is "GEMINI".
    gcloud_api_key: API key for Perspective, used when scorer_type is
      "PERSPECTIVE".
    scorer_type: Backend to use, either "GEMINI" or "PERSPECTIVE".
    model_name: Gemini model name, used when scorer_type is "GEMINI".

  Returns:
    A copy of df with one column per bridging attribute plus an
    AVERAGE_BRIDGING_COLUMN. Rows for which the scorer returned no score get
    NaN.

  Raises:
    ValueError: If scorer_type is not recognized.
  """
  score_columns = BRIDGING_ATTRIBUTES + [AVERAGE_BRIDGING_COLUMN]
  existing_columns = [col for col in score_columns if col in df.columns]
  if existing_columns:
    print(
        f"Warning: Existing bridging score columns {existing_columns} found "
        "in input. They will be replaced with newly computed scores."
    )

  texts = df[text_column].astype(str)
  unique_texts = pd.Series(texts.unique())
  print(
      f"Scoring {len(unique_texts)} unique texts across {len(df)} rows "
      f"({len(df) - len(unique_texts)} duplicate rows will reuse scores)."
  )

  if scorer_type == "GEMINI":
    print(f"Using Gemini ({model_name}) for bridging scoring...")
    scorer = ContentScorer(gemini_api_key=gemini_api_key, model_name=model_name)
    # Prepare batch for Gemini, keyed by position in unique_texts.
    texts_with_ids = [
        {"text": text, "row_id": idx}
        for idx, text in unique_texts.items()
    ]
    results = scorer.score(texts_with_ids, BRIDGING_ATTRIBUTES)
    scores_by_row_id = collections.defaultdict(dict)
    for res in results:
      rid = res["row_id"]
      scores_by_row_id[rid].update(res["scores"])
    unique_scores_df = pd.DataFrame.from_dict(scores_by_row_id, orient="index")
  elif scorer_type == "PERSPECTIVE":
    print("Using Perspective API for bridging scoring...")
    client = get_perspective_scores_lib.init_client(gcloud_api_key)
    scores_list = [
        get_perspective_scores_lib.score_text(
            client, text, BRIDGING_ATTRIBUTES
        )
        for text in unique_texts
    ]
    unique_scores_df = pd.DataFrame(scores_list, index=unique_texts.index)
  else:
    raise ValueError(f"Unknown scorer_type: {scorer_type}")

  # Always produce exactly the bridging attribute columns, in a fixed order.
  # Texts or attributes with no results (e.g. failed API calls) get NaN.
  unique_scores_df = unique_scores_df.reindex(
      index=unique_texts.index, columns=BRIDGING_ATTRIBUTES
  )
  # Key scores by text, then expand back out to one row per original row.
  unique_scores_df.index = unique_texts.to_numpy()
  row_scores_df = unique_scores_df.reindex(texts.to_numpy())

  df = df.copy()
  # Assign positionally so a non-unique df index is handled correctly.
  df[BRIDGING_ATTRIBUTES] = row_scores_df.to_numpy()
  # Create an average column, used for ranking.
  df[AVERAGE_BRIDGING_COLUMN] = df[BRIDGING_ATTRIBUTES].mean(axis=1)
  return df


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description=(
          "Scores quotes with bridging attributes and"
          " and selects recommended and backup GoV quotes."
      )
  )
  parser.add_argument(
      "--input_csv", required=True, help="Path to the input CSV file."
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to output CSV file."
  )
  parser.add_argument(
      "--gcloud_api_key",
      help="API key for the Perspective API.",
  )
  parser.add_argument(
      "--gemini_api_key",
      help="API key for Gemini (GenAI).",
  )
  parser.add_argument(
      "--text_column",
      default="quote",
      help="Text column in CSV to score.",
  )
  parser.add_argument(
      "--scorer_type",
      choices=["GEMINI", "PERSPECTIVE"],
      default="GEMINI",
      help="Backend to use for generating bridging scores.",
  )
  parser.add_argument(
      "--model_name",
      default="gemini-3.1-flash-lite-preview",
      help="Gemini model name to use when scorer_type is GEMINI.",
  )
  args = parser.parse_args()
  df = pd.read_csv(args.input_csv)
  print(f"Scoring {len(df)} rows from {args.input_csv}")
  gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
  gcloud_api_key = args.gcloud_api_key or os.getenv("GCLOUD_API_KEY")

  if args.scorer_type == "GEMINI" and not gemini_api_key:
    print("Error: --gemini_api_key or GEMINI_API_KEY environment variable missing.")
    exit(1)

  df = get_bridging_scores(
      df, args.text_column, gemini_api_key, gcloud_api_key, args.scorer_type, args.model_name
  )
  df.to_csv(args.output_csv, index=False)
  print(f"Wrote {args.output_csv}")
