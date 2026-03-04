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
Gets bridging scores from Perspective API or Gemini.
Example Usage:
 python3 -m src.get_bridging_scores \
     --input_csv <INPUT_CSV> \
     --output_csv <OUTPUT_CSV> \
     --api_key <API_KEY> \
     --scorer_type PERSPECTIVE
"""

import argparse
import collections
import pandas as pd
from src import get_perspective_scores_lib
from src.get_gemini_scores_lib import ContentScorer
AVERAGE_BRIDGING_COLUMN = "AVERAGE_OF_3_BRIDGING"
BRIDGING_ATTRIBUTES = [
    "CURIOSITY_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
]


def get_bridging_scores(df: pd.DataFrame, text_column: str, api_key: str, scorer_type: str, model_name: str):
  """Score df with bridging attributes using specified scorer."""
  if scorer_type == "GEMINI":
    print(f"Using Gemini ({model_name}) for bridging scoring...")
    scorer = ContentScorer(api_key=api_key, model_name=model_name)
    # Prepare batch for Gemini
    texts_with_ids = [
        {"text": str(text), "row_id": idx}
        for idx, text in df[text_column].items()
    ]
    results = scorer.score(texts_with_ids, BRIDGING_ATTRIBUTES)
    scores_by_row_id = collections.defaultdict(dict)
    for res in results:
      rid = res["row_id"]
      scores_by_row_id[rid].update(res["scores"])
    scores_df = pd.DataFrame.from_dict(scores_by_row_id, orient='index')
    df = df.join(scores_df)
  elif scorer_type == "PERSPECTIVE":
    print("Using Perspective API for bridging scoring...")
    client = get_perspective_scores_lib.init_client(api_key)
    scores_list = [
        get_perspective_scores_lib.score_text(
            client, str(text), BRIDGING_ATTRIBUTES
        )
        for text in df[text_column]
    ]
    scores_df = pd.DataFrame(scores_list, index=df.index)
    df = df.join(scores_df)
  else:
    raise ValueError(f"Unknown scorer_type: {scorer_type}")
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
      "--api_key",
      required=True,
      help="API key for the Perspective API or Gemini.",
  )
  parser.add_argument(
      "--text_column",
      default="quote",
      help="Text column in CSV to score.",
  )
  parser.add_argument(
      "--scorer_type",
      choices=["GEMINI", "PERSPECTIVE"],
      default="PERSPECTIVE",
      help="Backend to use for generating bridging scores.",
  )
  parser.add_argument(
      "--model_name",
      default="gemini-3-flash-preview",
      help="Gemini model name to use when scorer_type is GEMINI.",
  )
  args = parser.parse_args()
  df = pd.read_csv(args.input_csv)
  print(f"Scoring {len(df)} rows from {args.input_csv}")
  df = get_bridging_scores(
      df, args.text_column, args.api_key, args.scorer_type, args.model_name
  )
  df.to_csv(args.output_csv, index=False)
  print(f"Wrote {args.output_csv}")
