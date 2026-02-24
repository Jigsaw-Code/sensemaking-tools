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
"""
Gets briding scores from Perspective API.

Example Usage:
 python3 -m src.get_bridging_scores \
     --input_csv <INPUT_CSV> \
     --output_csv <OUTPUT_CSV> \
     --api_key <PERSPECTIVE_API_KEY>
"""

import argparse
from src import get_perspective_scores_lib
import pandas as pd

AVERAGE_BRIDGING_COLUMN = "AVERAGE_OF_3_BRIDGING"
BRIDGING_ATTRIBUTES = [
    "CURIOSITY_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
]


def get_perspective_scores(df: str, text_column: str, api_key: str):
  """Score df with Perspective API."""
  # Get BRIDGING_ATTRIBUTES for every row with Perspective API
  client = get_perspective_scores_lib.init_client(api_key)
  scores_list = [
      get_perspective_scores_lib.score_text(
          client, str(text), BRIDGING_ATTRIBUTES
      )
      for text in df[text_column]
  ]
  scores_df = pd.DataFrame(scores_list, index=df.index)
  df = df.join(scores_df)
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
      help="API key for the Perspective API.",
  )
  parser.add_argument(
      "--text_column",
      default="representative_text",
      help="Text column in CSV to score with Perspective API.",
  )
  args = parser.parse_args()

  df = pd.read_csv(args.input_csv)
  print(f"Scoring {len(df)} rows from {args.output_csv}")
  df = get_perspective_scores(df, args.text_column, args.api_key)
  df.to_csv(args.output_csv, index=False)
  print(f"Wrote {args.output_csv}")
