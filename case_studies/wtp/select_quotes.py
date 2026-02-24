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
Processes a CSV file with topics, opinions, and quotes to select a curated
set of top and backup quotes based on topic-opinion popularity and bridging
scores from the Perspective API.

The script generates three output files:
1.  `*_with_bridging_scores.csv`: A copy of the input data with all Perspective
    API bridging scores appended.
2.  `*_top_quotes.csv`: The single best quote (highest average bridging score)
    for each opinion.
3.  `*_backup_quotes.csv`: Backup quotes for the GoV

Example Usage:
 python3 -m case_studies.wtp.select_quotes \
     --input_csv topic_opinion_data.csv \
     --api_key <PERSPECTIVE_API_KEY>
"""

import argparse
import os
import random
import sys

from case_studies.wtp import get_perspective_scores_lib
import pandas as pd

AVERAGE_BRIDGING_COLUMN = "AVERAGE_OF_3_BRIDGING"
BRIDGING_ATTRIBUTES = [
    "CURIOSITY_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
]
RID_COLUMN = "rid"
OPINION_COLUMN = "opinion"
QUOTE_COLUMN = "representative_text"

ALL_ROWS_WITH_BRIDGING_FILE = "all_rows_with_bridging_scores.csv"
RECOMMENDED_GOV_QUOTES_FILE = "recommended_gov_quotes.csv"
BACKUP_GOV_QUOTES_FILE = "backup_gov_quotes.csv"


def get_perspective_scores(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
  """Scores quotes for each of the bridging attributes."""
  client = get_perspective_scores_lib.init_client(api_key)

  scores_list = [
      get_perspective_scores_lib.score_text(
          client, str(text), BRIDGING_ATTRIBUTES
      )
      for text in df[QUOTE_COLUMN]
  ]
  scores_df = pd.DataFrame(scores_list, index=df.index)
  df = df.join(scores_df)
  df[AVERAGE_BRIDGING_COLUMN] = df[BRIDGING_ATTRIBUTES].mean(axis=1)
  return df


def select_gov_quotes(df: pd.DataFrame) -> pd.DataFrame:
  """Selects one quote per opinion, ensuring unique rids."""
  # Remove "other" opinion
  df = df[df[OPINION_COLUMN] != "other"]

  # Randomize df so that if rows have equal bridging scores,
  # the winner is chosen randomly.
  df = df.sample(frac=1).reset_index(drop=True)

  # Sort by descending AVERAGE_BRIDGING score
  df.sort_values(by=AVERAGE_BRIDGING_COLUMN,
      ascending=False, inplace=True)

  # Process opinions in a random order, so that if the
  # same quote appears in multiple opinions, the winner
  # will be chosen randomly
  random_opinion_order = list(set(df[OPINION_COLUMN]))
  random.shuffle(random_opinion_order)

  # For each opinion, pick the top scoring row, and
  # eliminate all other rows for that rid.
  selected_rows = []
  for opinion in random_opinion_order:
    row = df[df[OPINION_COLUMN] == opinion].iloc[0]
    df = df[df[RID_COLUMN] != row[RID_COLUMN]]
    selected_rows.append(row)

  return pd.DataFrame(selected_rows)


def write_to_csv(df: pd.DataFrame, output_dir: str, filename: str):
  """Writes CSV file to disk, ensuring directory exists."""
  os.makedirs(output_dir, exist_ok=True)  # make sure dir exists
  full_filepath = os.path.join(output_dir, filename)
  df.to_csv(full_filepath, index=False)
  print(f"Wrote {full_filepath}")


def write_topic_and_opinion_count_csvs(df: pd.DataFrame, output_dir: str):
  """Writes CSVs containing counts of topics and opinions."""
  topic_counts_df = df.groupby('topic').size()
  topic_counts_df = topic_counts_df.reset_index()
  topic_counts_df.columns = ['topic', 'count']
  topic_counts_df = topic_counts_df.sort_values('count', ascending=False)
  write_to_csv(topic_counts_df, output_dir, 'topic_counts.csv')
  opinion_counts_df = df.groupby(['topic', 'opinion']).size()
  opinion_counts_df = opinion_counts_df.reset_index()
  opinion_counts_df.columns = ['topic', 'opinion', 'count']
  opinion_counts_df = opinion_counts_df.sort_values('count', ascending=False)
  write_to_csv(opinion_counts_df, output_dir, 'opinion_counts.csv')


def select_quotes_for_visual(input_df: pd.DataFrame, output_dir: str):
  """Selects quotes for visual, ensures all participants are represented."""
  # Randomize df so that if rows have equal bridging scores,
  # the winner is chosen randomly.
  df = input_df.sample(frac=1).reset_index(drop=True)

  # step 1: pick the top (max 100) quotes per opinion by
  # average bridging.  It allows for the same participant
  # to appear multiple times, and for the same quote to
  # appear in multiple opinions
  step_1_output_df = (
      df.groupby(OPINION_COLUMN)
      .apply(lambda x: x.nlargest(100, AVERAGE_BRIDGING_COLUMN)))

  # step 2: add all participants not chosen by step 1.  For each
  # participant, pick the quote with the highest average bridging
  step_2_rids = set(df[RID_COLUMN]) - set(step_1_output_df[RID_COLUMN])
  step_2_input_df = df[df[RID_COLUMN].isin(step_2_rids)]
  step_2_indices = step_2_input_df.groupby(RID_COLUMN)[AVERAGE_BRIDGING_COLUMN].idxmax()
  step_2_output_df = step_2_input_df.loc[step_2_indices]

  # Check that step 1 and step 2 together cover all unique rids
  step_1_unique_rid_count = step_1_output_df[RID_COLUMN].nunique()
  step_2_unique_rid_count = step_2_output_df[RID_COLUMN].nunique()
  total_unique_rid_count = df[RID_COLUMN].nunique()
  assert step_1_unique_rid_count + step_2_unique_rid_count == total_unique_rid_count

  # merge steps 1 and 2, then write to disk
  visual_df = pd.concat([step_1_output_df, step_1_output_df], axis=0, ignore_index=True)
  visual_df = visual_df.rename(columns={QUOTE_COLUMN: 'quote'})
  visual_df = visual_df[['topic', 'opinion', 'quote']]
  write_to_csv(visual_df, output_dir, 'quotes_for_visual.csv')


def process_csv(input_path: str, api_key: str, output_dir: str):
  """Processes input CSV to select quotes for GoV and Visual."""
  df = pd.read_csv(input_path)

  # Get Perspective bridging scores, if needed.
  if AVERAGE_BRIDGING_COLUMN not in df.columns:
    print("Fetching Perspective API bridging scores...")
    df = get_perspective_scores(df, api_key)
    print("Successfully fetched Perspective API bridging scores.")
    write_to_csv(df, output_dir, ALL_ROWS_WITH_BRIDGING_FILE)
  else:
    print("Bridging scores already present, skipping Perspective")

  # Get top recommended GoV quotes
  recommended_gov_quotes = select_gov_quotes(df)
  write_to_csv(recommended_gov_quotes, output_dir, RECOMMENDED_GOV_QUOTES_FILE)

  # Remove all rids from top recommend quotes, and get backups
  recommended_gov_quote_rids = set(recommended_gov_quotes[RID_COLUMN])
  df_filtered = df[~df[RID_COLUMN].isin(recommended_gov_quote_rids)]
  backup_gov_quotes = select_gov_quotes(df_filtered)
  write_to_csv(backup_gov_quotes, output_dir, BACKUP_GOV_QUOTES_FILE)

  select_quotes_for_visual(df, output_dir)

  write_topic_and_opinion_count_csvs(df, output_dir)


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
      "--output_dir", required=True, help="Path to output directory."
  )
  parser.add_argument(
      "--api_key",
      required=True,
      help="API key for the Perspective API.",
  )

  args = parser.parse_args()
  process_csv(args.input_csv, args.api_key, args.output_dir)
