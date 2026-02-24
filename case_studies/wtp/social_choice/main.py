"""
A command-line utility for running social choice analyses.

This script analyzes real-world participant ranking data from a CSV file and
outputs a detailed Schulze ranking.

Example Usage:
  python3 -m case_studies.wtp.social_choice.main --input_csv wtp_s1_r2_v3_processed.csv
"""

import argparse
import itertools
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from case_studies.wtp.social_choice import schulze
from typing import List, Dict


# Helper to extract rankings from the CSV
def extract_rankings_from_csv(file_path: str) -> Dict[int, Dict]:
  """
  Extracts participant rankings from a processed CSV file.

  This function reads a CSV file where each row corresponds to a participant's
  rankings of opinions for each topic. It parses column names to identify
  topics and opinions, and extracts the rankings.

  The topic name is extracted from the opinion text, which is expected to be
  in the format "Topic: [Topic Name] - [Opinion]".

  Args:
    file_path: The path to the input CSV file.

  Returns:
    A dictionary mapping each topic ID to its details, including the topic
    name and a list of participant rankings. For example:
    {
      1: {
        "topic_name": "Example Topic",
        "rankings": [
          ["Opinion A", "Opinion B"],
          ["Opinion B", "Opinion A"]
        ]
      },
      ...
    }
    Returns None if the file cannot be read.
  """
  try:
    df = pd.read_csv(file_path)
  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    return None

  ranking_col_pattern = re.compile(r"ranking_(\d+)_a_(\d+)")
  preferences_by_topic = {}

  for rid, group in df.groupby("rid"):
    participant_rankings_by_topic = defaultdict(list)
    for col_name in group.columns:
      match = ranking_col_pattern.match(col_name)
      if match:
        topic_id = int(match.group(1))
        opinion_col_name = col_name.replace("_a_", "_q_")
        if opinion_col_name in group.columns:
          opinion_text = group[opinion_col_name].iloc[0]
          if pd.isna(opinion_text):
            continue
          rank_value = group[col_name].iloc[0]
          if pd.notna(rank_value):
            try:
              rank = int(float(rank_value))
              participant_rankings_by_topic[topic_id].append(
                  (rank, opinion_text)
              )
            except (ValueError, TypeError):
              pass

    for topic_id, rankings in participant_rankings_by_topic.items():
      sorted_rankings = sorted(rankings, key=lambda x: x[0])
      if not sorted_rankings:
        continue

      if topic_id not in preferences_by_topic:
        # Extract topic name from the first ranked opinion
        first_opinion_text = sorted_rankings[0][1]
        topic_match = re.search(r"Topic:\s*\n(.*?)\s*-", first_opinion_text)
        topic_name = (
            topic_match.group(1).strip() if topic_match else "Unknown Topic"
        )
        preferences_by_topic[topic_id] = {
            "topic_name": topic_name,
            "rankings": [],
        }

      ranked_opinions = []
      for rank, opinion in sorted_rankings:
        # Clean the opinion text
        cleaned_opinion = re.sub(r"^Topic: \s*.*?\s*-\s*", "", opinion)
        ranked_opinions.append(cleaned_opinion)

      if ranked_opinions:
        preferences_by_topic[topic_id]["rankings"].append(ranked_opinions)

  return preferences_by_topic


def analyze_r2_data(csv_file_path: str):
  """
  Loads R2 data, runs Schulze, and prints detailed ranking results.
  """
  print(f"--- Analyzing Schulze Rankings for {csv_file_path} ---")

  rankings_by_topic = extract_rankings_from_csv(csv_file_path)
  if not rankings_by_topic:
    print("Could not extract any ranking data to analyze.")
    return

  for topic_id, topic_data in sorted(rankings_by_topic.items()):
    topic_name = topic_data["topic_name"]
    rankings = topic_data["rankings"]

    print(f"\n--- Topic {topic_id} - {topic_name} ---")

    try:
      # The Schulze implementation uses the opinion text as a unique
      # identifier for each item being ranked. While the algorithm itself
      # is based on the order of rankings, the text is used to map
      # opinions to indices in the preference matrix.
      result = schulze.get_schulze_ranking(rankings)
      d_matrix = result.get("preference_matrix")
      beatpaths = result.get("schulze_matrix", [])
      # Note: the schulze implementation uses somewhat inconsistent notation
      # around "statement" vs "proposition", but in this context, they are
      # what we call "opinions", so using that language here.
      ranked_opinions = result.get("top_propositions", [])
      all_opinions = result.get("orig_statements", [])

      if d_matrix is None or not ranked_opinions or not all_opinions:
        print("  Could not generate ranking or required matrices.")
        continue

      opinion_to_index = {s: i for i, s in enumerate(all_opinions)}

      print("\n  Full Ranking and Strength Details:")
      for i in range(len(ranked_opinions)):
        current_opinion = ranked_opinions[i]
        print(f"  {i+1}. {current_opinion}")

        if i < len(ranked_opinions) - 1:
          next_opinion = ranked_opinions[i + 1]
          current_idx = opinion_to_index[current_opinion]
          next_idx = opinion_to_index[next_opinion]

          p_strength = beatpaths[current_idx, next_idx]
          d_wins = d_matrix[current_idx, next_idx]
          d_losses = d_matrix[next_idx, current_idx]
          difference = d_wins - d_losses

          print(f"     - Strongest Path (vs next): {p_strength:.2f}")
          print(
              f"     - Direct Beat Strength (vs next): {d_wins} to"
              f" {d_losses} -> {difference}"
          )
        else:
          print("     - (End of ranking)")
        print()

    except Exception as e:
      print(
          "  An error occurred during Schulze calculation for topic"
          f" {topic_id}: {e}"
      )


def main():
  """Main entry point for the script."""
  parser = argparse.ArgumentParser(
      description="Run Schulze method analysis on participant ranking data."
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      type=str,
      help="Path to the input CSV file with processed R2 data.",
  )
  args = parser.parse_args()
  analyze_r2_data(args.input_csv)


if __name__ == "__main__":
  main()
