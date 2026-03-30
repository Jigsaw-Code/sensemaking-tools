"""
A command-line utility for running simulated jury analyses.

This script runs a simulated jury for a given set of participants and statements,
and outputs the aggregated results.

Example Usage:
  python3 -m src.simulated_jury.main \
    --participants_csv s1_r2_v3_processed.csv \
    --statements_csv statements.csv \
    --output_csv results.csv
"""

import argparse
import asyncio
import os
import pandas as pd
from src.simulated_jury import simulated_jury
from src.simulated_jury import sampling_utils
from src.models import genai_model


def main():
  """Main entry point for the script."""
  parser = argparse.ArgumentParser(description="Run a simulated jury analysis.")
  parser.add_argument(
      "--participants_csv",
      required=True,
      type=str,
      help="Path to the input CSV file with participant data.",
  )
  parser.add_argument(
      "--statements_csv",
      required=True,
      type=str,
      help="Path to the input CSV file with statements to be evaluated.",
  )
  parser.add_argument(
      "--output_csv",
      required=True,
      type=str,
      help="Path to the output CSV file for the results.",
  )
  parser.add_argument(
      "--statement_column",
      default=None,
      type=str,
      help=(
          "Name of the column containing the statements. Defaults to"
          " 'statement' or 'proposition'."
      ),
  )
  parser.add_argument(
      "--model_name",
      default="gemini-2.5-flash-lite",
      type=str,
      help="Name of the model to use.",
  )
  parser.add_argument(
      "--jury_size",
      type=float,
      help=(
          "The size of the simulated jury. If between 0 and 1, it's treated as"
          " a fraction of the total participants; if greater than 1, it's an"
          " absolute number."
      ),
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=15,
      help="The number of statements to process in each batch.",
  )
  parser.add_argument(
      "--approval_scale",
      type=str,
      default="agree_disagree",
      choices=[
          "agree_disagree",
          "agree_disagree_neither",
          "likert_5",
          "likert_5_somewhat",
          "likert_4",
          "likert_4_somewhat",
      ],
      help="The scale of approval voting to use.",
  )
  parser.add_argument(
      "--percent",
      action="store_true",
      help="Output the results as a percentage.",
  )
  parser.add_argument(
      "--true_agree_rate_column",
      type=str,
      help="Column with the true agree rate for error calculation.",
  )
  args = parser.parse_args()

  # --- Load Data ---
  try:
    participants_df = pd.read_csv(args.participants_csv)
    statements_df = pd.read_csv(args.statements_csv)
  except FileNotFoundError as e:
    print(f"Error: {e.filename} not found.")
    return

  if args.true_agree_rate_column:
    if args.true_agree_rate_column not in statements_df.columns:
      print(
          f"Error: Column '{args.true_agree_rate_column}' not found in"
          " statements CSV."
      )
      return

  # --- Sample Participants for the Jury ---
  if args.jury_size is not None:
    try:
      participants_df = sampling_utils.apply_jury_size_sampling(
          participants_df, args.jury_size, verbose=True
      )
    except ValueError as e:
      print(f"Error: {e}")
      return

  # --- Identify Statement Column ---
  statement_column = args.statement_column
  if not statement_column:
    if "statement" in statements_df.columns:
      statement_column = "statement"
    elif "proposition" in statements_df.columns:
      statement_column = "proposition"
    else:
      print(
          "Error: Could not find a 'statement' or 'proposition' column."
          " Please specify with --statement_column."
      )
      return

  statements = statements_df[statement_column].tolist()

  # --- Run Simulation ---
  approval_scale = simulated_jury.ApprovalScale(args.approval_scale)
  model = genai_model.GenaiModel(model_name=args.model_name)
  approval_results_df, _ = asyncio.run(
      simulated_jury.run_simulated_jury(
          participants_df=participants_df,
          statements=statements,
          voting_mode=simulated_jury.VotingMode.APPROVAL,
          model=model,
          batch_size=args.batch_size,
          approval_scale=approval_scale,
      )
  )

  # --- Process Results ---
  if approval_results_df.empty:
    print("No results from the simulated jury.")
    return

  approval_matrix = simulated_jury.build_approval_matrix(
      approval_results_df, approval_scale=approval_scale
  )
  agree_rate = approval_matrix.mean().rename("agree_rate")

  # --- Save Output ---
  results_df = statements_df.merge(
      agree_rate, left_on=statement_column, right_index=True
  )

  if args.true_agree_rate_column:
    true_rate = results_df[args.true_agree_rate_column]
    if true_rate.dtype == "object":
      true_rate = true_rate.str.rstrip("%").astype("float") / 100.0
    results_df["error"] = results_df["agree_rate"] - true_rate

  if args.percent:
    results_df["agree_rate"] = results_df["agree_rate"].apply(
        lambda x: f"{x*100:.1f}%"
    )
    if "error" in results_df.columns:
      results_df["error"] = results_df["error"].apply(lambda x: f"{x*100:.1f}%")

  results_df.to_csv(args.output_csv, index=False)
  print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
  main()
