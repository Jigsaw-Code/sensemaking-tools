import logging
import sys
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
from src.social_choice import schulze
from src.social_choice import proportional_approval_voting


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
      "--approval_batch_size",
      type=int,
      default=15,
      help="The number of statements to process in each batch for approval.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--voting_mode",
      type=str,
      default="approval",
      choices=["approval", "ranking", "schulze_pav"],
      help="The voting mode to use.",
  )
  parser.add_argument(
      "--group_by",
      type=str,
      default=None,
      help="Column to group statements by for ranking/PAV evaluations.",
  )
  parser.add_argument(
      "--max_ranking_candidates",
      type=int,
      default=simulated_jury.MAX_STATEMENTS_FOR_RANKING,
      help="Maximum number of statements to rank per group.",
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

  if args.batch_size is not None:
    args.approval_batch_size = args.batch_size

  # --- Load Data ---
  try:
    participants_df = pd.read_csv(args.participants_csv)
    statements_df = pd.read_csv(args.statements_csv)
  except FileNotFoundError as e:
    logging.fatal("Error: %s not found.", e.filename)

  if args.true_agree_rate_column:
    if args.true_agree_rate_column not in statements_df.columns:
      logging.fatal(
          "Error: Column '%s' not found in statements CSV.",
          args.true_agree_rate_column,
      )

  # --- Sample Participants for the Jury ---
  if args.jury_size is not None:
    try:
      participants_df = sampling_utils.apply_jury_size_sampling(
          participants_df, args.jury_size, verbose=True
      )
    except ValueError as e:
      logging.fatal("Error: %s", e)

  # --- Identify Statement Column ---
  statement_column = args.statement_column
  if not statement_column:
    if "statement" in statements_df.columns:
      statement_column = "statement"
    elif "proposition" in statements_df.columns:
      statement_column = "proposition"
    else:
      logging.fatal(
          "Error: Could not find a 'statement' or 'proposition' column. Please"
          " specify with --statement_column."
      )

  statements = statements_df[statement_column].tolist()

  # --- Setup Columns ---
  if args.group_by and args.group_by not in statements_df.columns:
    logging.fatal("Error: Grouping column '%s' not found.", args.group_by)

  # If an existing agree_rate column was supplied by the input, drop it to prevent duplicates
  if "agree_rate" in statements_df.columns:
    print(
        "Warning: Existing 'agree_rate' column found in statements CSV."
        " Overwriting."
    )
    statements_df = statements_df.drop(columns=["agree_rate"])

  # --- Run Simulation ---
  approval_scale = simulated_jury.ApprovalScale(args.approval_scale)
  model = genai_model.GenaiModel(model_name=args.model_name)

  approval_matrix = pd.DataFrame()

  if args.voting_mode in ("approval", "schulze_pav"):
    print(
        f"--- Running APPROVAL simulation on {len(statements)} statements ---"
    )
    approval_results_df, _ = asyncio.run(
        simulated_jury.run_simulated_jury(
            participants_df=participants_df,
            statements=statements,
            voting_mode=simulated_jury.VotingMode.APPROVAL,
            model=model,
            batch_size=args.approval_batch_size,
            approval_scale=approval_scale,
        )
    )

    if not approval_results_df.empty:
      approval_matrix = simulated_jury.build_approval_matrix(
          approval_results_df, approval_scale=approval_scale
      )
      agree_rate = approval_matrix.mean().rename("agree_rate")

      # Use standard left join to append averages
      statements_df = statements_df.merge(
          agree_rate, left_on=statement_column, right_index=True, how="left"
      )
    else:
      logging.fatal("Error: No results from the simulated approval jury.")

  if args.voting_mode in ("ranking", "schulze_pav"):
    # If no grouping column is specified, create a dummy temporary group
    # column so we can reuse the same group iteration loop below.
    group_col = args.group_by if args.group_by else "temp_group"
    if not args.group_by:
      statements_df["temp_group"] = "all"

    for group_name, group_df in statements_df.groupby(group_col):
      group_statements = group_df[statement_column].tolist()

      if len(group_statements) > args.max_ranking_candidates:
        print(
            f"Warning: Group '{group_name}' has {len(group_statements)}"
            " statements, exceeding max_ranking_candidates"
            f" ({args.max_ranking_candidates}). Skipping ranking."
        )
        continue

      print(
          f"--- Running RANK simulation for group: {group_name}"
          f" ({len(group_statements)} statements) ---"
      )
      jury_results_df, _ = asyncio.run(
          simulated_jury.run_simulated_jury(
              participants_df=participants_df,
              statements=group_statements,
              voting_mode=simulated_jury.VotingMode.RANK,
              model=model,
              topic_name=str(group_name) if args.group_by else "All Statements",
          )
      )

      if jury_results_df.empty:
        print(f"No ranking results for group: {group_name}.")
        continue

      jury_preferences = []
      for res in jury_results_df["result"]:
        if res and "ranking" in res:
          ranking = res["ranking"]
          if ranking:
            jury_preferences.append(ranking)

      if not jury_preferences:
        print(f"No valid rankings for group: {group_name}.")
        continue

      schulze_ranking = schulze.get_schulze_ranking(jury_preferences)
      final_ranking = schulze_ranking.get("top_propositions", [])

      for rank_idx, statement in enumerate(final_ranking):
        statements_df.loc[
            statements_df[statement_column] == statement, "schulze_rank"
        ] = (rank_idx + 1)

      if args.voting_mode == "schulze_pav":
        if approval_matrix.empty:
          logging.fatal("Error: Approval matrix is empty, cannot compute PAV.")

        group_approval_matrix = approval_matrix[group_statements]
        pav_slate = proportional_approval_voting.run_schulze_pav_selection(
            ranked_choice_results=jury_preferences,
            approval_matrix=group_approval_matrix,
            k=len(group_statements),
        )
        for rank_idx, statement in enumerate(pav_slate):
          statements_df.loc[
              statements_df[statement_column] == statement, "pav_rank"
          ] = (rank_idx + 1)

    # Clean up the dummy loop column so it doesn't clutter the output CSV
    if "temp_group" in statements_df.columns:
      statements_df = statements_df.drop(columns=["temp_group"])

  # --- Process Results & Save Output ---
  results_df = statements_df

  if args.true_agree_rate_column:
    true_rate = results_df[args.true_agree_rate_column]
    if true_rate.dtype == "object":
      true_rate = true_rate.str.rstrip("%").astype("float") / 100.0
    results_df["error"] = results_df["agree_rate"] - true_rate

  if args.percent and "agree_rate" in results_df.columns:
    results_df["agree_rate"] = results_df["agree_rate"].apply(
        lambda x: f"{x*100:.1f}%" if pd.notna(x) else x
    )
    if "error" in results_df.columns:
      results_df["error"] = results_df["error"].apply(
          lambda x: f"{x*100:.1f}%" if pd.notna(x) else x
      )

  results_df.to_csv(args.output_csv, index=False)
  print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
  main()
