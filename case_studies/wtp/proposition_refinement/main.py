"""
Command line utility for running simulated jury on propositions.
"""

import argparse
import asyncio
import os
import re
import pickle
import json
import time
import pandas as pd
import functools
import copy
import logging
import sys
import collections
from case_studies.wtp.simulated_jury import simulated_jury
from case_studies.wtp.simulated_jury import sampling_utils
from case_studies.wtp.social_choice import schulze
from case_studies.wtp.proposition_refinement import nuanced_propositions
from case_studies.wtp.proposition_refinement import topic_deduplication
from case_studies.wtp.proposition_refinement import deduplication
from case_studies.wtp.social_choice import proportional_approval_voting
from case_studies.wtp import participation
from case_studies.wtp import runner_utils
from case_studies.wtp.models import genai_model


def pipeline_stage(name=None, check_fn=None):
  """
  A decorator for a pipeline stage that handles logging, checkpointing,
  timing, and skipping execution.
  """

  def decorator(func):
    stage_name = name or func.__name__

    @functools.wraps(func)
    async def wrapper(world_model, args, *extra_args, **kwargs):
      # Determine if the stage should be skipped
      should_skip = False
      if check_fn:
        # Manual mode: Use the provided function to check for completion
        should_skip = check_fn(world_model)
      else:
        # Automatic mode: Check if the stage name is in the completed list
        should_skip = stage_name in world_model.get("completed_stages", [])

      if should_skip:
        print(f"--- Skipping '{stage_name}' (already completed) ---")
        return world_model

      # Run the actual pipeline stage logic
      print(f"\n--- Running Pipeline Stage: {stage_name} ---")
      start_time = time.monotonic()
      result_data = await func(world_model, args, *extra_args, **kwargs)
      end_time = time.monotonic()
      duration = end_time - start_time
      print(f"--- Stage '{stage_name}' finished in {duration:.2f} seconds ---")

      # In automatic mode, mark the stage as complete
      if not check_fn:
        if "completed_stages" not in result_data:
          result_data["completed_stages"] = []
        result_data["completed_stages"].append(stage_name)

      # Save the checkpoint
      checkpoint_filename = args.output_pkl.replace(
          ".pkl", f"_{stage_name}.pkl"
      )
      with open(checkpoint_filename, "wb") as f:
        pickle.dump(result_data, f)
      print(f"Checkpoint for '{stage_name}' saved to {checkpoint_filename}")

      return result_data

    return wrapper

  return decorator


def reconstitute_participant_data(by_opinion_df, args):
  """Merges R1 and R2 participant data into a single jury pool."""
  if args.verbose:
    print("\n--- Participant Data Reconstitution & Analysis ---")
  all_r1_dfs = [row["r1_df"] for _, row in by_opinion_df.iterrows()]
  # Because the R1 data comes to us broken down by topic, we need to
  # concatenate these dataframes together to get the full dataset.
  full_r1_df = (
      pd.concat(all_r1_dfs)
      .drop_duplicates(subset=["rid"])
      .reset_index(drop=True)
  )
  all_r2_dfs = [
      row["r2_df"]
      for _, row in by_opinion_df.iterrows()
      if not row["r2_df"].empty
  ]

  if not all_r2_dfs:
    merged_df = full_r1_df
  else:
    # The R2 data is subset by topic, containing only the columns relevant to
    # that topic's propositions. We need to merge these DataFrames column by
    # column to reconstruct the full R2 dataset. This is a bit tricky.
    # TODO: Ideally, this reconstituted data should be a first-class
    # entry in the world model structure so we don't have to compute it here.
    reconstituted_r2_df = all_r2_dfs[0].set_index("rid")
    for i in range(1, len(all_r2_dfs)):
      reconstituted_r2_df = reconstituted_r2_df.combine_first(
          all_r2_dfs[i].set_index("rid")
      )
    reconstituted_r2_df = reconstituted_r2_df.reset_index()
    # This is the merge of the R1 and R2 data. It is computed as an outer merge
    # to allow for missing entries from either collection, which supports
    # refilling the participant panel if needed.
    merged_df = pd.merge(
        full_r1_df,
        reconstituted_r2_df,
        on="rid",
        how="outer",
        indicator=True,
    )

  print(f"--- Reconstituted jury pool with {len(merged_df)} participants ---")
  return merged_df


def validate_jury_pool_data(jury_pool_df: pd.DataFrame):
  """
  Inspects the jury pool DataFrame to validate if it contains the expected
  data columns and structure for running a simulated jury.
  """
  print("\n--- Validating Structure of Jury Pool Data ---")
  if jury_pool_df.empty:
    print("  - ❌ Validation Failed: Jury pool is empty.")
    return

  sample_size = min(len(jury_pool_df), 3)
  print(f"  - Inspecting the first {sample_size} participant records...")

  spec = {
      "R1 Conversation Data": "survey_text",
      "Visual Question Response": "visual_question_1",
      "GOV Responses": r"question_\d+$",
      "Ranking Task Responses": r"ranking_\d+_a_\d+",
  }
  num_gov_responses_expected = 4
  num_ranking_tasks_expected = 4

  for i, row in jury_pool_df.head(sample_size).iterrows():
    print(f"\n  --- Participant {row.get('rid', 'N/A')} ---")

    # Check for R1 data
    if spec["R1 Conversation Data"] in row and pd.notna(
        row[spec["R1 Conversation Data"]]
    ):
      print("    - ✅ R1 Conversation Data: Present.")
    else:
      print("    - ❌ R1 Conversation Data: Missing or empty.")

    # Check for Visual Question
    if "visual_question_1" in row and pd.notna(row["visual_question_1"]):
      print("    - ✅ Visual Question Response: Present.")
    elif "meta_question_1" in row and pd.notna(row["meta_question_1"]):
      print("    - ✅ Visual Question Response (as meta_question_1): Present.")
    else:
      print("    - ❌ Visual Question Response: Missing.")

    # Check for GOV responses
    gov_cols = [
        col for col in row.index if re.match(spec["GOV Responses"], str(col))
    ]
    num_gov_found = sum(
        1 for col in gov_cols if col in row and pd.notna(row[col])
    )
    if num_gov_found >= num_gov_responses_expected:
      print(
          f"    - ✅ GOV Responses: Found {num_gov_found} (expected at least"
          f" {num_gov_responses_expected})."
      )
    else:
      print(
          f"    - ❌ GOV Responses: Found {num_gov_found}, expected at least"
          f" {num_gov_responses_expected}."
      )

    # Check for Ranking tasks
    ranking_cols = [
        col
        for col in row.index
        if re.match(spec["Ranking Task Responses"], str(col))
    ]
    ranking_task_ids = set(
        re.match(r"ranking_(\d+)_.*", str(col)).group(1)
        for col in ranking_cols
        if re.match(r"ranking_(\d+)_.*", str(col))
    )
    num_ranking_sets_found = len(ranking_task_ids)

    if num_ranking_sets_found >= num_ranking_tasks_expected:
      print(
          f"    - ✅ Ranking Tasks: Found {num_ranking_sets_found} sets"
          f" (expected at least {num_ranking_tasks_expected})."
      )
    else:
      print(
          f"    - ❌ Ranking Tasks: Found {num_ranking_sets_found} sets,"
          f" expected at least {num_ranking_tasks_expected}."
      )

  print("\n" + "-" * 60)


def _extract_full_r1_data(by_opinion_df: pd.DataFrame) -> pd.DataFrame:
  """Extracts and consolidates all R1 data from the world model."""
  all_r1_dfs = [row["r1_df"] for _, row in by_opinion_df.iterrows()]
  return (
      pd.concat(all_r1_dfs)
      .drop_duplicates(subset=["rid"])
      .reset_index(drop=True)
  )


def _create_all_participant_data(
    full_r1_df: pd.DataFrame, processed_r2_path: str
) -> pd.DataFrame:
  """Merges the full R1 data with the full R2 data from a CSV."""
  print(
      "--- Creating 'all_participant_data' by merging R1 data with"
      f" {processed_r2_path} ---"
  )

  # 1. Load the full R2 data
  full_r2_df = pd.read_csv(processed_r2_path)

  # 2. Validation Check
  r1_pids = set(full_r1_df["rid"])
  r2_pids = set(full_r2_df["rid"])
  print(f"  - Found {len(r1_pids)} unique participants in R1 data.")
  print(f"  - Found {len(r2_pids)} unique participants in R2 data.")
  print(
      f"  - {len(r1_pids.intersection(r2_pids))} participants are in both"
      " datasets."
  )

  # 3. Merge into the final DataFrame
  merged_df = pd.merge(
      full_r1_df, full_r2_df, on="rid", how="outer", suffixes=("_r1", "_r2")
  )
  print(
      f"--- Created 'all_participant_data' with {len(merged_df)} total"
      " participants. ---"
  )
  return merged_df


def get_jury_pool(
    world_model: dict, by_opinion_df: pd.DataFrame, args: argparse.Namespace
) -> (pd.DataFrame, dict):
  """
  Determines the definitive jury pool, creating it if necessary.
  Returns the jury pool DataFrame and the (potentially updated) world model.
  """
  if (
      "all_participant_data" in world_model
      and not world_model["all_participant_data"].empty
  ):
    print(
        "--- Found 'all_participant_data' in the world model. Using it for the"
        " jury pool. ---"
    )
    return world_model["all_participant_data"], world_model

  if args.processed_r2_data:
    try:
      full_r1_df = _extract_full_r1_data(by_opinion_df)
      jury_pool_df = _create_all_participant_data(
          full_r1_df, args.processed_r2_data
      )
      world_model["all_participant_data"] = jury_pool_df
      return jury_pool_df, world_model
    except FileNotFoundError:
      print(
          "  - ❌ ERROR: The file specified by --processed_r2_data was not"
          f" found: {args.processed_r2_data}"
      )
      sys.exit(1)
    except Exception as e:
      print(
          "  - ❌ ERROR: An unexpected error occurred while creating"
          f" participant data: {e}"
      )
      sys.exit(1)

  # Fallback case
  print("\n" + "=" * 80)
  print("  ⚠️ WARNING: Complete participant data not found.")
  print(
      "  'all_participant_data' was not in the world model, and"
      " --processed_r2_data was not provided."
  )
  print(
      "  Falling back to the original data reconstitution method, which may"
      " result in an"
  )
  print(
      "  incomplete jury pool. This could impact the quality of simulated jury"
      " results."
  )
  print("=" * 80 + "\n")
  jury_pool_df = reconstitute_participant_data(by_opinion_df, args)
  return jury_pool_df, world_model


@pipeline_stage(
    name="r2_opinion_ranking",
    check_fn=lambda data: "r2_opinion_ranking" in data,
)
async def run_r2_opinion_ranking(world_model, args, jury_pool_df):
  """
  Runs Schulze ranking on the real R2 participant opinion rankings.
  """
  print("--- Running Schulze ranking on real R2 participant data ---")
  world_model = copy.deepcopy(world_model)

  # Re-using the extraction logic from our analysis script
  ranking_col_pattern = re.compile(r"ranking_(\d+)_a_(\d+)")
  preferences_by_topic = collections.defaultdict(list)

  for rid, group in jury_pool_df.groupby("rid"):
    participant_rankings_by_topic = collections.defaultdict(list)
    for col_name in group.columns:
      match = ranking_col_pattern.match(str(col_name))
      if match:
        topic_id = int(match.group(1))
        opinion_col_name = col_name.replace("_a_", "_q_")
        if opinion_col_name in group.columns:
          # Get the actual value from the series using .iloc[0]
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
      ranked_opinions = [opinion for rank, opinion in sorted_rankings]
      if ranked_opinions:
        preferences_by_topic[topic_id].append(ranked_opinions)

  if not preferences_by_topic:
    print("  - No R2 ranking data found to analyze. Skipping.")
    world_model["r2_opinion_ranking"] = {}
    return world_model

  # Run Schulze for each topic and store results
  all_topic_rankings = {}
  for topic_id, preferences in sorted(preferences_by_topic.items()):
    if len(preferences) < 2:
      continue

    result = schulze.get_schulze_ranking(preferences)
    all_topic_rankings[f"topic_{topic_id}"] = result

    # Also print the results for immediate feedback
    topic_name = "Unknown Topic"
    if preferences and preferences[0]:
      match = re.search(r"Topic:\s*\n(.*?)\s*-", preferences[0][0])
      if match:
        topic_name = match.group(1).strip()
    print(f"\n  --- Topic {topic_id} - {topic_name} ---")
    for i, prop in enumerate(result.get("top_propositions", [])):
      cleaned_opinion = re.sub(r"^Topic: \s*.*?\s*-\s*", "", prop)
      print(f"  {i+1}. {cleaned_opinion}")

  world_model["r2_opinion_ranking"] = all_topic_rankings
  return world_model


@pipeline_stage(
    name="initial_approval_jury",
    check_fn=lambda data: "initial_approval_matrix" in data,
)
async def run_initial_approval_jury(
    world_model, args, jury_pool_df, sim_jury_model
):
  """
  Runs a single, batched approval vote across all simple propositions.
  """
  world_model = copy.deepcopy(world_model)
  by_opinion_df = world_model["world_model"]

  # 1. Gather all unique propositions from the entire model
  all_propositions = []
  for _, row in by_opinion_df.iterrows():
    if "propositions" in row and not row["propositions"].empty:
      all_propositions.extend(
          p_row["proposition"] for _, p_row in row["propositions"].iterrows()
      )

  unique_propositions = pd.Series(all_propositions).unique().tolist()

  if not unique_propositions:
    print(
        "No propositions found to run the initial approval jury on. Skipping."
    )
    world_model["initial_approval_matrix"] = pd.DataFrame()
    return world_model

  # 2. Run a single, large, batched approval simulation
  print(
      "\n--- Running initial batched APPROVAL simulation for"
      f" {len(unique_propositions)} unique propositions ---"
  )
  approval_results_df, stats_summary = await simulated_jury.run_simulated_jury(
      jury_pool_df,
      unique_propositions,
      simulated_jury.VotingMode.APPROVAL,
      model=sim_jury_model,
      topic_name="All Simple Propositions",
      batch_size=args.approval_batch_size,
  )
  world_model["simulated_jury_stats"].append(stats_summary)

  # 3. Build and store the matrix
  approval_matrix = simulated_jury.build_approval_matrix(approval_results_df)
  world_model["initial_approval_matrix"] = approval_matrix

  return world_model


@pipeline_stage(
    name="opinion_level_jury",
    check_fn=lambda data: "opinion_level_schulze_ranking"
    in data.get("world_model", {}).columns,
)
async def run_jury_by_opinion(world_model, args, jury_pool_df, sim_jury_model):
  """
  Runs a simulated jury for each (topic, opinion) group on their propositions.
  """
  print(
      "--- Jury pool size for all simulations:"
      f" {len(jury_pool_df)} participants ---"
  )
  world_model = copy.deepcopy(world_model)
  by_opinion_df = world_model["world_model"]

  # Initialize new columns to prevent errors when assigning list-like objects
  if "opinion_level_schulze_ranking" not in by_opinion_df.columns:
    by_opinion_df["opinion_level_schulze_ranking"] = pd.Series(dtype="object")
  if "opinion_level_simulation_results" not in by_opinion_df.columns:
    by_opinion_df["opinion_level_simulation_results"] = pd.Series(
        dtype="object"
    )
  if "opinion_level_pav_ranking" not in by_opinion_df.columns:
    by_opinion_df["opinion_level_pav_ranking"] = pd.Series(dtype="object")
  if "opinion_level_approval_results" not in by_opinion_df.columns:
    by_opinion_df["opinion_level_approval_results"] = pd.Series(dtype="object")

  for index, row in by_opinion_df.iterrows():
    if "propositions" not in row or row["propositions"].empty:
      print(
          f"No propositions for topic/opinion: {row['topic']}/{row['opinion']}."
          " Skipping."
      )
      continue
    propositions = [
        p_row["proposition"] for _, p_row in row["propositions"].iterrows()
    ]
    if not propositions:
      print(
          "Empty propositions list for topic/opinion:"
          f" {row['topic']}/{row['opinion']}. Skipping."
      )
      continue

    print(
        "\n--- Running RANK simulation for topic/opinion:"
        f" {row['topic']}/{row['opinion']} ---"
    )

    # Optimization: If there are few propositions, we skip the jury step.
    if len(propositions) <= args.propositions_per_opinion:
      print(
          f"  - Only {len(propositions)} propositions found (<= limit"
          f" {args.propositions_per_opinion}). Skipping jury."
      )
      # We intentionally do NOT populate the ranking columns with fake data.
      # The downstream aggregation stage will handle missing rankings by
      # falling back to the raw propositions.
      continue

    jury_results_df, stats_summary = await simulated_jury.run_simulated_jury(
        jury_pool_df,
        propositions,
        simulated_jury.VotingMode.RANK,
        model=sim_jury_model,
        topic_name=row["topic"],
        opinion_name=row["opinion"],
    )
    world_model["simulated_jury_stats"].append(stats_summary)

    if jury_results_df.empty:
      print(
          "No ranking results for topic/opinion:"
          f" {row['topic']}/{row['opinion']}. Skipping ranking."
      )
      continue

    jury_preferences = []
    for res in jury_results_df["result"]:
      if res and "ranking" in res:
        ranking = res["ranking"]
        if ranking:  # Ensure the ranking list is not empty
          jury_preferences.append(ranking)

    if not jury_preferences:
      print(
          "No valid rankings for topic/opinion:"
          f" {row['topic']}/{row['opinion']}. Skipping ranking."
      )
      continue

    schulze_ranking = schulze.get_schulze_ranking(jury_preferences)
    if "top_propositions" in schulze_ranking:
      by_opinion_df.at[index, "opinion_level_schulze_ranking"] = (
          schulze_ranking["top_propositions"]
      )
    by_opinion_df.at[index, "opinion_level_simulation_results"] = (
        jury_results_df
    )

    # --- Bridging Selection (if enabled) ---
    if args.run_pav_selection:
      if (
          "initial_approval_matrix" not in world_model
          or world_model["initial_approval_matrix"].empty
      ):
        print(
            "  - WARNING: Cannot run bridging selection for opinion"
            f" '{row['opinion']}' due to missing approval matrix."
        )
        continue

      # Filter the main approval matrix for the propositions in this opinion group
      opinion_approval_matrix = world_model["initial_approval_matrix"][
          propositions
      ]

      pav_slate = proportional_approval_voting.run_schulze_pav_selection(
          ranked_choice_results=jury_preferences,
          approval_matrix=opinion_approval_matrix,
          k=len(propositions),  # Get a full ranking
      )
      by_opinion_df.at[index, "opinion_level_pav_ranking"] = pav_slate

  return world_model


def _aggregate_propositions_for_topic(
    topic_group: pd.DataFrame, approval_matrix: pd.DataFrame
) -> pd.DataFrame:
  """
  Aggregates nested proposition DataFrames from a topic group into a
  single DataFrame with a clean, unique index, preserving metadata.
  """
  if not any("propositions" in row for _, row in topic_group.iterrows()):
    return pd.DataFrame()

  # Calculate approval rates for all propositions in the matrix once for efficiency
  all_approval_rates = approval_matrix.mean()

  enriched_props_list = []
  for _, row in topic_group.iterrows():
    if "propositions" in row and not row["propositions"].empty:
      props_df = row["propositions"].copy()
      props_df["topic"] = row["topic"]
      props_df["opinion"] = row["opinion"]
      props_df["r1_quotes_by_opinion"] = row["r1_df_length"]
      # Map the pre-calculated approval rates to the propositions
      props_df["approval_rate"] = props_df["proposition"].map(
          all_approval_rates
      )
      enriched_props_list.append(props_df)

  if not enriched_props_list:
    return pd.DataFrame()

  # The critical step: concatenate the enriched DataFrames and reset the index.
  return pd.concat(enriched_props_list).reset_index(drop=True)


@pipeline_stage(
    name="topic_level_jury",
    check_fn=lambda data: "topic_level_results" in data,
)
async def run_jury_by_topic(world_model, args, jury_pool_df, sim_jury_model):
  """
  Aggregates top propositions from opinions and runs a jury for each topic.
  """
  world_model = copy.deepcopy(world_model)
  by_opinion_df = world_model["world_model"]
  topic_results = []

  for topic, group in by_opinion_df.groupby("topic"):
    print(f"\n--- Aggregating and running jury for topic: {topic} ---")
    top_opinion_propositions = []

    ranking_column_to_use = "opinion_level_schulze_ranking"
    if args.run_pav_selection and "opinion_level_pav_ranking" in group.columns:
      print(f"  - Using PAV-based ranking for opinion-to-topic aggregation.")
      ranking_column_to_use = "opinion_level_pav_ranking"

    for _, row in group.iterrows():
      # ranking is the list of proposition strings ordered by the given column
      ranking = row.get(ranking_column_to_use, [])
      if isinstance(ranking, list) and ranking:
        top_opinion_propositions.extend(
            ranking[: args.propositions_per_opinion]
        )
      else:
        # Fallback: If no ranking exists (e.g., optimization skipped jury),
        # check if we should include all propositions.
        props_in_row = []
        if (
            "propositions" in row
            and isinstance(row["propositions"], pd.DataFrame)
            and not row["propositions"].empty
        ):
          props_in_row = row["propositions"]["proposition"].tolist()

        if props_in_row and len(props_in_row) <= args.propositions_per_opinion:
          print(
              f"  - No ranking for opinion '{row['opinion']}', but prop count"
              f" ({len(props_in_row)}) <= limit. Including all."
          )
          top_opinion_propositions.extend(props_in_row)
        else:
          raise ValueError(
              f"No valid ranking for opinion '{row['opinion']}' and prop count"
              f" ({len(props_in_row)}) > limit"
              f" ({args.propositions_per_opinion}). This should not happen."
          )

    # Remove duplicates that might have won in multiple opinion groups
    unique_propositions = pd.Series(top_opinion_propositions).unique().tolist()

    if not unique_propositions:
      print(f"No propositions to rank for topic: {topic}. Skipping.")
      continue

    jury_results_df, stats_summary = await simulated_jury.run_simulated_jury(
        jury_pool_df,
        unique_propositions,
        simulated_jury.VotingMode.RANK,
        model=sim_jury_model,
        topic_name=topic,
    )
    world_model["simulated_jury_stats"].append(stats_summary)

    if jury_results_df.empty:
      print(f"No jury results for topic: {topic}. Skipping ranking.")
      continue

    jury_preferences = []
    for res in jury_results_df["result"]:
      if res and "ranking" in res:
        ranking = res["ranking"]
        if ranking:  # Ensure the ranking list is not empty
          jury_preferences.append(ranking)

    if not jury_preferences:
      print(f"No valid rankings for topic: {topic}. Skipping ranking.")
      continue

    schulze_ranking = schulze.get_schulze_ranking(jury_preferences)
    final_ranking = schulze_ranking.get("top_propositions", [])

    # --- PAV Selection (if enabled) ---
    topic_pav_ranking = []
    if args.run_pav_selection:
      if (
          "initial_approval_matrix" not in world_model
          or world_model["initial_approval_matrix"].empty
      ):
        print(
            f"  - WARNING: Cannot run bridging selection for topic '{topic}'"
            " due to missing approval matrix."
        )
      else:
        # Filter the main approval matrix for the propositions in this topic group
        topic_approval_matrix = world_model["initial_approval_matrix"][
            unique_propositions
        ]

        topic_pav_ranking = (
            proportional_approval_voting.run_schulze_pav_selection(
                ranked_choice_results=jury_preferences,
                approval_matrix=topic_approval_matrix,
                k=len(unique_propositions),  # Get a full ranking
            )
        )

    # Find the original, nested proposition data for the final propositions
    all_props_in_topic = _aggregate_propositions_for_topic(
        group, world_model["initial_approval_matrix"]
    )
    # Use .loc to ensure we are selecting rows from the original DataFrame
    # with all its columns, preventing the 'opinion' column from being dropped.
    final_topic_propositions_df = all_props_in_topic.loc[
        all_props_in_topic["proposition"].isin(final_ranking)
    ].copy()

    topic_results.append({
        "topic": topic,
        "propositions": final_topic_propositions_df,
        "full_schulze_ranking": final_ranking,
        "topic_level_pav_ranking": topic_pav_ranking,
        "simulation_results": jury_results_df,
        "r1_quotes_by_topic": group["r1_df_length"].sum(),
    })

  world_model["topic_level_results"] = pd.DataFrame(topic_results)
  return world_model


def _check_if_deduplication_has_run(world_model):
  """Check function to see if deduplication results are present."""
  topic_df = world_model["topic_level_results"]
  if topic_df.empty or "propositions" not in topic_df.columns:
    return False

  if topic_df["propositions"].notna().any():
    first_prop_df = topic_df["propositions"].dropna().iloc[0]
    if isinstance(first_prop_df, pd.DataFrame) and "selected" in first_prop_df:
      return (
          topic_df["propositions"]
          .dropna()
          .apply(lambda df: "selected" in df.columns and df["selected"].any())
          .any()
      )
  return False


@pipeline_stage(name="deduplication", check_fn=_check_if_deduplication_has_run)
async def run_deduplication_stage(world_model, args, nuanced_props_model):
  """Runs the deduplication and refined selection stage."""
  world_model = copy.deepcopy(world_model)
  topic_level_data = world_model["topic_level_results"]

  ranking_column = "full_schulze_ranking"
  if args.run_pav_selection:
    if "topic_level_pav_ranking" in topic_level_data.columns:
      print("\n--- Using PAV-based ranking for deduplication stage ---")
      ranking_column = "topic_level_pav_ranking"
    else:
      raise ValueError(
          "PAV selection was requested, but 'topic_level_pav_ranking' column"
          " not found in topic_level_results. Was the topic_level_jury stage"
          " run with PAV enabled?"
      )
  else:
    print("\n--- Using Schulze ranking for deduplication stage ---")

  if args.deduplication_method == "rank_filling":
    dedup_results, _ = await deduplication.run_deduplication(
        topic_level_data,
        args.final_propositions_per_topic,
        nuanced_props_model,
        ranking_column=ranking_column,
    )
  elif args.deduplication_method == "topic_based":
    dedup_results, _ = await topic_deduplication.run_topic_deduplication(
        topic_level_data,
        args.final_propositions_per_topic,
        nuanced_props_model,
        ranking_column=ranking_column,
    )
  world_model["topic_level_results"] = dedup_results
  return world_model


@pipeline_stage(
    name="nuanced_proposition_generation",
    check_fn=lambda data: "nuanced_propositions" in data
    and not data["nuanced_propositions"].empty,
)
async def generate_nuanced_propositions(world_model, args, nuanced_props_model):
  """Generates nuanced propositions from the top-ranked simple ones."""
  world_model = copy.deepcopy(world_model)
  by_topic_df = world_model["topic_level_results"]
  # Reconstruct top_propositions_by_topic from the 'selected' column, which is
  # the definitive output of the deduplication stage.
  reconstructed_top_props = {}
  for _, row in by_topic_df.iterrows():
    props_df = row.get("propositions")
    if isinstance(props_df, pd.DataFrame) and "selected" in props_df.columns:
      selected_props = props_df[props_df["selected"]]["proposition"].tolist()
      reconstructed_top_props[row["topic"]] = selected_props
  top_propositions_by_topic = reconstructed_top_props

  generated_nuanced_propositions_df, _ = (
      await nuanced_propositions.combine_propositions(
          top_propositions_by_topic,
          model=nuanced_props_model,
          additional_context=args.additional_context,
      )
  )
  world_model["nuanced_propositions"] = generated_nuanced_propositions_df
  return world_model


def _get_nuanced_propositions_list(world_model):
  """Extracts the list of nuanced propositions from the world model."""
  generated_nuanced_propositions_df = world_model.get(
      "nuanced_propositions", pd.DataFrame()
  )
  nuanced_propositions_list = []
  if (
      not generated_nuanced_propositions_df.empty
      and "result" in generated_nuanced_propositions_df.columns
  ):
    res = generated_nuanced_propositions_df.iloc[0]["result"]
    if res and isinstance(res, list):
      nuanced_propositions_list = res
    elif res and isinstance(res, str):
      print("Parsing nuanced propositions from string format.")
      propositions = [line.strip() for line in res.split("\n") if line.strip()]
      nuanced_propositions_list = [
          re.sub(r"^\s*\d+\.\s*", "", prop) for prop in propositions
      ]
  return nuanced_propositions_list


@pipeline_stage(
    name="nuanced_approval_jury",
    check_fn=lambda data: "nuanced_approval_matrix" in data,
)
async def run_nuanced_approval_jury(
    world_model, args, jury_pool_df, sim_jury_model
):
  """Runs an APPROVAL simulated jury on the generated nuanced propositions."""
  world_model = copy.deepcopy(world_model)
  nuanced_propositions_list = _get_nuanced_propositions_list(world_model)

  if nuanced_propositions_list:
    print("\n--- Running APPROVAL simulation for nuanced propositions ---")
    nuanced_approval_results_df, stats_summary = (
        await simulated_jury.run_simulated_jury(
            jury_pool_df,
            nuanced_propositions_list,
            simulated_jury.VotingMode.APPROVAL,
            model=sim_jury_model,
            topic_name="Nuanced Propositions",
            batch_size=args.approval_batch_size,
        )
    )
    world_model["simulated_jury_stats"].append(stats_summary)

    nuanced_approval_matrix = simulated_jury.build_approval_matrix(
        nuanced_approval_results_df
    )
    world_model["nuanced_approval_matrix"] = nuanced_approval_matrix
  else:
    print("No nuanced propositions to run approval jury on.")
    world_model["nuanced_approval_matrix"] = pd.DataFrame()
  return world_model


@pipeline_stage(
    name="nuanced_ranking_jury",
    check_fn=lambda data: "nuanced_jury_results" in data
    and not data["nuanced_jury_results"].empty,
)
async def run_nuanced_ranking_jury(
    world_model, args, jury_pool_df, sim_jury_model
):
  """Runs a RANKING simulated jury on the generated nuanced propositions."""
  world_model = copy.deepcopy(world_model)
  nuanced_propositions_list = _get_nuanced_propositions_list(world_model)

  if nuanced_propositions_list:
    print("\n--- Running RANK simulation for nuanced propositions ---")
    nuanced_jury_results_df, stats_summary = (
        await simulated_jury.run_simulated_jury(
            jury_pool_df,
            nuanced_propositions_list,
            simulated_jury.VotingMode.RANK,
            model=sim_jury_model,
            topic_name="Nuanced Propositions",
        )
    )
    world_model["simulated_jury_stats"].append(stats_summary)
    world_model["nuanced_jury_results"] = nuanced_jury_results_df
  else:
    print("No nuanced propositions to run ranking jury on.")
  return world_model


@pipeline_stage(
    name="nuanced_ranking",
    check_fn=lambda data: "top_nuanced_propositions" in data
    and not data["top_nuanced_propositions"].empty,
)
async def rank_nuanced_propositions(world_model, args):
  """Performs Schulze and PAV ranking on the nuanced proposition jury results."""
  world_model = copy.deepcopy(world_model)
  nuanced_jury_results_df = world_model["nuanced_jury_results"]

  if nuanced_jury_results_df.empty:
    print("No nuanced jury results to rank.")
    world_model["top_nuanced_propositions"] = pd.DataFrame()
    return world_model

  nuanced_jury_preferences = []
  for res in nuanced_jury_results_df["result"]:
    if res and "ranking" in res:
      ranking = res["ranking"]
      if ranking:  # Ensure the ranking list is not empty
        nuanced_jury_preferences.append(ranking)

  if not nuanced_jury_preferences:
    print("No valid rankings for nuanced propositions.")
    world_model["top_nuanced_propositions"] = pd.DataFrame()
    return world_model

  # --- Schulze Ranking ---
  num_propositions_to_rank = len(nuanced_jury_preferences[0])
  full_schulze_ranking = schulze.get_schulze_ranking(
      nuanced_jury_preferences, k=num_propositions_to_rank
  )
  world_model["nuanced_schulze_ranking"] = full_schulze_ranking.get(
      "top_propositions", []
  )

  # --- PAV Selection (if enabled) ---
  primary_ranking = world_model["nuanced_schulze_ranking"]
  if args.run_pav_selection:
    if (
        "nuanced_approval_matrix" not in world_model
        or world_model["nuanced_approval_matrix"].empty
    ):
      print(
          "  - WARNING: Cannot run PAV selection for nuanced propositions due"
          " to missing approval matrix."
      )
    else:
      nuanced_approval_matrix = world_model["nuanced_approval_matrix"]
      pav_ranking = proportional_approval_voting.run_schulze_pav_selection(
          ranked_choice_results=nuanced_jury_preferences,
          approval_matrix=nuanced_approval_matrix,
          k=len(nuanced_approval_matrix.columns),  # Get a full ranking
      )
      world_model["nuanced_pav_ranking"] = pav_ranking
      primary_ranking = pav_ranking

  top_nuanced_propositions = primary_ranking[: args.num_nuanced_propositions]

  world_model["top_nuanced_propositions"] = pd.DataFrame(
      {"proposition": top_nuanced_propositions}
  )
  return world_model


async def main():
  start_time = time.monotonic()
  parser = argparse.ArgumentParser(
      description="Run simulated jury on propositions from a pickle file."
  )
  parser.add_argument(
      "--input_pkl",
      required=True,
      help="Path to the input pickle file containing the world model.",
  )
  parser.add_argument(
      "--output_pkl",
      required=True,
      help="Path to save the output pickle file with simulated jury rankings.",
  )
  parser.add_argument(
      "--propositions_per_opinion",
      type=int,
      default=3,
      help=(
          "The number of top propositions to select from each opinion group to"
          " advance to the topic-level jury."
      ),
  )
  parser.add_argument(
      "--final_propositions_per_topic",
      type=int,
      default=3,
      help=(
          "The final number of top propositions to select per topic after the"
          " topic-level jury and deduplication."
      ),
  )
  parser.add_argument(
      "--deduplication_method",
      type=str,
      default="rank_filling",
      choices=["rank_filling", "topic_based"],
      help="The deduplication algorithm to use.",
  )
  parser.add_argument(
      "--run_pav_selection",
      action="store_true",
      help="Run the Proportional Approval Voting (PAV) selection algorithm.",
  )
  parser.add_argument(
      "--approval_batch_size",
      type=int,
      default=15,
      help=(
          "Number of propositions to evaluate in a single approval vote batch."
      ),
  )
  parser.add_argument(
      "--simulated_jury_model_name",
      type=str,
      default="gemini-2.5-flash-lite",
      help="The name of the generative model to use for the simulated jury.",
  )
  parser.add_argument(
      "--gemini_api_key",
      type=str,
      default=None,
      help=(
          "The Gemini API key. If not provided, it will be read from the"
          " GEMINI_API_KEY environment variable."
      ),
  )
  parser.add_argument(
      "--num_nuanced_propositions",
      type=int,
      default=10,
      help="Number of nuanced propositions to generate.",
  )
  runner_utils.add_additional_context_args(
      parser,
      help_str=(
          "Additional context to provide when generating nuanced propositions."
      ),
  )
  parser.add_argument(
      "--nuanced_propositions_model_name",
      type=str,
      default="gemini-2.5-pro",
      help="The name of the generative model to use for nuanced propositions.",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Enable verbose output for debugging and data analysis.",
  )
  parser.add_argument(
      "--jury_size",
      type=float,
      default=1.0,
      help=(
          "Fraction of participants to sample for the jury (0.0 to 1.0), or "
          "an integer number of participants (> 1.0)."
      ),
  )
  parser.add_argument(
      "--processed_r2_data",
      type=str,
      help="Path to the full, processed R2 participant data CSV file.",
  )
  args = parser.parse_args()

  # Configure logging
  log_level = logging.DEBUG if args.verbose else logging.INFO
  logging.basicConfig(
      level=log_level,
      format="%(asctime)s - %(levelname)s - %(message)s",
      stream=sys.stdout,
  )

  gemini_api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
  if not gemini_api_key:
    raise ValueError(
        "Gemini API key not provided. Please set the GEMINI_API_KEY environment"
        " variable or use the --gemini_api_key argument."
    )

  sim_jury_model = genai_model.GenaiModel(
      api_key=gemini_api_key,
      model_name=args.simulated_jury_model_name,
  )
  nuanced_props_model = genai_model.GenaiModel(
      api_key=gemini_api_key,
      model_name=args.nuanced_propositions_model_name,
  )

  args.additional_context = runner_utils.get_additional_context(args)

  if args.verbose:
    print(f"--- Loading {args.input_pkl} ---")
  with open(args.input_pkl, "rb") as f:
    loaded_data = pickle.load(f)

  # Handle different input formats (DataFrame vs. dictionary from previous runs)
  if isinstance(loaded_data, pd.DataFrame):
    by_opinion_df = loaded_data
    world_model = {"world_model": by_opinion_df}
  elif isinstance(loaded_data, dict) and "world_model" in loaded_data:
    world_model = loaded_data
    by_opinion_df = world_model["world_model"]
  else:
    raise TypeError(
        "Unsupported format for input pickle file. Expected a pandas DataFrame"
        f" or a dictionary with a 'world_model' key. Got: {type(loaded_data)}"
    )

  if args.propositions_per_opinion > 0 and "opinion" not in by_opinion_df:
    raise ValueError(
        "An opinion-level jury was requested, but the input world model does"
        " not contain an 'opinion' column."
    )

  if args.verbose:
    print(f"Loaded data with {len(by_opinion_df)} topics (rows).")

  # --- Data Loading and Jury Pool Preparation ---
  jury_pool_df, world_model = get_jury_pool(world_model, by_opinion_df, args)

  # Apply jury size sampling to the determined jury pool
  jury_pool_df = sampling_utils.apply_jury_size_sampling(
      jury_pool_df, args.jury_size, verbose=args.verbose
  )

  print(f"--- Final jury pool contains {len(jury_pool_df)} participants. ---")

  if args.verbose:
    validate_jury_pool_data(jury_pool_df)

  if args.verbose:
    print("\n--- Sample Participant Records for Jury ---")
    for i, row in jury_pool_df.head(3).iterrows():
      print(f"\n--- Participant {row['rid']} ---")
      print(participation.get_prompt_representation(row))

  # --- Pipeline Execution ---
  # Initialize the stats list if it doesn't exist (for checkpoint compatibility)
  if "simulated_jury_stats" not in world_model:
    world_model["simulated_jury_stats"] = []

  world_model = await run_r2_opinion_ranking(world_model, args, jury_pool_df)
  world_model = await run_initial_approval_jury(
      world_model, args, jury_pool_df, sim_jury_model
  )
  world_model = await run_jury_by_opinion(
      world_model, args, jury_pool_df, sim_jury_model
  )
  world_model = await run_jury_by_topic(
      world_model, args, jury_pool_df, sim_jury_model
  )
  if args.deduplication_method:
    world_model = await run_deduplication_stage(
        world_model, args, nuanced_props_model
    )
  world_model = await generate_nuanced_propositions(
      world_model, args, nuanced_props_model
  )
  world_model = await run_nuanced_approval_jury(
      world_model, args, jury_pool_df, sim_jury_model
  )
  world_model = await run_nuanced_ranking_jury(
      world_model, args, jury_pool_df, sim_jury_model
  )
  world_model = await rank_nuanced_propositions(world_model, args)

  # --- Final Save and Print ---
  with open(args.output_pkl, "wb") as f:
    pickle.dump(world_model, f)
  print(f"\nFinal results saved to {args.output_pkl}")

  print("\n--- Top Nuanced Propositions ---")
  if (
      "top_nuanced_propositions" in world_model
      and not world_model["top_nuanced_propositions"].empty
  ):
    for i, row in world_model["top_nuanced_propositions"].iterrows():
      print(f"{i+1}. {row['proposition']}")
  else:
    print("No top nuanced propositions were selected.")

  end_time = time.monotonic()
  total_duration = end_time - start_time
  print(f"\n--- Pipeline finished in {total_duration:.2f} seconds ---")


if __name__ == "__main__":
  asyncio.run(main())
