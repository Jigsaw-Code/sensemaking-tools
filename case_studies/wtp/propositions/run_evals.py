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
This module provides a utility for running AI evaluations related to
the propositions and world model using GenaiModel.

Example Usage:
  python3 -m case_studies.wtp.propositions.run_evals \
    --eval_type=agreement_on_opinion \
    --r2_input_file=/path/to/r2_data.csv \
    --output_file=/path/to/output \
    --api_key=your-api-key

  python3 -m case_studies.wtp.propositions.run_evals \
    --eval_type=agreement_on_topic \
    --r2_input_file=/path/to/r2_data.csv \
    --output_file=/path/to/output.csv \
    --api_key=your-api-key

  python3 -m case_studies.wtp.propositions.run_evals \
    --eval_type=propositions_quality \
    --input_file=/path/to/world_model.pkl \
    --output_file=/path/to/output.pkl \
    --api_key=your-api-key

"""

import argparse
import asyncio
import logging
import os
import pickle
import random
import re
import json
from typing import Any

import pandas as pd
from case_studies.wtp.models import genai_model
from case_studies.wtp.evals import eval_metrics
from case_studies.wtp.evals import eval_runner
from case_studies.wtp.propositions import world_model_util

# The maximum number of times an evaluation call should be retried.
MAX_EVAL_RETRIES = 6
# How long in seconds to wait between evaluation calls.
RETRY_DELAY_SEC = 5
# Maximum number of concurrent evaluation calls.
MAX_CONCURRENT_EVALS = 100


def _map_agreement_score(score: float) -> bool | None:
  """Maps a numeric agreement score to True, False, or None.

  This mapping is based on a scale where scores above 3 indicate agreement,
  scores below 2 indicate disagreement, and scores between 2 and 3 (inclusive)
  are considered neutral or unclear, thus returning None.

  Args:
      score: The numeric agreement score.

  Returns:
      True if the score indicates agreement, False if it indicates disagreement,
      and None otherwise.
  """
  if score > 3:
    return True
  if score < 2:
    return False
  return None


def _prepare_eval_jobs(
    df: pd.DataFrame,
    metric: Any,
    input_mapping: dict[str, str],
    metadata_cols: list[str],
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
  """Prepares evaluation jobs for processing.

  Args:
      df: The dataframe containing the data to evaluate.
      metric: The metric object to use for creating prompts.
      input_mapping: A mapping of metric input variables to dataframe columns.
      metadata_cols: A list of columns to include in the metadata.
      extra_metadata: Optional additional metadata to include in all jobs.

  Returns:
      A list of dictionaries, each representing an evaluation job.
  """
  eval_jobs = []
  for idx, row in df.iterrows():
    row_data = {
        metric_var: row[df_col] for metric_var, df_col in input_mapping.items()
    }
    prompt = eval_runner.create_eval_prompt(
        metric.name,
        metric.criteria,
        metric.input_variables,
        metric.rating_rubric,
        row_data,
    )
    metadata = {col: row[col] for col in metadata_cols}
    if extra_metadata:
      metadata.update(extra_metadata)
    eval_jobs.append({"prompt": prompt, "metadata": metadata})
  return eval_jobs


async def run_agreement_evals_on_r2(
    df: pd.DataFrame, agreement_on: str, model: genai_model.GenaiModel
) -> pd.DataFrame:
  """
  Runs agreement evaluations on the r2 data.
  """
  eval_data = []
  for col in df.columns:
    match = re.match(rf"^question_(\d+)_{agreement_on}$", col)
    if match:
      q_num = match.group(1)
      answer_col = f"answer_{q_num}"
      if answer_col in df.columns:
        temp_df = df[["rid", col, answer_col]].copy()
        temp_df.rename(
            columns={col: "question", answer_col: "answer"}, inplace=True
        )
        temp_df["original_answer_col"] = answer_col
        temp_df.dropna(subset=["answer"], inplace=True)
        eval_data.append(temp_df)

  if not eval_data:
    logging.warning(
        f"No data found for agreement evaluation on '{agreement_on}'."
    )
    return df

  combined_eval_df = pd.concat(eval_data, ignore_index=True)
  logging.info(
      "Starting agreement evaluation for %d examples.", len(combined_eval_df)
  )

  metric = eval_metrics.AGREEMENT_METRICS.pointwise_metric

  eval_jobs = _prepare_eval_jobs(
      combined_eval_df,
      metric,
      input_mapping={"question": "question", "response": "answer"},
      metadata_cols=["rid"],
  )

  runner = eval_runner.EvalRunner(model)
  results_df = await runner.process_evals_concurrently(
      eval_jobs, max_concurrent_calls=100
  )

  if not results_df.empty:
    results_df = results_df.set_index("job_id")
    combined_eval_df["agrees"] = None

    for job_id, result_row in results_df.iterrows():
      score = result_row["score"]
      agrees = _map_agreement_score(score)
      combined_eval_df.at[job_id, "agrees"] = agrees

    for _, row in combined_eval_df.iterrows():
      if pd.isna(row["agrees"]):
        continue
      answer_col = row["original_answer_col"]
      agrees_col = f"{answer_col}_agrees"
      df.loc[df["rid"] == row["rid"], agrees_col] = row["agrees"]

    logging.info("Agreement evaluations completed successfully.")
  else:
    logging.error("Agreement evaluation task failed.")
  return df


async def run_evals_on_propositions(
    df: pd.DataFrame, model: genai_model.GenaiModel
) -> pd.DataFrame:
  """
  Runs evaluations on the generated propositions.
  """
  eval_runner_instance = eval_runner.EvalRunner(model)
  topic_scores_present, opinion_scores_present = False, False

  # --- Topic Evals ---
  eval_dataset_topic = (
      df.assign(
          proposition=df["propositions"].apply(
              lambda p: p["proposition"].tolist()
              if isinstance(p, pd.DataFrame)
              else []
          )
      )
      .explode("proposition")
      .dropna(subset=["proposition"])[["topic", "proposition"]]
      .reset_index(drop=True)
  )

  results_df_topic = pd.DataFrame()
  if not eval_dataset_topic.empty:
    logging.info("Preparing proposition-topic evaluations.")
    metric = eval_metrics.PROPOSITION_TOPIC_METRICS.pointwise_metric

    eval_jobs_topic = _prepare_eval_jobs(
        eval_dataset_topic,
        metric,
        input_mapping={"topic": "topic", "response": "proposition"},
        metadata_cols=[],
        extra_metadata={"type": "topic"},
    )
    # Add index to metadata manually since it's used in the original code
    for i, job in enumerate(eval_jobs_topic):
      job["metadata"]["index"] = i

    results_df_topic = await eval_runner_instance.process_evals_concurrently(
        eval_jobs_topic, max_concurrent_calls=100
    )

    if not results_df_topic.empty:
      results_df_topic = results_df_topic.sort_values("job_id")
      eval_dataset_topic["topic_score"] = results_df_topic["score"].values
      topic_scores_present = True

  # --- Opinion Evals ---
  eval_dataset_opinion = (
      df.assign(
          proposition=df["propositions"].apply(
              lambda p: p["proposition"].tolist()
              if isinstance(p, pd.DataFrame)
              else []
          )
      )
      .explode("proposition")
      .dropna(subset=["proposition"])[["opinion", "proposition"]]
      .reset_index(drop=True)
  )

  results_df_opinion = pd.DataFrame()
  if not eval_dataset_opinion.empty:
    logging.info("Preparing proposition-opinion evaluations.")
    metric = eval_metrics.PROPOSITION_OPINION_METRICS.pointwise_metric

    eval_jobs_opinion = _prepare_eval_jobs(
        eval_dataset_opinion,
        metric,
        input_mapping={"opinion": "opinion", "response": "proposition"},
        metadata_cols=[],
        extra_metadata={"type": "opinion"},
    )
    # Add index to metadata manually
    for i, job in enumerate(eval_jobs_opinion):
      job["metadata"]["index"] = i

    results_df_opinion = await eval_runner_instance.process_evals_concurrently(
        eval_jobs_opinion, max_concurrent_calls=100
    )

    if not results_df_opinion.empty:
      results_df_opinion = results_df_opinion.sort_values("job_id")
      eval_dataset_opinion["opinion_score"] = results_df_opinion["score"].values
      opinion_scores_present = True

  if not topic_scores_present and not opinion_scores_present:
    logging.error("All evaluation tasks failed. No scores to add.")
    return df

  all_scores = None
  if topic_scores_present and opinion_scores_present:
    topic_opinion_map = df[["topic", "opinion"]].drop_duplicates()
    opinion_scores_with_topic = pd.merge(
        eval_dataset_opinion,
        topic_opinion_map,
        on="opinion",
    )
    all_scores = pd.merge(
        opinion_scores_with_topic,
        eval_dataset_topic,
        on=["topic", "proposition"],
        how="outer",
    )
  elif topic_scores_present:
    all_scores = eval_dataset_topic
  elif opinion_scores_present:
    all_scores = eval_dataset_opinion

  if all_scores is not None:

    def update_propositions(row):
      propositions_df = row["propositions"]
      if (
          isinstance(propositions_df, pd.DataFrame)
          and not propositions_df.empty
      ):
        group_key = (
            "opinion"
            if "opinion" in row and "opinion" in all_scores.columns
            else "topic"
        )
        group_scores = all_scores[all_scores[group_key] == row[group_key]]
        score_cols = ["proposition"] + [
            c
            for c in ["topic_score", "opinion_score"]
            if c in group_scores.columns
        ]
        if len(score_cols) > 1:
          cols_to_drop = [
              c
              for c in ["topic_score", "opinion_score"]
              if c in propositions_df.columns
          ]
          if cols_to_drop:
            propositions_df.drop(columns=cols_to_drop, inplace=True)

          return pd.merge(
              propositions_df,
              group_scores[score_cols],
              on="proposition",
              how="left",
          )
      return propositions_df

    df["propositions"] = df.apply(update_propositions, axis=1)
  return df


async def main():
  """Main function to run evaluations."""
  parser = argparse.ArgumentParser(
      description="Run evaluations on survey data."
  )
  parser.add_argument(
      "--eval_type",
      type=str,
      required=True,
      choices=[
          "agreement_on_topic",
          "agreement_on_opinion",
          "propositions_quality",
      ],
      help="The type of evaluation to run.",
  )
  parser.add_argument(
      "--r2_input_file", type=str, help="Input R2 CSV file for agreement evals."
  )
  parser.add_argument(
      "--input_file", type=str, help="Input .pkl file for proposition evals."
  )
  parser.add_argument("--output_file", type=str, help="Output file path.")
  parser.add_argument(
      "--api_key",
      type=str,
      required=False,
      help="The API key for GenaiModel.",
  )
  parser.add_argument(
      "--eval_model_name",
      default="gemini-1.5-flash-latest",
      help="Model for evals.",
  )
  parser.add_argument(
      "--log_level",
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
  )
  args = parser.parse_args()

  logging.basicConfig(
      level=args.log_level.upper(),
      format="%(asctime)s - %(levelname)s - %(message)s",
  )

  logging.info("Starting evaluation runner. Using GenaiModel.")

  if not args.api_key and "GOOGLE_API_KEY" not in os.environ:
    logging.error(
        "API key for GenaiModel is required (either via --api_key or"
        " GOOGLE_API_KEY env var)."
    )
    return

  model = genai_model.GenaiModel(
      model_name=args.eval_model_name, api_key=args.api_key
  )

  if args.eval_type in ["agreement_on_topic", "agreement_on_opinion"]:
    if not args.r2_input_file or not args.output_file:
      parser.error(
          "--r2_input_file and --output_file are required for agreement evals."
      )
    df = world_model_util.read_csv_to_dataframe(args.r2_input_file)
    agreement_on = (
        "topic" if args.eval_type == "agreement_on_topic" else "opinion"
    )
    result_df = await run_agreement_evals_on_r2(df, agreement_on, model)
    result_df.to_csv(args.output_file, index=False)
    logging.info(f"Agreement evaluation results saved to {args.output_file}")

  elif args.eval_type == "propositions_quality":
    if not args.input_file or not args.output_file:
      parser.error(
          "--input_file and --output_file are required for proposition evals."
      )
    with open(args.input_file, "rb") as f:
      world_model_df = pickle.load(f)

    result_df = await run_evals_on_propositions(world_model_df, model)

    output_dir = os.path.dirname(args.output_file)
    output_name = os.path.splitext(os.path.basename(args.output_file))[0]

    pkl_file = os.path.join(output_dir, f"{output_name}.pkl")
    csv_file = os.path.join(output_dir, f"propositions_{output_name}.csv")

    world_model_util.save_dataframe_to_pickle(result_df, pkl_file)
    world_model_util.save_propositions_as_csv(
        result_df, csv_file, has_eval_data=True
    )
    logging.info(
        f"Proposition evaluation results saved to {pkl_file} and {csv_file}"
    )

  logging.info("Finished processing data.")


if __name__ == "__main__":
  asyncio.run(main())
