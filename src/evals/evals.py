# Copyright 2024 Google LLC
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

"""Evaluate the quote_extraction task for correctness, conciseness, and relevance.

Uses pointwise evaluation (scores result) if there is one input CSV.
Pairwise evaluation (comparing two models) is currently not supported with the GenAI runner.

Example Command:
python3 -m src.evals.evals \
  --baseline_csv ~/input.csv \
  --output_dir output \
  --project your-cloud-project \
  --api_key YOUR_API_KEY
"""

import argparse
import asyncio
import logging
import os
import sys

import pandas as pd
from src.models import genai_model
from src.evals import eval_metrics
from src.evals import eval_runner

_AVAILABLE_METRICS = {
    "quote_extraction": eval_metrics.QUOTE_EXTRACTION_METRICS,
    "input_evals": eval_metrics.INPUT_EVAL_METRICS,
    "opinion_quality": eval_metrics.OPINION_QUALITY_METRICS,
    "proposition_opinion": eval_metrics.PROPOSITION_OPINION_METRICS,
    "proposition_topic": eval_metrics.PROPOSITION_TOPIC_METRICS,
    "opinion_categorization": eval_metrics.OPINION_CATEGORIZATION_METRICS,
    "other_opinion": eval_metrics.OTHER_OPINION_METRICS,
    "agreement_metric": eval_metrics.AGREEMENT_METRICS,
}


async def main(args):
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
  )

  api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    logging.error(
        "API key is required. Provide it via --api_key or GOOGLE_API_KEY"
        " environment variable."
    )
    return

  # Initialize GenaiModel
  model = genai_model.GenaiModel(model_name=args.model_name, api_key=api_key)

  selected_metric = _AVAILABLE_METRICS.get(args.metric_name)
  if not selected_metric:
    raise ValueError(
        f"Unknown metric name: '{args.metric_name}'. Available metrics are:"
        f" {list(_AVAILABLE_METRICS.keys())}"
    )

  baseline_data = pd.read_csv(args.baseline_csv)
  is_pairwise_comparison = args.candidate_csv is not None

  if is_pairwise_comparison:
    logging.error(
        "Pairwise evaluation is not yet supported with the GenAI runner."
    )
    return

  # Pointwise Evaluation
  metric = selected_metric.pointwise_metric
  if not metric:
    logging.error(f"No pointwise metric defined for {args.metric_name}")
    return

  eval_dataset = metric.get_evaluation_data(baseline_data)

  if eval_dataset.empty:
    logging.warning(
        f"Evaluation dataset for metric '{args.metric_name}' is empty. Nothing"
        " to evaluate."
    )
    return

  logging.info(
      f"Starting evaluation for metric '{args.metric_name}' with"
      f" {len(eval_dataset)} examples."
  )

  # Prepare evaluation jobs
  eval_jobs = []
  # Include original indices to map back results if needed, though EvalRunner preserves order/ids
  for idx, row in eval_dataset.iterrows():
    row_dict = row.to_dict()
    prompt = eval_runner.create_eval_prompt(
        metric_name=metric.name,
        criteria=metric.criteria,
        input_variables=metric.input_variables,
        rating_rubric=metric.rating_rubric,
        row_data=row_dict,
    )
    eval_jobs.append(
        {"prompt": prompt, "metadata": {"row_index": idx, **row_dict}}
    )

  runner = eval_runner.EvalRunner(model)
  results_df = await runner.process_evals_concurrently(
      eval_jobs, max_concurrent_calls=50  # Adjust concurrency as needed
  )

  if results_df.empty:
    logging.error("Evaluation failed. No results returned.")
    return

  # Calculate summary metrics (Mean score)
  # Ensure score is numeric
  results_df["score"] = pd.to_numeric(results_df["score"], errors="coerce")
  mean_score = results_df["score"].mean()
  summary_metrics = pd.DataFrame(
      [{"mean_score": mean_score, "count": results_df.shape[0]}]
  )

  logging.info(f"Evaluation completed. Mean Score: {mean_score:.2f}")

  os.makedirs(args.output_dir, exist_ok=True)

  summary_metrics.to_csv(
      os.path.join(args.output_dir, "summary_metrics.csv"),
      index=False,
  )

  # Save detailed metrics
  # Merge metadata back if needed, but results_df should contain what we need or we can join
  # EvalRunner results_df has 'score', 'explanation', 'metadata' (dict)

  # Flatten metadata if present
  if "metadata" in results_df.columns:
    metadata_df = pd.json_normalize(results_df["metadata"])
    # Avoid column name collisions
    metadata_df.columns = [
        f"input_{c}" if c in results_df.columns else c
        for c in metadata_df.columns
    ]
    detailed_results = pd.concat(
        [results_df.drop(columns=["metadata"]), metadata_df], axis=1
    )
  else:
    detailed_results = results_df

  detailed_results.to_csv(
      os.path.join(args.output_dir, "metrics.csv"), index=False
  )
  logging.info(f"Results saved to {args.output_dir}")


def get_args():
  parser = argparse.ArgumentParser(description="Run evals on a task.")
  parser.add_argument(
      "--baseline_csv",
      required=True,
      help=(
          "Path to the input CSV file. If candidate_csv is not included,this is"
          " used for pointwise evaluation."
      ),
  )
  parser.add_argument(
      "--candidate_csv",
      required=False,
      help=(
          "Path to the second input CSV file. This is used for pairwise"
          " comparison (Not Supported)"
      ),
  )
  parser.add_argument(
      "--output_dir",
      required=True,
      help="Path to save the output files to.",
  )
  parser.add_argument(
      "--project",
      required=False,  # Made optional as we might not need it for GenaiModel if using API Key
      help="Google Cloud Project ID (Deprecated for GenAI runner).",
  )
  parser.add_argument(
      "--location",
      default="global",
      help="Google Cloud Location (Deprecated for GenAI runner).",
  )
  parser.add_argument(
      "--model_name",
      default="gemini-2.5-pro",
      help="GenAI model name (e.g., gemini-2.5-pro).",
  )
  parser.add_argument(
      "--metric_name",
      type=str,
      required=True,  # Enforce metric name
      help=(
          "Name of the evaluation metric to use. Available:"
          f" {list(_AVAILABLE_METRICS.keys())}."
      ),
  )
  parser.add_argument(
      "--api_key",
      type=str,
      help="Google GenAI API Key.",
  )

  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
