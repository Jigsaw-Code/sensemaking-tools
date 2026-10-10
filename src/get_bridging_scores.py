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
import numpy as np
import pandas as pd
from src import get_perspective_scores_lib
from src.get_gemini_scores_lib import ContentScorer
from src.models import decision

# Read by src/report_ui/data.js (BRIDGING_COLUMN) to order quotes; keep the
# two in sync if this is renamed.
AVERAGE_BRIDGING_COLUMN = "AVERAGE_OF_3_BRIDGING"
BRIDGING_ATTRIBUTES = [
    "CURIOSITY_EXPERIMENTAL",
    "PERSONAL_STORY_EXPERIMENTAL",
    "REASONING_EXPERIMENTAL",
]
# Explains why a row has no (or incomplete) bridging scores; empty otherwise.
SKIP_REASON_COLUMN = "BRIDGING_SKIP_REASON"
SKIP_REASON_EMPTY = "empty_text"
SKIP_REASON_TOO_SHORT = "too_short"
SKIP_REASON_FAILED = "scoring_failed"


def _score_texts(
    texts: pd.Series,
    gemini_api_key: str,
    gcloud_api_key: str,
    scorer_type: str,
    model_name: str,
) -> pd.DataFrame:
  """Scores each text with every bridging attribute.

  Args:
    texts: Unique texts to score.
    gemini_api_key: API key for Gemini, used when scorer_type is "GEMINI".
    gcloud_api_key: API key for Perspective, used when scorer_type is
      "PERSPECTIVE".
    scorer_type: Backend to use, either "GEMINI" or "PERSPECTIVE".
    model_name: Gemini model name, used when scorer_type is "GEMINI".

  Returns:
    A DataFrame indexed by text with exactly the BRIDGING_ATTRIBUTES columns.
    Texts or attributes with no result (e.g. failed API calls) get NaN.
  """
  texts = texts.reset_index(drop=True)
  if scorer_type == "GEMINI":
    if decision.decision_enabled():
      print(
          "Using System One"
          f" ({os.getenv('DECISION_MODEL')}) for bridging scoring..."
      )
    else:
      print(f"Using Gemini ({model_name}) for bridging scoring...")
    scorer = ContentScorer(gemini_api_key=gemini_api_key, model_name=model_name)
    # Prepare batch for Gemini, keyed by position in texts.
    texts_with_ids = [
        {"text": text, "row_id": idx} for idx, text in texts.items()
    ]
    results = scorer.score(texts_with_ids, BRIDGING_ATTRIBUTES)
    scores_by_row_id = collections.defaultdict(dict)
    for res in results:
      scores_by_row_id[res["row_id"]].update(res["scores"])
    scores_df = pd.DataFrame.from_dict(scores_by_row_id, orient="index")
  elif scorer_type == "PERSPECTIVE":
    print("Using Perspective API for bridging scoring...")
    client = get_perspective_scores_lib.init_client(gcloud_api_key)
    scores_list = [
        get_perspective_scores_lib.score_text(client, text, BRIDGING_ATTRIBUTES)
        for text in texts
    ]
    scores_df = pd.DataFrame(scores_list, index=texts.index)
  else:
    raise ValueError(f"Unknown scorer_type: {scorer_type}")

  # Always produce exactly the bridging attribute columns, in a fixed order.
  scores_df = scores_df.reindex(index=texts.index, columns=BRIDGING_ATTRIBUTES).set_axis(texts, axis="index")
  return scores_df


def get_bridging_scores(
    df: pd.DataFrame,
    text_column: str,
    gemini_api_key: str,
    gcloud_api_key: str,
    scorer_type: str,
    model_name: str,
    force_rerun: bool = False,
    min_text_length: int = 0,
) -> pd.DataFrame:
  """Score df with bridging attributes using specified scorer.

  To save API quota:
  - Bridging scores depend only on the text being scored, so each unique text
    is scored once and the scores are copied to every row containing that
    text (e.g. categorization output has one row per quote/topic/opinion).
  - Unless force_rerun is set, texts that already have all bridging scores in
    df (e.g. from an earlier, partly failed run) are not scored again; their
    existing scores are reused for every row with that text. Texts with only
    some scores are rescored in full.
  - Empty or blank texts, and texts shorter than min_text_length characters,
    are not scored.

  Rows without complete scores get NaN for the missing attributes and a value
  in SKIP_REASON_COLUMN explaining why; the column is empty for other rows.

  Args:
    df: DataFrame containing the texts to score.
    text_column: Name of the column in df containing the texts.
    gemini_api_key: API key for Gemini, used when scorer_type is "GEMINI".
    gcloud_api_key: API key for Perspective, used when scorer_type is
      "PERSPECTIVE".
    scorer_type: Backend to use, either "GEMINI" or "PERSPECTIVE".
    model_name: Gemini model name, used when scorer_type is "GEMINI".
    force_rerun: If True, ignore any scores already in df and rescore every
      eligible text.
    min_text_length: Texts with fewer characters than this (after stripping
      surrounding whitespace) are skipped. 0 skips only empty texts.

  Returns:
    A copy of df with one column per bridging attribute, an
    AVERAGE_BRIDGING_COLUMN (mean of the available attribute scores) and
    SKIP_REASON_COLUMN.

  Raises:
    ValueError: If scorer_type is not recognized or min_text_length is
      negative.
  """
  if scorer_type not in ("GEMINI", "PERSPECTIVE"):
    raise ValueError(f"Unknown scorer_type: {scorer_type}")
  if min_text_length < 0:
    raise ValueError(f"min_text_length must be >= 0, got {min_text_length}")

  raw_texts = df[text_column]
  texts = raw_texts.where(raw_texts.notna(), "").astype(str)
  # Boolean masks are kept as numpy arrays so they apply by position, which
  # stays correct when df has a non-unique index.
  lengths = texts.str.strip().str.len().to_numpy()
  is_empty = lengths == 0
  is_too_short = ~is_empty & (lengths < min_text_length)
  eligible = ~is_empty & ~is_too_short

  # Existing complete scores, keyed by text.
  score_columns = BRIDGING_ATTRIBUTES + [AVERAGE_BRIDGING_COLUMN]
  existing_columns = [col for col in score_columns if col in df.columns]
  existing = df.reindex(columns=BRIDGING_ATTRIBUTES)
  if force_rerun:
    if existing_columns:
      print(
          f"Warning: --force_rerun set; existing bridging score columns "
          f"{existing_columns} will be replaced with newly computed scores."
      )
    complete = np.zeros(len(df), dtype=bool)
  else:
    complete = eligible & existing.notna().all(axis=1).to_numpy()
  reused_df = (
      existing[complete]
      .set_axis(texts[complete].to_numpy())
      .groupby(level=0, sort=False)
      .first()
  )

  eligible_texts = pd.Series(texts[eligible].unique())
  to_score = eligible_texts[~eligible_texts.isin(reused_df.index)]
  print(
      f"{len(df)} rows: {int(is_empty.sum())} empty and "
      f"{int(is_too_short.sum())} shorter than {min_text_length} characters "
      f"skipped; {len(eligible_texts)} unique texts to score, of which "
      f"{len(reused_df)} already have scores and {len(to_score)} will be "
      "sent to the scorer."
  )
  if len(reused_df):
    print(
        "Reusing existing bridging scores. They must come from the same "
        "scorer and model; use --force_rerun to rescore everything."
    )

  if len(to_score):
    new_df = _score_texts(
        to_score, gemini_api_key, gcloud_api_key, scorer_type, model_name
    )
  else:
    new_df = pd.DataFrame(columns=BRIDGING_ATTRIBUTES, dtype=float)
  scores_by_text = pd.concat([reused_df, new_df])

  # Expand back out to one row per original row. Ineligible rows get NaN.
  row_scores_df = scores_by_text.reindex(texts.where(eligible).to_numpy())

  df = df.copy()
  # Assign positionally so a non-unique df index is handled correctly.
  df[BRIDGING_ATTRIBUTES] = row_scores_df.to_numpy(dtype=float)
  # Create an average column, used for ranking.
  df[AVERAGE_BRIDGING_COLUMN] = df[BRIDGING_ATTRIBUTES].mean(axis=1)
  missing_any = df[BRIDGING_ATTRIBUTES].isna().any(axis=1).to_numpy()
  reasons = pd.Series("", index=range(len(df)), dtype=object)
  reasons[eligible & missing_any] = SKIP_REASON_FAILED
  reasons[is_too_short] = SKIP_REASON_TOO_SHORT
  reasons[is_empty] = SKIP_REASON_EMPTY
  df[SKIP_REASON_COLUMN] = reasons
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
  parser.add_argument(
      "--force_rerun",
      action="store_true",
      help=(
          "Rescore every text, ignoring bridging scores already in the input."
          " By default, texts that already have all bridging scores are"
          " reused to save quota."
      ),
  )
  parser.add_argument(
      "--min_text_length",
      type=int,
      default=0,
      help=(
          "Skip texts with fewer than this many characters (after stripping"
          " whitespace) to save quota. Default 0 skips only empty texts."
      ),
  )
  args = parser.parse_args()
  df = pd.read_csv(args.input_csv)
  print(f"Scoring {len(df)} rows from {args.input_csv}")
  gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
  gcloud_api_key = args.gcloud_api_key or os.getenv("GCLOUD_API_KEY")

  if (
      args.scorer_type == "GEMINI"
      and not gemini_api_key
      and not decision.decision_enabled()
  ):
    print(
        "Error: --gemini_api_key or GEMINI_API_KEY environment variable"
        " missing."
    )
    exit(1)
  if decision.decision_enabled():
    print(f"Scoring with System One model {os.getenv('DECISION_MODEL')}.")

  df = get_bridging_scores(
      df,
      args.text_column,
      gemini_api_key or "",
      gcloud_api_key,
      args.scorer_type,
      args.model_name,
      force_rerun=args.force_rerun,
      min_text_length=args.min_text_length,
  )
  df.to_csv(args.output_csv, index=False)
  print(f"Wrote {args.output_csv}")
