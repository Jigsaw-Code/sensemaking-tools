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
"""Benchmark loader, runner, and evaluator for the bridging scores step.

Assumes an idealized output from the categorization step (using the wedding
music scenario with 3 topics and 7 opinions) and evaluates
``src/get_bridging_scores.py`` across four complementary dimensions:

1. **Per-Attribute Discrimination**:
   ROC-AUC, Average Precision (PR-AUC), positive/negative mean separation, and
   prompt-rubric band compliance (low band ``<= 0.35`` for negative examples,
   high band ``>= 0.60`` for positive examples) for ``CURIOSITY_EXPERIMENTAL``,
   ``PERSONAL_STORY_EXPERIMENTAL``, and ``REASONING_EXPERIMENTAL``.
2. **Cross-Attribute Orthogonality & Halo-Effect Leakage**:
   Measures whether a quote exhibiting one bridging quality (e.g. a personal
   story without reasoning or curiosity) causes the model to inflate off-target
   attributes, and checks top-1 attribute identification on single-attribute
   archetypes.
3. **Within-Opinion Quote Ranking**:
   Evaluates how well ``AVERAGE_OF_3_BRIDGING`` orders quotes within each
   ``(topic, opinion)`` cluster—the exact downstream task used by
   ``src/report_ui/data.js``—via Spearman's rho, Kendall's tau-b, pairwise tier
   concordance, NDCG@3, and top-1 / bottom-1 tier hit rates. Separate ranking
   evaluation on ``AVERAGE_OF_3_BRIDGING`` is necessary even when all three
   individual classifiers achieve strong per-attribute ROC-AUC: because ROC-AUC
   is scale-invariant, calibration offsets or cross-attribute leakage (e.g.
   ``pure_curiosity`` quotes leaking high ``REASONING_EXPERIMENTAL`` scores)
   can inflate the unweighted mean and misorder quotes across tiers. In
   addition, individual attributes are binary (``0``/``1``) whereas
   ``AVERAGE_OF_3_BRIDGING`` aggregates an ordinal bridging tier (``0..3``);
   pairwise tier concordance is the direct ordinal generalization of ROC-AUC
   across ordered tier pairs (``3 > 2 > 1 > 0``), while NDCG@3 and
   Top-1/Bottom-1 focus on the top and bottom positions surfaced in the UI.
4. **Operational Reliability**:
   Tracks ``scored_rate``, ``scoring_failed`` rate, skip reason breakdown, and
   score distribution statistics.

Dataset Creation
----------------
The default benchmark CSV
(``src/evals/benchmarks/wedding_music_bridging_quotes.csv``) is a synthetic
70-quote dataset authored with Gemini/Jetski to mirror the wedding music
scenario (3 topics and 7 opinions, 10 quotes per opinion). Within every
opinion, quotes follow a fixed distribution across 7 controlled rhetorical
archetypes spanning ordinal tiers 0–3:

- Tier 0 (0 attributes): 2 ``bare_assertion``
- Tier 1 (1 attribute): 1 ``pure_curiosity``, 1 ``pure_personal_story``,
  2 ``pure_reasoning``
- Tier 2 (2 attributes): 1 ``story_and_reasoning``,
  1 ``curiosity_and_reasoning``
- Tier 3 (3 attributes): 2 ``full_bridging``

Each quote is written to clearly express only its targeted bridging attribute(s)
under the ``src/get_bridging_scores.py`` rubric without inadvertently
introducing off-target attributes. The CSV stores the minimal 9 columns
(``quote_id``, ``participant_id``, ``topic``, ``opinion``, ``archetype``,
``expected_curiosity``, ``expected_personal_story``, ``expected_reasoning``,
``quote``), and ``expected_bridging_tier`` is computed dynamically as the sum of
the three binary flags. When extending the benchmark with additional opinions or
quotes, preserve this per-opinion archetype structure and avoid leaking
off-target attributes in single-attribute quotes so metrics remain comparable.
"""

import argparse
import dataclasses
import json
import math
import os
import pathlib
import sys
from typing import Any, Mapping, MutableMapping, Sequence
import urllib.parse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as sklearn_metrics

from src import get_bridging_scores as bridging_scores_lib


DEFAULT_BENCHMARK_CSV = (
    pathlib.Path(__file__).resolve().parent
    / "benchmarks"
    / "wedding_music_bridging_quotes.csv"
)

GCE_OLLAMA_ACTIVE_HOST_FILE = pathlib.Path("/tmp/gce_ollama_active_host")
DEFAULT_LOCAL_OLLAMA_URL = "http://127.0.0.1:11434/v1/"

LOW_SCORE_THRESHOLD = 0.35
HIGH_SCORE_THRESHOLD = 0.60

ATTRIBUTE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("curiosity", "CURIOSITY_EXPERIMENTAL", "expected_curiosity"),
    (
        "personal_story",
        "PERSONAL_STORY_EXPERIMENTAL",
        "expected_personal_story",
    ),
    ("reasoning", "REASONING_EXPERIMENTAL", "expected_reasoning"),
)

SINGLE_ATTRIBUTE_ARCHETYPES: dict[str, str] = {
    "pure_curiosity": "CURIOSITY_EXPERIMENTAL",
    "pure_personal_story": "PERSONAL_STORY_EXPERIMENTAL",
    "pure_reasoning": "REASONING_EXPERIMENTAL",
}

ARCHETYPE_ORDER: tuple[str, ...] = (
    "bare_assertion",
    "pure_curiosity",
    "pure_personal_story",
    "pure_reasoning",
    "story_and_reasoning",
    "curiosity_and_reasoning",
    "full_bridging",
)


@dataclasses.dataclass(frozen=True)
class BridgingBenchmarkConfig:
  """Column configuration and thresholds for bridging benchmark evaluation.

  Attributes:
    quote_id_col: Unique identifier column for each quote row.
    participant_id_col: Participant/comment identifier column.
    quote_col: Text column scored by ``get_bridging_scores.py``.
    topic_col: Assigned high-level topic column from categorization.
    opinion_col: Assigned subtopic/opinion column from categorization.
    archetype_col: Rhetorical archetype label for controlled evaluation.
    expected_curiosity_col: Binary ground-truth column for curiosity (0 or 1).
    expected_personal_story_col: Binary ground-truth column for personal story.
    expected_reasoning_col: Binary ground-truth column for reasoning.
    expected_tier_col: Expected bridging tier column computed dynamically as
      the sum of the three binary flags (0-3).
    pred_curiosity_col: Predicted curiosity score column.
    pred_personal_story_col: Predicted personal story score column.
    pred_reasoning_col: Predicted reasoning score column.
    pred_average_col: Predicted average bridging score column.
    skip_reason_col: Skip/failure reason column emitted by bridging scorer.
    low_score_threshold: Upper bound of the prompt's low score band (0.35).
    high_score_threshold: Lower bound of the prompt's high score band (0.60).
  """

  quote_id_col: str = "quote_id"
  participant_id_col: str = "participant_id"
  quote_col: str = "quote"
  topic_col: str = "topic"
  opinion_col: str = "opinion"
  archetype_col: str = "archetype"
  expected_curiosity_col: str = "expected_curiosity"
  expected_personal_story_col: str = "expected_personal_story"
  expected_reasoning_col: str = "expected_reasoning"
  expected_tier_col: str = "expected_bridging_tier"
  pred_curiosity_col: str = "CURIOSITY_EXPERIMENTAL"
  pred_personal_story_col: str = "PERSONAL_STORY_EXPERIMENTAL"
  pred_reasoning_col: str = "REASONING_EXPERIMENTAL"
  pred_average_col: str = "AVERAGE_OF_3_BRIDGING"
  skip_reason_col: str = "BRIDGING_SKIP_REASON"
  low_score_threshold: float = LOW_SCORE_THRESHOLD
  high_score_threshold: float = HIGH_SCORE_THRESHOLD


def load_bridging_benchmark_dataset(
    csv_path: str | pathlib.Path = DEFAULT_BENCHMARK_CSV,
    config: BridgingBenchmarkConfig | None = None,
) -> pd.DataFrame:
  """Loads and validates the bridging benchmark ground-truth dataset.

  Computes ``config.expected_tier_col`` dynamically as the sum of the three
  binary attribute flags if it is not already present in the CSV (and validates
  consistency if it is present).

  Args:
    csv_path: Path to the benchmark ground-truth CSV file.
    config: Optional column configuration.

  Returns:
    A validated pandas DataFrame containing the benchmark dataset, including
    the ``expected_bridging_tier`` column.

  Raises:
    FileNotFoundError: If ``csv_path`` does not exist.
    ValueError: If required columns are missing, IDs are duplicated, binary
      labels are outside ``{0, 1}``, or an explicitly provided
      ``expected_bridging_tier`` column does not equal the sum of the three
      expected attribute flags.
  """
  cfg = config or BridgingBenchmarkConfig()
  path = pathlib.Path(csv_path)
  if not path.exists():
    raise FileNotFoundError(f"Benchmark dataset not found: {path}")

  df = pd.read_csv(path)
  required_columns = [
      cfg.quote_id_col,
      cfg.participant_id_col,
      cfg.quote_col,
      cfg.topic_col,
      cfg.opinion_col,
      cfg.archetype_col,
      cfg.expected_curiosity_col,
      cfg.expected_personal_story_col,
      cfg.expected_reasoning_col,
  ]
  missing = [col for col in required_columns if col not in df.columns]
  if missing:
    raise ValueError(
        f"Benchmark dataset {path} is missing required columns: {missing}"
    )

  if df.empty:
    raise ValueError(f"Benchmark dataset {path} is empty.")

  if df[cfg.quote_id_col].isna().any():
    raise ValueError("Benchmark dataset contains null quote_id values.")

  duplicate_ids = (
      df[cfg.quote_id_col][df[cfg.quote_id_col].duplicated()].unique().tolist()
  )
  if duplicate_ids:
    raise ValueError(
        f"Benchmark dataset contains duplicate quote_id values: {duplicate_ids}"
    )

  quotes = (
      df[cfg.quote_col].astype("object").fillna("").astype(str).str.strip()
  )
  if quotes.eq("").any():
    raise ValueError("Benchmark dataset contains empty quote texts.")

  binary_cols = [
      cfg.expected_curiosity_col,
      cfg.expected_personal_story_col,
      cfg.expected_reasoning_col,
  ]
  for col in binary_cols:
    values = pd.to_numeric(df[col], errors="coerce")
    if values.isna().any() or not set(values.astype(int).unique()).issubset(
        {0, 1}
    ):
      raise ValueError(
          f"Column '{col}' must contain only binary values 0 or 1."
      )
    df[col] = values.astype(int)

  computed_tiers = (
      df[cfg.expected_curiosity_col]
      + df[cfg.expected_personal_story_col]
      + df[cfg.expected_reasoning_col]
  )
  if cfg.expected_tier_col in df.columns:
    tiers = pd.to_numeric(df[cfg.expected_tier_col], errors="coerce")
    if tiers.isna().any():
      raise ValueError(
          f"Column '{cfg.expected_tier_col}' contains non-numeric values."
      )
    df[cfg.expected_tier_col] = tiers.astype(int)
    mismatched = df[df[cfg.expected_tier_col] != computed_tiers]
    if not mismatched.empty:
      bad_ids = mismatched[cfg.quote_id_col].tolist()
      raise ValueError(
          f"Column '{cfg.expected_tier_col}' does not equal the sum of binary "
          f"attribute flags for quote_id(s): {bad_ids}"
      )
  else:
    df[cfg.expected_tier_col] = computed_tiers.astype(int)

  return df


def _compute_roc_auc(
    y_true: Sequence[int], y_score: Sequence[float]
) -> float | None:
  """Computes ROC-AUC via ``sklearn.metrics.roc_auc_score``.

  Args:
    y_true: Binary ground-truth labels (0 or 1).
    y_score: Continuous predicted scores.

  Returns:
    ROC-AUC in ``[0.0, 1.0]``, or ``None`` if fewer than two classes are
    present.
  """
  labels = np.asarray(y_true, dtype=int)
  scores = np.asarray(y_score, dtype=float)
  valid = ~np.isnan(scores)
  labels = labels[valid]
  scores = scores[valid]

  if not (np.any(labels == 1) and np.any(labels == 0)):
    return None
  return float(sklearn_metrics.roc_auc_score(labels, scores))


def _compute_average_precision(
    y_true: Sequence[int], y_score: Sequence[float]
) -> float | None:
  """Computes Average Precision via ``sklearn.metrics.average_precision_score``.

  Args:
    y_true: Binary ground-truth labels (0 or 1).
    y_score: Continuous predicted scores.

  Returns:
    Average precision in ``[0.0, 1.0]``, or ``None`` if no positive labels
    exist.
  """
  labels = np.asarray(y_true, dtype=int)
  scores = np.asarray(y_score, dtype=float)
  valid = ~np.isnan(scores)
  labels = labels[valid]
  scores = scores[valid]

  if len(labels) == 0 or not np.any(labels == 1):
    return None
  return float(sklearn_metrics.average_precision_score(labels, scores))


def _compute_spearman_rho(
    x: Sequence[float], y: Sequence[float]
) -> float | None:
  """Computes Spearman rank correlation via ``scipy.stats.spearmanr``."""
  x_arr = np.asarray(x, dtype=float)
  y_arr = np.asarray(y, dtype=float)
  valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
  x_arr = x_arr[valid]
  y_arr = y_arr[valid]
  if len(x_arr) < 2:
    return None
  if np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
    return 0.0
  result = stats.spearmanr(x_arr, y_arr)
  return 0.0 if math.isnan(result.statistic) else float(result.statistic)


def _compute_kendall_tau_b(
    x: Sequence[float], y: Sequence[float]
) -> float | None:
  """Computes Kendall's tau-b via ``scipy.stats.kendalltau``."""
  x_arr = np.asarray(x, dtype=float)
  y_arr = np.asarray(y, dtype=float)
  valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
  x_arr = x_arr[valid]
  y_arr = y_arr[valid]
  if len(x_arr) < 2:
    return None
  if np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
    return 0.0
  result = stats.kendalltau(x_arr, y_arr, variant="b")
  return 0.0 if math.isnan(result.statistic) else float(result.statistic)


def _compute_pairwise_tier_concordance(
    tiers: Sequence[int], scores: Sequence[float]
) -> dict[str, float | int | None]:
  """Computes pairwise concordance across pairs with strictly different tiers.

  For every pair ``(i, j)`` where ``tier[i] > tier[j]``:
  - Concordant if ``score[i] > score[j]``
  - Tied if ``score[i] == score[j]``
  - Discordant if ``score[i] < score[j]``

  Args:
    tiers: Ground-truth ordinal bridging tiers (0..3).
    scores: Predicted bridging scores.

  Returns:
    Dictionary with ``eligible_pairs``, ``concordant_pairs``,
    ``tied_pairs``, ``discordant_pairs``, ``strict_concordance``,
    ``concordance_with_half_ties``, and ``tie_rate``.
  """
  t_arr = np.asarray(tiers, dtype=float)
  s_arr = np.asarray(scores, dtype=float)
  valid = ~(np.isnan(t_arr) | np.isnan(s_arr))
  t_arr = t_arr[valid]
  s_arr = s_arr[valid]
  n = len(t_arr)

  eligible = 0
  concordant = 0
  tied = 0
  discordant = 0

  for i in range(n):
    for j in range(i + 1, n):
      if t_arr[i] == t_arr[j]:
        continue
      eligible += 1
      if t_arr[i] > t_arr[j]:
        diff = s_arr[i] - s_arr[j]
      else:
        diff = s_arr[j] - s_arr[i]
      if diff > 1e-12:
        concordant += 1
      elif abs(diff) <= 1e-12:
        tied += 1
      else:
        discordant += 1

  if eligible == 0:
    return {
        "eligible_pairs": 0,
        "concordant_pairs": 0,
        "tied_pairs": 0,
        "discordant_pairs": 0,
        "strict_concordance": None,
        "concordance_with_half_ties": None,
        "tie_rate": None,
    }

  return {
      "eligible_pairs": eligible,
      "concordant_pairs": concordant,
      "tied_pairs": tied,
      "discordant_pairs": discordant,
      "strict_concordance": float(concordant / eligible),
      "concordance_with_half_ties": float((concordant + 0.5 * tied) / eligible),
      "tie_rate": float(tied / eligible),
  }


def _compute_ndcg_at_k(
    relevances: Sequence[int | float],
    scores: Sequence[float],
    k: int = 3,
) -> float | None:
  """Computes tie-aware NDCG@k with exponential gain via ``sklearn.metrics``.

  Passes exponential gains ``2^rel - 1`` to ``sklearn.metrics.ndcg_score`` with
  ``ignore_ties=False`` so tied predicted scores receive averaged gains and
  NDCG@k is invariant to input row order.

  Args:
    relevances: Ground-truth non-negative relevance tiers (e.g. 0..3).
    scores: Predicted ranking scores.
    k: Rank cutoff.

  Returns:
    NDCG@k in ``[0.0, 1.0]``, or ``None`` if no valid positive-gain items exist.
  """
  rel_arr = np.asarray(relevances, dtype=float)
  score_arr = np.asarray(scores, dtype=float)
  valid = ~(np.isnan(rel_arr) | np.isnan(score_arr))
  rel_arr = rel_arr[valid]
  score_arr = score_arr[valid]
  n = len(rel_arr)
  if n == 0 or k <= 0:
    return None

  gains = np.power(2.0, rel_arr) - 1.0
  if float(np.sum(gains)) <= 1e-12:
    return None
  if n == 1:
    return 1.0

  cutoff = min(k, n)
  return float(
      sklearn_metrics.ndcg_score(
          [gains], [score_arr], k=cutoff, ignore_ties=False
      )
  )


def _compute_series_stats(series: pd.Series) -> dict[str, float | int | None]:
  """Computes summary distribution statistics for a numeric score series."""
  clean = pd.to_numeric(series, errors="coerce").dropna()
  if clean.empty:
    return {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "q25": None,
        "median": None,
        "q75": None,
        "iqr": None,
        "max": None,
    }
  q25 = float(np.percentile(clean, 25))
  q75 = float(np.percentile(clean, 75))
  return {
      "count": int(len(clean)),
      "mean": float(clean.mean()),
      "std": float(clean.std(ddof=0)),
      "min": float(clean.min()),
      "q25": q25,
      "median": float(np.median(clean)),
      "q75": q75,
      "iqr": float(q75 - q25),
      "max": float(clean.max()),
  }


def _merge_scored_and_ground_truth(
    scored_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None,
    config: BridgingBenchmarkConfig,
) -> pd.DataFrame:
  """Ensures ground-truth and predicted columns are present on one DataFrame."""
  if ground_truth_df is None:
    return scored_df.copy()

  gt = ground_truth_df.copy()
  pred_cols = [
      config.pred_curiosity_col,
      config.pred_personal_story_col,
      config.pred_reasoning_col,
      config.pred_average_col,
      config.skip_reason_col,
  ]
  available_pred_cols = [c for c in pred_cols if c in scored_df.columns]
  if config.quote_id_col in scored_df.columns:
    rhs = scored_df[
        [config.quote_id_col] + available_pred_cols
    ].drop_duplicates(subset=[config.quote_id_col])
    gt_without_preds = gt.drop(
        columns=[c for c in available_pred_cols if c in gt.columns]
    )
    return gt_without_preds.merge(rhs, on=config.quote_id_col, how="left")

  if config.quote_col in scored_df.columns:
    rhs = scored_df[[config.quote_col] + available_pred_cols].drop_duplicates(
        subset=[config.quote_col]
    )
    gt_without_preds = gt.drop(
        columns=[c for c in available_pred_cols if c in gt.columns]
    )
    return gt_without_preds.merge(rhs, on=config.quote_col, how="left")

  raise ValueError(
      f"Scored DataFrame must contain '{config.quote_id_col}' or "
      f"'{config.quote_col}' to align with ground truth."
  )


def evaluate_bridging_output(
    scored_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None = None,
    config: BridgingBenchmarkConfig | None = None,
) -> dict[str, Any]:
  """Evaluates bridging score predictions against the benchmark ground truth.

  Args:
    scored_df: DataFrame produced by ``src/get_bridging_scores.py`` (or a
      DataFrame containing both ground-truth and predicted columns).
    ground_truth_df: Optional ground-truth DataFrame if ``scored_df`` does not
      already include the expected ground-truth columns.
    config: Optional benchmark column and threshold configuration.

  Returns:
    Nested dictionary with four top-level metric sections:
    ``summary``, ``attribute_metrics``, ``orthogonality_metrics``, and
    ``ranking_metrics``.

  Raises:
    ValueError: If required ground-truth or predicted score columns are missing.
  """
  cfg = config or BridgingBenchmarkConfig()
  df = _merge_scored_and_ground_truth(scored_df, ground_truth_df, cfg)

  required_gt_cols = [
      cfg.quote_id_col,
      cfg.topic_col,
      cfg.opinion_col,
      cfg.archetype_col,
      cfg.expected_curiosity_col,
      cfg.expected_personal_story_col,
      cfg.expected_reasoning_col,
  ]
  missing_gt = [c for c in required_gt_cols if c not in df.columns]
  if missing_gt:
    raise ValueError(
        f"DataFrame is missing required ground-truth columns: {missing_gt}"
    )

  required_pred_cols = [
      cfg.pred_curiosity_col,
      cfg.pred_personal_story_col,
      cfg.pred_reasoning_col,
      cfg.pred_average_col,
  ]
  missing_pred = [c for c in required_pred_cols if c not in df.columns]
  if missing_pred:
    raise ValueError(
        f"DataFrame is missing required predicted score columns: {missing_pred}"
    )

  for col in required_pred_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

  for col in [
      cfg.expected_curiosity_col,
      cfg.expected_personal_story_col,
      cfg.expected_reasoning_col,
  ]:
    df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

  if cfg.expected_tier_col in df.columns:
    df[cfg.expected_tier_col] = pd.to_numeric(
        df[cfg.expected_tier_col], errors="raise"
    ).astype(int)
  else:
    df[cfg.expected_tier_col] = (
        df[cfg.expected_curiosity_col]
        + df[cfg.expected_personal_story_col]
        + df[cfg.expected_reasoning_col]
    ).astype(int)

  total_quotes = len(df)
  valid_all_mask = (
      df[cfg.pred_curiosity_col].notna()
      & df[cfg.pred_personal_story_col].notna()
      & df[cfg.pred_reasoning_col].notna()
      & df[cfg.pred_average_col].notna()
  )
  scored_quotes = int(valid_all_mask.sum())
  scored_rate = float(scored_quotes / total_quotes) if total_quotes else 0.0

  if cfg.skip_reason_col in df.columns:
    raw_reasons = (
        df[cfg.skip_reason_col]
        .astype("object")
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"": "scored", "nan": "scored", "None": "scored"})
    )
    # If a row has NaN scores even though skip_reason was empty, mark it as
    # missing_score.
    effective_reasons = raw_reasons.where(
        valid_all_mask | (raw_reasons != "scored"), "missing_score"
    )
  else:
    effective_reasons = pd.Series(
        np.where(valid_all_mask, "scored", "missing_score"), index=df.index
    )

  skip_reason_counts = {
      str(k): int(v) for k, v in effective_reasons.value_counts().items()
  }
  scoring_failed_count = int(
      skip_reason_counts.get("scoring_failed", 0)
      + skip_reason_counts.get("missing_score", 0)
  )
  scoring_failed_rate = (
      float(scoring_failed_count / total_quotes) if total_quotes else 0.0
  )

  summary = {
      "total_quotes": total_quotes,
      "scored_quotes": scored_quotes,
      "scored_rate": scored_rate,
      "scoring_failed_count": scoring_failed_count,
      "scoring_failed_rate": scoring_failed_rate,
      "skip_reason_counts": skip_reason_counts,
      "score_distributions": {
          "average_bridging": _compute_series_stats(df[cfg.pred_average_col]),
          "curiosity": _compute_series_stats(df[cfg.pred_curiosity_col]),
          "personal_story": _compute_series_stats(
              df[cfg.pred_personal_story_col]
          ),
          "reasoning": _compute_series_stats(df[cfg.pred_reasoning_col]),
      },
  }

  # 1. Per-attribute discrimination and cross-attribute leakage metrics.
  attr_pairs = [
      ("curiosity", cfg.pred_curiosity_col, cfg.expected_curiosity_col),
      (
          "personal_story",
          cfg.pred_personal_story_col,
          cfg.expected_personal_story_col,
      ),
      ("reasoning", cfg.pred_reasoning_col, cfg.expected_reasoning_col),
  ]
  per_attribute: dict[str, dict[str, Any]] = {}

  for short_name, pred_col, exp_col in attr_pairs:
    sub = df[[pred_col, exp_col, cfg.expected_tier_col]].dropna(
        subset=[pred_col]
    )
    y_true = sub[exp_col].to_numpy(dtype=int)
    y_score = sub[pred_col].to_numpy(dtype=float)
    tiers = sub[cfg.expected_tier_col].to_numpy(dtype=int)

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    off_target_scores = y_score[(y_true == 0) & (tiers >= 1)]
    zero_tier_scores = y_score[tiers == 0]

    mean_pos = float(np.mean(pos_scores)) if len(pos_scores) > 0 else None
    mean_neg = float(np.mean(neg_scores)) if len(neg_scores) > 0 else None
    mean_sep = (
        float(mean_pos - mean_neg)
        if (mean_pos is not None and mean_neg is not None)
        else None
    )
    low_band_rate = (
        float(np.mean(neg_scores <= cfg.low_score_threshold))
        if len(neg_scores) > 0
        else None
    )
    high_band_rate = (
        float(np.mean(pos_scores >= cfg.high_score_threshold))
        if len(pos_scores) > 0
        else None
    )
    off_target_mean = (
        float(np.mean(off_target_scores))
        if len(off_target_scores) > 0
        else None
    )
    off_target_high_rate = (
        float(np.mean(off_target_scores >= cfg.high_score_threshold))
        if len(off_target_scores) > 0
        else None
    )
    zero_tier_mean = (
        float(np.mean(zero_tier_scores)) if len(zero_tier_scores) > 0 else None
    )

    per_attribute[short_name] = {
        "column": pred_col,
        "expected_column": exp_col,
        "scored_count": int(len(sub)),
        "positive_count": int(len(pos_scores)),
        "negative_count": int(len(neg_scores)),
        "roc_auc": _compute_roc_auc(y_true, y_score),
        "average_precision": _compute_average_precision(y_true, y_score),
        "mean_positive": mean_pos,
        "mean_negative": mean_neg,
        "mean_separation": mean_sep,
        "low_band_compliance_rate": low_band_rate,
        "high_band_compliance_rate": high_band_rate,
        "off_target_leakage_mean": off_target_mean,
        "off_target_high_rate": off_target_high_rate,
        "baseline_zero_tier_mean": zero_tier_mean,
    }

  def _macro_mean(metric_key: str) -> float | None:
    vals = [
        m[metric_key]
        for m in per_attribute.values()
        if m.get(metric_key) is not None
    ]
    return float(np.mean(vals)) if vals else None

  attribute_metrics = {
      "per_attribute": per_attribute,
      "macro_average": {
          "roc_auc": _macro_mean("roc_auc"),
          "average_precision": _macro_mean("average_precision"),
          "mean_positive": _macro_mean("mean_positive"),
          "mean_negative": _macro_mean("mean_negative"),
          "mean_separation": _macro_mean("mean_separation"),
          "low_band_compliance_rate": _macro_mean("low_band_compliance_rate"),
          "high_band_compliance_rate": _macro_mean("high_band_compliance_rate"),
          "off_target_leakage_mean": _macro_mean("off_target_leakage_mean"),
          "off_target_high_rate": _macro_mean("off_target_high_rate"),
      },
  }

  # 2. Cross-attribute orthogonality and archetype breakdown.
  single_attr_hits: list[bool] = []
  per_single_archetype_acc: dict[str, float | None] = {}
  all_pred_attr_cols = [
      cfg.pred_curiosity_col,
      cfg.pred_personal_story_col,
      cfg.pred_reasoning_col,
  ]

  archetype_to_col = {
      "pure_curiosity": cfg.pred_curiosity_col,
      "pure_personal_story": cfg.pred_personal_story_col,
      "pure_reasoning": cfg.pred_reasoning_col,
  }

  for arch_name, target_col in archetype_to_col.items():
    other_cols = [c for c in all_pred_attr_cols if c != target_col]
    arch_rows = df[(df[cfg.archetype_col] == arch_name) & valid_all_mask]
    if arch_rows.empty:
      per_single_archetype_acc[arch_name] = None
      continue
    hits = (arch_rows[target_col] > arch_rows[other_cols[0]]) & (
        arch_rows[target_col] > arch_rows[other_cols[1]]
    )
    hit_list = hits.astype(bool).tolist()
    single_attr_hits.extend(hit_list)
    per_single_archetype_acc[arch_name] = float(np.mean(hit_list))

  single_attr_top1_acc = (
      float(np.mean(single_attr_hits)) if single_attr_hits else None
  )

  tier_means: dict[int, float | None] = {}
  for tier_val in sorted(df[cfg.expected_tier_col].unique()):
    tier_scores = df.loc[
        df[cfg.expected_tier_col] == tier_val, cfg.pred_average_col
    ].dropna()
    tier_means[int(tier_val)] = (
        float(tier_scores.mean()) if not tier_scores.empty else None
    )

  non_null_tier_means = [
      tier_means[t] for t in sorted(tier_means) if tier_means[t] is not None
  ]
  tier_means_strictly_increasing = bool(
      len(non_null_tier_means) >= 2
      and all(
          non_null_tier_means[i] < non_null_tier_means[i + 1]
          for i in range(len(non_null_tier_means) - 1)
      )
  )

  observed_archetypes = list(ARCHETYPE_ORDER) + [
      a
      for a in df[cfg.archetype_col].dropna().unique()
      if a not in ARCHETYPE_ORDER
  ]
  archetype_breakdown: list[dict[str, Any]] = []
  for arch_name in observed_archetypes:
    arch_df = df[df[cfg.archetype_col] == arch_name]
    if arch_df.empty:
      continue
    scored_arch = arch_df[valid_all_mask.loc[arch_df.index]]
    archetype_breakdown.append({
        "archetype": str(arch_name),
        "expected_tier": int(arch_df[cfg.expected_tier_col].iloc[0]),
        "total_quotes": int(len(arch_df)),
        "scored_quotes": int(len(scored_arch)),
        "mean_curiosity": (
            float(scored_arch[cfg.pred_curiosity_col].mean())
            if not scored_arch.empty
            else None
        ),
        "mean_personal_story": (
            float(scored_arch[cfg.pred_personal_story_col].mean())
            if not scored_arch.empty
            else None
        ),
        "mean_reasoning": (
            float(scored_arch[cfg.pred_reasoning_col].mean())
            if not scored_arch.empty
            else None
        ),
        "mean_average_bridging": (
            float(scored_arch[cfg.pred_average_col].mean())
            if not scored_arch.empty
            else None
        ),
    })

  orthogonality_metrics = {
      "single_attribute_top1_accuracy": single_attr_top1_acc,
      "per_single_attribute_top1_accuracy": per_single_archetype_acc,
      "tier_mean_average_bridging": {
          str(k): v for k, v in tier_means.items()
      },
      "tier_means_strictly_increasing": tier_means_strictly_increasing,
      "archetype_breakdown": archetype_breakdown,
  }

  # 3. Within-opinion quote ranking metrics.
  per_opinion_ranking: list[dict[str, Any]] = []
  grouped = df.groupby([cfg.topic_col, cfg.opinion_col], sort=False)

  for (topic_val, opinion_val), group_df in grouped:
    valid_group = group_df.dropna(subset=[cfg.pred_average_col])
    tiers = valid_group[cfg.expected_tier_col].to_numpy(dtype=int)
    scores = valid_group[cfg.pred_average_col].to_numpy(dtype=float)

    spearman = _compute_spearman_rho(tiers, scores)
    kendall = _compute_kendall_tau_b(tiers, scores)
    concordance_info = _compute_pairwise_tier_concordance(tiers, scores)
    ndcg3 = _compute_ndcg_at_k(tiers, scores, k=3)

    if len(valid_group) > 0:
      max_tier = int(np.max(group_df[cfg.expected_tier_col]))
      min_tier = int(np.min(group_df[cfg.expected_tier_col]))
      max_score = float(np.max(scores))
      min_score = float(np.min(scores))
      top_tied_tiers = tiers[np.isclose(scores, max_score, atol=1e-12)]
      bottom_tied_tiers = tiers[np.isclose(scores, min_score, atol=1e-12)]
      top1_hit = float(np.mean(top_tied_tiers == max_tier))
      bottom1_hit = float(np.mean(bottom_tied_tiers == min_tier))
    else:
      top1_hit = None
      bottom1_hit = None

    per_opinion_ranking.append({
        "topic": str(topic_val),
        "opinion": str(opinion_val),
        "total_quotes": int(len(group_df)),
        "scored_quotes": int(len(valid_group)),
        "spearman_rho": spearman,
        "kendall_tau": kendall,
        "pairwise_tier_concordance": concordance_info["strict_concordance"],
        "pairwise_concordance_with_half_ties": concordance_info[
            "concordance_with_half_ties"
        ],
        "pairwise_tie_rate": concordance_info["tie_rate"],
        "ndcg_at_3": ndcg3,
        "top1_is_max_tier": top1_hit,
        "bottom1_is_min_tier": bottom1_hit,
    })

  def _opinion_mean(key: str) -> float | None:
    vals = [
        row[key] for row in per_opinion_ranking if row.get(key) is not None
    ]
    return float(np.mean(vals)) if vals else None

  global_concordance = _compute_pairwise_tier_concordance(
      df[cfg.expected_tier_col].to_numpy(dtype=int),
      df[cfg.pred_average_col].to_numpy(dtype=float),
  )
  global_spearman = _compute_spearman_rho(
      df[cfg.expected_tier_col].to_numpy(dtype=int),
      df[cfg.pred_average_col].to_numpy(dtype=float),
  )

  ranking_metrics = {
      "opinion_count": len(per_opinion_ranking),
      "mean_spearman_rho": _opinion_mean("spearman_rho"),
      "mean_kendall_tau": _opinion_mean("kendall_tau"),
      "mean_pairwise_tier_concordance": _opinion_mean(
          "pairwise_tier_concordance"
      ),
      "mean_pairwise_concordance_with_half_ties": _opinion_mean(
          "pairwise_concordance_with_half_ties"
      ),
      "mean_pairwise_tie_rate": _opinion_mean("pairwise_tie_rate"),
      "mean_ndcg_at_3": _opinion_mean("ndcg_at_3"),
      "top1_is_max_tier_rate": _opinion_mean("top1_is_max_tier"),
      "bottom1_is_min_tier_rate": _opinion_mean("bottom1_is_min_tier"),
      "global_spearman_rho": global_spearman,
      "global_pairwise_tier_concordance": global_concordance[
          "strict_concordance"
      ],
      "per_opinion_ranking": per_opinion_ranking,
  }

  return {
      "summary": summary,
      "attribute_metrics": attribute_metrics,
      "orthogonality_metrics": orthogonality_metrics,
      "ranking_metrics": ranking_metrics,
  }


def _normalize_ollama_v1_url(raw_host: str) -> str:
  """Normalizes an Ollama host or URL to an OpenAI-compatible ``/v1/`` URL."""
  value = raw_host.strip()
  if not value:
    return DEFAULT_LOCAL_OLLAMA_URL
  if "://" not in value:
    value = f"http://{value}"
  parsed = urllib.parse.urlsplit(value)
  netloc = parsed.netloc or parsed.path
  path = parsed.path if parsed.netloc else ""
  if ":" not in netloc:
    netloc = f"{netloc}:11434"
  clean_path = path.rstrip("/")
  if not clean_path.endswith("/v1"):
    clean_path = f"{clean_path}/v1"
  return urllib.parse.urlunsplit(
      (parsed.scheme or "http", netloc, f"{clean_path}/", "", "")
  )


def configure_model_endpoint_env(
    model_name: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
  """Configures environment variables for ``get_bridging_scores`` in-process.

  Automatically sets ``MODEL_ENDPOINT_TYPE=openai_api_compatible`` and
  ``OPENAI_API_ENDPOINT_URL`` when ``model_name`` starts with ``gemma``.

  Args:
    model_name: Optional model identifier passed to ``get_bridging_scores``.
    env: Optional mutable environment mapping (defaults to ``os.environ``).

  Returns:
    The updated environment mapping.
  """
  target_env = os.environ if env is None else env
  if not model_name:
    return target_env

  if model_name.strip().lower().startswith("gemma"):
    target_env.setdefault("MODEL_ENDPOINT_TYPE", "openai_api_compatible")
    if not target_env.get("OPENAI_API_ENDPOINT_URL"):
      ollama_host = target_env.get("OLLAMA_HOST", "").strip()
      if ollama_host:
        target_env["OPENAI_API_ENDPOINT_URL"] = _normalize_ollama_v1_url(
            ollama_host
        )
      elif GCE_OLLAMA_ACTIVE_HOST_FILE.exists():
        active_ip = GCE_OLLAMA_ACTIVE_HOST_FILE.read_text(
            encoding="utf-8"
        ).strip()
        if active_ip:
          target_env["OPENAI_API_ENDPOINT_URL"] = _normalize_ollama_v1_url(
              active_ip
          )
        else:
          target_env["OPENAI_API_ENDPOINT_URL"] = DEFAULT_LOCAL_OLLAMA_URL
      else:
        target_env["OPENAI_API_ENDPOINT_URL"] = DEFAULT_LOCAL_OLLAMA_URL
  return target_env


def run_bridging_pipeline(
    dataset: pd.DataFrame | str | pathlib.Path,
    output_csv: str | pathlib.Path | None = None,
    text_column: str = bridging_scores_lib.DEFAULT_TEXT_COLUMN,
    model_name: str | None = None,
    scorer_type: str = bridging_scores_lib.DEFAULT_SCORER_TYPE,
    gemini_api_key: str | None = None,
    gcloud_api_key: str | None = None,
    min_text_length: int = 0,
    force_rerun: bool = True,
) -> pd.DataFrame:
  """Invokes ``get_bridging_scores`` directly as a library on the dataset.

  Args:
    dataset: Input DataFrame or path to the input benchmark CSV.
    output_csv: Optional path where the scored CSV will be saved.
    text_column: Column containing the quote text to score.
    model_name: Optional model name override (defaults to
      ``get_bridging_scores.DEFAULT_MODEL_NAME``).
    scorer_type: Scorer backend (``"GEMINI"`` or ``"PERSPECTIVE"``).
    gemini_api_key: Optional Gemini API key.
    gcloud_api_key: Optional Google Cloud API key for Perspective API.
    min_text_length: Minimum character length filter.
    force_rerun: Whether to recompute scores even if columns already exist.

  Returns:
    Scored DataFrame returned by ``get_bridging_scores.get_bridging_scores``.
  """
  if isinstance(dataset, pd.DataFrame):
    input_df = dataset.copy()
  else:
    input_df = pd.read_csv(dataset, keep_default_na=False, na_values=[""])

  resolved_model = model_name or bridging_scores_lib.DEFAULT_MODEL_NAME
  configure_model_endpoint_env(model_name=resolved_model)

  scored_df = bridging_scores_lib.get_bridging_scores(
      df=input_df,
      text_column=text_column,
      gemini_api_key=gemini_api_key,
      gcloud_api_key=gcloud_api_key,
      scorer_type=scorer_type.upper(),
      model_name=resolved_model,
      force_rerun=force_rerun,
      min_text_length=min_text_length,
  )

  if output_csv is not None:
    output_path = pathlib.Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(output_path, index=False)

  return scored_df


def _fmt_float(val: float | None, digits: int = 3) -> str:
  """Formats an optional float for human-readable report tables."""
  if val is None or (isinstance(val, float) and math.isnan(val)):
    return "N/A"
  return f"{val:.{digits}f}"


def format_bridging_evaluation_report(metrics: Mapping[str, Any]) -> str:
  """Formats the bridging evaluation metrics into a human-readable summary.

  Args:
    metrics: Dictionary returned by ``evaluate_bridging_output``.

  Returns:
    Multi-line human-readable report string.
  """
  summary = metrics["summary"]
  attr_metrics = metrics["attribute_metrics"]
  ortho = metrics["orthogonality_metrics"]
  ranking = metrics["ranking_metrics"]

  lines = [
      "======================================================================",
      "BRIDGING SCORES BENCHMARK EVALUATION REPORT",
      "======================================================================",
      "",
      "1. Operational Summary",
      "----------------------",
      (
          f"  Total quotes:          {summary['total_quotes']}\n"
          f"  Scored quotes:         {summary['scored_quotes']} "
          f"({_fmt_float(summary['scored_rate'] * 100, 1)}%)\n"
          f"  Scoring failed count:  {summary['scoring_failed_count']} "
          f"({_fmt_float(summary['scoring_failed_rate'] * 100, 1)}%)\n"
          f"  Skip reason breakdown: {summary['skip_reason_counts']}"
      ),
      "",
      "2. Per-Attribute Discrimination & Leakage",
      "-----------------------------------------",
      (
          f"  {'Attribute':<16} {'ROC-AUC':>8} {'PR-AUC':>8} {'Mean+':>8} "
          f"{'Mean-':>8} {'Sep':>8} {'Low<=.35':>9} {'High>=.60':>10} "
          f"{'LeakMean':>9}"
      ),
  ]

  for attr_key, info in attr_metrics["per_attribute"].items():
    lines.append(
        f"  {attr_key:<16} "
        f"{_fmt_float(info['roc_auc']):>8} "
        f"{_fmt_float(info['average_precision']):>8} "
        f"{_fmt_float(info['mean_positive']):>8} "
        f"{_fmt_float(info['mean_negative']):>8} "
        f"{_fmt_float(info['mean_separation']):>8} "
        f"{_fmt_float(info['low_band_compliance_rate']):>9} "
        f"{_fmt_float(info['high_band_compliance_rate']):>10} "
        f"{_fmt_float(info['off_target_leakage_mean']):>9}"
    )

  macro = attr_metrics["macro_average"]
  lines.append(
      f"  {'MACRO AVERAGE':<16} "
      f"{_fmt_float(macro['roc_auc']):>8} "
      f"{_fmt_float(macro['average_precision']):>8} "
      f"{_fmt_float(macro['mean_positive']):>8} "
      f"{_fmt_float(macro['mean_negative']):>8} "
      f"{_fmt_float(macro['mean_separation']):>8} "
      f"{_fmt_float(macro['low_band_compliance_rate']):>9} "
      f"{_fmt_float(macro['high_band_compliance_rate']):>10} "
      f"{_fmt_float(macro['off_target_leakage_mean']):>9}"
  )

  lines.extend([
      "",
      "3. Cross-Attribute Orthogonality & Archetype Profile",
      "----------------------------------------------------",
      (
          "  Single-attribute top-1 accuracy: "
          f"{_fmt_float(ortho['single_attribute_top1_accuracy'])}"
      ),
      (
          "  Per-single-archetype top-1:      "
          f"{ortho['per_single_attribute_top1_accuracy']}"
      ),
      (
          "  Tier mean AVERAGE_OF_3_BRIDGING: "
          f"{ortho['tier_mean_average_bridging']}"
      ),
      (
          "  Tier means strictly increasing:  "
          f"{ortho['tier_means_strictly_increasing']}"
      ),
      "",
      (
          f"  {'Archetype':<24} {'Tier':>4} {'N':>4} {'Curiosity':>10} "
          f"{'Story':>10} {'Reasoning':>10} {'AvgBridge':>10}"
      ),
  ])

  for row in ortho["archetype_breakdown"]:
    lines.append(
        f"  {row['archetype']:<24} "
        f"{row['expected_tier']:>4} "
        f"{row['scored_quotes']:>4} "
        f"{_fmt_float(row['mean_curiosity']):>10} "
        f"{_fmt_float(row['mean_personal_story']):>10} "
        f"{_fmt_float(row['mean_reasoning']):>10} "
        f"{_fmt_float(row['mean_average_bridging']):>10}"
    )

  lines.extend([
      "",
      "4. Within-Opinion Quote Ranking (Downstream UI Task)",
      "----------------------------------------------------",
      (
          "  Opinions evaluated:                "
          f"{ranking['opinion_count']}"
      ),
      (
          "  Mean within-opinion Spearman rho:  "
          f"{_fmt_float(ranking['mean_spearman_rho'])}"
      ),
      (
          "  Mean within-opinion Kendall tau:   "
          f"{_fmt_float(ranking['mean_kendall_tau'])}"
      ),
      (
          "  Mean pairwise tier concordance:    "
          f"{_fmt_float(ranking['mean_pairwise_tier_concordance'])}"
      ),
      (
          "  Mean pairwise tie rate:            "
          f"{_fmt_float(ranking['mean_pairwise_tie_rate'])}"
      ),
      (
          "  Mean within-opinion NDCG@3:        "
          f"{_fmt_float(ranking['mean_ndcg_at_3'])}"
      ),
      (
          "  Top-1 quote is Tier 3 rate:        "
          f"{_fmt_float(ranking['top1_is_max_tier_rate'])}"
      ),
      (
          "  Bottom-1 quote is Tier 0 rate:     "
          f"{_fmt_float(ranking['bottom1_is_min_tier_rate'])}"
      ),
      "======================================================================",
  ])
  return "\n".join(lines)


def save_bridging_evaluation_artifacts(
    metrics: Mapping[str, Any],
    output_dir: str | pathlib.Path,
) -> dict[str, pathlib.Path]:
  """Writes JSON, text, and CSV evaluation artifacts to ``output_dir``.

  Args:
    metrics: Evaluation metrics dictionary from ``evaluate_bridging_output``.
    output_dir: Directory where artifacts will be written.

  Returns:
    Dictionary mapping artifact names to their written file paths.
  """
  out_dir = pathlib.Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  json_path = out_dir / "bridging_evaluation_report.json"
  json_path.write_text(
      json.dumps(metrics, indent=2, sort_keys=False) + "\n", encoding="utf-8"
  )

  summary_text = format_bridging_evaluation_report(metrics)
  txt_path = out_dir / "bridging_evaluation_summary.txt"
  txt_path.write_text(summary_text + "\n", encoding="utf-8")

  attr_rows = []
  for attr_name, info in metrics["attribute_metrics"]["per_attribute"].items():
    row = {"attribute": attr_name}
    row.update(info)
    attr_rows.append(row)
  attr_csv_path = out_dir / "bridging_attribute_metrics.csv"
  pd.DataFrame(attr_rows).to_csv(attr_csv_path, index=False)

  arch_csv_path = out_dir / "bridging_archetype_breakdown.csv"
  pd.DataFrame(
      metrics["orthogonality_metrics"]["archetype_breakdown"]
  ).to_csv(arch_csv_path, index=False)

  opinion_csv_path = out_dir / "bridging_opinion_ranking.csv"
  pd.DataFrame(metrics["ranking_metrics"]["per_opinion_ranking"]).to_csv(
      opinion_csv_path, index=False
  )

  return {
      "report_json": json_path,
      "summary_txt": txt_path,
      "attribute_csv": attr_csv_path,
      "archetype_csv": arch_csv_path,
      "opinion_ranking_csv": opinion_csv_path,
  }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for the bridging benchmark tool."""
  parser = argparse.ArgumentParser(
      description=(
          "Validate, run, and evaluate the Sensemaking bridging scores "
          "benchmark."
      )
  )
  parser.add_argument(
      "--benchmark_csv",
      type=str,
      default=str(DEFAULT_BENCHMARK_CSV),
      help="Path to the ground-truth bridging benchmark CSV.",
  )
  parser.add_argument(
      "--scored_csv",
      type=str,
      default=None,
      help=(
          "Path to an existing CSV scored by src/get_bridging_scores.py, or "
          "the destination path when --run_bridging is set."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=None,
      help="Directory to write JSON, text, and CSV evaluation reports.",
  )
  parser.add_argument(
      "--validate_only",
      action="store_true",
      help="Only validate the ground-truth benchmark dataset and exit.",
  )
  parser.add_argument(
      "--run_bridging",
      action="store_true",
      help="Run src.get_bridging_scores before evaluating.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default=None,
      help="Optional model name passed to src.get_bridging_scores.",
  )
  parser.add_argument(
      "--scorer_type",
      type=str.upper,
      default=bridging_scores_lib.DEFAULT_SCORER_TYPE,
      choices=["GEMINI", "PERSPECTIVE"],
      help="Scorer backend to use when --run_bridging is set.",
  )
  parser.add_argument(
      "--gemini_api_key",
      type=str,
      default=None,
      help="Optional Gemini API key passed to get_bridging_scores.",
  )
  parser.add_argument(
      "--gcloud_api_key",
      type=str,
      default=None,
      help="Optional Google Cloud API key for Perspective API scoring.",
  )
  parser.add_argument(
      "--min_text_length",
      type=int,
      default=0,
      help="Optional minimum text length filter passed to get_bridging_scores.",
  )
  parser.add_argument(
      "--no_force_rerun",
      action="store_true",
      help="Do not set force_rerun=True when invoking get_bridging_scores.",
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """CLI entry point for validating, running, and evaluating the benchmark."""
  args = parse_args(argv)
  gt_df = load_bridging_benchmark_dataset(args.benchmark_csv)

  if args.validate_only:
    print(
        f"Validated bridging benchmark dataset: {len(gt_df)} quotes across "
        f"{gt_df['topic'].nunique()} topics and "
        f"{gt_df['opinion'].nunique()} opinions."
    )
    return 0

  scored_csv_path: pathlib.Path | None = (
      pathlib.Path(args.scored_csv) if args.scored_csv else None
  )
  if args.run_bridging:
    if scored_csv_path is None and args.output_dir:
      scored_csv_path = (
          pathlib.Path(args.output_dir) / "scored_bridging_quotes.csv"
      )
    scored_df = run_bridging_pipeline(
        dataset=gt_df,
        output_csv=scored_csv_path,
        model_name=args.model_name,
        scorer_type=args.scorer_type,
        gemini_api_key=args.gemini_api_key,
        gcloud_api_key=args.gcloud_api_key,
        min_text_length=args.min_text_length,
        force_rerun=not args.no_force_rerun,
    )
  else:
    if scored_csv_path is None:
      raise ValueError(
          "Provide --scored_csv to evaluate an existing output, or set "
          "--run_bridging or --validate_only."
      )
    scored_df = pd.read_csv(scored_csv_path)

  metrics = evaluate_bridging_output(scored_df, ground_truth_df=gt_df)
  report_text = format_bridging_evaluation_report(metrics)
  print(report_text)

  if args.output_dir:
    save_bridging_evaluation_artifacts(metrics, args.output_dir)

  return 0


if __name__ == "__main__":
  sys.exit(main())
