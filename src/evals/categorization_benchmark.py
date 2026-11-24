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

"""Modular benchmark loader and evaluator for `categorization_runner`.

This module provides a reusable framework for evaluating Sensemaker's
categorization step (`src/categorization_runner.py`) against labeled
ground-truth datasets. It includes a default 100-comment benchmark simulating an
8-person discussion about wedding reception music (covering house music, 80s
pop, disco, reception timeline scheduling, dinner atmosphere, and DJ vs. live
band performance) and supports swapping in any custom labeled CSV via
`BenchmarkConfig` or CLI arguments. See `src/evals/benchmarks/README.md` for
dataset creation notes, CLI examples, and how this benchmark complements the
LLM-as-a-judge autoraters in `src/evals/evals.py`.

Supported evaluation modes:
1. Unconstrained Discovery (`learn_all`): Sensemaker learns topics and opinions
   from scratch. Evaluated using label-agnostic clustering metrics (Cluster
   Purity, Inverse Purity / Recall, Purity F1, Pairwise Co-assignment F1,
   Homogeneity, Completeness, V-Measure) plus expected-keyword sanity checks on
   the discovered `topic + opinion` cluster labels (not quote extraction) to
   verify that learned labels remain semantically meaningful rather than
   placeholder strings.
2. Topic-Guided Discovery (`guided_topics`): Sensemaker receives ground-truth
   top-level topics (`--topics`) and learns opinions within each topic.
3. Full Taxonomy Guided (`guided_taxonomy`): Sensemaker receives both
   ground-truth topics and opinions (`--topic_and_opinion_csv`) and assigns
   statements to them, enabling exact-match accuracy alongside cluster metrics.
"""

import argparse
import collections
import dataclasses
import itertools
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
from sklearn import metrics as sklearn_metrics


_BENCHMARKS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmarks"
)
DEFAULT_WEDDING_BENCHMARK_CSV = os.path.join(
    _BENCHMARKS_DIR, "wedding_music_100.csv"
)
DEFAULT_WEDDING_CONTEXT_FILE = os.path.join(
    _BENCHMARKS_DIR, "wedding_music_context.md"
)
GCE_OLLAMA_ACTIVE_HOST_FILE = "/tmp/gce_ollama_active_host"

# Column names that `categorization_runner._convert_csv_rows_to_statements`
# interprets as pre-populated topic assignments on input rows.
RESERVED_RUNNER_INPUT_COLUMNS = frozenset(
    {"topics", "topic", "quote_with_brackets", "quote_id"}
)


def build_runner_env(
    model_name: str,
    base_env: Optional[Dict[str, str]] = None,
    active_host_file: str = GCE_OLLAMA_ACTIVE_HOST_FILE,
) -> Dict[str, str]:
  """Builds environment variables for invoking `src.categorization_runner`.

  When `model_name` starts with 'gemma' (and `MODEL_ENDPOINT_TYPE` is not
  explicitly set), automatically configures the OpenAI-compatible Ollama adapter
  variables, including reading `/tmp/gce_ollama_active_host` if `OLLAMA_HOST` is
  not set in the environment.

  Args:
    model_name: Name of the model passed to `--model_name`.
    base_env: Optional starting environment dict (defaults to `os.environ`).
    active_host_file: Path to the active GCE Ollama tunnel host file.

  Returns:
    Environment dictionary suitable for `subprocess.run(..., env=...)`.
  """
  env = dict(os.environ if base_env is None else base_env)
  is_gemma = model_name.strip().lower().startswith("gemma")
  endpoint_type = env.get("MODEL_ENDPOINT_TYPE", "")

  if is_gemma and not endpoint_type:
    ollama_host = env.get("OLLAMA_HOST", "").strip()
    if (
        not ollama_host
        and active_host_file
        and os.path.exists(active_host_file)
    ):
      try:
        with open(active_host_file, "r") as f:
          candidate = f.read().strip()
        if candidate:
          ollama_host = candidate
      except OSError:
        pass
    if not ollama_host:
      ollama_host = "127.0.0.1:11434"

    env["OLLAMA_HOST"] = ollama_host
    env["MODEL_ENDPOINT_TYPE"] = "openai_api_compatible"
    env.setdefault("OPENAI_API_ENDPOINT_URL", f"http://{ollama_host}/v1")
    env.setdefault("OPENAI_API_ENDPOINT_KEY", "ollama")
    env.setdefault("OPENAI_RESPONSE_FORMAT_STYLE", "openai_json_schema")

    if model_name.strip().lower() == "gemma3:270m":
      env.setdefault("OLLAMA_CONTEXT_LENGTH", "8192")
      env.setdefault("OPENAI_MAX_TOKENS", "1024")
      env.setdefault("OPENAI_FREQUENCY_PENALTY", "0.5")
      env.setdefault("MAX_BATCH_TOKENS", "1500")
      env.setdefault("MAX_ITEMS_PER_BATCH", "2")
    else:
      env.setdefault("OLLAMA_CONTEXT_LENGTH", "65536")
      env.setdefault("OPENAI_MAX_TOKENS", "8192")
      env.setdefault("OPENAI_REASONING_EFFORT", "none")
      env.setdefault("OPENAI_FREQUENCY_PENALTY", "0.0")
      env.setdefault("MAX_BATCH_TOKENS", "3000")
      env.setdefault("MAX_ITEMS_PER_BATCH", "5")

  return env


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
  """Configuration mapping column names for a labeled categorization benchmark.

  Attributes:
    id_col: Unique statement identifier column required by
      `categorization_runner` (default: 'participant_id').
    text_col: Statement text column required by `categorization_runner`
      (default: 'survey_text').
    expected_topic_col: Ground-truth top-level topic column name.
    expected_opinion_col: Ground-truth opinion/subtopic column name.
    expected_stance_col: Optional concise stance/category slug column name.
    expected_keywords_col: Optional column with semicolon-separated domain
      keywords checked against discovered `topic + opinion` cluster labels (not
      extracted quotes) as a semantic label sanity check.
    predicted_topic_col: Predicted topic column produced by
      `categorization_runner` (default: 'topic').
    predicted_opinion_col: Predicted opinion column produced by
      `categorization_runner` (default: 'opinion').
    other_label: Catch-all category name used by Sensemaker (default: 'Other').
  """

  id_col: str = "participant_id"
  text_col: str = "survey_text"
  expected_topic_col: str = "expected_topic"
  expected_opinion_col: str = "expected_opinion"
  expected_stance_col: Optional[str] = "expected_stance"
  expected_keywords_col: Optional[str] = "expected_keywords"
  predicted_topic_col: str = "topic"
  predicted_opinion_col: str = "opinion"
  other_label: str = "Other"


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
  """Returns a copy of `df` with lowercase, stripped column headers."""
  normalized = df.copy()
  normalized.columns = [str(col).strip().lower() for col in normalized.columns]
  return normalized


def _is_other(value: Any, other_label: str = "Other") -> bool:
  """Returns True if `value` is empty or matches `other_label` case-insensitively."""
  text = str(value or "").strip()
  return not text or text.lower() == other_label.strip().lower()


def _safe_harmonic_mean(a: float, b: float) -> float:
  """Computes the harmonic mean of two non-negative numbers, or 0.0 if zero."""
  if a + b <= 0.0:
    return 0.0
  return 2.0 * a * b / (a + b)


def load_benchmark_dataset(
    csv_path: str = DEFAULT_WEDDING_BENCHMARK_CSV,
    config: Optional[BenchmarkConfig] = None,
    allow_reserved_runner_columns: bool = False,
) -> pd.DataFrame:
  """Loads and validates a labeled benchmark CSV file using pandas.

  Args:
    csv_path: Path to the benchmark CSV file.
    config: Column configuration for the benchmark. Defaults to
      `BenchmarkConfig()`.
    allow_reserved_runner_columns: Whether to allow columns like `topics` or
      `topic` that `categorization_runner` interprets as pre-existing topics.

  Returns:
    A validated `pd.DataFrame` with lowercase column names.

  Raises:
    FileNotFoundError: If `csv_path` does not exist.
    ValueError: If required columns are missing, values are blank,
      `participant_id` values are not unique, or reserved columns are present.
  """
  cfg = config or BenchmarkConfig()
  expanded_path = os.path.expanduser(csv_path)
  if not os.path.exists(expanded_path):
    raise FileNotFoundError(f"Benchmark CSV file not found: {expanded_path}")

  df = pd.read_csv(expanded_path, dtype=str, keep_default_na=False)
  df = _normalize_headers(df)

  required_cols = [
      cfg.id_col.lower(),
      cfg.text_col.lower(),
      cfg.expected_topic_col.lower(),
      cfg.expected_opinion_col.lower(),
  ]
  missing_cols = [col for col in required_cols if col not in df.columns]
  if missing_cols:
    raise ValueError(
        f"Benchmark CSV '{expanded_path}' is missing required columns:"
        f" {missing_cols}. Present columns: {list(df.columns)}"
    )

  if df.empty:
    raise ValueError(f"Benchmark CSV '{expanded_path}' contains no rows.")

  if not allow_reserved_runner_columns:
    colliding = sorted(RESERVED_RUNNER_INPUT_COLUMNS.intersection(df.columns))
    if colliding:
      raise ValueError(
          f"Benchmark CSV '{expanded_path}' contains reserved"
          f" categorization_runner input columns {colliding} which would"
          " override topic learning unless --force_rerun is passed. Rename"
          " ground-truth columns (e.g., 'expected_topic', 'expected_opinion')."
      )

  for col in required_cols:
    blank_mask = df[col].astype(str).str.strip().eq("")
    if blank_mask.any():
      first_bad_row = int(blank_mask.idxmax()) + 1
      raise ValueError(
          f"Benchmark CSV '{expanded_path}' has an empty value in required"
          f" column '{col}' at row {first_bad_row}."
      )

  id_col = cfg.id_col.lower()
  if df[id_col].nunique() != len(df):
    duplicates = df[df[id_col].duplicated()][id_col].unique().tolist()
    raise ValueError(
        f"Benchmark CSV '{expanded_path}' has duplicate '{id_col}' values:"
        f" {duplicates}. Each comment must have a unique '{id_col}' so"
        " categorization_runner does not deduplicate them."
    )

  return df


def export_guided_topics_arg(
    df: pd.DataFrame,
    config: Optional[BenchmarkConfig] = None,
) -> str:
  """Returns a comma-separated string of unique ground-truth topics for `--topics`.

  Args:
    df: Benchmark DataFrame loaded via `load_benchmark_dataset`.
    config: Optional `BenchmarkConfig`.

  Returns:
    Comma-separated string of unique expected topic names in appearance order.
  """
  cfg = config or BenchmarkConfig()
  topic_col = cfg.expected_topic_col.lower()
  unique_topics = [
      str(t).strip()
      for t in df[topic_col].drop_duplicates().tolist()
      if str(t).strip()
  ]
  return ",".join(unique_topics)


def export_topic_and_opinion_csv(
    df: pd.DataFrame,
    output_csv_path: str,
    config: Optional[BenchmarkConfig] = None,
) -> pd.DataFrame:
  """Writes a `topic,opinion` taxonomy CSV for `--topic_and_opinion_csv`.

  Args:
    df: Benchmark DataFrame loaded via `load_benchmark_dataset`.
    output_csv_path: Destination file path for the generated CSV.
    config: Optional `BenchmarkConfig`.

  Returns:
    The exported `pd.DataFrame` with columns `['topic', 'opinion']`.
  """
  cfg = config or BenchmarkConfig()
  topic_col = cfg.expected_topic_col.lower()
  opinion_col = cfg.expected_opinion_col.lower()

  taxonomy_df = (
      df[[topic_col, opinion_col]]
      .drop_duplicates()
      .rename(columns={topic_col: "topic", opinion_col: "opinion"})
      .reset_index(drop=True)
  )
  parent_dir = os.path.dirname(os.path.abspath(output_csv_path))
  if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)
  taxonomy_df.to_csv(output_csv_path, index=False)
  return taxonomy_df


def _compute_pairwise_coassignment(
    ground_truth_by_id: Dict[str, str],
    predicted_sets_by_id: Dict[str, Set[str]],
) -> Dict[str, float]:
  """Computes pairwise co-assignment precision, recall, and F1 across statements.

  Two statements `(a, b)` are a ground-truth positive pair if they share the
  same expected category. They are a predicted positive pair if they share at
  least one substantive (non-Other) predicted category.

  Args:
    ground_truth_by_id: Mapping from statement ID to ground-truth label.
    predicted_sets_by_id: Mapping from statement ID to set of non-Other
      predicted labels assigned to that statement.

  Returns:
    Dictionary with `pairwise_precision`, `pairwise_recall`, and `pairwise_f1`.
  """
  statement_ids = sorted(ground_truth_by_id.keys())
  true_positives = 0
  false_positives = 0
  false_negatives = 0

  for id_a, id_b in itertools.combinations(statement_ids, 2):
    same_expected = ground_truth_by_id[id_a] == ground_truth_by_id[id_b]
    preds_a = predicted_sets_by_id.get(id_a, set())
    preds_b = predicted_sets_by_id.get(id_b, set())
    same_predicted = bool(preds_a and preds_b and (preds_a & preds_b))

    if same_expected and same_predicted:
      true_positives += 1
    elif not same_expected and same_predicted:
      false_positives += 1
    elif same_expected and not same_predicted:
      false_negatives += 1

  precision = (
      true_positives / (true_positives + false_positives)
      if (true_positives + false_positives) > 0
      else 0.0
  )
  recall = (
      true_positives / (true_positives + false_negatives)
      if (true_positives + false_negatives) > 0
      else 0.0
  )
  return {
      "pairwise_precision": precision,
      "pairwise_recall": recall,
      "pairwise_f1": _safe_harmonic_mean(precision, recall),
  }


def _compute_v_measure(
    pairs: Sequence[Tuple[str, str]],
) -> Dict[str, float]:
  """Computes Homogeneity, Completeness, and V-Measure via ``sklearn.metrics``.

  Args:
    pairs: Sequence of `(expected_label, predicted_label)` tuples.

  Returns:
    Dictionary with `homogeneity`, `completeness`, and `v_measure`.
  """
  if not pairs:
    return {"homogeneity": 0.0, "completeness": 0.0, "v_measure": 0.0}

  y_true = [exp for exp, _ in pairs]
  y_pred = [pred for _, pred in pairs]
  homogeneity, completeness, v_measure = (
      sklearn_metrics.homogeneity_completeness_v_measure(y_true, y_pred)
  )
  return {
      "homogeneity": float(homogeneity),
      "completeness": float(completeness),
      "v_measure": float(v_measure),
  }


def _matches_any_keyword(text: str, keywords_str: str) -> bool:
  """Returns True if `text` contains any semicolon-separated keyword in `keywords_str`."""
  if not keywords_str or not text:
    return False
  lowered_text = text.lower()
  keywords = [
      kw.strip().lower() for kw in keywords_str.split(";") if kw.strip()
  ]
  return any(kw in lowered_text for kw in keywords)


def _compute_level_metrics(
    merged_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    id_col: str,
    expected_col: str,
    predicted_col: str,
    other_label: str = "Other",
    stance_col: Optional[str] = None,
    keywords_col: Optional[str] = None,
    aux_predicted_col: Optional[str] = None,
) -> Dict[str, Any]:
  """Computes clustering, exact-match, and keyword alignment metrics at one level.

  Args:
    merged_df: DataFrame of categorized rows joined with ground-truth columns.
    ground_truth_df: Deduplicated ground-truth DataFrame (one row/statement).
    id_col: Statement identifier column name.
    expected_col: Ground-truth category column name.
    predicted_col: Predicted category column name.
    other_label: Catch-all category label ('Other').
    stance_col: Optional short stance column name for reporting.
    keywords_col: Optional semicolon-separated expected keywords column name.
    aux_predicted_col: Optional parent/auxiliary predicted column (e.g., 'topic'
      when evaluating 'opinion') to include in keyword checks and cluster names.

  Returns:
    Dictionary of computed metrics and per-ground-truth-group mapping details.
  """
  total_gt = len(ground_truth_df)
  gt_by_id: Dict[str, str] = dict(
      zip(
          ground_truth_df[id_col].astype(str),
          ground_truth_df[expected_col].astype(str).str.strip(),
      )
  )
  stance_by_expected: Dict[str, str] = {}
  if stance_col and stance_col in ground_truth_df.columns:
    stance_rows = ground_truth_df[[expected_col, stance_col]].drop_duplicates()
    for _, row in stance_rows.iterrows():
      stance_by_expected[str(row[expected_col]).strip()] = str(
          row[stance_col]
      ).strip()

  keywords_by_id: Dict[str, str] = {}
  keywords_by_expected: Dict[str, str] = {}
  if keywords_col and keywords_col in ground_truth_df.columns:
    keywords_by_id = dict(
        zip(
            ground_truth_df[id_col].astype(str),
            ground_truth_df[keywords_col].astype(str).str.strip(),
        )
    )
    kw_rows = ground_truth_df[[expected_col, keywords_col]].drop_duplicates()
    for _, row in kw_rows.iterrows():
      keywords_by_expected[str(row[expected_col]).strip()] = str(
          row[keywords_col]
      ).strip()

  # Build per-statement substantive (non-Other) predictions.
  predicted_sets_by_id: Dict[str, Set[str]] = {sid: set() for sid in gt_by_id}
  combined_text_sets_by_id: Dict[str, Set[str]] = {
      sid: set() for sid in gt_by_id
  }
  # Tuples of (statement_id, expected_label, predicted_label).
  substantive_pairs: List[Tuple[str, str, str]] = []

  for _, row in merged_df.iterrows():
    sid = str(row.get(id_col, "")).strip()
    if sid not in gt_by_id:
      continue
    pred_val = str(row.get(predicted_col, "")).strip()
    aux_val = (
        str(row.get(aux_predicted_col, "")).strip()
        if aux_predicted_col and aux_predicted_col in merged_df.columns
        else ""
    )
    is_pred_other = _is_other(pred_val, other_label) or (
        bool(aux_predicted_col) and _is_other(aux_val, other_label)
    )
    if not is_pred_other:
      predicted_sets_by_id[sid].add(pred_val)
      combined_label = (
          f"{aux_val} :: {pred_val}".strip(" :") if aux_val else pred_val
      )
      combined_text_sets_by_id[sid].add(combined_label)
      substantive_pairs.append((sid, gt_by_id[sid], pred_val))

  # Deduplicate (statement_id, expected, predicted) so multi-quote repeats of
  # the same opinion on the same statement do not inflate counts.
  unique_substantive_pairs = list(dict.fromkeys(substantive_pairs))
  expected_pred_tuples = [
      (exp, pred) for _, exp, pred in unique_substantive_pairs
  ]

  # Cluster Purity across all non-Other predicted clusters:
  pred_cluster_to_expected: Dict[str, collections.Counter[str]] = {}
  for _, exp, pred in unique_substantive_pairs:
    if pred not in pred_cluster_to_expected:
      pred_cluster_to_expected[pred] = collections.Counter()
    pred_cluster_to_expected[pred][exp] += 1

  total_substantive_assignments = len(unique_substantive_pairs)
  if total_substantive_assignments > 0:
    pure_assignments = sum(
        counter.most_common(1)[0][1]
        for counter in pred_cluster_to_expected.values()
    )
    cluster_purity = pure_assignments / total_substantive_assignments
  else:
    cluster_purity = 0.0

  # Inverse Purity (Recall) across all ground-truth groups:
  expected_to_pred_cluster: Dict[str, collections.Counter[str]] = {
      exp: collections.Counter()
      for exp in ground_truth_df[expected_col].unique()
  }
  for _, exp, pred in unique_substantive_pairs:
    expected_to_pred_cluster[exp][pred] += 1

  recalled_in_dominant_clusters = sum(
      counter.most_common(1)[0][1] if counter else 0
      for counter in expected_to_pred_cluster.values()
  )
  inverse_purity_recall = (
      recalled_in_dominant_clusters / total_gt if total_gt > 0 else 0.0
  )
  non_other_statement_count = sum(
      1 for preds in predicted_sets_by_id.values() if preds
  )
  conditional_inverse_purity = (
      recalled_in_dominant_clusters / non_other_statement_count
      if non_other_statement_count > 0
      else 0.0
  )
  purity_f1 = _safe_harmonic_mean(cluster_purity, inverse_purity_recall)

  # Pairwise co-assignment metrics across all ground-truth statement pairs:
  pairwise_metrics = _compute_pairwise_coassignment(
      gt_by_id, predicted_sets_by_id
  )

  # V-Measure (Homogeneity, Completeness, V-Measure) on substantive pairs:
  v_measure_metrics = _compute_v_measure(expected_pred_tuples)

  # Exact Match Rate (case-insensitive) across all ground-truth statements:
  exact_matches = sum(
      1
      for sid, exp in gt_by_id.items()
      if any(
          p.lower() == exp.lower() for p in predicted_sets_by_id.get(sid, set())
      )
  )
  exact_match_rate = exact_matches / total_gt if total_gt > 0 else 0.0

  # Statement-level keyword hit rate:
  keyword_hits = 0
  if keywords_by_id:
    for sid, kw_str in keywords_by_id.items():
      combined_preds = " ".join(combined_text_sets_by_id.get(sid, set()))
      if _matches_any_keyword(combined_preds, kw_str):
        keyword_hits += 1
  statement_keyword_hit_rate = (
      keyword_hits / total_gt if (total_gt > 0 and keywords_by_id) else 0.0
  )

  # Build dominant cluster mapping per expected group:
  dominant_mapping: List[Dict[str, Any]] = []
  group_keyword_matches = 0
  unique_expected_groups = (
      ground_truth_df[expected_col].drop_duplicates().tolist()
  )

  # Also track dominant aux_predicted_col (e.g. predicted topic) for each pred
  pred_to_aux: Dict[str, collections.Counter[str]] = {}
  if aux_predicted_col and aux_predicted_col in merged_df.columns:
    for _, row in merged_df.iterrows():
      pred_val = str(row.get(predicted_col, "")).strip()
      aux_val = str(row.get(aux_predicted_col, "")).strip()
      if not _is_other(pred_val, other_label) and not _is_other(
          aux_val, other_label
      ):
        if pred_val not in pred_to_aux:
          pred_to_aux[pred_val] = collections.Counter()
        pred_to_aux[pred_val][aux_val] += 1

  for exp_label in unique_expected_groups:
    exp_ids = [sid for sid, exp in gt_by_id.items() if exp == exp_label]
    group_total = len(exp_ids)
    group_non_other = sum(1 for sid in exp_ids if predicted_sets_by_id.get(sid))
    group_other_or_missing = group_total - group_non_other

    pred_counter = expected_to_pred_cluster.get(
        exp_label, collections.Counter()
    )
    if pred_counter:
      dominant_pred, dominant_count = pred_counter.most_common(1)[0]
      dominant_cluster_total = sum(
          pred_cluster_to_expected[dominant_pred].values()
      )
      dominant_precision = (
          dominant_count / dominant_cluster_total
          if dominant_cluster_total > 0
          else 0.0
      )
      dominant_aux = (
          pred_to_aux[dominant_pred].most_common(1)[0][0]
          if dominant_pred in pred_to_aux and pred_to_aux[dominant_pred]
          else ""
      )
    else:
      dominant_pred = other_label
      dominant_count = 0
      dominant_precision = 0.0
      dominant_aux = ""

    dominant_recall = dominant_count / group_total if group_total > 0 else 0.0
    kw_str = keywords_by_expected.get(exp_label, "")
    combined_dominant_text = (
        f"{dominant_aux} {dominant_pred}".strip()
        if dominant_aux
        else dominant_pred
    )
    kw_matched = (
        _matches_any_keyword(combined_dominant_text, kw_str)
        if dominant_count > 0 and kw_str
        else False
    )
    if kw_matched:
      group_keyword_matches += 1

    entry: Dict[str, Any] = {
        "expected_label": exp_label,
        "expected_stance": stance_by_expected.get(exp_label, ""),
        "total_statements": group_total,
        "non_other_statements": group_non_other,
        "other_or_missing_statements": group_other_or_missing,
        "dominant_predicted_label": dominant_pred,
        "dominant_cluster_matches": dominant_count,
        "dominant_cluster_recall": dominant_recall,
        "dominant_cluster_precision": dominant_precision,
        "keyword_matched_in_dominant_label": kw_matched,
    }
    if aux_predicted_col:
      entry["dominant_predicted_parent"] = dominant_aux
    dominant_mapping.append(entry)

  group_keyword_coverage = (
      group_keyword_matches / len(unique_expected_groups)
      if (unique_expected_groups and keywords_by_expected)
      else 0.0
  )

  return {
      "cluster_purity": cluster_purity,
      "inverse_purity_recall": inverse_purity_recall,
      "conditional_inverse_purity": conditional_inverse_purity,
      "purity_f1": purity_f1,
      "pairwise_precision": pairwise_metrics["pairwise_precision"],
      "pairwise_recall": pairwise_metrics["pairwise_recall"],
      "pairwise_f1": pairwise_metrics["pairwise_f1"],
      "homogeneity": v_measure_metrics["homogeneity"],
      "completeness": v_measure_metrics["completeness"],
      "v_measure": v_measure_metrics["v_measure"],
      "exact_match_rate": exact_match_rate,
      "statement_keyword_hit_rate": statement_keyword_hit_rate,
      "group_keyword_coverage": group_keyword_coverage,
      "num_expected_categories": len(unique_expected_groups),
      "num_predicted_non_other_categories": len(pred_cluster_to_expected),
      "dominant_cluster_mapping": dominant_mapping,
  }


def evaluate_categorization_output(
    categorized_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame] = None,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
  """Evaluates a categorized DataFrame against ground-truth benchmark labels.

  If `ground_truth_df` is omitted, the ground-truth columns (`expected_topic`,
  `expected_opinion`, etc.) are read directly from `categorized_df` (which
  `categorization_runner` preserves automatically from the input CSV).

  Args:
    categorized_df: DataFrame read from `categorized_with_other.csv`.
    ground_truth_df: Optional benchmark input DataFrame containing all
      ground-truth rows. Providing this ensures any statements dropped before
      output are still counted in denominator coverage metrics.
    config: Optional `BenchmarkConfig` specifying column names.

  Returns:
    Dictionary containing overall coverage rates (`summary`), `topic_metrics`,
    and `opinion_metrics`.

  Raises:
    ValueError: If required prediction or ground-truth columns are missing.
  """
  cfg = config or BenchmarkConfig()
  cat_df = _normalize_headers(categorized_df)

  id_col = cfg.id_col.lower()
  pred_topic_col = cfg.predicted_topic_col.lower()
  pred_opinion_col = cfg.predicted_opinion_col.lower()
  exp_topic_col = cfg.expected_topic_col.lower()
  exp_opinion_col = cfg.expected_opinion_col.lower()
  stance_col = (
      cfg.expected_stance_col.lower() if cfg.expected_stance_col else None
  )
  keywords_col = (
      cfg.expected_keywords_col.lower() if cfg.expected_keywords_col else None
  )

  for req_pred_col in (id_col, pred_topic_col, pred_opinion_col):
    if req_pred_col not in cat_df.columns:
      raise ValueError(
          f"Categorized DataFrame is missing required column '{req_pred_col}'."
          f" Present columns: {list(cat_df.columns)}"
      )

  if ground_truth_df is not None:
    gt_df = _normalize_headers(ground_truth_df)
    for req_gt_col in (id_col, exp_topic_col, exp_opinion_col):
      if req_gt_col not in gt_df.columns:
        raise ValueError(
            "Ground-truth DataFrame is missing required column"
            f" '{req_gt_col}'. Present columns: {list(gt_df.columns)}"
        )
    gt_dedup = gt_df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    gt_cols_to_merge = [
        col
        for col in [
            id_col,
            exp_topic_col,
            exp_opinion_col,
            stance_col,
            keywords_col,
        ]
        if col and col in gt_dedup.columns
    ]
    cat_without_gt = cat_df.drop(
        columns=[
            c for c in gt_cols_to_merge if c != id_col and c in cat_df.columns
        ],
        errors="ignore",
    )
    merged_df = pd.merge(
        cat_without_gt,
        gt_dedup[gt_cols_to_merge],
        on=id_col,
        how="inner",
    )
  else:
    for req_gt_col in (exp_topic_col, exp_opinion_col):
      if req_gt_col not in cat_df.columns:
        raise ValueError(
            "Categorized DataFrame is missing ground-truth column"
            f" '{req_gt_col}' and no separate ground_truth_df was provided."
        )
    merged_df = cat_df.copy()
    gt_dedup = cat_df.drop_duplicates(subset=[id_col]).reset_index(drop=True)

  total_gt_statements = len(gt_dedup)
  gt_ids = set(gt_dedup[id_col].astype(str).str.strip())
  present_ids = set(merged_df[id_col].astype(str).str.strip()) & gt_ids
  missing_ids = sorted(gt_ids - present_ids)

  # Statement-level non-Other coverage:
  non_other_topic_ids: Set[str] = set()
  non_other_opinion_ids: Set[str] = set()
  topic_other_rows = 0
  opinion_other_rows = 0
  total_output_rows = len(merged_df)

  for _, row in merged_df.iterrows():
    sid = str(row[id_col]).strip()
    t_val = str(row[pred_topic_col]).strip()
    o_val = str(row[pred_opinion_col]).strip()
    t_is_other = _is_other(t_val, cfg.other_label)
    o_is_other = _is_other(o_val, cfg.other_label)

    if t_is_other:
      topic_other_rows += 1
    else:
      non_other_topic_ids.add(sid)

    if t_is_other or o_is_other:
      opinion_other_rows += 1
    else:
      non_other_opinion_ids.add(sid)

  topic_metrics = _compute_level_metrics(
      merged_df=merged_df,
      ground_truth_df=gt_dedup,
      id_col=id_col,
      expected_col=exp_topic_col,
      predicted_col=pred_topic_col,
      other_label=cfg.other_label,
      stance_col=None,
      keywords_col=None,
      aux_predicted_col=None,
  )

  opinion_metrics = _compute_level_metrics(
      merged_df=merged_df,
      ground_truth_df=gt_dedup,
      id_col=id_col,
      expected_col=exp_opinion_col,
      predicted_col=pred_opinion_col,
      other_label=cfg.other_label,
      stance_col=stance_col,
      keywords_col=keywords_col,
      aux_predicted_col=pred_topic_col,
  )

  return {
      "summary": {
          "total_ground_truth_statements": total_gt_statements,
          "categorized_statements_present": len(present_ids),
          "missing_statements_count": len(missing_ids),
          "missing_statement_ids": missing_ids,
          "statement_coverage_rate": (
              len(present_ids) / total_gt_statements
              if total_gt_statements > 0
              else 0.0
          ),
          "non_other_topic_statement_rate": (
              len(non_other_topic_ids) / total_gt_statements
              if total_gt_statements > 0
              else 0.0
          ),
          "non_other_opinion_statement_rate": (
              len(non_other_opinion_ids) / total_gt_statements
              if total_gt_statements > 0
              else 0.0
          ),
          "total_output_rows": total_output_rows,
          "topic_other_row_rate": (
              topic_other_rows / total_output_rows
              if total_output_rows > 0
              else 0.0
          ),
          "opinion_other_row_rate": (
              opinion_other_rows / total_output_rows
              if total_output_rows > 0
              else 0.0
          ),
      },
      "topic_metrics": topic_metrics,
      "opinion_metrics": opinion_metrics,
  }


def evaluate_categorization_files(
    categorized_csv_path: str,
    benchmark_csv_path: Optional[str] = DEFAULT_WEDDING_BENCHMARK_CSV,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
  """Loads `categorized_with_other.csv` and benchmark CSV and evaluates them.

  Args:
    categorized_csv_path: Path to `categorized_with_other.csv`.
    benchmark_csv_path: Optional path to the ground-truth benchmark CSV.
    config: Optional `BenchmarkConfig`.

  Returns:
    Evaluation metrics dictionary from `evaluate_categorization_output`.

  Raises:
    FileNotFoundError: If `categorized_csv_path` does not exist.
  """
  cfg = config or BenchmarkConfig()
  expanded_cat_path = os.path.expanduser(categorized_csv_path)
  if not os.path.exists(expanded_cat_path):
    raise FileNotFoundError(
        f"Categorized output CSV not found: {expanded_cat_path}"
    )

  cat_df = pd.read_csv(expanded_cat_path, dtype=str, keep_default_na=False)
  gt_df = (
      load_benchmark_dataset(benchmark_csv_path, config=cfg)
      if benchmark_csv_path
      else None
  )
  return evaluate_categorization_output(
      categorized_df=cat_df,
      ground_truth_df=gt_df,
      config=cfg,
  )


def format_evaluation_report(metrics: Dict[str, Any]) -> str:
  """Formats the evaluation metrics dictionary into a human-readable report.

  Args:
    metrics: Dictionary returned by `evaluate_categorization_output`.

  Returns:
    Formatted multi-line summary report string.
  """
  summary = metrics["summary"]
  t_metrics = metrics["topic_metrics"]
  o_metrics = metrics["opinion_metrics"]

  lines = [
      "=" * 78,
      "CATEGORIZATION BENCHMARK EVALUATION REPORT",
      "=" * 78,
      (
          "Statement Coverage:       "
          f" {summary['categorized_statements_present']}/{summary['total_ground_truth_statements']}"
          f" ({summary['statement_coverage_rate']:.1%})"
      ),
      (
          "Non-'Other' Topic Rate:   "
          f" {summary['non_other_topic_statement_rate']:.1%} of statements"
          f" (row 'Other' rate: {summary['topic_other_row_rate']:.1%})"
      ),
      (
          "Non-'Other' Opinion Rate: "
          f" {summary['non_other_opinion_statement_rate']:.1%} of statements"
          f" (row 'Other' rate: {summary['opinion_other_row_rate']:.1%})"
      ),
      "-" * 78,
      (
          "TOPIC-LEVEL ALIGNMENT"
          f" (Expected: {t_metrics['num_expected_categories']},"
          " Predicted non-Other:"
          f" {t_metrics['num_predicted_non_other_categories']})"
      ),
      f"  Cluster Purity:         {t_metrics['cluster_purity']:.3f}",
      f"  Inverse Purity (Recall):{t_metrics['inverse_purity_recall']:.3f}",
      f"  Purity F1:              {t_metrics['purity_f1']:.3f}",
      (
          f"  Pairwise Co-assignment: P={t_metrics['pairwise_precision']:.3f},"
          f" R={t_metrics['pairwise_recall']:.3f},"
          f" F1={t_metrics['pairwise_f1']:.3f}"
      ),
      (
          f"  V-Measure:              {t_metrics['v_measure']:.3f}"
          f" (Homogeneity={t_metrics['homogeneity']:.3f},"
          f" Completeness={t_metrics['completeness']:.3f})"
      ),
      f"  Exact Label Match Rate: {t_metrics['exact_match_rate']:.1%}",
      "-" * 78,
      (
          "OPINION-LEVEL ALIGNMENT"
          f" (Expected: {o_metrics['num_expected_categories']},"
          " Predicted non-Other:"
          f" {o_metrics['num_predicted_non_other_categories']})"
      ),
      f"  Cluster Purity:         {o_metrics['cluster_purity']:.3f}",
      f"  Inverse Purity (Recall):{o_metrics['inverse_purity_recall']:.3f}",
      f"  Purity F1:              {o_metrics['purity_f1']:.3f}",
      (
          f"  Pairwise Co-assignment: P={o_metrics['pairwise_precision']:.3f},"
          f" R={o_metrics['pairwise_recall']:.3f},"
          f" F1={o_metrics['pairwise_f1']:.3f}"
      ),
      (
          f"  V-Measure:              {o_metrics['v_measure']:.3f}"
          f" (Homogeneity={o_metrics['homogeneity']:.3f},"
          f" Completeness={o_metrics['completeness']:.3f})"
      ),
      f"  Exact Label Match Rate: {o_metrics['exact_match_rate']:.1%}",
      (
          "  Keyword Hit Rate:      "
          f" {o_metrics['statement_keyword_hit_rate']:.1%}"
      ),
      f"  Group Keyword Coverage: {o_metrics['group_keyword_coverage']:.1%}",
      "-" * 78,
      "DOMINANT PREDICTED CLUSTER PER EXPECTED OPINION GROUP:",
  ]

  for item in o_metrics["dominant_cluster_mapping"]:
    stance_tag = (
        f"[{item['expected_stance']}] " if item.get("expected_stance") else ""
    )
    parent_prefix = (
        f"{item.get('dominant_predicted_parent')} -> "
        if item.get("dominant_predicted_parent")
        else ""
    )
    dom_pred = item["dominant_predicted_label"]
    dom_matches = item["dominant_cluster_matches"]
    total_stmts = item["total_statements"]
    non_other_stmts = item["non_other_statements"]
    lines.append(
        f"  * {stance_tag}{item['expected_label']}\n"
        f"    -> Dominant Predicted: {parent_prefix}{dom_pred}\n"
        f"    -> Recall: {dom_matches}/{total_stmts}"
        f" ({item['dominant_cluster_recall']:.1%}),"
        f" Cluster Precision: {item['dominant_cluster_precision']:.1%},"
        f" Non-Other: {non_other_stmts}/{total_stmts},"
        f" Keyword Match: {item['keyword_matched_in_dominant_label']}"
    )

  lines.append("=" * 78)
  return "\n".join(lines)


def save_evaluation_artifacts(
    metrics: Dict[str, Any], output_dir: str
) -> Tuple[str, str]:
  """Writes JSON metrics and a pandas CSV table of dominant opinion mappings.

  Args:
    metrics: Dictionary returned by `evaluate_categorization_output`.
    output_dir: Directory where artifacts will be written.

  Returns:
    Tuple of `(json_report_path, opinion_mapping_csv_path)`.
  """
  os.makedirs(output_dir, exist_ok=True)
  json_path = os.path.join(output_dir, "benchmark_evaluation_report.json")
  with open(json_path, "w") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)

  mapping_df = pd.DataFrame(
      metrics["opinion_metrics"]["dominant_cluster_mapping"]
  )
  csv_path = os.path.join(output_dir, "benchmark_opinion_mapping.csv")
  mapping_df.to_csv(csv_path, index=False)
  return json_path, csv_path


def main(argv: Optional[Sequence[str]] = None) -> int:
  """CLI entrypoint for validating, running, and evaluating benchmarks."""
  parser = argparse.ArgumentParser(
      description=(
          "Validate, run, and evaluate Sensemaker's categorization_runner on"
          " labeled benchmark datasets."
      )
  )
  parser.add_argument(
      "--benchmark_csv",
      type=str,
      default=DEFAULT_WEDDING_BENCHMARK_CSV,
      help=(
          "Path to the labeled ground-truth benchmark CSV (default:"
          f" {DEFAULT_WEDDING_BENCHMARK_CSV})."
      ),
  )
  parser.add_argument(
      "--categorized_csv",
      type=str,
      default=None,
      help=(
          "Path to an existing categorized_with_other.csv to evaluate."
          " Required unless --validate_only, --export_taxonomy_csv, or"
          " --run_categorization is set."
      ),
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=None,
      help="Directory to write evaluation artifacts or categorization outputs.",
  )
  parser.add_argument(
      "--validate_only",
      action="store_true",
      help="Validate the benchmark CSV and print dataset summary statistics.",
  )
  parser.add_argument(
      "--export_taxonomy_csv",
      type=str,
      default=None,
      help=(
          "Export unique (expected_topic, expected_opinion) pairs to this CSV"
          " path for use with categorization_runner --topic_and_opinion_csv."
      ),
  )
  parser.add_argument(
      "--run_categorization",
      action="store_true",
      help=(
          "Invoke `python -m src.categorization_runner` on --benchmark_csv"
          " before evaluating the output."
      ),
  )
  parser.add_argument(
      "--mode",
      choices=["learn_all", "guided_topics", "guided_taxonomy"],
      default="learn_all",
      help=(
          "Categorization mode when --run_categorization is set: 'learn_all'"
          " learns topics and opinions from scratch; 'guided_topics' passes"
          " ground-truth --topics; 'guided_taxonomy' passes"
          " --topic_and_opinion_csv."
      ),
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemma4:26b",
      help="Model name passed to categorization_runner when running E2E.",
  )
  parser.add_argument(
      "--gemini_api_key",
      type=str,
      default=None,
      help="Optional Gemini API key passed to categorization_runner.",
  )
  parser.add_argument(
      "--max_concurrent_calls",
      type=int,
      default=None,
      help="Optional --max_concurrent_calls passed to categorization_runner.",
  )
  parser.add_argument(
      "--skip_autoraters",
      action="store_true",
      default=False,
      help="Pass --skip_autoraters to categorization_runner.",
  )
  parser.add_argument(
      "--recategorize_other",
      action="store_true",
      default=False,
      help="Pass --recategorize_other to categorization_runner.",
  )
  parser.add_argument(
      "--additional_context_file",
      type=str,
      default=DEFAULT_WEDDING_CONTEXT_FILE,
      help="Additional context file passed to categorization_runner.",
  )
  parser.add_argument(
      "--skip_quote_extraction",
      action="store_true",
      default=True,
      help="Pass --skip_quote_extraction to categorization_runner (default).",
  )
  parser.add_argument(
      "--expected_topic_col",
      type=str,
      default="expected_topic",
      help="Column name for expected topics in --benchmark_csv.",
  )
  parser.add_argument(
      "--expected_opinion_col",
      type=str,
      default="expected_opinion",
      help="Column name for expected opinions in --benchmark_csv.",
  )
  parser.add_argument(
      "--expected_stance_col",
      type=str,
      default="expected_stance",
      help="Column name for expected stance slug in --benchmark_csv.",
  )
  parser.add_argument(
      "--expected_keywords_col",
      type=str,
      default="expected_keywords",
      help="Column name for semicolon-separated expected keywords.",
  )

  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

  config = BenchmarkConfig(
      expected_topic_col=args.expected_topic_col,
      expected_opinion_col=args.expected_opinion_col,
      expected_stance_col=args.expected_stance_col,
      expected_keywords_col=args.expected_keywords_col,
  )

  benchmark_df = load_benchmark_dataset(args.benchmark_csv, config=config)

  if args.export_taxonomy_csv:
    taxonomy_df = export_topic_and_opinion_csv(
        benchmark_df, args.export_taxonomy_csv, config=config
    )
    logging.info(
        "Exported %d topic-opinion pairs to %s",
        len(taxonomy_df),
        args.export_taxonomy_csv,
    )

  if args.validate_only:
    print(f"Validated benchmark dataset: {args.benchmark_csv}")
    print(f"Total statements: {len(benchmark_df)}")
    print(
        "Unique expected topics:"
        f" {benchmark_df[config.expected_topic_col].nunique()}"
    )
    print(
        "Unique expected opinions:"
        f" {benchmark_df[config.expected_opinion_col].nunique()}"
    )
    return 0

  categorized_csv_path = args.categorized_csv
  if args.run_categorization:
    if not args.output_dir:
      raise ValueError(
          "--output_dir is required when --run_categorization is enabled."
      )
    os.makedirs(args.output_dir, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.categorization_runner",
        "--input_file",
        args.benchmark_csv,
        "--output_dir",
        args.output_dir,
        "--model_name",
        args.model_name,
    ]
    if args.gemini_api_key:
      cmd.extend(["--gemini_api_key", args.gemini_api_key])
    if args.max_concurrent_calls is not None:
      cmd.extend(["--max_concurrent_calls", str(args.max_concurrent_calls)])
    if args.skip_autoraters:
      cmd.append("--skip_autoraters")
    if args.recategorize_other:
      cmd.append("--recategorize_other")
    if args.skip_quote_extraction:
      cmd.append("--skip_quote_extraction")
    if args.additional_context_file and os.path.exists(
        args.additional_context_file
    ):
      cmd.extend(["--additional_context_file", args.additional_context_file])

    if args.mode == "guided_topics":
      cmd.extend(["--topics", export_guided_topics_arg(benchmark_df, config)])
    elif args.mode == "guided_taxonomy":
      taxonomy_path = os.path.join(args.output_dir, "guided_taxonomy.csv")
      export_topic_and_opinion_csv(benchmark_df, taxonomy_path, config=config)
      cmd.extend(["--topic_and_opinion_csv", taxonomy_path])

    runner_env = build_runner_env(args.model_name)
    logging.info(
        "Model endpoint: %s (%s)",
        runner_env.get("MODEL_ENDPOINT_TYPE", "google_genai"),
        runner_env.get("OPENAI_API_ENDPOINT_URL", "default"),
    )
    logging.info("Running categorization_runner: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=runner_env)
    categorized_csv_path = os.path.join(
        args.output_dir, "categorized_with_other.csv"
    )

  if not categorized_csv_path:
    parser.error(
        "Provide --categorized_csv to evaluate an existing output, or pass"
        " --validate_only / --run_categorization."
    )

  metrics = evaluate_categorization_files(
      categorized_csv_path=categorized_csv_path,
      benchmark_csv_path=args.benchmark_csv,
      config=config,
  )
  print(format_evaluation_report(metrics))

  if args.output_dir:
    json_path, csv_path = save_evaluation_artifacts(metrics, args.output_dir)
    logging.info("Saved evaluation JSON to %s", json_path)
    logging.info("Saved opinion mapping CSV to %s", csv_path)

  return 0


if __name__ == "__main__":
  sys.exit(main())
