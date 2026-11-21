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

"""Bridging consensus metrics across participant segments.

Computes domain-agnostic bridging scores from a participant-by-candidate
approval matrix and segment membership definitions (either a single categorical
segment column or multiple potentially overlapping binary indicator columns).

Supported bridging metrics:
  - MinMax approval: minimum approval rate across valid segments.
  - Group-Informed Consensus (GIC) product: product of segment approval rates.
  - Group-Informed Consensus (GIC) geometric mean: scale-invariant geometric
    mean of segment approval rates across valid segments.
"""

import argparse
from collections import abc
import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd

BRIDGING_METRICS: tuple[str, ...] = (
    "minmax",
    "gic_product",
    "gic_geomean",
)
DEFAULT_BRIDGING_METRIC = "minmax"

BRIDGING_OUTPUT_COLUMNS: tuple[str, ...] = (
    "minmax_approval",
    "bottleneck_segment",
    "gic_product",
    "gic_geomean",
    "bridging_rank",
)

_METRIC_TO_COLUMN: abc.Mapping[str, str] = {
    "minmax": "minmax_approval",
    "gic_product": "gic_product",
    "gic_geomean": "gic_geomean",
}
_CORE_RESULT_COLUMNS: tuple[str, ...] = (
    "candidate",
    "minmax_approval",
    "bottleneck_segment",
    "gic_product",
    "gic_geomean",
    "overall_approval",
    "bridging_rank",
)
_TRUTHY_STRINGS = frozenset({"1", "1.0", "true", "t", "yes", "y"})
_FALSY_STRINGS = frozenset({
    "",
    "0",
    "0.0",
    "false",
    "f",
    "no",
    "n",
    "na",
    "nan",
    "none",
    "null",
})


@dataclasses.dataclass(frozen=True)
class BridgingScores:
  """Bridging consensus scores for a single candidate statement.

  Attributes:
    candidate: The candidate statement or identifier.
    minmax_approval: Minimum raw approval rate across valid segments, or NaN if
      no segment meets min_segment_size.
    bottleneck_segment: Name of the segment achieving minmax_approval (with ties
      broken deterministically by segment name), or None if no valid segment.
    gic_product: Product of (optionally smoothed) approval rates across valid
      segments, or NaN if no valid segment.
    gic_geomean: Geometric mean of (optionally smoothed) approval rates across
      valid segments, or NaN if no valid segment.
    overall_approval: Mean approval across all non-null votes for the candidate,
      or NaN if no non-null votes exist.
    segment_approvals: Raw approval rate for each valid segment.
    segment_sizes: Count of non-null votes for each valid segment.
  """

  candidate: str
  minmax_approval: float
  bottleneck_segment: str | None
  gic_product: float
  gic_geomean: float
  overall_approval: float
  segment_approvals: abc.Mapping[str, float]
  segment_sizes: abc.Mapping[str, int]


@dataclasses.dataclass(frozen=True)
class BridgingConfig:
  """Normalized bridging CLI configuration extracted from parsed args."""

  segment_column: str | None
  segment_columns: tuple[str, ...] | None
  metric: str | None
  min_segment_size: int


def bridging_config_from_args(
    args: argparse.Namespace,
) -> BridgingConfig | None:
  """Extracts normalized bridging configuration from CLI args, or None."""
  segment_column = getattr(args, "segment_column", None)
  segment_columns = parse_segment_columns(
      getattr(args, "segment_columns", None)
  )
  if segment_column is None and segment_columns is None:
    return None
  return BridgingConfig(
      segment_column=segment_column,
      segment_columns=tuple(segment_columns) if segment_columns else None,
      metric=getattr(args, "bridging_metric", None),
      min_segment_size=getattr(args, "min_segment_size", 1),
  )


def extract_segment_memberships_from_args(
    participants_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    participant_id_col: str = "participant_id",
) -> dict[str, set[Any]]:
  """Validates CLI bridging flags and extracts segment memberships."""
  config = bridging_config_from_args(args)
  if config is None:
    raise ValueError("Must specify either segment_column or segment_columns.")
  return extract_segment_memberships(
      participants_df,
      participant_id_col=participant_id_col,
      segment_column=config.segment_column,
      segment_columns=config.segment_columns,
  )


def min_segment_size_arg(value: str) -> int:
  """Argparse type validator requiring an integer >= 1.

  Args:
    value: Raw command-line argument string.

  Returns:
    Parsed positive integer.

  Raises:
    argparse.ArgumentTypeError: If value is not an integer >= 1.
  """
  try:
    parsed = int(value)
  except ValueError as err:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    ) from err
  if parsed < 1:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    )
  return parsed


def add_bridging_args(parser: argparse.ArgumentParser) -> None:
  """Adds standard CLI flags for segment-based bridging evaluation.

  Args:
    parser: The ArgumentParser to populate with bridging flags.
  """
  parser.add_argument(
      "--segment_column",
      type=str,
      default=None,
      help=(
          "Name of a single categorical segment column in the participants"
          " data for bridging consensus evaluation."
      ),
  )
  parser.add_argument(
      "--segment_columns",
      nargs="+",
      default=None,
      help=(
          "Binary indicator column names in the participants data for"
          " bridging consensus evaluation (supports overlapping segments;"
          " accepts space- or comma-separated names)."
      ),
  )
  parser.add_argument(
      "--bridging_metric",
      type=str,
      default=DEFAULT_BRIDGING_METRIC,
      choices=list(BRIDGING_METRICS),
      help=(
          "Bridging consensus metric used to rank candidates when segment"
          " columns are provided."
      ),
  )
  parser.add_argument(
      "--min_segment_size",
      type=min_segment_size_arg,
      default=1,
      help=(
          "Minimum number of non-null votes required for a segment to be"
          " included in bridging calculations."
      ),
  )


def parse_segment_columns(
    raw_segment_columns: abc.Sequence[str] | str | None,
) -> list[str] | None:
  """Normalizes CLI or programmatic segment_columns specifications.

  Supports either comma-separated strings (e.g. ``"seg_a,seg_b"`` or
  ``["seg_a,seg_b"]``) or sequences of individual column names (e.g.
  ``["seg_a", "seg_b"]``), stripping whitespace and deduplicating while
  preserving order.

  Args:
    raw_segment_columns: Raw segment column specification, or None.

  Returns:
    Deduplicated list of non-empty column names, or None when input is None.

  Raises:
    ValueError: If raw_segment_columns is provided but yields no valid names.
  """
  if raw_segment_columns is None:
    return None

  tokens = (
      [raw_segment_columns]
      if isinstance(raw_segment_columns, str)
      else list(raw_segment_columns)
  )
  parsed: list[str] = []
  for token in tokens:
    for part in token.split(","):
      cleaned = part.strip()
      if cleaned:
        parsed.append(cleaned)

  if not parsed:
    raise ValueError(
        "segment_columns must contain at least one non-empty column name."
    )
  return list(dict.fromkeys(parsed))


def _is_truthy_indicator(value: Any, column_name: str) -> bool:
  """Converts a single binary segment indicator cell to a boolean."""
  if pd.isna(value):
    return False
  if isinstance(value, (bool, np.bool_)):
    return bool(value)
  if isinstance(value, (int, float, np.integer, np.floating)):
    if value in (0, 1):
      return bool(value)
    raise ValueError(
        f"Expected binary indicator (0/1 or bool) in column {column_name!r},"
        f" got {value!r}."
    )
  if isinstance(value, str):
    normalized = value.strip().lower()
    if normalized in _TRUTHY_STRINGS:
      return True
    if normalized in _FALSY_STRINGS:
      return False
  raise ValueError(
      f"Expected binary indicator (0/1 or bool) in column {column_name!r},"
      f" got {value!r}."
  )


def extract_segment_memberships(
    participants_df: pd.DataFrame,
    *,
    participant_id_col: str = "participant_id",
    segment_column: str | None = None,
    segment_columns: abc.Sequence[str] | None = None,
) -> dict[str, set[Any]]:
  """Extracts segment-to-participant memberships from a participants DataFrame.

  Supports either a single categorical column (mutually exclusive segments) or
  a sequence of binary/boolean indicator columns (supporting overlapping
  segments or one-hot cluster indicators).

  Args:
    participants_df: DataFrame with participant metadata and segment columns.
    participant_id_col: Column containing participant identifiers matching the
      approval matrix index.
    segment_column: Name of a single categorical segment column.
    segment_columns: Sequence of binary/boolean indicator column names.

  Returns:
    Dictionary mapping each segment name to the set of member participant IDs.

  Raises:
    ValueError: If neither or both of segment_column and segment_columns are
      specified, if required columns are missing, or if a binary indicator
      column contains non-binary values.
  """
  if segment_column is not None and segment_columns is not None:
    raise ValueError(
        "Specify exactly one of segment_column or segment_columns, not both."
    )
  if segment_column is None and segment_columns is None:
    raise ValueError("Must specify either segment_column or segment_columns.")

  if participant_id_col not in participants_df.columns:
    raise ValueError(
        f"Participant ID column {participant_id_col!r} not found in"
        " participants DataFrame."
    )

  if segment_column is not None:
    if not isinstance(segment_column, str) or not segment_column.strip():
      raise ValueError("segment_column must be a non-empty string.")
    if segment_column not in participants_df.columns:
      raise ValueError(
          f"Segment column {segment_column!r} not found in participants"
          " DataFrame."
      )

    valid_df = participants_df[[participant_id_col, segment_column]].dropna()
    labels = valid_df[segment_column].astype(str).str.strip()
    non_empty = valid_df[labels != ""].assign(
        _segment_label=labels[labels != ""]
    )
    grouped = non_empty.groupby("_segment_label", sort=True)[participant_id_col]
    return {str(label): set(group) for label, group in grouped}

  if isinstance(segment_columns, str):
    raise ValueError(
        "segment_columns must be a sequence of column names, not a string."
    )
  column_list = list(segment_columns)
  if not column_list:
    raise ValueError("segment_columns must contain at least one column name.")

  missing_columns = [
      col for col in column_list if col not in participants_df.columns
  ]
  if missing_columns:
    raise ValueError(
        "Segment column(s) not found in participants DataFrame:"
        f" {missing_columns}."
    )

  valid_df = participants_df.dropna(subset=[participant_id_col])
  memberships: dict[str, set[Any]] = {}
  for col in column_list:
    mask = valid_df[col].map(lambda val, c=col: _is_truthy_indicator(val, c))
    memberships[col] = set(valid_df.loc[mask, participant_id_col])
  return memberships


def _validate_numeric_votes(series: pd.Series, candidate: str) -> pd.Series:
  """Returns non-null votes as floats after validating they lie in [0, 1]."""
  non_null = series.dropna()
  if non_null.empty:
    return pd.Series(dtype=float)
  try:
    votes = non_null.astype(float)
  except (TypeError, ValueError) as err:
    raise ValueError(
        f"Approval values for candidate {candidate!r} must be boolean or"
        " numeric in [0, 1]."
    ) from err
  if not np.isfinite(votes).all() or ((votes < 0.0) | (votes > 1.0)).any():
    raise ValueError(
        f"Approval values for candidate {candidate!r} must be finite and in"
        " [0, 1]."
    )
  return votes


def _compute_gic_metrics(
    smoothed_rates: abc.Sequence[float],
) -> tuple[float, float]:
  """Computes GIC product and geometric mean with underflow protection."""
  gic_product = math.prod(smoothed_rates)
  if any(rate == 0.0 for rate in smoothed_rates):
    return 0.0, 0.0
  num_segments = len(smoothed_rates)
  if gic_product > 0.0:
    gic_geomean = gic_product ** (1.0 / num_segments)
  else:
    # Fall back to log-space when many segments cause product float underflow.
    gic_geomean = math.exp(
        sum(math.log(rate) for rate in smoothed_rates) / num_segments
    )
  return gic_product, gic_geomean


def compute_bridging_scores(
    approval_matrix: pd.DataFrame,
    segment_memberships: abc.Mapping[str, abc.Collection[Any]],
    *,
    min_segment_size: int = 1,
    smoothing: float = 0.0,
) -> dict[str, BridgingScores]:
  """Computes per-candidate bridging consensus metrics across segments.

  For each candidate column in approval_matrix, evaluates non-null approval
  votes within each segment meeting min_segment_size. Raw segment approval
  rates are ``k_g / n_g`` where ``k_g`` is the sum of approvals and ``n_g`` is
  the count of non-null votes in segment ``g``. When ``smoothing > 0``,
  Group-Informed Consensus (GIC) metrics apply Laplace/Beta smoothing
  ``(k_g + smoothing) / (n_g + 2 * smoothing)`` per segment to prevent zero
  collapse while remaining bounded in ``(0, 1)``.

  Args:
    approval_matrix: DataFrame indexed by participant ID with candidate
      statements as columns and boolean or numeric values in [0, 1] (NaN for
      missing votes).
    segment_memberships: Mapping from segment name to collection of member
      participant IDs. Segments may overlap.
    min_segment_size: Minimum number of non-null votes required for a segment
      to be included for a candidate. Must be >= 1.
    smoothing: Non-negative additive Laplace smoothing parameter applied to GIC
      product and geometric mean calculations. Defaults to 0.0 (unsmoothed).

  Returns:
    Dictionary mapping each candidate column name to its BridgingScores.

  Raises:
    ValueError: If segment_memberships is empty, parameters are out of range,
      approval_matrix has duplicate indices or columns, or votes fall outside
      [0, 1].
  """
  if min_segment_size < 1:
    raise ValueError(f"min_segment_size must be >= 1, got {min_segment_size}.")
  if not math.isfinite(smoothing) or smoothing < 0.0:
    raise ValueError(f"smoothing must be finite and >= 0.0, got {smoothing}.")
  if not segment_memberships:
    raise ValueError("segment_memberships must contain at least one segment.")
  if approval_matrix.index.has_duplicates:
    raise ValueError(
        "approval_matrix index contains duplicate participant IDs."
    )
  if approval_matrix.columns.has_duplicates:
    raise ValueError(
        "approval_matrix columns contain duplicate candidate names."
    )

  # Materialize membership sets once in deterministic segment order.
  sorted_segment_names = sorted(segment_memberships)
  segment_sets = {
      name: frozenset(segment_memberships[name])
      for name in sorted_segment_names
  }

  scores_by_candidate: dict[str, BridgingScores] = {}
  for candidate in approval_matrix.columns:
    valid_votes = _validate_numeric_votes(approval_matrix[candidate], candidate)
    overall_approval = (
        float(valid_votes.mean()) if not valid_votes.empty else float("nan")
    )

    segment_approvals: dict[str, float] = {}
    segment_sizes: dict[str, int] = {}
    smoothed_rates: list[float] = []

    for seg_name in sorted_segment_names:
      if valid_votes.empty:
        continue
      seg_votes = valid_votes[valid_votes.index.isin(segment_sets[seg_name])]
      seg_size = seg_votes.shape[0]
      if seg_size < min_segment_size:
        continue
      approvals_sum = float(seg_votes.sum())
      segment_approvals[seg_name] = approvals_sum / seg_size
      segment_sizes[seg_name] = seg_size
      smoothed_rates.append(
          (approvals_sum + smoothing) / (seg_size + 2.0 * smoothing)
      )

    if segment_approvals:
      # Break ties deterministically by segment name via sorted key order.
      bottleneck_segment, minmax_approval = min(
          segment_approvals.items(), key=lambda item: (item[1], item[0])
      )
      gic_product, gic_geomean = _compute_gic_metrics(smoothed_rates)
    else:
      bottleneck_segment = None
      minmax_approval = float("nan")
      gic_product = float("nan")
      gic_geomean = float("nan")

    scores_by_candidate[candidate] = BridgingScores(
        candidate=candidate,
        minmax_approval=minmax_approval,
        bottleneck_segment=bottleneck_segment,
        gic_product=gic_product,
        gic_geomean=gic_geomean,
        overall_approval=overall_approval,
        segment_approvals=segment_approvals,
        segment_sizes=segment_sizes,
    )

  return scores_by_candidate


def rank_by_bridging(
    approval_matrix: pd.DataFrame,
    segment_memberships: abc.Mapping[str, abc.Collection[Any]],
    *,
    metric: str | None = None,
    min_segment_size: int = 1,
    smoothing: float = 0.0,
    include_segment_rates: bool = False,
    segment_column_prefix: str = "segment_",
) -> pd.DataFrame:
  """Ranks candidates by a bridging consensus metric with deterministic ties.

  Candidates are ordered by the selected metric descending, with
  ``overall_approval`` descending as tie-breaker and candidate name ascending
  for remaining ties. Candidates with NaN in the selected metric are placed
  last and receive a missing ``bridging_rank``.

  Args:
    approval_matrix: Participant-by-candidate approval DataFrame.
    segment_memberships: Mapping from segment name to collection of participant
      IDs.
    metric: Primary ranking metric; one of ``"minmax"``, ``"gic_product"``, or
      ``"gic_geomean"``. If None, uses ``DEFAULT_BRIDGING_METRIC``.
    min_segment_size: Minimum non-null votes per segment for inclusion.
    smoothing: Non-negative Laplace smoothing parameter for GIC metrics.
    include_segment_rates: Whether to append one column per segment containing
      its raw approval rate.
    segment_column_prefix: Prefix for per-segment rate columns when
      ``include_segment_rates`` is True.

  Returns:
    DataFrame sorted by bridging rank with columns ``candidate``,
    ``minmax_approval``, ``bottleneck_segment``, ``gic_product``,
    ``gic_geomean``, ``overall_approval``, ``bridging_rank``, and optional
    per-segment approval rate columns.

  Raises:
    ValueError: If metric is unsupported, segment column names collide with
      core result columns, or underlying validation fails.
  """
  resolved_metric = DEFAULT_BRIDGING_METRIC if metric is None else metric
  if resolved_metric not in _METRIC_TO_COLUMN:
    raise ValueError(
        f"Unknown bridging metric {resolved_metric!r}. Expected one of"
        f" {BRIDGING_METRICS}."
    )

  scores_by_candidate = compute_bridging_scores(
      approval_matrix,
      segment_memberships,
      min_segment_size=min_segment_size,
      smoothing=smoothing,
  )

  sorted_segments = sorted(segment_memberships)
  segment_col_names: list[str] = []
  if include_segment_rates:
    core_column_set = frozenset(_CORE_RESULT_COLUMNS)
    for seg_name in sorted_segments:
      col_name = f"{segment_column_prefix}{seg_name}"
      if col_name in core_column_set:
        raise ValueError(
            f"Segment rate column {col_name!r} collides with a core result"
            " column. Use a different segment_column_prefix."
        )
      segment_col_names.append(col_name)

  all_columns = list(_CORE_RESULT_COLUMNS) + segment_col_names
  if not scores_by_candidate:
    return pd.DataFrame(columns=all_columns)

  rows = []
  for score in scores_by_candidate.values():
    row = {
        "candidate": score.candidate,
        "minmax_approval": score.minmax_approval,
        "bottleneck_segment": score.bottleneck_segment,
        "gic_product": score.gic_product,
        "gic_geomean": score.gic_geomean,
        "overall_approval": score.overall_approval,
    }
    if include_segment_rates:
      for seg_name, col_name in zip(sorted_segments, segment_col_names):
        row[col_name] = score.segment_approvals.get(seg_name, float("nan"))
    rows.append(row)

  result_df = pd.DataFrame(rows)
  metric_col = _METRIC_TO_COLUMN[resolved_metric]
  result_df = result_df.sort_values(
      by=[metric_col, "overall_approval", "candidate"],
      ascending=[False, False, True],
      na_position="last",
      kind="stable",
  ).reset_index(drop=True)

  valid_mask = result_df[metric_col].notna()
  if valid_mask.all():
    result_df["bridging_rank"] = result_df.index + 1
  else:
    result_df["bridging_rank"] = pd.Series(
        result_df.index + 1, index=result_df.index, dtype="Int64"
    ).where(valid_mask)

  return result_df[all_columns]
