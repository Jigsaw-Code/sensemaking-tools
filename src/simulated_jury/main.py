r"""Command-line utility for running simulated jury analyses.

Runs a simulated jury for a set of participants and statements and outputs
aggregated approval, ranking, PAV, and optional bridging metrics.

Example Usage:
  python3 -m src.simulated_jury.main \
    --participants_csv s1_r2_v3_processed.csv \
    --statements_csv statements.csv \
    --output_csv results.csv \
    --gemini_api_key "$GEMINI_API_KEY"
"""

import argparse
import asyncio
from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import logging
from typing import Any

import pandas as pd

from src import runner_utils
from src.models import base_model
from src.models import genai_model
from src.models import model_factory
from src.models import openai_compatible_model
from src.simulated_jury import sampling_utils
from src.simulated_jury import simulated_jury
from src.social_choice import bridging
from src.social_choice import proportional_approval_voting
from src.social_choice import schulze

_VOTING_MODES: tuple[str, ...] = ("approval", "ranking", "schulze_pav")
_DEFAULT_APPROVAL_BATCH_SIZE = 15

# Rate-valued bridging columns formatted as percentages when --percent is set.
# gic_product remains numeric because products across many segments can be
# very small.
_BRIDGING_PERCENT_COLUMNS: tuple[str, ...] = ("minmax_approval", "gic_geomean")


def _append_bridging_metrics(
    statements_df: pd.DataFrame,
    approval_matrix: pd.DataFrame,
    segment_memberships: Mapping[str, Collection[Any]],
    *,
    statement_column: str,
    group_by: str | None,
    bridging_metric: str | None,
    min_segment_size: int,
) -> pd.DataFrame:
  """Computes bridging metrics from approval_matrix and merges into statements_df."""
  output_cols = list(bridging.BRIDGING_OUTPUT_COLUMNS)
  existing_cols = [col for col in output_cols if col in statements_df.columns]
  if existing_cols:
    logging.warning(
        "Existing bridging columns %s found in statements CSV. Overwriting.",
        existing_cols,
    )
    statements_df = statements_df.drop(columns=existing_cols)

  if not group_by:
    bridging_df = bridging.rank_by_bridging(
        approval_matrix,
        segment_memberships,
        metric=bridging_metric,
        min_segment_size=min_segment_size,
    )
    return statements_df.merge(
        bridging_df.set_index("candidate")[output_cols],
        left_on=statement_column,
        right_index=True,
        how="left",
    )

  grouped_frames: list[pd.DataFrame] = []
  for group_name, group_df in statements_df.groupby(group_by, sort=False):
    group_statements = list(
        dict.fromkeys(
            stmt
            for stmt in group_df[statement_column].tolist()
            if stmt in approval_matrix.columns
        )
    )
    if not group_statements:
      continue
    group_bridging = bridging.rank_by_bridging(
        approval_matrix[group_statements],
        segment_memberships,
        metric=bridging_metric,
        min_segment_size=min_segment_size,
    )
    group_bridging[group_by] = group_name
    grouped_frames.append(group_bridging[[group_by, "candidate", *output_cols]])

  if not grouped_frames:
    return statements_df

  combined = pd.concat(grouped_frames, ignore_index=True)
  return statements_df.merge(
      combined.set_index([group_by, "candidate"])[output_cols],
      left_on=[group_by, statement_column],
      right_index=True,
      how="left",
  )


@dataclasses.dataclass(frozen=True, slots=True)
class SimulatedJuryConfig:
  """Immutable configuration for a simulated jury analysis run.

  Holds pure algorithmic and formatting settings (no file paths, model names,
  API keys, or environment lookups) so `run_analysis` can be invoked
  deterministically from tests, notebooks, or orchestration pipelines with an
  explicitly injected `BaseModel`.

  Attributes:
    statement_column: Column containing statements, or None to auto-detect
      ``"statement"`` or ``"proposition"``.
    jury_size: Optional participant sample size (fraction in ``(0, 1)`` or
      absolute count ``> 1``).
    approval_batch_size: Number of statements to process per approval batch.
    voting_mode: One of ``"approval"``, ``"ranking"``, or ``"schulze_pav"``.
    group_by: Optional statement grouping column for ranking, PAV, and bridging.
    max_ranking_candidates: Maximum number of statements to rank per group.
    max_concurrent_calls: Optional concurrency limit forwarded to the model.
    approval_scale: Approval voting scale, or None to use the library default.
    additional_context: Optional trusted prompt framing context.
    bridging_config: Optional normalized bridging configuration.
    percent: Whether to format rate columns as percentage strings.
    true_agree_rate_column: Optional ground-truth rate column for error
      calculation.
  """

  statement_column: str | None = None
  jury_size: float | None = None
  approval_batch_size: int = _DEFAULT_APPROVAL_BATCH_SIZE
  voting_mode: str = "approval"
  group_by: str | None = None
  max_ranking_candidates: int = simulated_jury.MAX_STATEMENTS_FOR_RANKING
  max_concurrent_calls: int | None = None
  approval_scale: simulated_jury.ApprovalScale | None = None
  additional_context: str | None = None
  bridging_config: bridging.BridgingConfig | None = None
  percent: bool = False
  true_agree_rate_column: str | None = None

  def __post_init__(self) -> None:
    if self.voting_mode not in _VOTING_MODES:
      raise ValueError(
          f"Unsupported voting_mode {self.voting_mode!r}; expected one of"
          f" {_VOTING_MODES}."
      )
    if self.approval_batch_size < 1:
      raise ValueError(
          f"approval_batch_size must be >= 1, got {self.approval_batch_size}."
      )
    if self.max_ranking_candidates < 1:
      raise ValueError(
          "max_ranking_candidates must be >= 1, got"
          f" {self.max_ranking_candidates}."
      )
    if self.max_concurrent_calls is not None:
      base_model.validate_max_concurrent_calls(self.max_concurrent_calls)


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line argument parser for simulated jury runs."""
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
      "--gemini_api_key",
      type=str,
      help="The Gemini API key.",
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
      default=_DEFAULT_APPROVAL_BATCH_SIZE,
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
      choices=list(_VOTING_MODES),
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
      "-c",
      "--max_concurrent_calls",
      type=runner_utils.max_concurrent_calls_arg,
      default=None,
      help=(
          "Maximum number of concurrent LLM API calls"
          f" (default: {genai_model.MAX_CONCURRENT_CALLS} for Gemini,"
          f" {openai_compatible_model.DEFAULT_MAX_CONCURRENT_CALLS} for open"
          " model adapter)."
      ),
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
  runner_utils.add_additional_context_args(
      parser,
      help_str=(
          "Optional additional context added to simulated jury voting prompts."
          " Treated as privileged prompt framing; provide only trusted author"
          " or researcher content and do not expose to untrusted public input."
      ),
  )
  bridging.add_bridging_args(parser)
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
  return parser


def config_from_args(args: argparse.Namespace) -> SimulatedJuryConfig:
  """Normalizes parsed CLI arguments into an immutable `SimulatedJuryConfig`."""
  approval_batch_size = (
      args.batch_size
      if args.batch_size is not None
      else args.approval_batch_size
  )
  additional_context = runner_utils.get_additional_context(args)
  if additional_context is not None:
    additional_context = additional_context.strip() or None
  approval_scale = (
      simulated_jury.ApprovalScale(args.approval_scale)
      if args.approval_scale
      else None
  )
  return SimulatedJuryConfig(
      statement_column=args.statement_column,
      jury_size=args.jury_size,
      approval_batch_size=approval_batch_size,
      voting_mode=args.voting_mode,
      group_by=args.group_by,
      max_ranking_candidates=args.max_ranking_candidates,
      max_concurrent_calls=args.max_concurrent_calls,
      approval_scale=approval_scale,
      additional_context=additional_context,
      bridging_config=bridging.bridging_config_from_args(args),
      percent=args.percent,
      true_agree_rate_column=args.true_agree_rate_column,
  )


def _resolve_statement_column(
    statements_df: pd.DataFrame, statement_column: str | None
) -> str:
  """Returns the validated statement column name from `statements_df`."""
  if statement_column:
    if statement_column not in statements_df.columns:
      raise ValueError(
          f"Statement column {statement_column!r} not found in statements CSV."
      )
    return statement_column
  if "statement" in statements_df.columns:
    return "statement"
  if "proposition" in statements_df.columns:
    return "proposition"
  raise ValueError(
      "Could not find a 'statement' or 'proposition' column. Please specify"
      " --statement_column."
  )


async def run_analysis(
    participants_df: pd.DataFrame,
    statements_df: pd.DataFrame,
    *,
    model: base_model.BaseModel,
    config: SimulatedJuryConfig,
) -> pd.DataFrame:
  """Executes simulated jury analysis on in-memory DataFrames.

  Does not mutate caller-supplied DataFrames, read environment variables, or
  touch the filesystem.

  Args:
    participants_df: DataFrame of participant records.
    statements_df: DataFrame of candidate statements to evaluate.
    model: Explicit `BaseModel` instance used for all jury LLM calls.
    config: Immutable algorithmic and formatting configuration.

  Returns:
    Aggregated results DataFrame with approval, ranking, PAV, and/or bridging
    columns appended.

  Raises:
    ValueError: If required columns, jury size, or bridging segments are
      invalid.
    RuntimeError: If approval voting produces no results when required.
  """
  working_statements_df = statements_df.copy()

  if (
      config.true_agree_rate_column
      and config.true_agree_rate_column not in working_statements_df.columns
  ):
    raise ValueError(
        f"Column {config.true_agree_rate_column!r} not found in statements CSV."
    )

  sampled_participants_df = participants_df
  if config.jury_size is not None:
    sampled_participants_df = sampling_utils.apply_jury_size_sampling(
        participants_df, config.jury_size, verbose=True
    )

  segment_memberships = None
  if config.bridging_config is not None:
    if config.voting_mode == "ranking":
      logging.warning(
          "Bridging metrics require approval voting; ignoring segment flags in"
          " 'ranking' mode."
      )
    else:
      segment_memberships = bridging.extract_segment_memberships(
          sampled_participants_df,
          segment_column=config.bridging_config.segment_column,
          segment_columns=config.bridging_config.segment_columns,
      )

  statement_column = _resolve_statement_column(
      working_statements_df, config.statement_column
  )
  statements = working_statements_df[statement_column].tolist()

  if config.group_by and config.group_by not in working_statements_df.columns:
    raise ValueError(
        f"Grouping column {config.group_by!r} not found in statements CSV."
    )

  # Drop any pre-existing agree_rate column to prevent duplicate columns on
  # merge.
  if "agree_rate" in working_statements_df.columns:
    logging.warning(
        "Existing 'agree_rate' column found in statements CSV. Overwriting."
    )
    working_statements_df = working_statements_df.drop(columns=["agree_rate"])

  approval_scale = (
      config.approval_scale
      if config.approval_scale is not None
      else simulated_jury.ApprovalScale.AGREE_DISAGREE
  )
  approval_matrix = pd.DataFrame()

  if config.voting_mode in ("approval", "schulze_pav"):
    logging.info(
        "--- Running APPROVAL simulation on %d statements ---",
        len(statements),
    )
    approval_results_df, _ = await simulated_jury.run_simulated_jury(
        participants_df=sampled_participants_df,
        statements=statements,
        voting_mode=simulated_jury.VotingMode.APPROVAL,
        model=model,
        batch_size=config.approval_batch_size,
        approval_scale=approval_scale,
        max_concurrent_calls=config.max_concurrent_calls,
        additional_context=config.additional_context,
    )

    if approval_results_df.empty:
      raise RuntimeError("No results from the simulated approval jury.")

    approval_matrix = simulated_jury.build_approval_matrix(
        approval_results_df, approval_scale=approval_scale
    )
    agree_rate = approval_matrix.mean().rename("agree_rate")
    working_statements_df = working_statements_df.merge(
        agree_rate, left_on=statement_column, right_index=True, how="left"
    )
    if segment_memberships is not None and config.bridging_config is not None:
      working_statements_df = _append_bridging_metrics(
          working_statements_df,
          approval_matrix,
          segment_memberships,
          statement_column=statement_column,
          group_by=config.group_by,
          bridging_metric=config.bridging_config.metric,
          min_segment_size=config.bridging_config.min_segment_size,
      )

  if config.voting_mode in ("ranking", "schulze_pav"):
    grouped_statements = (
        working_statements_df.groupby(config.group_by)
        if config.group_by
        else [("All Statements", working_statements_df)]
    )

    for group_name, group_df in grouped_statements:
      group_statements = group_df[statement_column].tolist()

      if len(group_statements) > config.max_ranking_candidates:
        logging.warning(
            "Group '%s' has %d statements, exceeding max_ranking_candidates"
            " (%d). Skipping ranking.",
            group_name,
            len(group_statements),
            config.max_ranking_candidates,
        )
        continue

      logging.info(
          "--- Running RANK simulation for group: %s (%d statements) ---",
          group_name,
          len(group_statements),
      )
      jury_results_df, _ = await simulated_jury.run_simulated_jury(
          participants_df=sampled_participants_df,
          statements=group_statements,
          voting_mode=simulated_jury.VotingMode.RANK,
          model=model,
          topic_name=str(group_name),
          max_concurrent_calls=config.max_concurrent_calls,
          additional_context=config.additional_context,
      )

      if jury_results_df.empty:
        logging.warning("No ranking results for group: %s.", group_name)
        continue

      jury_preferences = [
          res["ranking"]
          for res in jury_results_df["result"]
          if res and res.get("ranking")
      ]
      if not jury_preferences:
        logging.warning("No valid rankings for group: %s.", group_name)
        continue

      schulze_ranking = schulze.get_schulze_ranking(jury_preferences)
      final_ranking = schulze_ranking.get("top_propositions", [])

      for rank_idx, statement in enumerate(final_ranking):
        working_statements_df.loc[
            working_statements_df[statement_column] == statement, "schulze_rank"
        ] = (rank_idx + 1)

      if config.voting_mode == "schulze_pav":
        if approval_matrix.empty:
          raise RuntimeError("Approval matrix is empty; cannot compute PAV.")

        group_approval_matrix = approval_matrix[group_statements]
        pav_slate = proportional_approval_voting.run_schulze_pav_selection(
            ranked_choice_results=jury_preferences,
            approval_matrix=group_approval_matrix,
            k=len(group_statements),
        )
        for rank_idx, statement in enumerate(pav_slate):
          working_statements_df.loc[
              working_statements_df[statement_column] == statement, "pav_rank"
          ] = (rank_idx + 1)

  if config.true_agree_rate_column:
    true_rate = working_statements_df[config.true_agree_rate_column]
    if true_rate.dtype == "object":
      true_rate = true_rate.str.rstrip("%").astype("float") / 100.0
    working_statements_df["error"] = (
        working_statements_df["agree_rate"] - true_rate
    )

  if config.percent and "agree_rate" in working_statements_df.columns:
    working_statements_df["agree_rate"] = working_statements_df[
        "agree_rate"
    ].apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else x)
    if "error" in working_statements_df.columns:
      working_statements_df["error"] = working_statements_df["error"].apply(
          lambda x: f"{x * 100:.1f}%" if pd.notna(x) else x
      )
    for col in _BRIDGING_PERCENT_COLUMNS:
      if col in working_statements_df.columns:
        working_statements_df[col] = working_statements_df[col].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else x
        )

  return working_statements_df


async def run_from_csv(
    participants_csv: str,
    statements_csv: str,
    output_csv: str,
    *,
    model: base_model.BaseModel,
    config: SimulatedJuryConfig,
) -> pd.DataFrame:
  """Loads CSV inputs preserving literal strings, runs analysis, writes CSV."""
  participants_df = pd.read_csv(
      participants_csv, keep_default_na=False, na_values=[""]
  )
  statements_df = pd.read_csv(
      statements_csv, keep_default_na=False, na_values=[""]
  )
  results_df = await run_analysis(
      participants_df,
      statements_df,
      model=model,
      config=config,
  )
  results_df.to_csv(output_csv, index=False)
  logging.info("Results saved to %s.", output_csv)
  return results_df


def main(argv: Sequence[str] | None = None) -> None:
  """Main entry point for the script."""
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
  )
  args = build_arg_parser().parse_args(argv)
  config = config_from_args(args)
  model = model_factory.get_model(
      model_name=args.model_name,
      gemini_api_key=args.gemini_api_key,
  )
  asyncio.run(
      run_from_csv(
          args.participants_csv,
          args.statements_csv,
          args.output_csv,
          model=model,
          config=config,
      )
  )


if __name__ == "__main__":
  main()
