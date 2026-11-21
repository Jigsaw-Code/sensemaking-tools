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

r"""Runs readability-asserted simplification on a CSV of propositions.

Example input CSV (`input.csv`):
  proposition
  Establish an interagency infrastructure modernization committee.

Example output CSV (`output.csv`) preserves all input columns and adds:
  simplification, pre_simplified_statement, post_simplified_statement,
  pre_fkgl, post_fkgl, readability_compliant, simplification_attempts

Sample command:
  python -m src.proposition_simplification_runner \
      --input_csv input.csv \
      --output_csv output.csv \
      --model_name gemini-3.5-flash
"""

import argparse
import asyncio
from collections import abc
import logging
import math
import os

import pandas as pd

from src import runner_utils
from src.models import base_model
from src.models import model_factory
from src.propositions import proposal_simplifier

_DEFAULT_MODEL_NAME = "gemini-3.5-flash"
_DEFAULT_PROPOSITION_COLUMN = "proposition"


def _positive_int_arg(value: str) -> int:
  """Argparse `type=` validating an integer >= 1."""
  try:
    parsed = int(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    ) from exc
  if parsed < 1:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be an integer >= 1."
    )
  return parsed


def _positive_float_arg(value: str) -> float:
  """Argparse `type=` validating a float > 0."""
  try:
    parsed = float(value)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be a number > 0."
    ) from exc
  if not math.isfinite(parsed) or parsed <= 0:
    raise argparse.ArgumentTypeError(
        f"invalid value {value!r}: must be a finite number > 0."
    )
  return parsed


async def run_simplification(
    args: argparse.Namespace,
    *,
    model: base_model.BaseModel | None = None,
) -> pd.DataFrame:
  """Loads the input CSV, simplifies propositions, and writes the output CSV.

  Args:
    args: Parsed command-line arguments from `get_args`.
    model: Optional BaseModel override (primarily for testing). When None, a
      model is constructed via `model_factory.get_model`.

  Returns:
    The simplified output DataFrame that was written to `args.output_csv`.
  """
  additional_context = runner_utils.get_additional_context(args)
  input_df = pd.read_csv(args.input_csv, keep_default_na=False, na_values=[""])

  active_model = model
  if active_model is None:
    active_model = model_factory.get_model(
        model_name=args.model_name,
        gemini_api_key=args.gemini_api_key,
    )

  output_df = await proposal_simplifier.simplify_proposals_dataframe(
      active_model,
      input_df,
      statement_column=args.proposition_column,
      max_fkgl=args.max_fkgl,
      max_sentences=args.max_sentences,
      max_words=args.max_words,
      max_attempts=args.max_attempts,
      max_concurrent_calls=args.max_concurrent_calls,
      additional_context=additional_context,
      system_prompt=args.prompt,
  )

  output_dir = os.path.dirname(args.output_csv)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  output_df.to_csv(args.output_csv, index=False)
  logging.info(
      "Wrote %d simplified rows to %s.",
      output_df.shape[0],
      args.output_csv,
  )
  return output_df


async def main(args: argparse.Namespace) -> None:
  """Entry point for running proposition simplification from parsed CLI args."""
  await run_simplification(args)


def get_args(argv: abc.Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments for proposition simplification."""
  parser = argparse.ArgumentParser(
      description=(
          "Simplify propositions from a CSV file with deterministic"
          " readability assertions."
      )
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help="Path to the input CSV file.",
  )
  parser.add_argument(
      "--output_csv",
      required=True,
      help="Path to save the output CSV file.",
  )
  parser.add_argument(
      "--gemini_api_key",
      required=False,
      help=(
          "Google AI Studio API Key. If not provided, uses the GEMINI_API_KEY"
          " environment variable."
      ),
  )
  parser.add_argument(
      "--prompt",
      type=str,
      default=None,
      help=(
          "Optional custom system prompt overriding the default plain-English"
          " simplification prompt."
      ),
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default=_DEFAULT_MODEL_NAME,
      help=f"The name of the AI model to use. Default: {_DEFAULT_MODEL_NAME}.",
  )
  parser.add_argument(
      "--proposition_column",
      type=str,
      default=_DEFAULT_PROPOSITION_COLUMN,
      help=(
          "The name of the proposition column to simplify. Default:"
          f" {_DEFAULT_PROPOSITION_COLUMN}."
      ),
  )
  parser.add_argument(
      "--max_fkgl",
      type=_positive_float_arg,
      default=proposal_simplifier.DEFAULT_MAX_FKGL,
      help=(
          "Maximum allowable Flesch-Kincaid Grade Level. Default:"
          f" {proposal_simplifier.DEFAULT_MAX_FKGL}."
      ),
  )
  parser.add_argument(
      "--max_sentences",
      type=_positive_int_arg,
      default=proposal_simplifier.DEFAULT_MAX_SENTENCES,
      help=(
          "Maximum allowable sentence count. Default:"
          f" {proposal_simplifier.DEFAULT_MAX_SENTENCES}."
      ),
  )
  parser.add_argument(
      "--max_words",
      type=_positive_int_arg,
      default=None,
      help=(
          "Optional maximum allowable word count. Unset by default so"
          " substantive details are not truncated arbitrarily."
      ),
  )
  parser.add_argument(
      "--max_attempts",
      type=_positive_int_arg,
      default=proposal_simplifier.DEFAULT_MAX_ATTEMPTS,
      help=(
          "Maximum rewrite attempts per proposition. Default:"
          f" {proposal_simplifier.DEFAULT_MAX_ATTEMPTS}."
      ),
  )
  parser.add_argument(
      "-c",
      "--max_concurrent_calls",
      type=runner_utils.max_concurrent_calls_arg,
      default=proposal_simplifier.DEFAULT_MAX_CONCURRENT_CALLS,
      help=(
          "Maximum concurrent simplification requests. Default:"
          f" {proposal_simplifier.DEFAULT_MAX_CONCURRENT_CALLS}."
      ),
  )
  runner_utils.add_additional_context_args(parser)
  return parser.parse_args(argv)


if __name__ == "__main__":
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
  )
  asyncio.run(main(get_args()))
