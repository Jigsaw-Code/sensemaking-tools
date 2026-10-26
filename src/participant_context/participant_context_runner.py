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

"""Adds participant context columns to a survey CSV.

Example:
  export CONTEXT_ENDPOINT_TYPE=clairic CLOUDFLARE_ACCOUNT_ID=... \\
      CONTEXT_MODEL_API_KEY=...
  export DECISION_ENDPOINT_TYPE=systemone DECISION_MODEL=clef
  python3 -m src.participant_context.participant_context_runner \\
      --input_csv processed.csv --output_csv with_context.csv

The input needs "participant_id" and "survey_text" columns. The extraction
model comes from CONTEXT_* (see model_factory.get_extractor) and labels come
from the DECISION_* decision model; without one, only evidence is written.
"""

import argparse
import asyncio
import logging

import pandas as pd

from src import runner_utils
from src.models import decision
from src.models import model_factory
from src.participant_context import participant_context_lib

ID_COL = "participant_id"
TEXT_COL = "survey_text"


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses command-line arguments.

  Args:
    argv: Arguments to parse; None reads sys.argv.

  Returns:
    The parsed arguments.
  """
  parser = argparse.ArgumentParser(
      description="Add participant context columns to a survey CSV."
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help=f"CSV with '{ID_COL}' and '{TEXT_COL}' columns.",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Where to write the result."
  )
  parser.add_argument(
      "--attributes_file",
      help=(
          "JSON list of attributes (name, instructions, labels). Defaults to"
          " one 'role' attribute."
      ),
  )
  runner_utils.add_additional_context_args(
      parser, help_str="Survey context added to both models' instructions."
  )
  return parser.parse_args(argv)


async def main(args: argparse.Namespace) -> pd.DataFrame:
  """Reads the survey, describes participants, and writes the output CSV.

  Args:
    args: Parsed arguments from get_args.

  Returns:
    The output table.

  Raises:
    ValueError: If the input lacks the required columns or no extraction
      model is configured.
  """
  table = pd.read_csv(args.input_csv)
  missing = [col for col in (ID_COL, TEXT_COL) if col not in table.columns]
  if missing:
    raise ValueError(f"{args.input_csv} is missing columns: {missing}")
  extractor = model_factory.get_extractor("CONTEXT")
  if extractor is None:
    raise ValueError(
        "Set CONTEXT_ENDPOINT_TYPE to 'clairic' or 'systemone' to choose an"
        " extraction model."
    )
  attributes = (
      participant_context_lib.load_attributes(args.attributes_file)
      if args.attributes_file
      else participant_context_lib.DEFAULT_ATTRIBUTES
  )
  decision_client = decision.decision_client()
  logging.info(
      "Describing %d participants with %s%s.",
      len(table),
      extractor.model,
      f" and labels from {decision_client.model}" if decision_client else "",
  )
  results = await participant_context_lib.describe_participants(
      extractor,
      decision_client,
      table[TEXT_COL].fillna("").astype(str).tolist(),
      attributes,
      runner_utils.get_additional_context(args),
  )
  columns = participant_context_lib.result_columns(results, attributes)
  output = table.assign(**columns)
  output.to_csv(args.output_csv, index=False)
  logging.info("Wrote %s.", args.output_csv)
  return output


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  asyncio.run(main(get_args()))
