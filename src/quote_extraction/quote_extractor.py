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
#
# This quote extracts representative quotes from long statements in a CSV file.
# Example usage:
# export GOOGLE_API_KEY="your_actual_key_here" && \
# python3 -m src.quote_extraction.quote_extractor \
#     --input_csv input.csv \
#     --output_csv output.csv \
#     --output_dir your/output/dir \
#     --additional_context "This is a response from a survey asking \
#      what people think about ____ issues in the US" \
#     --model_name gemini-3-pro-preview


import argparse
import asyncio
import csv
import logging
from typing import Any, Dict, List

from src.models.genai_model import GenaiModel
from src.models.custom_types import Statement
from src.input_parsing_lib import parse_topics_string
from src import runner_utils

from .quote_extraction_lib import extract_quotes_from_text

SURVEY_TEXT_COL = "survey_text"
TOPICS_COL = "topics"
TOPIC_COL = "topic"
QUOTE_COL = "quote"


async def main(args):

  logging.basicConfig(
      level=logging.INFO,
      format="%(message)s",  # print log messages only without any extra info
  )

  additional_context = runner_utils.get_additional_context(args)

  model = GenaiModel(model_name=args.model_name)

  statements_to_process: List[Statement] = []
  original_rows: Dict[int, Dict[str, Any]] = {}
  input_fieldnames = []
  try:
    with open(args.input_csv, "r", newline="", encoding="utf-8") as infile:
      reader = csv.DictReader(infile)
      input_fieldnames = list(reader.fieldnames) if reader.fieldnames else []
      if SURVEY_TEXT_COL not in reader.fieldnames:
        logging.error(
            f"Input CSV '{args.input_csv}' must have a '{SURVEY_TEXT_COL}'"
            f" header. Found: {reader.fieldnames}"
        )
        raise ValueError(f"Missing '{SURVEY_TEXT_COL}' header in input CSV.")
      if TOPICS_COL not in reader.fieldnames:
        logging.error(
            f"Input CSV '{args.input_csv}' must have a 'topics' header. Found:"
            f" {reader.fieldnames}"
        )
        raise ValueError("Missing 'topics' header in input CSV.")

      for i, row in enumerate(reader):
        survey_text = row.get(SURVEY_TEXT_COL, "").strip()
        topics_str = row.get(TOPICS_COL, "")

        if not survey_text:
          logging.warning(f"Skipping row due to empty text: {row}")
          continue

        individual_topics = parse_topics_string(topics_str)

        if not individual_topics:
          logging.warning(
              f"No topics found for text: '{survey_text[:50]}...'. Skipping"
              " this text for topic-specific extraction."
          )
          continue

        statement_id = i
        statements_to_process.append(
            Statement(
                id=str(statement_id),
                text=survey_text,
                topics=individual_topics,
            )
        )
        original_rows[statement_id] = row

  except FileNotFoundError:
    logging.error(f"Input CSV file not found: {args.input_csv}")
    return
  except Exception as e:
    logging.error(f"Error reading CSV file '{args.input_csv}': {e}")
    return

  if not statements_to_process:
    logging.info("No statements to process from the input CSV.")
    return

  logging.info(
      f"Starting processing of {len(statements_to_process)} statements."
  )

  updated_statements = await extract_quotes_from_text(
      statements=statements_to_process,
      model=model,
      additional_context=additional_context,
      output_dir=args.output_dir,
  )

  processed_results = []
  for statement in updated_statements:
    if statement.quotes:
      for quote in statement.quotes:
        original_row = original_rows[int(statement.id)]
        output_row = original_row.copy()
        output_row[QUOTE_COL] = quote.text
        output_row[TOPIC_COL] = quote.topic.name
        processed_results.append(output_row)

  try:
    with open(args.output_csv, "w", newline="", encoding="utf-8") as outfile:
      # Ensure all original columns are present, plus the new quote
      output_fieldnames = list(input_fieldnames)
      # Rename 'topics' -> 'topic' column
      if TOPICS_COL in output_fieldnames:
        output_fieldnames.remove(TOPICS_COL)
      if TOPIC_COL not in output_fieldnames:
        output_fieldnames.append(TOPIC_COL)
      if QUOTE_COL not in output_fieldnames:
        output_fieldnames.append(QUOTE_COL)

      writer = csv.DictWriter(
          outfile, fieldnames=output_fieldnames, extrasaction="ignore"
      )
      writer.writeheader()
      writer.writerows(processed_results)
    logging.info(
        f"Successfully processed {len(processed_results)} topic-specific"
        f" quotes. Output written to {args.output_csv}"
    )
  except Exception as e:
    logging.error(f"Error writing output CSV file '{args.output_csv}': {e}")


def get_args():
  parser = argparse.ArgumentParser(
      description="Extract representative quotes from statements in a CSV file."
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help=(
          f"Path to the input CSV file with a '{SURVEY_TEXT_COL}' and"
          f" '{TOPICS_COL}' header."
      ),
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  runner_utils.add_additional_context_args(
      parser,
      help_str=(
          "Additional context to add to LLM prompts. (e.g., 'This is a response"
          " from a survey asking what people think about immigration issues in"
          " the US.')."
      ),
  )
  parser.add_argument(
      "-o",
      "--output_dir",
      type=str,
      required=True,
      help="The output directory for checkpointing and categorized files.",
  )
  parser.add_argument(
      "--model_name",
      required=True,
      help="Gemini model name (e.g., gemini-2.5-pro).",
  )

  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
