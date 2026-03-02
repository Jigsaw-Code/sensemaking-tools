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
Learns and assigns topics and opinions to a CSV of statements.

The input CSV must contain "survey_text" and "participant_id" fields.
The output CSV will contain all input fields plus new "topic" and "opinion" fields.
The additional_context_file is optional but can be used to clarify goals and context.

Sample Usage:
python -m src.categorization_runner \
    --output_dir ~/categorization_outputs \
    --input_file ~/input.csv \
    --model_name gemini-3-pro-preview \
    --additional_context_file ~/additional_context.md
"""

import argparse
import asyncio
import collections
import csv
import logging
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Union, cast
from src.models import genai_model
from src import runner_utils, sensemaker
from src.models import custom_types
import pandas as pd

# Define a type for the rows read from CSV, expecting specific keys.
StatementCsvRow = Dict[str, str]
# Override the default CSV field size limit to handle larger files.
csv.field_size_limit(1000000)


def _filter_csv_columns(
    input_file: str, output_file: str, columns_to_keep: List[str]
):
  """Filters a CSV file to keep only specified columns."""
  try:
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
      reader = csv.DictReader(infile)

      # Find which of the desired columns are present in the input file
      present_columns = [
          col for col in columns_to_keep if col in reader.fieldnames
      ]

      if not present_columns:
        logging.warning(
            "None of the specified columns were found in the input file."
        )
        return

      with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=present_columns)
        writer.writeheader()

        for row in reader:
          # Create a new dictionary with only the desired columns
          filtered_row = {col: row[col] for col in present_columns}
          writer.writerow(filtered_row)

      logging.info(
          f"Successfully created '{output_file}' with columns:"
          f" {', '.join(present_columns)}"
      )

  except FileNotFoundError:
    logging.error(f"Error: The file '{input_file}' was not found.")
  except Exception as e:
    logging.error(f"An error occurred: {e}")


def _read_csv_to_dicts(input_file_path: str) -> List[Dict[str, str]]:
  """Reads a CSV file into a list of dictionaries."""
  if not input_file_path:
    raise ValueError("Input file path is missing!")

  file_path = os.path.expanduser(input_file_path)
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")

  with open(file_path, mode="r", encoding="utf-8") as csvfile:
    # Read the header row first using a regular reader
    reader = csv.reader(csvfile)
    header = next(reader)
    # Convert all header names to lowercase
    lowercase_header = [h.lower() for h in header]
    reader = csv.DictReader(csvfile, fieldnames=lowercase_header)
    return list(reader)


def _convert_csv_rows_to_statements(
    csv_rows: List[StatementCsvRow],
) -> List[custom_types.Statement]:
  """Converts CSV row dictionaries to Statement Pydantic models."""
  statements_map: Dict[str, custom_types.Statement] = {}

  for i, row in enumerate(csv_rows):
    statement_id = row.get("participant_id")
    statement_text = row.get("survey_text")
    csv_topics_string = row.get("topics")
    quote_text = row.get("quote_with_brackets")
    quote_topic_name = row.get("topic")
    quote_id = row.get("quote_id")

    if not statement_id:
      raise ValueError(f"Row {i+1} is missing 'participant_id'.")
    if not statement_text:
      raise ValueError(
          f"Row {i+1} (ID: {statement_id}) is missing 'survey_text'."
      )

    parsed_topics_from_csv: Optional[List[custom_types.Topic]] = None
    if csv_topics_string:
      try:
        parsed_topics_from_csv = runner_utils.parse_topics_string(
            csv_topics_string
        )
      except Exception as e:
        raise ValueError(
            f"Failed to parse topics string '{csv_topics_string}' for statement"
            f" ID {statement_id}: {e}"
        )

    if statement_id not in statements_map:
      statements_map[statement_id] = custom_types.Statement(
          id=statement_id,
          text=statement_text,
          topics=parsed_topics_from_csv if parsed_topics_from_csv else [],
          quotes=[],
      )
    elif parsed_topics_from_csv:
      existing_statement = statements_map[statement_id]
      if existing_statement.topics is None:
        existing_statement.topics = []

      existing_topic_names = {t.name for t in existing_statement.topics}
      for new_topic in parsed_topics_from_csv:
        if new_topic.name not in existing_topic_names:
          existing_statement.topics.append(new_topic)
          existing_topic_names.add(new_topic.name)

    if quote_text and quote_topic_name:
      if not quote_id:
        raise ValueError(
            f"Row {i+1} (ID: {statement_id}) is missing 'quote_id' for a quote."
        )
      quote_topic = custom_types.FlatTopic(name=quote_topic_name)
      quote = custom_types.Quote(
          id=quote_id, text=quote_text, topic=quote_topic
      )

      statement = statements_map[statement_id]
      statement.quotes.append(quote)

      # Also add the topic to the statement's topic list
      if statement.topics is None:
        statement.topics = []

      existing_topic_names = {t.name for t in statement.topics}
      if quote_topic.name not in existing_topic_names:
        statement.topics.append(quote_topic)

  return list(statements_map.values())


def _set_topics_on_csv_rows(
    original_csv_rows: List[StatementCsvRow],
    categorized_statements: Iterable[custom_types.Statement],
) -> List[StatementCsvRow]:
  """Prepares CSV output with one row per quote."""
  categorized_statements_map: Dict[str, custom_types.Statement] = {
      s.id: s for s in categorized_statements
  }
  output_csv_rows: List[StatementCsvRow] = []

  original_rows_map: Dict[str, StatementCsvRow] = {}
  for row in original_csv_rows:
    statement_id = row.get("participant_id")
    if statement_id and statement_id not in original_rows_map:
      original_rows_map[statement_id] = row

  logging.info(
      "Formatting output CSV for opinions (one row per quote-opinion pair)."
  )
  for (
      statement_id,
      categorized_statement,
  ) in categorized_statements_map.items():
    if not categorized_statement.quotes:
      logging.error(
          f"Statement ID {categorized_statement.id} has no quotes, skipping"
          " output for this statement."
      )
      raise ValueError("Quotes cannot be empty")

    base_row_data = original_rows_map.get(statement_id, {})
    new_row = base_row_data.copy()
    new_row["participant_id"] = categorized_statement.id
    new_row["survey_text"] = categorized_statement.text

    for quote in categorized_statement.quotes:
      topic_name = quote.topic.name
      opinions = getattr(quote.topic, "subtopics", [])

      for opinion in opinions:
        new_row_with_opinion = new_row.copy()
        new_row_with_opinion["quote_with_brackets"] = quote.text
        new_row_with_opinion["quote"] = re.sub(
            r"[\[\]]", "", quote.text
        )
        new_row_with_opinion["topic"] = topic_name
        new_row_with_opinion["opinion"] = opinion.name
        output_csv_rows.append(new_row_with_opinion)

  return output_csv_rows


def _get_topics_from_arg(
    comma_separated_topics: str,
) -> List[custom_types.Topic]:
  """Converts a comma-separated string of topic names into a list of FlatTopic."""
  topics: List[custom_types.Topic] = []
  if comma_separated_topics:
    for topic_name in comma_separated_topics.split(","):
      topic_name = topic_name.strip()
      if topic_name:
        topics.append(custom_types.FlatTopic(name=topic_name))
  return topics


def _get_topics_and_opinions_from_csv(csv_path: str):
  return_topics = []
  df = pd.read_csv(csv_path)
  for topic_str, group_df in df.groupby("topic"):
    new_topic = custom_types.NestedTopic(name=topic_str)
    for opinion_str in group_df["opinion"].unique():
      new_topic.subtopics.append(custom_types.FlatTopic(name=opinion_str))
    return_topics.append(new_topic)
  return return_topics


def _drop_other(rows):
  """"Drops both Other topic and opinion rows."""
  rows = [row for row in rows if row.get("topic") != "Other"]
  return [row for row in rows if row.get("opinion") != "Other"]


def _process_and_print_topic_tree(
    output_csv_rows: List[Dict[str, Any]], output_file_base: str
) -> None:
  """Processes the output CSV data to generate a topic tree, saves it to a TXT file,

  and prints a formatted version to the console.
  """
  topics = collections.defaultdict(
      lambda: collections.defaultdict(
          lambda: {"count": 0, "quotes": []}
      )
  )
  for row in output_csv_rows:
    topic = row.get("topic", "").strip()
    opinion = row.get("opinion", "").strip()
    quote = row.get("quote", "").strip()

    if topic and opinion:
      topics[topic][opinion]["count"] += 1
      topics[topic][opinion]["quotes"].append(quote)

  # Prepare data for topic tree generation
  topic_tree_data = []
  for topic, opinions in topics.items():
    opinions_list = []
    for opinion, opinion_data in opinions.items():
      opinion_entry = {
          "opinion_text": opinion,
          "quotes": opinion_data["quotes"],
      }
      opinions_list.append(opinion_entry)
    topic_data = {"topic_name": topic, "opinions": opinions_list}
    topic_tree_data.append(topic_data)

  runner_utils.generate_and_save_topic_tree(
      topic_tree_data, f"{output_file_base}_topic_tree"
  )


async def main():
  """Main function to run the categorization runner."""
  parser = argparse.ArgumentParser(
      description="Categorize statements using Sensemaker."
  )
  parser.add_argument(
      "-o",
      "--output_dir",
      type=str,
      required=True,
      help="The output directory for categorized files.",
  )
  parser.add_argument(
      "-i",
      "--input_file",
      type=str,
      required=True,
      help="The input file name (CSV).",
  )
  parser.add_argument(
      "-t",
      "--topics",
      type=str,
      help="Optional comma-separated list of top-level topics.",
  )
  parser.add_argument(
      "-x",
      "--topic_and_opinion_csv",
      type=str,
      help="Optional csv containing topics and opinions.",
  )
  runner_utils.add_additional_context_args(
      parser,
      help_str="Additional context for the categorization.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.5-pro",
      help="The name of the AI model to use. Default: gemini-2.5-pro.",
  )
  parser.add_argument(
      "-f",
      "--force_rerun",
      action="store_true",
      help=(
          "Force rerun of categorization, ignoring existing topics in the input"
          " file."
      ),
  )
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level.",
  )
  parser.add_argument(
      "--skip_autoraters",
      action="store_true",
      help="If set, skip autorater evaluations as part of categorization.",
  )

  args = parser.parse_args()

  runner_utils.setup_logging(args.log_level, args.output_dir)

  original_csv_rows = _read_csv_to_dicts(args.input_file)
  statements_to_process = _convert_csv_rows_to_statements(original_csv_rows)

  # Filter out statements that are too long to process.
  # In an unlikely event that a single statement is larger than the token limit,
  # we skip it and write the skipped rows to a separate CSV file.
  statements_to_process, skipped_statements = (
      runner_utils.filter_large_statements(statements_to_process)
  )

  if skipped_statements:
    skipped_count = len(skipped_statements)
    logging.error(
        f"Skipped {skipped_count} statements because they exceeded the token"
        " limit."
    )
    skipped_csv_path = os.path.join(args.output_dir, "skipped_rows.csv")

    skipped_rows = []
    skipped_ids = {s.id for s in skipped_statements}
    for row in original_csv_rows:
      if row.get("participant_id") in skipped_ids:
        skipped_rows.append(row)

    runner_utils.write_dicts_to_csv(skipped_rows, skipped_csv_path)
    logging.info(f"Skipped rows written to {skipped_csv_path}")
    return

  if args.force_rerun:
    for statement_item in statements_to_process:
      statement_item.topics = []
      statement_item.quotes = []

  genai_llm = genai_model.GenaiModel(
      model_name=args.model_name,
  )

  sensemaker_instance = sensemaker.Sensemaker(
      genai_model=genai_llm,
  )

  cli_topics = None
  if args.topics:
    cli_topics = _get_topics_from_arg(args.topics)
  elif args.topic_and_opinion_csv:
    cli_topics = _get_topics_and_opinions_from_csv(args.topic_and_opinion_csv)

  guiding_topics: Optional[List[custom_types.Topic]] = None
  if cli_topics:
    guiding_topics = cli_topics
    logging.info(
        "Using topics from CLI for guidance:"
        f" {[t.name for t in guiding_topics]}"
    )
  else:
    unique_topic_names_from_csv = set()
    for statement_item in statements_to_process:
      if statement_item.topics:
        for topic_obj in statement_item.topics:
          unique_topic_names_from_csv.add(topic_obj.name)
      if statement_item.quotes:
        for quote_obj in statement_item.quotes:
          unique_topic_names_from_csv.add(quote_obj.topic.name)

    if unique_topic_names_from_csv:
      guiding_topics = [
          custom_types.FlatTopic(name=name)
          for name in sorted(list(unique_topic_names_from_csv))
      ]
      logging.info(
          "Using unique top-level topics derived from input CSV for guidance:"
          f" {[t.name for t in guiding_topics]}"
      )
    else:
      logging.info(
          "No topics from CLI and no topics found in input CSV. Sensemaker will"
          " learn topics from scratch."
      )

  additional_context = runner_utils.get_additional_context(args)

  logging.debug("Performing categorization.")
  categorized_statements = await sensemaker_instance.categorize_statements(
      statements=statements_to_process,
      topics=guiding_topics,
      additional_context=additional_context,
      original_csv_rows=original_csv_rows,
      output_dir=args.output_dir,
      run_autoraters=not args.skip_autoraters,
  )

  output_csv_rows = _set_topics_on_csv_rows(
      original_csv_rows, categorized_statements
  )

  filtered_columns = [
      "participant_id",
      "survey_text",
      "quote",
      "topic",
      "opinion",
  ]

  # Write version of data with "Other" topics and opinions
  categorized_csv_path = os.path.join(args.output_dir, "categorized_with_other.csv")
  runner_utils.write_dicts_to_csv(output_csv_rows, categorized_csv_path)
  output_csv_path = os.path.join(args.output_dir, "categorized_with_other_filtered.csv")
  _filter_csv_columns(categorized_csv_path, output_csv_path, filtered_columns)
  output_file_base = os.path.join(args.output_dir, "categorized_with_other")
  _process_and_print_topic_tree(output_csv_rows, output_file_base)

  # Create another version without "Other" topics and opinions
  csv_rows_without_other = _drop_other(output_csv_rows)
  categorized_csv_path = os.path.join(args.output_dir, "categorized_without_other.csv")
  runner_utils.write_dicts_to_csv(csv_rows_without_other, categorized_csv_path)
  output_csv_path = os.path.join(args.output_dir, "categorized_without_other_filtered.csv")
  _filter_csv_columns(categorized_csv_path, output_csv_path, filtered_columns)


if __name__ == "__main__":
  start_time = time.time()
  asyncio.run(main())
  end_time = time.time()
  logging.info(f"Categorization completed in {end_time - start_time} seconds.")
