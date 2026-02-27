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
Runs opinion learning on a CSV of statements and quotes.

Sample command:
python opinion_learning_runner.py --input_file input.csv --output_file opinion_learning_output.csv --subject "Freedom and Equality" --vertex_project YOUR_PROJECT --vertex_location global --model_name gemini-2.5-pro --runs 5
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from typing import Dict, List

# Add the 'src' directory to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from src.models import genai_model
from src.models import custom_types
from src.tasks import topic_modeling
from src import runner_utils


def read_csv_to_dicts(input_file_path: str) -> List[Dict[str, str]]:
  """Reads a CSV file into a list of dictionaries."""
  if not input_file_path:
    raise ValueError("Input file path is missing!")

  file_path = os.path.expanduser(input_file_path)
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")

  with open(file_path, mode="r", encoding="utf-8") as csvfile:
    return list(csv.DictReader(csvfile))


def convert_csv_rows_to_statements(
    csv_rows: List[Dict[str, str]],
) -> List[custom_types.Statement]:
  """Converts CSV row dictionaries to Statement Pydantic models."""
  statements: List[custom_types.Statement] = []
  for i, row in enumerate(csv_rows):
    survey_text = row.get("survey_text")
    quote = row.get("quote_with_brackets")
    topic_name = row.get("topic")
    participant_id = row.get("participant_id")

    if not survey_text:
      raise ValueError(f"Row {i+1} is missing 'survey_text'.")
    if not quote:
      raise ValueError(
          f"Row {i+1} is missing 'quote_with_brackets'."
      )
    if not topic_name:
      raise ValueError(f"Row {i+1} is missing 'topic'.")
    if not rid:
      raise ValueError(f"Row {i+1} is missing 'participant_id'.")

    quote = custom_types.Quote(
        id=rid,
        text=quote,
        topic=custom_types.FlatTopic(name=topic_name),
    )
    statement = custom_types.Statement(id=rid, text=survey_text, quotes=[quote])
    statements.append(statement)
  return statements


async def main():
  """Main function to run the opinion learning runner."""
  parser = argparse.ArgumentParser(
      description="Learn opinions from statements and quotes."
  )
  parser.add_argument(
      "-i",
      "--input_file",
      type=str,
      required=True,
      help="The input file name (CSV).",
  )
  parser.add_argument(
      "-o",
      "--output_file",
      type=str,
      required=True,
      help="The output file name (CSV).",
  )
  runner_utils.add_additional_context_args(
      parser,
      help_str="Additional context for the opinion learning.",
  )
  parser.add_argument(
      "-r",
      "--runs",
      type=int,
      default=1,
      help="Number of times to run the opinion learning process.",
  )
  parser.add_argument(
      "--api_key",
      type=str,
      help="The Google AI Studio API Key.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.5-pro",
      help="The name of the Vertex AI model to use. Default: gemini-2.5-pro.",
  )
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level. Default: INFO.",
  )

  args = parser.parse_args()

  logging.basicConfig(
      level=args.log_level.upper(),
      format="%(asctime)s - %(levelname)s - %(message)s",
  )

  def convert_to_flat_if_no_subtopics(
      topic: custom_types.Topic,
  ) -> custom_types.Topic:
    if isinstance(topic, custom_types.NestedTopic):
      if not topic.subtopics:
        return custom_types.FlatTopic(name=topic.name)

      new_subtopics = []
      for subtopic in topic.subtopics:
        new_subtopics.append(convert_to_flat_if_no_subtopics(subtopic))
      topic.subtopics = new_subtopics
    return topic

  csv_rows = read_csv_to_dicts(args.input_file)
  statements = convert_csv_rows_to_statements(csv_rows)

  additional_context = runner_utils.get_additional_context(args)

  genai_model_instance = genai_model.GenaiModel(
      model_name=args.model_name,
      api_key=args.api_key,
  )

  topics = sorted(
      list(set(row.get("topic", "") for row in csv_rows if row.get("topic")))
  )

  output_rows = []
  for i in range(args.runs):
    logging.info(f"Run {i+1}/{args.runs}")
    for topic_name in topics:
      topic = custom_types.FlatTopic(name=topic_name)
      opinions = await topic_modeling.learn_opinions(
          statements=statements,
          model=genai_model_instance,
          topic=topic,
          additional_context=additional_context,
      )
      converted_opinions = convert_to_flat_if_no_subtopics(opinions)
      output_rows.append({
          "Library Runs": json.dumps(converted_opinions.model_dump(), indent=2)
      })

  with open(args.output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Library Runs"])
    writer.writeheader()
    writer.writerows(output_rows)

  logging.info(f"Results written to {args.output_file}")


if __name__ == "__main__":
  asyncio.run(main())
