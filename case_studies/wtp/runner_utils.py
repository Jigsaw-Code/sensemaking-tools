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

"""Utilities for processing and formatting topic modeling data for categorization runners."""

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from case_studies.wtp.models import custom_types


def setup_logging(log_level_str: str, output_dir: Optional[str] = None) -> None:
  """Configures logging to both console and multi-level files.

  Args:
      log_level_str: The desired console logging level (e.g., "INFO").
      output_dir: Optional directory to store the .logs folder.
  """
  log_level = getattr(logging, log_level_str.upper(), logging.INFO)

  # Root logger configuration
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)  # Catch all levels; handlers will filter.

  # Remove any existing handlers (like those from basicConfig)
  for handler in logger.handlers[:]:
    logger.removeHandler(handler)

  # Console Handler
  console_handler = logging.StreamHandler()
  console_handler.setLevel(log_level)
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
  console_handler.setFormatter(formatter)
  logger.addHandler(console_handler)

  # File Handlers setup
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  base_log_dir = os.path.join(output_dir, ".logs") if output_dir else ".logs"
  log_dir = os.path.join(base_log_dir, timestamp)
  os.makedirs(log_dir, exist_ok=True)

  levels = {
      "debug": logging.DEBUG,
      "info": logging.INFO,
      "warning": logging.WARNING,
      "error": logging.ERROR,
  }

  for name, level in levels.items():
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

  logging.info(f"Logging initialized. Files are saved in: {log_dir}")


def estimate_tokens(text: str) -> int:
  """Heuristic estimation of tokens."""
  return _estimate_tokens(len(text))


def _estimate_tokens(char_length: int) -> int:
  """Heuristic estimation of tokens."""
  return int(char_length / 4)


def filter_large_statements(
    statements: List[custom_types.Statement],
    token_limit: int = 1000000,
) -> Tuple[List[custom_types.Statement], List[custom_types.Statement]]:
  """Filters out statements that exceed the token limit.

  Args:
      statements: List of statements to filter.
      token_limit: Maximum number of tokens allowed per statement.

  Returns:
      Tuple of (valid_statements, skipped_statements).
  """
  valid_statements = []
  skipped_statements = []

  for s in statements:
    # Calculate total length for token estimation.
    total_len = len(s.text)
    if s.quotes:
      # Add length of each quote + 1 space for each quote
      total_len += sum(len(q.text) + 1 for q in s.quotes)

    if _estimate_tokens(total_len) > token_limit:
      skipped_statements.append(s)
    else:
      valid_statements.append(s)

  return valid_statements, skipped_statements


def get_additional_context(
    args: argparse.Namespace, default_context: Optional[str] = None
) -> Optional[str]:
  """
  Extracts additional context from argparse arguments, handling both string
  and file inputs, and applying a default if neither is provided.
  """
  additional_context = getattr(args, "additional_context", None)
  additional_context_file = getattr(args, "additional_context_file", None)

  if additional_context and additional_context_file:
    raise ValueError(
        "Cannot specify both --additional_context and --additional_context_file"
    )

  if additional_context_file:
    with open(additional_context_file, "r") as f:
      additional_context = f.read().strip()

  if not additional_context and default_context:
    additional_context = default_context

  return additional_context


def add_additional_context_args(
    parser: argparse.ArgumentParser,
    help_str: str = "Optional additional context to be added to the prompt.",
) -> None:
  """
  Adds standard flags for additional context to an argparse parser.

  Args:
      parser: The ArgumentParser instance to add arguments to.
      help_str: The help string for the --additional_context argument.
  """
  parser.add_argument(
      "--additional_context",
      type=str,
      help=help_str,
  )
  parser.add_argument(
      "--additional_context_file",
      type=str,
      help="Path to a file containing additional context.",
  )


def write_dicts_to_csv(
    csv_rows: List[Dict[str, Any]], output_file_path: str
) -> None:
  """Writes a list of dictionaries to a CSV file."""
  if not csv_rows:
    logging.warning("No data to write to CSV.")
    return

  file_path = os.path.expanduser(output_file_path)

  output_dir = os.path.dirname(file_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  all_keys = set()
  for row in csv_rows:
    all_keys.update(row.keys())

  preferred_order = [
      "rid",
      "survey_text",
      "response_text",
      "representative_text_with_brackets",
      "representative_text",
      "topics",
      "topic",
      "opinion",
  ]
  headers = [h for h in preferred_order if h in all_keys]
  remaining_keys = sorted([k for k in all_keys if k not in headers])
  headers.extend(remaining_keys)

  with open(file_path, mode="w", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(csv_rows)
  logging.info(f"CSV file written successfully to {file_path}.")


def concat_topics(topics: Optional[List[custom_types.Topic]]) -> str:
  """Returns topics and opinions concatenated together like

  "Topic1:Opinion1;Topic2:Opinion2"
  Handles up to 2 levels of nesting (topic:opinion).
  """
  pairs_array: List[str] = []

  for topic in topics:
    # Check if it's a NestedTopic AND has non-empty subtopics (opinions) to process
    if isinstance(topic, custom_types.NestedTopic) and topic.subtopics:
      pairs_array.extend(
          f"{topic.name}:{opinion_item.name}"
          for opinion_item in topic.subtopics
      )
    else:
      # Treat as a flat topic if it's a FlatTopic or a NestedTopic with no subtopics
      pairs_array.append(f"{topic.name}")

  return ";".join(pairs_array)


def parse_topics_string(topics_string: str) -> List[custom_types.Topic]:
  """Parse a topics string (e.g., from a CSV) into a list of Topic objects.

  Args:
      topics_string: A string in the format "Topic1:Opinion1;Topic2:Opinion2".

  Returns:
      A list of Topic objects (NestedTopic for topics with opinions, or
      FlatTopic).
  """
  if not topics_string:
    return []

  # parsed_structure will store: topic_name -> [opinion_names]
  parsed_structure: Dict[str, List[str]] = defaultdict(list)

  topic_entries = topics_string.split(";")
  for entry in topic_entries:
    # Only split the string at the first colon.
    parts = entry.strip().split(":", 1)
    topic_name = parts[0]
    if not topic_name:
      continue  # Skip empty entries part before first colon.

    opinions = parsed_structure[topic_name]
    if len(parts) == 2:  # Topic with opinion: "TopicName:OpinionName"
      opinion_name = parts[1]
      opinions.append(opinion_name)

  final_topics: List[custom_types.Topic] = []
  for topic_name, opinions_list in parsed_structure.items():
    if not opinions_list:  # No opinions, it's a FlatTopic.
      final_topics.append(custom_types.FlatTopic(name=topic_name))
    else:
      # This topic has opinions, so it's a NestedTopic with FlatTopic children.
      opinions = [
          custom_types.FlatTopic(name=opinion_name)
          for opinion_name in sorted(set(opinions_list))
      ]  # Sort for consistency.
      final_topics.append(
          custom_types.NestedTopic(name=topic_name, subtopics=opinions)
      )

  # Sort final topics by name for consistent output.
  final_topics.sort(key=lambda t: t.name)
  return final_topics


def generate_and_save_topic_tree(
    topic_tree_data: List[Dict[str, Any]],
    output_file_base: str,
    tree_title: str = "Topic Tree",
) -> None:
  """Generates a topic tree summary, saves it to a TXT file,

  and prints a formatted version to the console.
  """
  # First, let's calculate the counts and prepare for sorting
  for topic in topic_tree_data:
    topic["total_quotes"] = 0
    for opinion in topic["opinions"]:
      count = len(opinion.get("representative_texts", []))
      opinion["count"] = count
      topic["total_quotes"] += count

  # Sort topics by total quotes
  sorted_topics = sorted(
      topic_tree_data, key=lambda t: t["total_quotes"], reverse=True
  )

  output_lines = []
  topic_counter = 1
  for topic in sorted_topics:
    output_lines.append(
        f"{topic_counter}."
        f" {topic['topic_name']} ({topic['total_quotes']} quotes)"
    )
    topic_counter += 1

    # Sort opinions by count
    sorted_opinions = sorted(
        topic["opinions"], key=lambda o: o["count"], reverse=True
    )
    opinion_counter = 1
    for opinion in sorted_opinions:
      output_lines.append(
          f"  {opinion_counter}."
          f" {opinion['opinion_text']} ({opinion['count']} quotes)"
      )
      opinion_counter += 1

  topic_tree_string = "\n".join(output_lines)

  # Calculate and add the total number of unique opinions
  unique_opinions = Counter()
  for topic in topic_tree_data:
    for opinion in topic["opinions"]:
      unique_opinions[opinion["opinion_text"]] += 1
  unique_opinions_string = (
      f"\n\nTotal number of unique opinions: {len(unique_opinions)}"
  )

  # Prepare content for TXT file
  txt_file_content = topic_tree_string + unique_opinions_string

  # Write TXT file
  output_txt_file = f"{output_file_base}.txt"
  with open(output_txt_file, "w", encoding="utf-8") as txtfile:
    txtfile.write(txt_file_content)

  # Print to console
  print(f"\n--- {tree_title} ---")
  print(topic_tree_string)
  print(unique_opinions_string)

  # Log file saving messages
  logging.info(f"Topic tree in text format saved to {output_txt_file}")
