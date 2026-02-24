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
Runs proposition simplification based on a CSV of propositions and prompts.

The input CSV file must have a specific format:
- The first data row must contain the prompt texts in columns that start with
  "Prompt". For example, "Prompt 1", "Prompt 2", etc. The "Original" column
  for this row should be empty.
- Subsequent rows must contain the original propositions to be simplified in a
  column named "Original".
- The script will generate a simplified proposition for each original
  proposition using each of the prompts.
- The output CSV file will have the same structure as the input, with the
  simplified propositions filling the respective prompt columns for each
  proposition row.

Example input CSV (`input.csv`):
Original,Prompt 1,Prompt 2
,"Make this simpler","Make this more formal"
"the cat sat on the mat",,
"the dog chased the ball",,

Example output CSV (`output.csv`):
Original,Prompt 1,Prompt 2
,"Make this simpler","Make this more formal"
"the cat sat on the mat","The cat was on the mat.","The feline was situated upon the rug."
"the dog chased the ball","The dog ran after the ball.","The canine pursued the sphere."


Sample command:
python -m case_studies.wtp.proposition_simplification_runner \
    --input_csv case_studies/wtp/proposition_simplification.csv \
    --output_csv case_studies/wtp/proposition_simplification_output.csv \
    --project YOUR_PROJECT \
    --location global \
    --model_name gemini-2.5-pro
"""

import argparse
import asyncio
import csv
import logging
from typing import Any, Dict, List, Union

from case_studies.wtp.models import genai_model


async def simplify_one_proposition(
    task_item: Dict[str, Union[str, int]], model: genai_model.GenaiModel
) -> Dict[str, Union[str, int]]:
  """
  Processes a single proposition with a given prompt.

  Args:
    task_item: A dictionary containing the prompt key, prompt text, the original
      proposition, and its original row index.
    model: The VertexModel instance to use for generation.

  Returns:
    A dictionary containing the prompt key, original row index, and the
    simplified proposition.
  """
  prompt_key = str(task_item["prompt_key"])
  prompt_text = str(task_item["prompt_text"])
  proposition = str(task_item["proposition"])
  original_row_index = int(task_item["original_row_index"])

  full_prompt = (
      f"{prompt_text}\n\n"
      "Here is the proposition to rewrite:\n"
      f"<proposition>\n{proposition}\n</proposition>\n\n"
      "Provide only the rewritten proposition as your output. "
      "Do not include any other text or markdown."
  )

  logging.info(
      f"Simplifying proposition {original_row_index} for {prompt_key}..."
  )
  response = await model.call_gemini(prompt=full_prompt, run_name=prompt_key)
  simplified_proposition = response.get("text", "") if response else ""
  if not simplified_proposition:
    logging.error(
        f"Failed to generate text for {prompt_key} row {original_row_index}"
    )

  return {
      "prompt_key": prompt_key,
      "original_row_index": original_row_index,
      "simplified_proposition": simplified_proposition.strip(),
  }


async def main(args):
  """Main function to run the proposition simplification experiment."""
  logging.basicConfig(
      level=args.log_level.upper(),
      format="%(asctime)s - %(levelname)s - %(message)s",
  )

  model = genai_model.GenaiModel(
      model_name=args.model_name,
      api_key=args.api_key,
  )

  try:
    with open(args.input_csv, "r", newline="", encoding="utf-8") as infile:
      reader = csv.DictReader(infile)
      # fieldnames is None if the file is empty.
      if reader.fieldnames is None:
        logging.error(f"Input CSV file '{args.input_csv}' is empty or invalid.")
        return
      fieldnames = reader.fieldnames
      all_rows = list(reader)
  except FileNotFoundError:
    logging.error(f"Input CSV file not found: {args.input_csv}")
    return
  except Exception as e:
    logging.error(f"Error reading CSV file '{args.input_csv}': {e}")
    return

  if not all_rows:
    logging.error("CSV file has no data rows.")
    return

  # The first data row contains the prompts.
  prompt_row = all_rows[0]
  prompt_keys = [f for f in fieldnames if f.startswith("Prompt")]
  prompts = {key: prompt_row[key] for key in prompt_keys if prompt_row.get(key)}

  # The rest of the rows contain the original propositions.
  proposition_rows = all_rows[1:]
  if not proposition_rows:
    logging.info("No propositions found to process.")
    return

  original_propositions = [row["Original"] for row in proposition_rows]

  tasks_to_process = []
  for p_key, p_text in prompts.items():
    for i, original_prop in enumerate(original_propositions):
      tasks_to_process.append({
          "prompt_key": p_key,
          "prompt_text": p_text,
          "proposition": original_prop,
          "original_row_index": i,  # Index in the proposition_rows list
      })

  if not tasks_to_process:
    logging.info("No proposition-prompt pairs to process.")
    return

  logging.info(
      f"Starting processing of {len(tasks_to_process)} proposition-prompt pairs"
      f" with parallelism limit of {args.parallelism}."
  )

  semaphore = asyncio.Semaphore(args.parallelism)

  async def sem_task(item):
    async with semaphore:
      return await simplify_one_proposition(item, model)

  processed_results: List[Dict[str, Any]] = await asyncio.gather(
      *[sem_task(item) for item in tasks_to_process]
  )

  # Create a copy to store the results
  output_rows = [row.copy() for row in all_rows]

  for result in processed_results:
    prompt_key = result["prompt_key"]
    original_row_index = result["original_row_index"]
    simplified_prop = result["simplified_proposition"]

    # Proposition rows start at index 1 in the `output_rows` list
    # (index 0 is the prompt row).
    output_rows[original_row_index + 1][prompt_key] = simplified_prop

  try:
    with open(args.output_csv, "w", newline="", encoding="utf-8") as outfile:
      writer = csv.DictWriter(outfile, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(output_rows)
    logging.info(f"Successfully processed. Output written to {args.output_csv}")
  except Exception as e:
    logging.error(f"Error writing output CSV file '{args.output_csv}': {e}")


def get_args():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Simplify propositions from a CSV file."
  )
  parser.add_argument(
      "--input_csv",
      required=True,
      help="Path to the input CSV file.",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--api_key",
      required=False,
      help=(
          "Google AI Studio API Key. If not provided, uses GOOGLE_API_KEY env"
          " var."
      ),
  )
  parser.add_argument(
      "--parallelism",
      type=int,
      default=100,
      help="Number of concurrent LLM calls (default: 100).",
  )
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level (default: INFO).",
  )

  return parser.parse_args()


if __name__ == "__main__":
  asyncio.run(main(get_args()))
