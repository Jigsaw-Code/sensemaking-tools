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
This script processes survey data from two CSV files (round 1 and round 2)
to generate a world model of propositions.

Example Usage:
  python3 -m case_studies.wtp.propositions.proposition_generator \
    --prop_count=10 \
    --r1_input_file=/path/to/r1_data.csv \
    --r2_input_file=/path/to/r2_data.csv \
    --output_dir=/path/to/output \
    --gemini_api_key=your-api-key \
    --output_file_name=world_model \
    --save_prompt \
    --sample_print_shard_count=1 \
    --sample_save_dir=/path/to/print \
    --sample_save_file_name=prompt_sample
"""

import argparse
import asyncio
import collections
import csv
import logging
import os
import pickle
import random
import re
import typing

import pandas as pd

from case_studies.wtp.models import genai_model
from case_studies.wtp.propositions import input_csv_validation
from case_studies.wtp.propositions import prompts
from case_studies.wtp.propositions import prompts_util
from case_studies.wtp.propositions import world_model_util
from case_studies.wtp import runner_utils


async def _get_r2_data_by_opinion(
    df: pd.DataFrame,
    opinion: str,
) -> pd.DataFrame:
  """
  Filters and restructures a DataFrame to extract data for a specific opinion.

  Args:
      df: The input DataFrame with ranking data.
      opinion: The opinion string to filter by.

  Returns:
      A new DataFrame with 'opinion' and 'vote' columns for the specified opinion.
      Returns an empty DataFrame if the opinion is not found.
  """
  gallery_nums = []

  # Find the column containing the opinion name to identify the ranking number
  for col in df.columns:
    # Check if the opinion exists in the current column
    try:
      col_data = df[col]
      is_string_and_contains_opinion = (
          pd.api.types.is_string_dtype(col_data)
          and col_data.head(1).str.contains(opinion, case=False).any()
      )
      if is_string_and_contains_opinion:
        # Build the regex pattern dynamically using the prefix
        gallery_match = re.search(r"question_(\d+)_opinion", col)
        if gallery_match:
          gallery_nums.append(gallery_match.group(1))

    except Exception as e:
      logging.error(
          f"Error when comparing column: {col} to opinion: {opinion}: {e}"
      )
      continue

  # If the topic wasn't found, return an empty DataFrame
  if len(gallery_nums) == 0:
    logging.warning(f"Opinion '{opinion}' not found in the R2 DataFrame.")
    return pd.DataFrame(columns=["opinion", "vote"])

  # Filter the DataFrame to get only the rows for the specified opinion
  filtered_columns = ["rid"]
  # Track the answer columns to use to filter the rows down if there is no data.
  answer_columns = []
  for i in gallery_nums:
    filtered_columns.append(f"question_{i}")
    filtered_columns.append(f"answer_{i}")
    if f"answer_{i}_agrees" in df.columns:
      filtered_columns.append(f"answer_{i}_agrees")
    answer_columns.append(f"answer_{i}")

  restructured_data = df[filtered_columns].copy()
  restructured_data.dropna(subset=answer_columns, how="all", inplace=True)

  # Create and return the final DataFrame
  return pd.DataFrame(restructured_data)


async def analyze_and_allocate_by_opinion(
    r1_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    topic_column_name: str,
    opinion_column_name: str,
    num_propositions_to_allocate: int = 5,
    make_every_opinion_same: bool = False,
) -> pd.DataFrame:
  """
  Splits a DataFrame by a specified opinion column, calculates weighted
  allocations for each resulting child DataFrame, and returns a single
  DataFrame containing the opinion, child DataFrame length, calculated
  allocations, and the child DataFrame itself.

  Args:
      r1_df (pd.DataFrame): The input pandas DataFrame.
      r2_df (pd.DataFrame): The input pandas DataFrame for round 2.
      topic_column_name (str): The name of the column from R1 data containing
        the topics.
      opinion_column_name (str): The name of the column from R1 data containing
        the opinions to split by.
      num_propositions_to_allocate (int): The total number of propositions to
        allocate across all opinions, allocated proportionately by child
        DataFrame length.
      make_every_opinion_same (bool): Should give every opinion
        {num_propositions_to_allocate} allocations.

  Returns:
      pd.DataFrame: A DataFrame with the following columns:
        - 'topic': The unique value from the topic_column_name.
        - 'opinion': The unique value from the opinion_column_name.
        - 'r1_df_length': The number of rows in the R1 DataFrame for that
          opinion.
        - 'r2_df_length': The number of rows in the R2 DataFrame for that
          opinion.
        - 'allocations': The number of propositions allocated to this opinion
          based on its length.
        - 'r1_df': The actual pandas DataFrame containing rows for that opinion
          from R1.
        - 'r2_df': The actual pandas DataFrame containing rows for that opinion
          from R2.
  Raises:
      ValueError: If the opinion_column_name is not found in the DataFrame.
  """

  if opinion_column_name not in r1_df.columns:
    raise ValueError(
        f"Column '{opinion_column_name}' not found in the DataFrame."
    )

  unique_opinions = (
      r1_df[opinion_column_name]
      .dropna()
      .loc[lambda s: s.str.lower() != "other"]
      .loc[lambda s: ~s.str.lower().duplicated()]
      .tolist()
  )

  # Split DataFrame and gather initial info
  temp_df_info: typing.List[typing.Dict] = []

  for opinion in unique_opinions:
    r1_filtered_df = r1_df[r1_df[opinion_column_name] == opinion].copy()

    # For a given opinion, find all topics it belongs to in R1 data
    topics_for_opinion = r1_filtered_df[topic_column_name].unique()[0]

    r2_filtered_df = await _get_r2_data_by_opinion(
        r2_df.copy(),
        opinion,
    )

    temp_df_info.append({
        "topic": topics_for_opinion,
        "opinion": opinion,
        "r1_filtered_df": r1_filtered_df,
        "r1_length": len(r1_filtered_df),
        "r2_filtered_df": r2_filtered_df,
        "r2_length": len(r2_filtered_df),
    })

  if not temp_df_info:
    logging.warning("No valid DataFrames found after splitting by opinion.")
    return pd.DataFrame()

  # Prepare for weighted sampling
  population_opinions = [item["opinion"] for item in temp_df_info]
  weights = [item["r1_length"] for item in temp_df_info]

  if sum(weights) == 0:
    logging.warning(
        "All child DataFrames have a length of 0. Cannot perform weighted"
        " sampling."
    )
    # Fallback to uniform sampling if no lengths, or return empty if no examples
    if num_propositions_to_allocate > 0:
      logging.warning(
          "Falling back to uniform random sampling as all weights are zero."
      )
      sampled_opinions = random.choices(
          population=population_opinions, k=num_propositions_to_allocate
      )
    else:
      return pd.DataFrame()
  else:
    sampled_opinions = random.choices(
        population=population_opinions,
        weights=weights,
        k=num_propositions_to_allocate,
    )

  # Tally the allocations
  allocation_counts = collections.Counter(sampled_opinions)

  # Combine all information into the final DataFrame
  final_results_data = []
  for item in temp_df_info:
    topic = item["topic"]
    opinion = item["opinion"]
    r1_filtered_df = item["r1_filtered_df"]
    r1_length = item["r1_length"]
    r2_filtered_df = item["r2_filtered_df"]
    r2_length = item["r2_length"]

    allocations = (
        allocation_counts.get(opinion, 0)
        if not make_every_opinion_same
        else num_propositions_to_allocate
    )

    final_results_data.append({
        "topic": topic,
        "opinion": opinion,
        "r1_df_length": r1_length,
        "r2_df_length": r2_length,
        "allocations": allocations,
        "r1_df": r1_filtered_df,
        "r2_df": r2_filtered_df,
    })

  result_df = pd.DataFrame(final_results_data)

  # Sort for better readability
  result_df = result_df.sort_values(by="topic", ascending=False)

  logging.info(
      "Generated %d total examples, distributed by opinion length.",
      num_propositions_to_allocate,
  )
  logging.info(
      "\nCombined Allocation Summary (Sorted by Child DataFrame Length"
      " Descending):"
  )
  logging.info(
      "\n"
      + result_df[
          ["opinion", "r1_df_length", "r2_df_length", "allocations"]
      ].to_string(index=False)
  )

  # Verify total allocations
  total_allocated = result_df["allocations"].sum()
  logging.info("\nTotal examples allocated: %d", total_allocated)

  return result_df


def _save_sample_prompt_text(
    preamble: str,
    r1_prompt: str,
    r2_prompt: str,
    print_save_dir: str,
    print_save_file_name: str,
    r1_char_length: int = 4000,
    r2_char_length: int = 3000,
):
  """
  Save a sample of the prompt text to a file. This method will print the
  full preamble but it will shorten the r1 and r2 prompts.

  Args:
      preamble (str): The preamble text.
      r1_prompt (str): The R1 prompt text.
      r2_prompt (str): The R2 prompt text.
      footer (str): The footer text.
      print_save_dir (str): The directory to save the print file.
      print_save_file_name (str): The file name for the print file.
      r1_char_length (int): The maximum number of characters to print for R1.
      r2_char_length (int): The maximum number of characters to print for R2.
  """
  print_file_contents = "=" * 50
  print_file_contents += "\n" + preamble
  print_file_contents += "=" * 20
  print_file_contents += "\n" + "\tR1"
  print_file_contents += "\n" + "=" * 20
  print_file_contents += (
      "\n" + r1_prompt[:r1_char_length] + "..." + r1_prompt[-11:]
  )
  if r2_prompt and len(r2_prompt) > 0:
    print_file_contents += "=" * 20
    print_file_contents += "\n" + "\tR2"
    print_file_contents += "\n" + "=" * 20
    print_file_contents += (
        "\n" + r2_prompt[:r2_char_length] + "..." + r2_prompt[-11:]
    )
  print_file_contents += "=" * 50

  if print_save_dir:
    os.makedirs(print_save_dir, exist_ok=True)
  with open(
      os.path.join(
          print_save_dir,
          print_save_file_name + ".txt",
      ),
      "w",
  ) as f:
    f.write(print_file_contents)


async def main():
  """Main function to run the world model builder."""
  parser = argparse.ArgumentParser(
      description="Build a world model from survey data."
  )
  parser.add_argument(
      "--prop_count",
      type=int,
      default=5,
      help="How many propositions to generate per topic.",
  )
  parser.add_argument(
      "--r1_input_file",
      type=str,
      required=True,
      help="The input CSV file for round 1 data.",
  )
  parser.add_argument(
      "--r2_input_file",
      type=str,
      required=False,
      help="The input CSV file for round 2 data.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="The output directory for the generated files.",
  )
  parser.add_argument(
      "--gemini_api_key",
      type=str,
      required=True,
      help="The Gemini API key.",
  )
  parser.add_argument(
      "--output_file_name",
      type=str,
      default="world_model",
      help=(
          "The output file name (without file format) for the generated world"
          " model."
      ),
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="gemini-2.5-pro",
      help="The name of the generative model to use.",
  )
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
      help="Set the logging level.",
  )
  parser.add_argument(
      "--save_prompt",
      action="store_true",
      help="Save a sample of the prompt text.",
  )
  parser.add_argument(
      "--reasoning",
      action="store_true",
      help="Save a sample of the prompt text.",
  )
  parser.add_argument(
      "--sample_save_dir",
      type=str,
      default="case_studies/wtp/testdata/prompts",
      help="The output directory for the generated print file.",
  )
  parser.add_argument(
      "--sample_save_file_name",
      type=str,
      default="prompt_sample",
      help="The file name for the prompt print file.",
  )
  parser.add_argument(
      "--sample_print_shard_count",
      type=int,
      default=1,
      help="How many topics to print a sample for.",
  )
  parser.add_argument(
      "--include_opinion",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Whether to include the verbatem opinion text in the final"
          " set of statements generated."
      ),
  )
  runner_utils.add_additional_context_args(
      parser,
      help_str="Optional additional context to be added to the prompt.",
  )

  args = parser.parse_args()

  logging.basicConfig(
      level=args.log_level.upper(),
      format="%(asctime)s - %(levelname)s - %(message)s",
  )

  additional_context = runner_utils.get_additional_context(args)

  logging.info("Starting world model builder.")
  logging.info("Reading input files.")

  # Validate R1 data
  df_r1 = world_model_util.read_csv_to_dataframe(args.r1_input_file)
  logging.info("Validating Round 1 data.")
  missing_col_r1 = input_csv_validation.is_r1_df_missing_required_column(df_r1)
  if missing_col_r1:
    raise ValueError(
        f"Round 1 data is missing required column: {missing_col_r1}"
    )

  # Validate R2 data
  if args.r2_input_file:
    df_r2 = world_model_util.read_csv_to_dataframe(args.r2_input_file)
    logging.info("Validating Round 2 data.")
    missing_col_r2 = input_csv_validation.is_r2_df_missing_required_column(
        df_r2
    )
    if missing_col_r2:
      raise ValueError(
          f"Round 2 data is missing required column: {missing_col_r2}"
      )
  else:
    df_r2 = pd.DataFrame()

  logging.info(
      "Analyzing the DataFrame structure and allocating proposisiton count by"
      " opinion."
  )

  split_dfs = await analyze_and_allocate_by_opinion(
      df_r1,
      df_r2,
      "topic",
      "opinion",
      args.prop_count,
      True,
  )

  print_shard_count = args.sample_print_shard_count
  turn_on_reasoning = args.reasoning

  if args.include_opinion and args.prop_count == 1:
    logging.info(
        "Optimization triggered: prop_count=1 and include_opinion=True."
        " Returning opinions verbatim without model execution."
    )
    llm_response_rows = []
    for _, row in split_dfs.iterrows():
      opinion = row["opinion"]
      # Create propositions DataFrame
      props_data = [{"proposition": opinion}]
      if turn_on_reasoning:
        props_data[0][
            "reasoning"
        ] = "Original opinion returned verbatim due to optimization."
      props_df = pd.DataFrame(props_data)

      llm_response_rows.append({
          "opinion": opinion,
          "propositions": props_df,
          "total_token_used": 0,
          "prompt_token_count": 0,
          "candidates_token_count": 0,
          "tool_use_prompt_token_count": 0,
          "thoughts_token_count": 0,
      })
    llm_response = pd.DataFrame(llm_response_rows)
    llm_response_stats = pd.DataFrame([{"combined_tokens": 0}])

  else:
    model = genai_model.GenaiModel(
        api_key=args.gemini_api_key, model_name=args.model_name
    )

    all_prompts = []
    for index, row in split_dfs.iterrows():

      preamble = prompts.generate_preamble_prompt(
          opinion_list=split_dfs["opinion"].tolist(),
          additional_context=additional_context,
      )
      instructions_prompt = prompts.generate_instructions_prompt(
          number_of_propositions=args.prop_count,
          reasoning=turn_on_reasoning,
          include_opinion=args.include_opinion,
      )
      r1_prompt = prompts.generate_r1_prompt_string(
          df=row["r1_df"],
          user_id_column_name="rid",
          topic_column_name="topic",
          opinion_column_name="opinion",
          representative_text_column_name="representative_text",
      )
      r2_prompt = prompts.generate_r2_prompt_string(
          df=row["r2_df"],
          include_non_gov_sections=False,
      )
      prompt = preamble + instructions_prompt + r1_prompt + r2_prompt

      schema_items = {"type": "STRING"}
      if turn_on_reasoning:
        schema_items = {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING"},
                "reasoning": {"type": "STRING"},
            },
        }

      all_prompts.append({
          "topic": row["topic"],
          "opinion": row["opinion"],
          "prompt": prompt,
          "allocations": row["allocations"],
          "stats": {
              "combined_tokens": model.calculate_token_count_needed(
                  prompt=prompt, run_name=row["opinion"][:20]
              )
          },
          "system_prompt": (
              "You are assisting us in forming consensus opinions on an"
              " important question."
          ),
          "response_mime_type": "application/json",
          "response_schema": {
              "type": "ARRAY",
              "items": schema_items,
          },
      })

      # Log the prompts for human review.
      if args.save_prompt and index < print_shard_count + 1:
        file_name = (
            args.sample_save_file_name
            if print_shard_count == 1
            else f"{args.sample_save_file_name}_T{index}"
        )
        _save_sample_prompt_text(
            preamble + instructions_prompt,
            r1_prompt,
            r2_prompt,
            args.sample_save_dir,
            file_name,
        )

    # Start the LLM call process.
    (
        llm_response,
        llm_response_stats,
        _,
        _,
    ) = await model.process_prompts_concurrently(
        all_prompts,
        prompts_util.parse_proposition_response_json_reasoning
        if turn_on_reasoning
        else prompts_util.parse_proposition_response_json,
        len(all_prompts),
    )

  # Ensure the opinion text itself is included in the proposition list.
  if args.include_opinion:
    for index, row in llm_response.iterrows():
      propositions_df = row["propositions"]
      opinion = row["opinion"]
      if opinion not in propositions_df["proposition"].values:
        new_row_data = {"proposition": opinion}
        if turn_on_reasoning:
          new_row_data["reasoning"] = (
              "Original opinion added because it was not present in the LLM"
              " response."
          )
        new_row = pd.DataFrame([new_row_data])
        llm_response.at[index, "propositions"] = pd.concat(
            [propositions_df, new_row], ignore_index=True
        )

  logging.info(
      f"Total tokens used across {len(split_dfs)} calls:"
      f" {llm_response_stats['combined_tokens'].sum()}"
  )
  logging.info(f"\nStats: \n{llm_response_stats.to_string(index=False)}")

  merged_df = pd.merge(
      split_dfs,
      llm_response[[
          "opinion",
          "propositions",
          "total_token_used",
          "prompt_token_count",
          "candidates_token_count",
          "tool_use_prompt_token_count",
          "thoughts_token_count",
      ]],
      on="opinion",
      how="left",
  )

  output_pkl_file = os.path.join(
      args.output_dir, args.output_file_name + ".pkl"
  )
  propositions_csv_file = os.path.join(
      args.output_dir, "propositions_" + args.output_file_name + ".csv"
  )
  logging.info(
      f"Saving world model to {output_pkl_file} and propositions to"
      f" {propositions_csv_file}"
  )
  world_model_util.save_dataframe_to_pickle(merged_df, output_pkl_file)
  world_model_util.save_propositions_as_csv(
      df=merged_df,
      file_path=propositions_csv_file,
      reasoning=turn_on_reasoning,
      has_eval_data=False,
  )

  logging.info("Finished processing data.")


if __name__ == "__main__":
  asyncio.run(main())
