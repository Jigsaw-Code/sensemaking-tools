import pandas as pd
import argparse
import asyncio
from src.propositions import world_model_util


def is_r1_df_missing_required_column(df: pd.DataFrame) -> str:
  """
  Checks if the DataFrame provided for R1 survey results is missing required
  columns for world model generation.

  Args:
      df (pd.DataFrame): The input pandas DataFrame from R1.

  Returns:
      missing column name or None if all columns are present.
  """
  required_r1_columns = [
      "rid",
      "topic",
      "opinion",
      "representative_text",
  ]
  df_r1_columns_lower = {col.lower() for col in df.columns}
  for col in required_r1_columns:
    if col.lower() not in df_r1_columns_lower:
      return col

  return None


def is_r2_df_missing_required_column(df: pd.DataFrame) -> str:
  """
  Checks if the DataFrame provided for R2 survey results is missing required
  columns for world model generation.

  Args:
      df (pd.DataFrame): The input pandas DataFrame from R2.

  Returns:
      missing column name or None if all columns are present.
  """
  required_r2_columns = ["rid"] + [
      item
      for i in range(2, 7)
      for item in (
          f"question_{i}_topic",
          f"question_{i}_opinion",
          f"question_{i}",
          f"answer_{i}",
      )
  ]
  df_r2_columns_lower = {col.lower() for col in df.columns}
  for col in required_r2_columns:
    if col.lower() not in df_r2_columns_lower:
      return col

  return None


async def main():
  """Main function to run the world model builder."""
  parser = argparse.ArgumentParser(
      description="Validate survey data for world model generation."
  )
  parser.add_argument(
      "--r1_input_file",
      type=str,
      help="The input CSV file for round 1 data.",
  )
  parser.add_argument(
      "--r2_input_file",
      type=str,
      help="The input CSV file for round 2 data.",
  )

  args = parser.parse_args()

  if not args.r1_input_file and not args.r2_input_file:
    parser.error(
        "At least one of --r1_input_file or --r2_input_file is required."
    )

  if args.r1_input_file:
    df_r1 = world_model_util.read_csv_to_dataframe(args.r1_input_file)
    missing_col_r1 = is_r1_df_missing_required_column(df_r1)
    if missing_col_r1:
      raise ValueError(
          f"Round 1 data is missing required column: {missing_col_r1}"
      )
  if args.r2_input_file:
    df_r2 = world_model_util.read_csv_to_dataframe(args.r2_input_file)
    missing_col_r2 = is_r2_df_missing_required_column(df_r2)
    if missing_col_r2:
      raise ValueError(
          f"Round 2 data is missing required column: {missing_col_r2}"
      )

  print("Validation successful!")


if __name__ == "__main__":
  asyncio.run(main())
