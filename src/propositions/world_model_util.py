import csv
import logging
import os
import pickle
import pandas as pd


def read_csv_to_dataframe(file_path: str) -> pd.DataFrame:
  """Reads a CSV file into a pandas DataFrame."""
  if not file_path:
    raise ValueError("Input file path is missing!")

  file_path = os.path.expanduser(file_path)
  if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input file not found: {file_path}")

  return pd.read_csv(file_path)


def save_dataframe_to_pickle(df: pd.DataFrame, file_path: str):
  """Saves a pandas DataFrame to a pickle file."""
  if df.empty:
    logging.warning("DataFrame is empty. No file will be saved.")
    return

  file_path = os.path.expanduser(file_path)
  output_dir = os.path.dirname(file_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  with open(file_path, "wb") as f:
    pickle.dump(df, f)

  logging.info(f"DataFrame saved to {file_path}")


def save_propositions_as_csv(
    df: pd.DataFrame,
    file_path: str,
    reasoning: bool = False,
    has_eval_data: bool = False,
):
  """
  Saves proposition list DataFrame to a csv file to make it easier for human
  review.

  Args:
      df: The input DataFrame with proposition data.
      file_path: path of the file to be saved as csv.
      reasoning: Bool flag indicating the proposition data includes reasoning.
      has_eval_data: Bool flag indicating the proposition data includes evals.
  """
  if df.empty:
    logging.warning(
        "World model DataFrame is empty. No proposition file will be saved."
    )
    return

  file_path = os.path.expanduser(file_path)
  output_dir = os.path.dirname(file_path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  df_copy = df.copy()
  df_copy.sort_values(by="topic", ascending=False, inplace=True)
  output_rows_columns = ["topic", "opinion", "proposition"]
  if reasoning:
    output_rows_columns.append("reasoning")
  if has_eval_data:
    output_rows_columns.extend(["topic_score", "opinion_score"])

  output_rows = [output_rows_columns]
  for _, row in df_copy.iterrows():
    if "propositions" in row and isinstance(row["propositions"], pd.DataFrame):
      for _, prow in row["propositions"].iterrows():
        row_to_append = [
            row["topic"],
            row["opinion"],
            prow["proposition"],
        ]
        if reasoning:
          row_to_append.extend([prow.get("reasoning", "")])
        if has_eval_data:
          row_to_append.extend([
              prow["topic_score"],
              prow["opinion_score"],
          ])
        output_rows.append(row_to_append)
    else:
      row_to_append = [
          row["topic"],
          row["opinion"],
          "No propositions found for this opinion.",
      ]
      if reasoning:
        row_to_append.extend([""])
      if has_eval_data:
        row_to_append.extend(["", ""])
      output_rows.append(row_to_append)

  with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(output_rows)

  logging.info(f"DataFrame saved to {file_path}")
