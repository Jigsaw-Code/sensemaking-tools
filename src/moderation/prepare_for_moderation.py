# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reads a CSV with a "response_text" column and calls the Perspective API
to get toxicity, severe_toxicity, and profanity scores.

Example Usage:
  python3 -m src.prepare_for_moderation \
    --input_csv /path/to/data.csv \
    --output_csv /path/to/data_with_scores.csv \
    --data_type ROUND_1 \
    --api_key "$API_KEY"
"""

import argparse
import collections
import os
import re
import sys
from typing import Any, Callable

from google.cloud import dlp_v2
from src.evals.eval_metrics import INPUT_EVAL_METRICS
from src.get_gemini_scores_lib import ContentScorer
from src.get_perspective_scores_lib import init_client
from src.get_perspective_scores_lib import score_text
from src.qualtrics.process_qualtrics_output import DataType
from src.qualtrics.process_qualtrics_output import DURATION
from src.qualtrics.process_qualtrics_output import END_QUESTION_TAG
from src.qualtrics.process_qualtrics_output import END_RESPONSE_TAG
from src.qualtrics.process_qualtrics_output import RESPONSE_TEXT
from src.qualtrics.process_qualtrics_output import START_QUESTION_TAG
from src.qualtrics.process_qualtrics_output import START_RESPONSE_TAG
from src.qualtrics.process_qualtrics_output import SURVEY_TEXT
import pandas as pd


# The text column is renamed 'response' by the eval library we use.
_INPUT_EVAL_TEXT_COL = "response"
ORIGINAL_COLUMNS_TO_USE = [RESPONSE_TEXT, DURATION]

# Output column names
_TOXICITY_SCORE = "Toxicity Score (of the worst response)"
_SEVERE_TOXICITY_SCORE = "Severe Toxicity Score (of the worst response)"
_PROFANITY_SCORE = "Profanity Score (of the worst response)"
_SPAM_SCORE = "Spam Score"
_TOO_FAST_SCORE = "Too Fast Score"
_MAX_SCORE = "Max Score"
_DLP_FINDINGS = "DLP"


def get_csv(csv_path: str, cols_to_import: list[str] | None) -> pd.DataFrame:
  try:
    if cols_to_import:
      df = pd.read_csv(csv_path, usecols=cols_to_import)
    else:
      df = pd.read_csv(csv_path)
    print(f"Successfully read {len(df)} rows from {csv_path}.")
  except FileNotFoundError:
    print(f"Error: Input file not found at {csv_path}", file=sys.stderr)
    sys.exit(1)
  except Exception as e:
    print(f"Error reading CSV file: {e}", file=sys.stderr)
    sys.exit(1)
  return df


def split_round_1_text(text: str) -> list[str]:
  if (
      START_QUESTION_TAG not in text
      or END_QUESTION_TAG not in text
      or START_RESPONSE_TAG not in text
      or END_RESPONSE_TAG not in text
  ):
    raise ValueError(
        f"Expected text to be in the form {START_QUESTION_TAG} Question 1?"
        f" {END_QUESTION_TAG} {START_RESPONSE_TAG} Answer 1 {END_RESPONSE_TAG}"
    )
  pattern = re.compile(
      rf"{START_QUESTION_TAG}(.*?){END_QUESTION_TAG}\s*{START_RESPONSE_TAG}(.*?){END_RESPONSE_TAG}"
  )
  return [answer for _, answer in pattern.findall(text)]


def split_round_2_text(text: str) -> list[str]:
  """Splits round 2 response text into a list of answers."""
  answers = []
  # Responses are in the format "Response X: answer" and separated by double newlines.
  parts = text.split("\n\n")
  for part in parts:
    if ":\n" in part:
      answers.append(part.split(":\n", 1)[1])
  return answers


def get_max_scores(
    client: Any,
    text: str,
    attributes: list[str],
    text_splitter: Callable[[str], list[str]],
) -> dict[str, float]:
  """Calculates the max perspective score for a text block.

  Splits text using the provided splitter, gets a score for each part, and
  returns the maximum score for each attribute. If splitting fails, scores the
  entire text.
  """
  try:
    answers = text_splitter(text)
    if not answers:
      # If there are no answers, score the entire text.
      return score_text(client, text, attributes)

    scores = [score_text(client, answer, attributes) for answer in answers]

    max_scores = {}
    for attr in attributes:
      max_scores[attr] = (
          max(s.get(attr, 0.0) for s in scores) if scores else 0.0
      )
    return max_scores

  except (ValueError, TypeError):
    # If text is not in the expected format or not a string, score the whole text.
    return score_text(client, text, attributes)


def get_dlp_scores(
    dlp_client: dlp_v2.DlpServiceClient,
    text: str,
    text_splitter: Callable[[str], list[str]],
) -> dict[str, str]:
  """Calculates DLP scores.

  Splits text using the provided splitter, gets a score for each part, and
  returns the maximum score for each attribute. If splitting fails, scores the
  entire text.
  """
  answers = text_splitter(text) or text

  result = dlp_client.inspect_content(
      request={
          "parent": "projects/conversation-ai-experiments/locations/global",
          "inspect_config": {
              "info_types": [
                  {"name": "PERSON_NAME"},
                  {"name": "DOD_ID_NUMBER"},
                  {"name": "US_ADOPTION_TAXPAYER_IDENTIFICATION_NUMBER"},
                  {"name": "US_SOCIAL_SECURITY_NUMBER"},
                  {"name": "US_INDIVIDUAL_TAXPAYER_IDENTIFICATION_NUMBER"},
                  {"name": "US_DEA_NUMBER"},
                  {"name": "US_PASSPORT"},
                  {"name": "US_HEALTHCARE_NPI"},
                  {"name": "US_DRIVERS_LICENSE_NUMBER"},
                  {"name": "US_PREPARER_TAXPAYER_IDENTIFICATION_NUMBER"},
                  {"name": "US_VEHICLE_IDENTIFICATION_NUMBER"},
                  {"name": "US_MEDICARE_BENEFICIARY_ID_NUMBER"},
                  {"name": "FINANCIAL_ID"},
                  {"name": "STORAGE_SIGNED_POLICY_DOCUMENT"},
                  {"name": "EMAIL_ADDRESS"},
                  {"name": "SSL_CERTIFICATE"},
                  {"name": "AZURE_AUTH_TOKEN"},
                  {"name": "STREET_ADDRESS"},
                  {"name": "GCP_CREDENTIALS"},
                  {"name": "VEHICLE_IDENTIFICATION_NUMBER"},
                  {"name": "AUTH_TOKEN"},
                  {"name": "AWS_CREDENTIALS"},
                  {"name": "ADVERTISING_ID"},
                  {"name": "OAUTH_CLIENT_SECRET"},
                  {"name": "MAC_ADDRESS_UNIVERSAL"},
                  {"name": "HTTP_COOKIE"},
                  {"name": "MAC_ADDRESS"},
                  {"name": "ENCRYPTION_KEY"},
                  {"name": "TINK_KEYSET"},
                  {"name": "IMSI_ID"},
                  {"name": "SECURITY_DATA"},
                  {"name": "GOVERNMENT_ID"},
                  {"name": "IMEI_HARDWARE_ID"},
                  {"name": "JSON_WEB_TOKEN"},
                  {"name": "CREDIT_CARD_TRACK_NUMBER"},
                  {"name": "MAC_ADDRESS_LOCAL"},
                  {"name": "CREDIT_CARD_EXPIRATION_DATE"},
                  {"name": "PHONE_NUMBER"},
                  {"name": "PASSPORT"},
                  {"name": "STORAGE_SIGNED_URL"},
                  {"name": "MEDICAL_RECORD_NUMBER"},
                  {"name": "OBJECT_TYPE/LICENSE_PLATE"},
                  {"name": "XSRF_TOKEN"},
                  {"name": "CREDIT_CARD_DATA"},
                  {"name": "MEDICAL_ID"},
                  {"name": "BASIC_AUTH_HEADER"},
                  {"name": "PASSWORD"},
                  {"name": "ICCID_NUMBER"},
                  {"name": "DATE_OF_BIRTH"},
                  {"name": "CREDIT_CARD_NUMBER"},
                  {"name": "TECHNICAL_ID"},
                  {"name": "GCP_API_KEY"},
                  {"name": "CVV_NUMBER"},
                  {"name": "DRIVERS_LICENSE_NUMBER"},
                  {"name": "IP_ADDRESS"},
                  {"name": "FINANCIAL_ACCOUNT_NUMBER"},
                  {"name": "IBAN_CODE"},
              ],
              "min_likelihood": dlp_v2.Likelihood.POSSIBLE,
              "limits": {"max_findings_per_item": 3},
              "include_quote": True,
          },
          "item": {"value": str(answers)},
      }
  )

  findings = collections.defaultdict(set)
  for finding in result.result.findings:
    findings[finding.info_type.name].add(finding.quote)
  return {k: "\n".join(v) for k, v in findings.items()}


def main() -> None:
  """
  Main function to process the CSV file.
  """
  parser = argparse.ArgumentParser(
      description=(
          "Analyze text from a CSV file for toxicity using the Perspective API."
      )
  )
  parser.add_argument(
      "--input_csv", required=True, help="Path to the input CSV file."
  )
  parser.add_argument(
      "--input_evals_csv",
      help="Path to the CSV file that contains per row InputCriteria metrics.",
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to save the output CSV file."
  )
  parser.add_argument(
      "--data_type",
      required=True,
      type=DataType,
      help="The type of data being processed, either ROUND_1 or ROUND_2.",
  )
  parser.add_argument(
      "--text_column",
      help=(
          "Name of the column containing the text to analyze. If not provided,"
          " SURVEY_TEXT is used for ROUND_1 and RESPONSE_TEXT for ROUND_2."
      ),
  )
  parser.add_argument(
      "--api_key",
      required=True,
      help="API key for the Perspective API.",
  )
  parser.add_argument(
      "--scorer_type",
      choices=["GEMINI", "PERSPECTIVE"],
      default="PERSPECTIVE",
      help="Backend to use for generating moderation scores.",
  )
  parser.add_argument(
      "--model_name",
      default="gemini-3-flash-preview",
      help="Gemini model name to use when scorer_type is GEMINI.",
  )
  args = parser.parse_args()

  api_key = args.api_key
  if not api_key:
    print(
        "Error: --api_key missing.",
        file=sys.stderr,
    )
    sys.exit(1)

  dlp_client = dlp_v2.DlpServiceClient(client_options={"api_key": api_key})

  if args.data_type == DataType.ROUND_1:
    text_splitter = split_round_1_text
    text_column = args.text_column or SURVEY_TEXT
  else:  # ROUND_2
    text_splitter = split_round_2_text
    text_column = args.text_column or RESPONSE_TEXT

  df = get_csv(args.input_csv, set(ORIGINAL_COLUMNS_TO_USE + [text_column]))
  if text_column not in df.columns:
    print(
        f"Error: '{text_column}' column not found in {args.input_csv}",
        file=sys.stderr,
    )
    sys.exit(1)

  if args.input_evals_csv:
    # The input data evaluation output includes a score that we want to use for
    # moderation since it's a good proxy for data quality. We merge this data
    # into the existing input data.
    score_col = "score"
    input_evals_df = get_csv(
        args.input_evals_csv, [score_col, _INPUT_EVAL_TEXT_COL]
    )
    if (
        score_col not in input_evals_df.columns
        or _INPUT_EVAL_TEXT_COL not in input_evals_df.columns
    ):
      print(
          f"Error: '{score_col}' and '{_INPUT_EVAL_TEXT_COL}' columns required"
          f" in {args.input_evals_csv}",
          file=sys.stderr,
      )
      sys.exit(1)
    # The data quality score is used as a proxy for spam scores, and the lowest
    # quality data is considered to be spam while high quality data is
    # considered not spam.
    input_evals_df = input_evals_df.rename(columns={score_col: _SPAM_SCORE})

    df = df.merge(
        input_evals_df,
        left_on=text_column,
        right_on=_INPUT_EVAL_TEXT_COL,
        how="left",
        validate="1:1",
    )
    df = df.drop(columns=_INPUT_EVAL_TEXT_COL)

  dlp_df = df[text_column].apply(
      lambda t: pd.Series(get_dlp_scores(dlp_client, str(t), text_splitter))
  )

  df = pd.concat([df, dlp_df], axis=1)

  attributes_to_score = ["TOXICITY", "SEVERE_TOXICITY", "PROFANITY"]

  if args.scorer_type == "GEMINI":
    print(f"Using Gemini ({args.model_name}) for moderation scoring...")
    scorer = ContentScorer(api_key=api_key, model_name=args.model_name)

    # Batch Scoring with Gemini
    scoring_tasks = []
    for idx, row in df.iterrows():
      text = str(row[text_column])
      snippets = text_splitter(text)
      for snippet in snippets:
        scoring_tasks.append({"text": snippet, "row_id": idx})

    batch_results = scorer.score(scoring_tasks, attributes_to_score)

    # Aggregate results (MAX per row)
    row_scores = collections.defaultdict(lambda: collections.defaultdict(list))
    for res in batch_results:
      rid = res["row_id"]
      for attr, val in res["scores"].items():
        row_scores[rid][attr].append(val)

    for attr in attributes_to_score:
      df[attr] = 0.0
      for rid, scores_dict in row_scores.items():
        if attr in scores_dict:
          df.at[rid, attr] = max(scores_dict[attr])

    df[_PROFANITY_SCORE] = df["PROFANITY"]
    df[_TOXICITY_SCORE] = df["TOXICITY"]
    df[_SEVERE_TOXICITY_SCORE] = df["SEVERE_TOXICITY"]

  elif args.scorer_type == "PERSPECTIVE":
    print("Using Perspective API for moderation scoring...")
    client = init_client(api_key=api_key)

    scores_df = df[text_column].apply(
        lambda t: pd.Series(
            get_max_scores(client, str(t), attributes_to_score, text_splitter)
        )
    )
    df[[_PROFANITY_SCORE, _TOXICITY_SCORE, _SEVERE_TOXICITY_SCORE]] = scores_df[
        ["PROFANITY", "TOXICITY", "SEVERE_TOXICITY"]
    ]
  else:
    raise ValueError(f"Unknown scorer_type: {args.scorer_type}")

  # Force scores to be from 0-1 with higher scores being worse.
  # Use the 90th percentile logest duration to be more robust to outliers.
  df[_TOO_FAST_SCORE] = (1 - df[DURATION] / df[DURATION].quantile(0.9)).clip(
      0, 1
  )

  scores_to_consider = [
      _PROFANITY_SCORE,
      _TOXICITY_SCORE,
      _SEVERE_TOXICITY_SCORE,
      _TOO_FAST_SCORE,
  ]
  if args.input_evals_csv:
    # The input data evals measure how high quality data is so we need to
    # reverse the scores.
    df[_SPAM_SCORE] = 1 - df[_SPAM_SCORE] / df[_SPAM_SCORE].max()
    scores_to_consider.append(_SPAM_SCORE)

  # Create a column that can be used for sorting.
  df[_MAX_SCORE] = df[scores_to_consider].max(axis=1)
  df = df.sort_values(by=_MAX_SCORE, ascending=False)

  output_dir = os.path.dirname(args.output_csv)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)

  df.to_csv(args.output_csv, index=False)
  print(f"Processed data with toxicity scores saved to {args.output_csv}")


if __name__ == "__main__":
  main()
