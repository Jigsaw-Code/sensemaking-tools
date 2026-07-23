#!/usr/bin/env python3
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

"""Validation script for Sensemaking inputs and API keys."""

import argparse
import json
import os
import sys
import urllib.request
import pandas as pd


def check_csv(file_path: str) -> bool:
  """Validates the structure of the input CSV for Sensemaking."""
  if not os.path.exists(file_path):
    print(f"ERROR: File '{file_path}' does not exist.")
    return False

  try:
    # Read only the header first to check columns
    df = pd.read_csv(file_path, nrows=5)
  except Exception as e:
    print(f"ERROR: Failed to read CSV file: {e}")
    return False

  columns = list(df.columns)
  print(f"SUCCESS: Read file '{file_path}'. Found columns: {columns}")

  # Check standard columns
  has_participant_id = "participant_id" in columns
  has_survey_text = "survey_text" in columns

  if has_participant_id and has_survey_text:
    print("VALIDATION: File contains standard columns ('participant_id' and 'survey_text').")
    print("RECOMMENDATION: Ready to proceed directly to Topic Discovery (Step 2).")
    return True
  else:
    print("VALIDATION: File does NOT contain standard 'participant_id' and 'survey_text' columns.")
    print("RECOMMENDATION: Requires survey processing (Step 1).")
    return True


def check_gemini_key(api_key: str) -> bool:
  """Validates Google Gemini API key by making a minimal request."""
  if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    return False

  try:
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Ping"
    )
    if response.text:
      print("SUCCESS: GEMINI_API_KEY (Gemini API) is valid.")
      return True
  except Exception as e:
    print(f"ERROR: GEMINI_API_KEY validation failed: {e}")
  return False


def check_perspective_key(api_key: str) -> bool:
  """Validates Perspective API key by making a minimal request."""
  if not api_key:
    print("ERROR: GCLOUD_API_KEY environment variable is not set.")
    return False

  url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
  data = {
      "comment": {"text": "hello"},
      "requestedAttributes": {"TOXICITY": {}}
  }
  req = urllib.request.Request(
      url,
      data=json.dumps(data).encode("utf-8"),
      headers={"Content-Type": "application/json"}
  )
  try:
    with urllib.request.urlopen(req, timeout=10) as response:
      if response.status == 200:
        print("SUCCESS: GCLOUD_API_KEY (Perspective API) is valid.")
        return True
  except Exception as e:
    print(f"ERROR: GCLOUD_API_KEY validation failed: {e}")
  return False


def main() -> None:
  parser = argparse.ArgumentParser(description="Validate input CSV and API keys for Sensemaking.")
  parser.add_argument("--input_csv", required=True, help="Path to input CSV file")
  args = parser.parse_args()

  print("=== Pre-run Validation Checks ===")
  csv_ok = check_csv(args.input_csv)

  # Fetch keys from environment
  gemini_key = os.getenv("GEMINI_API_KEY")
  gcloud_key = os.getenv("GCLOUD_API_KEY")

  print("\n--- Validating API Keys ---")
  gemini_ok = check_gemini_key(gemini_key)
  perspective_ok = check_perspective_key(gcloud_key)

  print("\n=== Validation Summary ===")
  if csv_ok and (gemini_ok or perspective_ok):
    print("ALL CHECKS PASSED: Ready to run the pipeline.")
    sys.exit(0)
  else:
    print("SOME CHECKS FAILED: Please resolve the issues noted above.")
    sys.exit(1)


if __name__ == "__main__":
  main()
