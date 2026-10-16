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

"""Library for language detection and translation using Gemini."""

import argparse
import asyncio
import json
import logging
import re
from typing import Any
import pandas as pd
from src import prompts
from src.models import custom_types
from src.models import genai_model

# Global translation cache to avoid re-translating same strings across calls
_TRANSLATION_CACHE: dict[str, dict[str, Any]] = {}

_RESPONSE_TAG_PATTERN = re.compile(
    r"(<response>)(.*?)(</response>)", flags=re.DOTALL
)


def _extract_translatable_strings(value: Any) -> list[str]:
  """Extracts strings to translate from a cell value.

  If the cell contains `<response>...</response>` XML tags (such as in Qualtrics
  `survey_text`), extracts the inner text of each `<response>` tag so questions
  and XML markup are not sent to the translator. Otherwise, returns the stripped
  cell string itself.

  Args:
    value: The cell value from a DataFrame column.

  Returns:
    A list of non-empty stripped strings to translate.
  """
  if not isinstance(value, str):
    return []

  matches = _RESPONSE_TAG_PATTERN.findall(value)
  if matches:
    extracted = []
    for _, inner_text, _ in matches:
      cleaned = inner_text.strip()
      if cleaned:
        extracted.append(cleaned)
    return extracted

  cleaned = value.strip()
  return [cleaned] if cleaned else []


def _apply_mapping_to_value(value: Any, mapping: dict[str, str]) -> Any:
  """Applies a translation mapping to a cell value.

  If the cell contains `<response>...</response>` XML tags, replaces the inner
  text of each matching `<response>` tag while preserving surrounding XML and
  `<question>` tags. Otherwise, replaces the stripped cell string directly.

  Args:
    value: The cell value to update.
    mapping: Mapping from original stripped string to translated string.

  Returns:
    The updated cell value.
  """
  if not isinstance(value, str):
    return value

  if _RESPONSE_TAG_PATTERN.search(value):
    return _RESPONSE_TAG_PATTERN.sub(
        lambda m: (
            f"{m.group(1)}"
            f"{mapping.get(m.group(2).strip(), m.group(2))}"
            f"{m.group(3)}"
        ),
        value,
    )

  stripped = value.strip()
  return mapping.get(stripped, stripped)


async def translate_text_batch(
    texts: list[str], model: genai_model.GenaiModel, target_language: str
) -> dict[str, dict[str, Any]]:
  """Translates a batch of texts using Gemini, utilizing a global cache.

  Args:
    texts: List of text strings to detect and translate.
    model: Initialized GenaiModel instance.
    target_language: Target language code or name (e.g., 'en_us').

  Returns:
    Dictionary mapping each input text to its parsed TranslationResponse dict.
  """
  # Filter out texts already in cache
  to_translate = [t for t in texts if t not in _TRANSLATION_CACHE]
  if not to_translate:
    return _TRANSLATION_CACHE

  prompt_objs = [
      {
          "prompt": prompts.get_translation_prompt(text, target_language),
          "text_id": text,
          "stats": {},
          "response_mime_type": "application/json",
          "response_schema": custom_types.TranslationResponse,
      }
      for text in to_translate
  ]

  def response_parser(resp, job):
    try:
      res = resp["text"]
      return json.loads(res) if isinstance(res, str) else res
    except Exception as e:
      logging.error("Failed to parse translation for %s: %s", job["text_id"], e)
      return {"is_target_language": True, "translation": ""}

  response_df, _, _, _ = await model.process_prompts_concurrently(
      prompt_objs, response_parser
  )

  for _, row in response_df.iterrows():
    _TRANSLATION_CACHE[row["text_id"]] = row["result"]

  return _TRANSLATION_CACHE


async def translate_dataframe(
    df: pd.DataFrame,
    columns: list[str],
    model: genai_model.GenaiModel,
    target_language: str,
) -> pd.DataFrame:
  """Detects language and translates non-target-language responses in a DataFrame.

  For each target column, preserves the original values in an
  `original_text_<column>` column whenever translations are applied. Automatically
  handles both plain-text columns (e.g., `quote` or plain `survey_text`) and
  XML-formatted `survey_text` columns containing `<response>...</response>` tags.

  Args:
    df: Input pandas DataFrame.
    columns: List of column names to translate.
    model: Initialized GenaiModel instance.
    target_language: Target language code or name (e.g., 'en_us').

  Returns:
    The DataFrame with translated columns and `original_text_*` backup columns.
  """
  active_cols = [c for c in columns if c in df.columns]
  if not active_cols:
    logging.warning(
        "None of the specified columns %s found in DataFrame. Skipping"
        " translation.",
        columns,
    )
    return df

  original_cols = {col: df[col].copy() for col in active_cols}

  unique_to_translate: set[str] = set()
  for col in active_cols:
    df[col] = df[col].str.strip()
    for val in df[col].dropna():
      unique_to_translate.update(_extract_translatable_strings(val))

  unique_to_translate.discard("")
  if not unique_to_translate:
    return df
  logging.info("Translating %d unique responses...", len(unique_to_translate))
  translation_map = await translate_text_batch(
      list(unique_to_translate),
      model,
      target_language,
  )

  # Mapping original text to its translation for non-target language entries
  mapping: dict[str, str] = {}
  for text, res in translation_map.items():
    if not res.get("is_target_language", False):
      translated = (res.get("translation") or "").strip()
      if translated and translated != text:
        mapping[text] = translated

  if not mapping:
    return df

  for col in active_cols:
    df[f"original_text_{col}"] = original_cols[col]
    df[col] = df[col].apply(lambda v: _apply_mapping_to_value(v, mapping))

  # If individual question columns were translated (rather than survey_text
  # directly), also update matching <response> tags in survey_text.
  if "survey_text" in df.columns and "survey_text" not in active_cols:
    updated_survey_text = df["survey_text"].apply(
        lambda v: _apply_mapping_to_value(v, mapping)
    )
    if not updated_survey_text.equals(df["survey_text"]):
      df["original_text_survey_text"] = df["survey_text"].copy()
      df["survey_text"] = updated_survey_text

  return df


def main():
  """Main function for standalone translation of a CSV file."""
  parser = argparse.ArgumentParser(
      description="Translate CSV columns using Gemini."
  )
  parser.add_argument(
      "--input_csv", required=True, help="Path to input CSV file"
  )
  parser.add_argument(
      "--output_csv", required=True, help="Path to output CSV file"
  )
  parser.add_argument(
      "--columns",
      default="survey_text",
      help=(
          "Comma-separated list of columns to translate (default: survey_text)"
      ),
  )
  parser.add_argument("--gemini_api_key", help="Gemini API key")
  parser.add_argument(
      "--model_name",
      default="gemini-3.8-flash",
      help="Gemini model name",
  )
  parser.add_argument(
      "--target_language",
      default="en_us",
      help="The language to translate to",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  async def run_translation():
    # Load data
    df = pd.read_csv(args.input_csv)
    columns = [c.strip() for c in args.columns.split(",") if c.strip()]

    # Initialize model
    model = genai_model.GenaiModel(
        model_name=args.model_name, gemini_api_key=args.gemini_api_key
    )

    # Run translation
    translated_df = await translate_dataframe(
        df, columns, model, args.target_language
    )

    # Save results
    translated_df.to_csv(args.output_csv, index=False)
    logging.info("Successfully saved translated CSV to %s", args.output_csv)

  asyncio.run(run_translation())


if __name__ == "__main__":
  main()
