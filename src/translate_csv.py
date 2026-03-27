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
from typing import Any
import pandas as pd
from src import prompts
from src.models import custom_types
from src.models import genai_model

# Global translation cache to avoid re-translating same strings across calls
_TRANSLATION_CACHE = {}


async def translate_text_batch(
    texts: list[str], model: genai_model.GenaiModel, target_language: str
) -> dict[str, dict[str, Any]]:
  """Translates a batch of texts using Gemini, utilizing a global cache."""
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
      logging.error(f"Failed to parse translation for {job['text_id']}: {e}")
      return {"is_english": True, "translation": ""}

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
  """Detects language and translates non-English responses in a DataFrame."""
  active_cols = [c for c in columns if c in df.columns]
  if not active_cols:
    logging.warning(
        f"None of the specified columns {columns} found in DataFrame."
        " Skipping translation."
    )
    return df

  unique_to_translate = set()
  for col in active_cols:
    df[col] = df[col].str.strip()
    unique_to_translate.update(df[col].dropna().unique())

  unique_to_translate.discard("")
  if not unique_to_translate:
    return df
  logging.info(f"Translating {len(unique_to_translate)} unique responses...")
  translation_map = await translate_text_batch(
      list(unique_to_translate),
      model,
      target_language
  )

  # Mapping original text to its translation for non-target language entries
  mapping = {}
  for text, res in translation_map.items():
    if not res.get("is_target_language", False):
      mapping[text] = res.get("translation") or text

  for col in active_cols:
    df[col] = df[col].replace(mapping)

  # Update concatenated survey columns with translated text
  if mapping:
    if "survey_text" in df.columns:
      df["original_text_survey_text"] = df["survey_text"]

    for original, translated in mapping.items():
      original = str(original).strip()
      translated = str(translated).strip()
      if not original or original == translated:
        continue

      if "survey_text" in df.columns:
        df["survey_text"] = df["survey_text"].str.replace(
            f"<response>{original}</response>",
            f"<response>{translated}</response>",
            regex=False
        )

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
      required=True,
      help="Comma-separated list of columns to translate",
  )
  parser.add_argument("--gemini_api_key", help="Gemini API key")
  parser.add_argument(
      "--model_name",
      default="gemini-3.1-flash-lite-preview",
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
    columns = [c.strip() for c in args.columns.split(",")]

    # Initialize model
    model = genai_model.GenaiModel(
        model_name=args.model_name, api_key=args.gemini_api_key
    )

    # Run translation
    translated_df = await translate_dataframe(
        df, columns, model, args.target_language
    )

    # Save results
    translated_df.to_csv(args.output_csv, index=False)
    logging.info(f"Successfully saved translated CSV to {args.output_csv}")

  asyncio.run(run_translation())


if __name__ == "__main__":
  main()
