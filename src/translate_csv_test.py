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

"""Unit tests for src.translate_csv."""

import json
import unittest
from unittest import mock

import pandas as pd
from src import translate_csv
from src.models import genai_model


class TranslateCsvTest(unittest.IsolatedAsyncioTestCase):
  """Tests for language detection and CSV column translation."""

  def setUp(self):
    super().setUp()
    translate_csv._TRANSLATION_CACHE.clear()

  def _create_mock_model(self, responses_by_text: dict[str, dict]):
    """Creates a mock GenaiModel that returns predetermined translations."""
    mock_model = mock.create_autospec(genai_model.GenaiModel, instance=True)

    async def fake_process_prompts(prompt_objs, response_parser, **kwargs):
      rows = []
      for job in prompt_objs:
        text_id = job["text_id"]
        payload = responses_by_text.get(
            text_id, {"is_target_language": True, "translation": ""}
        )
        parsed = response_parser({"text": json.dumps(payload)}, job)
        rows.append({"text_id": text_id, "result": parsed})
      return pd.DataFrame(rows), None, None, None

    mock_model.process_prompts_concurrently.side_effect = fake_process_prompts
    return mock_model

  async def test_translate_survey_text_with_response_tags(self):
    """Translates inner <response> tags in survey_text without altering questions."""
    df = pd.DataFrame({
        "participant_id": [1, 2],
        "survey_text": [
            (
                "<question>What do you think?</question>\n"
                "<response>Me gusta mucho</response>\n\n"
                "<question>Why?</question>\n"
                "<response>Because it works</response>"
            ),
            (
                "<question>What do you think?</question>\n"
                "<response>Already in English</response>"
            ),
        ],
    })
    original_row_0 = df.loc[0, "survey_text"]

    mock_model = self._create_mock_model({
        "Me gusta mucho": {
            "is_target_language": False,
            "translation": "I like it a lot",
        },
        "Because it works": {
            "is_target_language": True,
            "translation": "",
        },
        "Already in English": {
            "is_target_language": True,
            "translation": "",
        },
    })

    result_df = await translate_csv.translate_dataframe(
        df, ["survey_text"], mock_model, "en_us"
    )

    # Verify only inner <response> strings were sent to the model (not <question>)
    # and that the corresponding <question> was included as context in the prompt.
    prompt_objs = mock_model.process_prompts_concurrently.call_args[0][0]
    prompts_by_text = {job["text_id"]: job["prompt"] for job in prompt_objs}
    self.assertEqual(
        set(prompts_by_text.keys()),
        {"Me gusta mucho", "Because it works", "Already in English"},
    )
    self.assertIn(
        "Survey question the text is responding to: What do you think?",
        prompts_by_text["Me gusta mucho"],
    )
    self.assertIn(
        "Survey question the text is responding to: Why?",
        prompts_by_text["Because it works"],
    )

    self.assertIn("original_text_survey_text", result_df.columns)
    self.assertEqual(result_df.loc[0, "original_text_survey_text"], original_row_0)
    self.assertEqual(
        result_df.loc[0, "survey_text"],
        (
            "<question>What do you think?</question>\n"
            "<response>I like it a lot</response>\n\n"
            "<question>Why?</question>\n"
            "<response>Because it works</response>"
        ),
    )

  async def test_translate_plain_quote_column(self):
    """Translates plain-text quote column at the end of the pipeline."""
    df = pd.DataFrame({
        "participant_id": [1, 2],
        "survey_text": [
            "<question>Q1</question>\n<response>Bonjour le monde</response>",
            "<question>Q1</question>\n<response>Hello world</response>",
        ],
        "quote": ["[Bonjour] le monde", "Hello world"],
    })
    original_survey_text = df["survey_text"].copy()

    mock_model = self._create_mock_model({
        "[Bonjour] le monde": {
            "is_target_language": False,
            "translation": "[Hello] world",
        },
        "Hello world": {
            "is_target_language": True,
            "translation": "",
        },
    })

    result_df = await translate_csv.translate_dataframe(
        df, ["quote"], mock_model, "en_us"
    )

    prompt_objs = mock_model.process_prompts_concurrently.call_args[0][0]
    prompts_by_text = {job["text_id"]: job["prompt"] for job in prompt_objs}
    self.assertNotIn(
        "Additional context", prompts_by_text["[Bonjour] le monde"]
    )

    self.assertEqual(result_df.loc[0, "quote"], "[Hello] world")
    self.assertEqual(result_df.loc[0, "original_text_quote"], "[Bonjour] le monde")
    self.assertEqual(result_df.loc[1, "quote"], "Hello world")
    # survey_text was not modified because the edited quote didn't match <response>
    pd.testing.assert_series_equal(result_df["survey_text"], original_survey_text)

  async def test_translate_individual_question_columns_updates_survey_text(self):
    """Translating raw question columns also updates matching <response> in survey_text."""
    df = pd.DataFrame({
        "Q1": ["Hola"],
        "survey_text": ["<question>Greeting?</question>\n<response>Hola</response>"],
    })

    mock_model = self._create_mock_model({
        "Hola": {"is_target_language": False, "translation": "Hello"},
    })

    result_df = await translate_csv.translate_dataframe(
        df, ["Q1"], mock_model, "en_us"
    )

    self.assertEqual(result_df.loc[0, "Q1"], "Hello")
    self.assertEqual(result_df.loc[0, "original_text_Q1"], "Hola")
    self.assertEqual(
        result_df.loc[0, "survey_text"],
        "<question>Greeting?</question>\n<response>Hello</response>",
    )
    self.assertEqual(
        result_df.loc[0, "original_text_survey_text"],
        "<question>Greeting?</question>\n<response>Hola</response>",
    )

  async def test_no_translation_needed_does_not_add_backup_columns(self):
    """Does not add original_text_* columns when all text is already in target language."""
    df = pd.DataFrame({
        "survey_text": ["Already English"],
    })
    mock_model = self._create_mock_model({
        "Already English": {"is_target_language": True, "translation": ""},
    })

    result_df = await translate_csv.translate_dataframe(
        df, ["survey_text"], mock_model, "en_us"
    )

    self.assertNotIn("original_text_survey_text", result_df.columns)
    self.assertEqual(result_df.loc[0, "survey_text"], "Already English")


if __name__ == "__main__":
  unittest.main()
