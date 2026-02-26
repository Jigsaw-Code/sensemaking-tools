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

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch
from src.models.genai_model import GenaiModel
from src.models import custom_types
from src.quote_extraction import quote_extraction_lib
import pandas as pd


class QuoteExtractionLibTest(unittest.IsolatedAsyncioTestCase):

  def test_extract_quotes_from_text_generates_unique_ids(self):
    # Setup
    statements = [
        custom_types.Statement(
            id="statement1",
            text="This is a statement about topic A and B.",
            topics=[
                custom_types.FlatTopic(name="Topic A"),
                custom_types.FlatTopic(name="Topic B"),
            ],
        ),
        custom_types.Statement(
            id="statement2",
            text="This is another statement about topic A.",
            topics=[custom_types.FlatTopic(name="Topic A")],
        ),
    ]

    mock_model = MagicMock()

    # Mock process_prompts_concurrently to return a DataFrame with results
    # Each result should correspond to the parsed output
    mock_results_df = pd.DataFrame([
        {
            "result": {"text": "This is a quote.", "error": None},
            "statement_id": "statement1",
            "topic": custom_types.FlatTopic(name="Topic A"),
        },
        {
            "result": {"text": "This is a quote.", "error": None},
            "statement_id": "statement1",
            "topic": custom_types.FlatTopic(name="Topic B"),
        },
        {
            "result": {"text": "This is a quote.", "error": None},
            "statement_id": "statement2",
            "topic": custom_types.FlatTopic(name="Topic A"),
        },
    ])
    mock_model.process_prompts_concurrently = AsyncMock(
        return_value=(mock_results_df, None)
    )

    # Execution
    result = asyncio.run(
        quote_extraction_lib.extract_quotes_from_text(
            statements=statements,
            model=mock_model,
        )
    )

    # Assertion
    expected_result = [
        custom_types.Statement(
            id="statement1",
            text="This is a statement about topic A and B.",
            topics=[
                custom_types.FlatTopic(name="Topic A"),
                custom_types.FlatTopic(name="Topic B"),
            ],
            quotes=[
                custom_types.Quote(
                    id="statement1-Topic A",
                    text="This is a quote.",
                    topic=custom_types.FlatTopic(name="Topic A"),
                ),
                custom_types.Quote(
                    id="statement1-Topic B",
                    text="This is a quote.",
                    topic=custom_types.FlatTopic(name="Topic B"),
                ),
            ],
        ),
        custom_types.Statement(
            id="statement2",
            text="This is another statement about topic A.",
            topics=[custom_types.FlatTopic(name="Topic A")],
            quotes=[
                custom_types.Quote(
                    id="statement2-Topic A",
                    text="This is a quote.",
                    topic=custom_types.FlatTopic(name="Topic A"),
                )
            ],
        ),
    ]
    self.assertEqual(result, expected_result)

    all_quote_ids = []
    for statement in result:
      self.assertIsNotNone(statement.quotes)
      for quote in statement.quotes:
        self.assertIsNotNone(quote.id)
        self.assertTrue(quote.id.startswith(statement.id))
        self.assertIn(quote.topic.name, quote.id)
        all_quote_ids.append(quote.id)

    self.assertEqual(
        len(all_quote_ids), 3
    )  # 2 for statement1, 1 for statement2
    self.assertEqual(
        len(set(all_quote_ids)), len(all_quote_ids)
    )  # Check for uniqueness


if __name__ == "__main__":
  unittest.main()
