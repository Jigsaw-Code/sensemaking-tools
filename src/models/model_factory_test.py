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

"""Tests for model_factory."""

import os
import unittest
from unittest.mock import patch
from src.models.model_factory import get_model


class ModelFactoryTest(unittest.TestCase):

  @patch("os.getenv")
  @patch("src.models.model_factory.GenaiModel")
  def test_get_model_genai_default(self, mock_genai, mock_getenv):
    """Tests that GenaiModel is created by default (google_genai)."""
    mock_getenv.side_effect = lambda k, default=None: {}.get(k, default)

    get_model(model_name="gemini-2.5-flash")
    mock_genai.assert_called_once_with(model_name="gemini-2.5-flash")

  @patch("os.getenv")
  @patch("src.models.model_factory.OpenAICompatibleModel")
  def test_get_model_openai(self, mock_openai, mock_getenv):
    """Tests that OpenAICompatibleModel is created when specified."""
    mock_getenv.side_effect = lambda k, default=None: {
        "MODEL_ENDPOINT_TYPE": "openai_api_compatible",
        "OPENAI_API_ENDPOINT_URL": "http://localhost:1234",
    }.get(k, default)

    get_model(model_name="gemma-4")
    mock_openai.assert_called_once_with(
        model_name="gemma-4", endpoint_url="http://localhost:1234", api_key=None
    )

  @patch("os.getenv")
  def test_get_model_missing_url_for_openai(self, mock_getenv):
    """Tests that ValueError is raised if URL is missing for OpenAI type."""
    mock_getenv.side_effect = lambda k, default=None: {
        "MODEL_ENDPOINT_TYPE": "openai_api_compatible",
    }.get(k, default)

    with self.assertRaises(ValueError) as context:
      get_model(model_name="gemma-4")
    self.assertIn("OPENAI_API_ENDPOINT_URL must be set", str(context.exception))

  @patch("os.getenv")
  def test_get_model_unknown_type(self, mock_getenv):
    """Tests that ValueError is raised for unknown model type."""
    mock_getenv.side_effect = lambda k, default=None: {
        "MODEL_ENDPOINT_TYPE": "unknown_type",
    }.get(k, default)

    with self.assertRaises(ValueError) as context:
      get_model(model_name="test_model")
    self.assertIn("Unknown MODEL_ENDPOINT_TYPE", str(context.exception))




if __name__ == "__main__":
  unittest.main()
