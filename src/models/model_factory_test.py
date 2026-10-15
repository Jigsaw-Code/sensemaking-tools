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
from src.models import model_factory
from src.models import systemone
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



  @patch.dict(os.environ, {}, clear=True)
  def test_get_decision_model_is_none_when_unset(self):
    """Tests that the decision path is off by default."""
    self.assertIsNone(model_factory.get_decision_model())

  @patch.dict(os.environ, {"DECISION_ENDPOINT_TYPE": "chat"}, clear=True)
  def test_get_decision_model_unknown_type(self):
    """Tests that an unknown DECISION_ENDPOINT_TYPE is a config error."""
    with self.assertRaises(systemone.SystemOneError) as context:
      model_factory.get_decision_model()
    self.assertEqual(context.exception.kind, systemone.FailureKind.CONFIG)
    self.assertIn("Unknown DECISION_ENDPOINT_TYPE", str(context.exception))

  @patch.dict(os.environ, {"DECISION_ENDPOINT_TYPE": "systemone"}, clear=True)
  def test_get_decision_model_missing_model(self):
    """Tests that DECISION_MODEL is required when the path is on."""
    with self.assertRaises(systemone.SystemOneError) as context:
      model_factory.get_decision_model()
    self.assertEqual(context.exception.kind, systemone.FailureKind.CONFIG)
    self.assertIn("DECISION_MODEL must be set", str(context.exception))

  @patch.dict(
      os.environ,
      {"DECISION_ENDPOINT_TYPE": "systemone", "DECISION_MODEL": "clef"},
      clear=True,
  )
  def test_get_decision_model_defaults(self):
    """Tests the client defaults when only the required variables are set."""
    client = model_factory.get_decision_model()
    self.assertEqual(client.model, "clef")
    self.assertEqual(
        client.endpoint,
        f"{systemone.OLLAMA_DEFAULT_BASE_URL}/v1/systemone",
    )
    self.assertEqual(client._max_concurrent, 1)

  @patch.dict(
      os.environ,
      {
          "DECISION_ENDPOINT_TYPE": "systemone",
          "DECISION_MODEL": "clef",
          "OLLAMA_HOST": "gpu-box",
          "DECISION_MAX_CONCURRENT": "2",
      },
      clear=True,
  )
  def test_get_decision_model_reads_host_and_concurrency(self):
    """Tests that OLLAMA_HOST is read like the Ollama CLI reads it."""
    client = model_factory.get_decision_model()
    self.assertEqual(client.endpoint, "http://gpu-box:11434/v1/systemone")
    self.assertEqual(client._max_concurrent, 2)

  @patch.dict(
      os.environ,
      {
          "DECISION_ENDPOINT_TYPE": "systemone",
          "DECISION_MODEL": "clef",
          "DECISION_MAX_CONCURRENT": "many",
      },
      clear=True,
  )
  def test_get_decision_model_bad_concurrency(self):
    """Tests that a non-integer DECISION_MAX_CONCURRENT is a config error."""
    with self.assertRaises(systemone.SystemOneError) as context:
      model_factory.get_decision_model()
    self.assertEqual(context.exception.kind, systemone.FailureKind.CONFIG)

  @patch.dict(os.environ, {}, clear=True)
  def test_get_decision_threshold_default(self):
    """Tests the default decision threshold."""
    self.assertEqual(model_factory.get_decision_threshold(), 0.5)

  def test_get_decision_threshold_validates(self):
    """Tests that the threshold must be a float in [0, 1]."""
    with patch.dict(os.environ, {"DECISION_THRESHOLD": "0.7"}, clear=True):
      self.assertEqual(model_factory.get_decision_threshold(), 0.7)
    for raw in ("high", "1.5"):
      with patch.dict(os.environ, {"DECISION_THRESHOLD": raw}, clear=True):
        with self.assertRaises(systemone.SystemOneError):
          model_factory.get_decision_threshold()


if __name__ == "__main__":
  unittest.main()
