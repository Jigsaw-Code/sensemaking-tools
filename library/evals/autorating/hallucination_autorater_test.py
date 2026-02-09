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

import os
import shutil
from unittest.mock import AsyncMock, MagicMock

from library.evals.autorating.autorating_utils import EvalInput
from library.evals.autorating.hallucination_autorater import HallucinationAutorater
from models.genai_model import GenaiModel
import pytest


@pytest.fixture
def mock_model():
  """Fixture to create a mocked GenaiModel."""
  model = MagicMock(spec=GenaiModel)
  model.call_gemini = AsyncMock()
  model.llm = MagicMock()
  model.llm.generate_content_async = AsyncMock()
  return model


@pytest.fixture
def mock_output_dir():
  """Fixture to create a mock output directory."""
  output_dir = "test_output"
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)
  yield output_dir  # Provide the directory path to the test
  shutil.rmtree(output_dir)  # Clean up after the test


@pytest.mark.asyncio
async def test_correctly_process_summaries_and_generate_report(
    mock_model, mock_output_dir
):
  summaries = [
      EvalInput(summary="Statement 1", source="Comment 1"),
      EvalInput(summary="Statement 2", source="Comment 2"),
  ]
  mock_model.call_gemini.return_value = {
      "text": (
          '{"analysis": "Test analysis", "answer": "YES", "explanation": "Test'
          ' explanation"}'
      )
  }

  autorater = HallucinationAutorater(mock_model, mock_output_dir)
  await autorater.rate_hallucination(summaries)

  csv_path = os.path.join(mock_output_dir, "hallucination_autoratings.csv")
  report_path = os.path.join(mock_output_dir, "hallucination_report.txt")
  assert os.path.exists(csv_path)
  assert os.path.exists(report_path)

  with open(csv_path, "r") as f:
    csv_content = f.read()
    assert "Statement 1" in csv_content
    assert "YES" in csv_content

  with open(report_path, "r") as f:
    report_content = f.read()
    assert "Summary Evaluation Report" in report_content
    assert "Total summary claims: 2" in report_content
    assert "Yes: 100%" in report_content
