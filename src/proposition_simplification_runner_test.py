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

"""Behavioral tests for proposition_simplification_runner."""

import asyncio
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src import proposition_simplification_runner
from src.models import fake_base_model
from src.models import model_factory
from src.propositions import proposal_simplifier

_COMPLIANT_PROPOSAL = (
    "Add shaded bus shelters at busy stops so riders stay dry."
)
_COMPLEX_PROPOSAL = (
    "Establish an intergovernmental administrative modernization consortium"
    " dedicated to systematically restructuring bureaucratic procurement"
    " methodologies, infrastructural capitalization mechanisms, and"
    " multijurisdictional accountability frameworks."
)
_SIMPLIFIED_PROPOSAL = (
    "Create a joint group to make buying rules simpler. Share yearly reports on"
    " how road and building funds are spent."
)


class PropositionSimplificationRunnerTest(unittest.TestCase):
  """End-to-end CSV-to-CSV tests for proposition_simplification_runner."""

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self._temp_dir.cleanup)

  def test_end_to_end_csv_simplification_preserves_columns_and_metrics(self):
    input_csv = os.path.join(self._temp_dir.name, "input.csv")
    output_csv = os.path.join(self._temp_dir.name, "nested", "output.csv")
    pd.DataFrame({
        "topic": ["Transit", "Governance"],
        "tag": ["None", "priority"],
        "proposition": [_COMPLIANT_PROPOSAL, _COMPLEX_PROPOSAL],
    }).to_csv(input_csv, index=False)

    fake_model = fake_base_model.FakeBaseModel([{"text": _SIMPLIFIED_PROPOSAL}])
    args = proposition_simplification_runner.get_args([
        "--input_csv",
        input_csv,
        "--output_csv",
        output_csv,
        "--gemini_api_key",
        "test-key",
        "--model_name",
        "gemini-3.5-flash-lite",
        "--additional_context",
        "Keep wording neutral and accessible.",
        "-c",
        "2",
    ])

    with mock.patch.object(
        model_factory, "get_model", return_value=fake_model
    ) as mock_get_model:
      asyncio.run(proposition_simplification_runner.main(args))

    mock_get_model.assert_called_once_with(
        model_name="gemini-3.5-flash-lite",
        gemini_api_key="test-key",
    )
    self.assertEqual(len(fake_model.calls), 1)
    self.assertIn(
        "Keep wording neutral and accessible.", fake_model.calls[0]["prompt"]
    )

    result_df = pd.read_csv(output_csv, keep_default_na=False, na_values=[""])
    self.assertEqual(list(result_df["topic"]), ["Transit", "Governance"])
    # Literal "None" values must not be mangled into NaN during CSV I/O.
    self.assertEqual(list(result_df["tag"]), ["None", "priority"])
    self.assertEqual(
        list(result_df["pre_simplified_statement"]),
        [_COMPLIANT_PROPOSAL, _COMPLEX_PROPOSAL],
    )
    self.assertEqual(
        list(result_df["post_simplified_statement"]),
        [_COMPLIANT_PROPOSAL, _SIMPLIFIED_PROPOSAL],
    )
    self.assertEqual(
        list(result_df["simplification"]),
        [_COMPLIANT_PROPOSAL, _SIMPLIFIED_PROPOSAL],
    )
    self.assertEqual(list(result_df["readability_compliant"]), [True, True])
    self.assertEqual(list(result_df["simplification_attempts"]), [0, 1])
    self.assertLessEqual(
        result_df["post_fkgl"].iloc[0], proposal_simplifier.DEFAULT_MAX_FKGL
    )
    self.assertGreater(
        result_df["pre_fkgl"].iloc[1], proposal_simplifier.DEFAULT_MAX_FKGL
    )
    self.assertLessEqual(
        result_df["post_fkgl"].iloc[1], proposal_simplifier.DEFAULT_MAX_FKGL
    )


if __name__ == "__main__":
  unittest.main()
