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

import unittest
from typing import Any
from case_studies.wtp.models import custom_types
from case_studies.wtp.evals import autorater_evals


class AutoraterEvalsTest(unittest.TestCase):

  def test_prepare_opinion_eval_prompts(self):
    input_statements = [
        custom_types.Statement(
            id="s1",
            text="Statement 1",
            quotes=[
                custom_types.Quote(
                    id="q1",
                    text="Quote 1",
                    topic=custom_types.FlatTopic(name="Topic A"),
                )
            ],
        )
    ]
    categorized_records = [
        custom_types.StatementRecord(
            id="s1",
            quote_id="q1",
            topics=[custom_types.FlatTopic(name="Opinion 1")],
        )
    ]
    all_opinions = [custom_types.FlatTopic(name="Opinion 1")]
    parent_topic_name = "Topic A"

    prompts = autorater_evals.prepare_opinion_eval_prompts(
        categorized_records,
        input_statements,
        all_opinions,
        parent_topic_name,
    )

    self.assertEqual(len(prompts), 1)
    prompt_data = prompts[0]
    self.assertIn("prompt", prompt_data)
    self.assertIn("metadata", prompt_data)
    self.assertEqual(prompt_data["metadata"]["record_id"], "s1")
    self.assertEqual(prompt_data["metadata"]["quote_id"], "q1")
    self.assertEqual(prompt_data["response_mime_type"], "application/json")

  def test_parse_eval_response_success(self):
    job = {}
    resp = {"text": '```json\n{"score": 4, "explanation": "Good"}\n```'}
    result = autorater_evals.parse_eval_response(resp, job)
    self.assertEqual(result["score"], 4.0)
    self.assertEqual(result["explanation"], "Good")

  def test_parse_eval_response_invalid_json(self):
    job = {}
    resp = {"text": "Invalid JSON"}
    with self.assertRaises(ValueError):
      autorater_evals.parse_eval_response(resp, job)

  def test_process_opinion_eval_results_success(self):
    record = custom_types.StatementRecord(id="s1", topics=[])
    results_list = [{
        "result": {"score": 4, "explanation": "Good"},
        "metadata": {"original_record": record},
    }]
    processed = autorater_evals.process_opinion_eval_results(results_list)
    self.assertEqual(len(processed["passed"]), 1)
    self.assertEqual(len(processed["failed"]), 0)
    self.assertEqual(processed["passed"][0].id, "s1")

  def test_process_opinion_eval_results_failure_low_score(self):
    record = custom_types.StatementRecord(id="s1", topics=[])
    results_list = [{
        "result": {"score": 2, "explanation": "Bad"},
        "metadata": {"original_record": record},
    }]
    processed = autorater_evals.process_opinion_eval_results(results_list)
    self.assertEqual(len(processed["passed"]), 0)
    self.assertEqual(len(processed["failed"]), 1)
    self.assertEqual(processed["failed"][0].id, "s1")

  def test_process_opinion_eval_results_execution_error(self):
    record = custom_types.StatementRecord(id="s1", topics=[])
    results_list = [{
        "result": {"error": "API Error"},  # Or error in top level
        "error": "API Error",
        "metadata": {"original_record": record},
    }]
    processed = autorater_evals.process_opinion_eval_results(results_list)
    self.assertEqual(len(processed["passed"]), 0)
    self.assertEqual(len(processed["failed"]), 1)
    self.assertEqual(processed["failed"][0].id, "s1")


if __name__ == "__main__":
  unittest.main()
