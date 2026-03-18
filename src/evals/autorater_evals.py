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

"""Handles autorater evaluations for opinion categorization."""

import json
import logging
from typing import Any

from src import prompts
from src.evals.eval_metrics import OPINION_CATEGORIZATION_METRICS
from src.models import custom_types


def parse_eval_response(
    resp: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
  """Parses the evaluation response."""
  text = resp.get("text", "")
  try:
    # The model is instructed to return JSON.
    # We might need to strip markdown block markers
    clean_text = text.strip()
    clean_text = (
        clean_text.removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
    )
    json_result = json.loads(clean_text)
    score = float(json_result.get("score", 0))
    explanation = json_result.get("explanation", "")
    return {"score": score, "explanation": explanation}
  except (json.JSONDecodeError, ValueError) as e:
    raise ValueError(f"Failed to parse JSON response: {text}. Error: {e}")
    return {"error": str(e)}


def prepare_opinion_eval_prompts(
    categorized_records: list[custom_types.StatementRecord],
    input_statements: list[custom_types.Statement],
    all_opinions_for_topic: list[custom_types.Topic],
    parent_topic_name: str,
) -> list[dict[str, Any]]:
  """Prepares prompts for opinion evaluation."""
  prompt_jobs = []
  if not categorized_records:
    return prompt_jobs

  # Prepare data for evaluation
  quote_id_to_text_map = {
      quote.id: quote.text
      for stmt in input_statements
      if stmt.quotes
      for quote in stmt.quotes
  }
  all_opinions_str = "\n".join(
      [f"- {op.name}" for op in all_opinions_for_topic]
  )

  pointwise_metric = OPINION_CATEGORIZATION_METRICS.pointwise_metric
  if not pointwise_metric:
    raise ValueError("Pointwise metric for opinion categorization not found.")

  rubric_str = json.dumps(pointwise_metric.rating_rubric, indent=2)

  for record in categorized_records:
    if not record.quote_id:
      continue

    quote_text = quote_id_to_text_map.get(record.quote_id)
    if not quote_text:
      logging.warning(f"Could not find text for quote id: {record.quote_id}")
      continue

    assigned_opinions_str = "\n".join([f"- {op.name}" for op in record.topics])

    # Manually format criteria as they contain placeholders
    formatted_criteria = {}
    for key, template in pointwise_metric.criteria.items():
      formatted_criteria[key] = template.format(
          topic=parent_topic_name,
          quote=quote_text,
          all_opinions=all_opinions_str,
      )
    criteria_str = json.dumps(formatted_criteria, indent=2)

    final_prompt = prompts.get_autorater_opinion_eval_prompt(criteria_str, rubric_str, assigned_opinions_str)

    prompt_jobs.append({
        "prompt": final_prompt,
        "metadata": {
            "record_id": record.id,
            "quote_id": record.quote_id,
            "original_record": record,
        },
        "response_mime_type": "application/json",
    })

  return prompt_jobs


def process_opinion_eval_results(
    results_list: list[dict[str, Any]],
) -> dict[str, list[custom_types.StatementRecord]]:
  """Processes the evaluation results."""
  passed_records = []
  failed_records = []

  for row in results_list:
    # row is the job dict merged with result dict
    # GenaiModel returns 'result' key which contains the output of _parse_eval_response
    result = row.get("result")
    metadata = row.get("metadata", {})
    original_record = metadata.get("original_record")

    # Check for execution error first (in row or result)
    error = row.get("error") or (
        result.get("error") if isinstance(result, dict) else None
    )

    # If result is just the dict from parser, it has 'score' and 'explanation'
    score = None
    explanation = None
    if isinstance(result, dict) and "score" in result:
      score = result["score"]
      explanation = result.get("explanation")

    if original_record:
      if error:
        # Failed execution
        failed_records.append(original_record)
        logging.debug(
            f"Opinion categorization for quote {original_record.quote_id}"
            f" failed eval with execution error: {error}"
        )
      elif score is not None and score >= 4:
        passed_records.append(original_record)
      else:
        failed_records.append(original_record)
        logging.debug(
            f"Opinion categorization for quote {original_record.quote_id}"
            f" failed eval with score {score}. Explanation:"
            f" {explanation}"
        )

  return {"passed": passed_records, "failed": failed_records}
