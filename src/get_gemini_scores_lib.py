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

import asyncio
import logging
import pydantic
from typing import Any
from src import attribute_prompt_config
from src import prompts
from src.models import genai_model
from src.models import custom_types


class ContentScorer:
  """Scorer implementation using GenaiModel for efficient content moderation and bridging."""

  def __init__(self, api_key: str, model_name: str):
    self.temperature = attribute_prompt_config.MODEL_CONFIG.get("temperature", 0.0)

    self.client = genai_model.GenaiModel(
        model_name=model_name,
        api_key=api_key
    )

  async def score_async(
      self,
      texts_with_ids: list[dict[str, Any]],
      attributes: list[str]
  ) -> list[dict[str, Any]]:
    """Scores attributes independently using concurrent Gemini calls."""

    def parse_gemini_response(resp: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
      attr = job.get("target_attr")
      response_text = resp.get("text", "{}")
      try:
        parsed_response = pydantic.TypeAdapter(custom_types.ScoreResponse).validate_json(response_text)
        return {attr: float(parsed_response.score)}
      except pydantic.ValidationError as e:
        logging.error(f"Failed Pydantic validation for {attr}: {e}. Raw: {response_text}")
        return {attr: 0.0}
      except Exception as e:
        logging.error(f"Failed to parse Gemini response for {attr}: {e}. Raw: {response_text}")
        return {attr: 0.0}

    jobs = []
    for item in texts_with_ids:
      text = item["text"]
      row_id = item["row_id"]
      for attr in attributes:
        if attr not in attribute_prompt_config.ATTRIBUTES:
          continue

        cat_info = attribute_prompt_config.ATTRIBUTES[attr]
        cal_ex_str = "\n".join([
            f"- \"{ex['text']}\" (Agreement Probability: {ex['score']}) - Reasoning: {ex['reasoning']}"
            for ex in cat_info.get("calibrated_examples", [])
        ])

        additional_instr = f"\nAdditional Guidance for {cat_info['label']}:\n{cat_info['additional_instruction']}\n" if "additional_instruction" in cat_info else ""

        system_prompt = prompts.scoring_system_prompt_template.format(
            system_instruction=prompts.scoring_system_instruction,
            label=cat_info['label'],
            definition=cat_info['definition'],
            additional_instr=additional_instr,
            calibrated_examples=cal_ex_str
        )

        jobs.append({
            "prompt": f"Text to evaluate: {text}",
            "system_prompt": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": custom_types.ScoreResponse,
            "temperature": self.temperature,
            "row_id": row_id,
            "target_attr": attr,
        })

    results_df, _, _, _ = await self.client.process_prompts_concurrently(
        jobs,
        parse_gemini_response
    )

    if results_df.empty:
      return []

    # Aggregate results by row_id
    aggregated = {}
    for _, row in results_df.iterrows():
      rid = row["row_id"]
      attr = row.get("target_attr")
      if rid not in aggregated:
        aggregated[rid] = {"row_id": rid, "scores": {}}

      result_dict = row["result"]
      if attr in result_dict:
        aggregated[rid]["scores"][attr] = result_dict[attr]

    return list(aggregated.values())

  def score(self, texts_with_ids: list[dict[str, Any]], attributes: list[str]) -> list[dict[str, Any]]:
    """Synchronous entry point for scoring a batch of texts."""
    try:
      loop = asyncio.get_event_loop()
      return loop.run_until_complete(self.score_async(texts_with_ids, attributes))
    except RuntimeError:
      return asyncio.run(self.score_async(texts_with_ids, attributes))
