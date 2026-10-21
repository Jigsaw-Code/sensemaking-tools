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
from src.models import decision
from src.models import genai_model
from src.models import custom_types
from src.models import systemone

# Maximum number of missing (row_id, attribute) pairs listed in the warning.
_MAX_MISSING_TO_LOG = 10
# Maximum characters of raw model output included in parse error messages.
_MAX_RAW_CHARS_IN_ERROR = 200
# Default maximum attempts per scoring call, lower than GenaiModel's default
# (genai_model.MAX_LLM_RETRIES). Every failed attempt that returned a response
# is a billed call, and a systematic problem (e.g. the model answering on the
# wrong scale) would otherwise cost every (text, attribute) pair the full
# wrapper default. A few attempts are enough for transient malformed output.
DEFAULT_SCORING_MAX_LLM_RETRIES = 5


def parse_score_response(
    resp: dict[str, Any], job: dict[str, Any]
) -> dict[str, float]:
  """Parses a Gemini scoring response into a score for the job's attribute.

  Malformed responses raise instead of returning a default score. The
  GenaiModel worker treats a raising parser as a failed attempt and retries
  the call; if every attempt fails, the job is reported as an error and no
  score is recorded for it, rather than a fabricated 0.0.

  Args:
    resp: Response dict from GenaiModel.call_gemini. The model output is
      expected under the "text" key as JSON matching ScoreResponse.
    job: The job dict, with the attribute being scored under "target_attr".

  Returns:
    A dict mapping the job's attribute to its parsed score.

  Raises:
    KeyError: If job has no "target_attr".
    ValueError: If the response text is missing, does not match
      ScoreResponse, or the score is not a finite number in [0.0, 1.0].
  """
  attr = job["target_attr"]
  response_text = resp.get("text") or ""
  try:
    parsed_response = pydantic.TypeAdapter(
        custom_types.ScoreResponse
    ).validate_json(response_text)
  except pydantic.ValidationError as e:
    # The full raw response is already kept in GenaiModel's failed_tries.
    raise ValueError(
        f"Invalid score response for {attr}: {e}. Raw (truncated):"
        f" {response_text[:_MAX_RAW_CHARS_IN_ERROR]!r}"
    ) from e
  score = parsed_response.score
  # Scores are probabilities. This also rejects NaN (all comparisons with NaN
  # are False) and infinities.
  if not 0.0 <= score <= 1.0:
    raise ValueError(
        f"Score for {attr} must be a probability in [0.0, 1.0], got {score!r}."
    )
  return {attr: score}


class ContentScorer:
  """Scorer implementation using GenaiModel for efficient content moderation and bridging."""

  def __init__(
      self,
      gemini_api_key: str,
      model_name: str,
      max_llm_retries: int = DEFAULT_SCORING_MAX_LLM_RETRIES,
  ):
    """Initializes the scorer.

    Args:
      gemini_api_key: API key for Gemini.
      model_name: Gemini model name.
      max_llm_retries: Maximum attempts per (text, attribute) call that
        returns an error or an unusable response, after which no score is
        recorded. Each such attempt is a billed call. Quota (429) and
        unavailability (503) errors pause and retry without consuming
        attempts, so they are not limited by this.
    """
    self.temperature = attribute_prompt_config.MODEL_CONFIG.get("temperature", 0.0)

    self._decision = decision.decision_client()
    if self._decision is None:
      self.client = genai_model.GenaiModel(
          model_name=model_name,
          gemini_api_key=gemini_api_key,
          max_llm_retries=max_llm_retries,
      )
    else:
      self.client = None

  async def score_async(
      self,
      texts_with_ids: list[dict[str, Any]],
      attributes: list[str]
  ) -> list[dict[str, Any]]:
    """Scores attributes independently using concurrent Gemini calls.

    Each (text, attribute) pair is scored by a separate Gemini call. Calls
    whose response cannot be parsed are retried by GenaiModel.

    Args:
      texts_with_ids: Dicts with the "text" to score and a caller-chosen
        "row_id" used to key the results.
      attributes: Attribute names to score. Names not present in
        attribute_prompt_config.ATTRIBUTES are skipped.

    Returns:
      A list of {"row_id": ..., "scores": {attribute: score}} dicts. Callers
      must expect missing data: an attribute that could not be scored after
      all retries is omitted from "scores" (it is never defaulted to 0.0), and
      a row_id with no results at all may be absent from the list. A warning
      summarizing missing pairs is logged.
    """
    if self._decision is not None:
      return await self._score_with_decision(
          self._decision, texts_with_ids, attributes
      )

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
        parse_score_response
    )

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

    # The wrapper's own error log only identifies an internal job index, so
    # summarize which (row_id, attribute) pairs ended up without a score.
    missing = []
    for job in jobs:
      row_id = job["row_id"]
      attr = job["target_attr"]
      if attr not in aggregated.get(row_id, {}).get("scores", {}):
        missing.append((row_id, attr))
    if missing:
      logging.warning(
          "No score obtained for %d of %d (row_id, attribute) pair(s) after"
          " all retries; these will be missing from the results. First %d: %s",
          len(missing),
          len(jobs),
          min(len(missing), _MAX_MISSING_TO_LOG),
          missing[:_MAX_MISSING_TO_LOG],
      )

    return list(aggregated.values())

  async def _score_with_decision(
      self,
      client: systemone.SystemOneClient,
      texts_with_ids: list[dict[str, Any]],
      attributes: list[str],
  ) -> list[dict[str, Any]]:
    """Scores each attribute as a System One yes/no probability."""
    specs = []
    for attr in attributes:
      info = attribute_prompt_config.ATTRIBUTES.get(attr)
      if not info:
        continue
      specs.append(
          decision.AttributeSpec(
              name=attr,
              label=info["label"],
              definition=info["definition"],
              guidance=info.get("additional_instruction", ""),
          )
      )
    aggregated = []
    for item in texts_with_ids:
      scores = await decision.score_attributes(client, item["text"], specs)
      aggregated.append({"row_id": item["row_id"], "scores": scores})
    return aggregated

  def score(self, texts_with_ids: list[dict[str, Any]], attributes: list[str]) -> list[dict[str, Any]]:
    """Synchronous entry point for scoring a batch of texts.

    Args:
      texts_with_ids: See score_async.
      attributes: See score_async.

    Returns:
      See score_async.
    """
    try:
      loop = asyncio.get_event_loop()
      return loop.run_until_complete(self.score_async(texts_with_ids, attributes))
    except RuntimeError:
      return asyncio.run(self.score_async(texts_with_ids, attributes))
