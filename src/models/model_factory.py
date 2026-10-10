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

"""Factory for creating model instances."""

import os
import logging
from src.models import systemone
from src.models.base_model import BaseModel
from src.models.genai_model import GenaiModel
from src.models.openai_compatible_model import OpenAICompatibleModel

DECISION_ENDPOINT_SYSTEMONE = "systemone"
_DEFAULT_DECISION_THRESHOLD = 0.5


def get_model(model_name: str, **kwargs) -> BaseModel:
  """Creates and returns a model instance.

  Args:
      model_name: Name of the model (e.g., 'gemini-2.5-flash', 'gemma-4').
      **kwargs: Additional arguments to pass to the model constructor.

  Environment Variables:
      MODEL_ENDPOINT_TYPE: 'google_genai' or 'openai_api_compatible'. Defaults to 'google_genai'.
      OPENAI_API_ENDPOINT_URL: URL for OpenAI API compatible endpoint (required if type is 'openai_api_compatible').
      OPENAI_API_ENDPOINT_KEY: API key for OpenAI endpoint (optional).

  Returns:
      An instance of a class inheriting from BaseModel.
  """
  endpoint_type = os.getenv("MODEL_ENDPOINT_TYPE", "google_genai")

  if endpoint_type == "google_genai":
    logging.info(f"Creating GenaiModel with model: {model_name}")
    return GenaiModel(model_name=model_name, **kwargs)

  elif endpoint_type == "openai_api_compatible":
    endpoint_url = os.getenv("OPENAI_API_ENDPOINT_URL")
    if not endpoint_url:
      raise ValueError(
          "OPENAI_API_ENDPOINT_URL must be set when MODEL_ENDPOINT_TYPE is"
          " 'openai_api_compatible'."
      )

    api_key = os.getenv("OPENAI_API_ENDPOINT_KEY")

    logging.info(
        f"Creating OpenAICompatibleModel with model: {model_name},"
        f" endpoint: {endpoint_url}"
    )
    return OpenAICompatibleModel(
        model_name=model_name,
        endpoint_url=endpoint_url,
        api_key=api_key,
        **kwargs,
    )

  else:
    raise ValueError(f"Unknown MODEL_ENDPOINT_TYPE: {endpoint_type}")


def get_decision_model() -> systemone.SystemOneClient | None:
  """Creates the System One client for judgment stages, if one is configured.

  Environment Variables:
      DECISION_ENDPOINT_TYPE: 'systemone' to send judgment stages to a local
        System One model. Unset keeps them on the generative model.
      DECISION_MODEL: Ollama model name, e.g. 'clef'. Required when
        DECISION_ENDPOINT_TYPE is 'systemone'.
      OLLAMA_HOST: Ollama server, read the same way as the Ollama CLI. Unset
        means the local server.
      DECISION_MAX_CONCURRENT: Maximum requests in flight (optional, default
        1, since a local runner has one slot).

  Returns:
      A SystemOneClient, or None when DECISION_ENDPOINT_TYPE is unset.

  Raises:
      SystemOneError: If DECISION_ENDPOINT_TYPE is unknown, DECISION_MODEL is
        missing, or DECISION_MAX_CONCURRENT is not a positive integer.
  """
  endpoint_type = os.getenv("DECISION_ENDPOINT_TYPE")
  if not endpoint_type:
    return None
  if endpoint_type != DECISION_ENDPOINT_SYSTEMONE:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"Unknown DECISION_ENDPOINT_TYPE: {endpoint_type}",
    )
  model = os.getenv("DECISION_MODEL")
  if not model:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        "DECISION_MODEL must be set when DECISION_ENDPOINT_TYPE is"
        f" '{DECISION_ENDPOINT_SYSTEMONE}'.",
    )
  kwargs = {}
  host = os.getenv("OLLAMA_HOST")
  if host and host.strip():
    kwargs["base_url"] = systemone.ollama_base_url(host)
  raw_limit = os.getenv("DECISION_MAX_CONCURRENT")
  if raw_limit:
    try:
      kwargs["max_concurrent"] = int(raw_limit)
    except ValueError as exc:
      raise systemone.SystemOneError(
          systemone.FailureKind.CONFIG,
          f"DECISION_MAX_CONCURRENT must be an integer, got {raw_limit!r}.",
      ) from exc
  logging.info(f"Creating SystemOneClient with model: {model}")
  return systemone.SystemOneClient(model=model, **kwargs)


def get_decision_threshold() -> float:
  """Returns the probability cutoff for System One labels and equivalence.

  Environment Variables:
      DECISION_THRESHOLD: Cutoff in [0, 1] (optional, default 0.5).

  Raises:
      SystemOneError: If DECISION_THRESHOLD is not a float in [0, 1].
  """
  raw = os.getenv("DECISION_THRESHOLD")
  if not raw:
    return _DEFAULT_DECISION_THRESHOLD
  try:
    value = float(raw)
  except ValueError as exc:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"DECISION_THRESHOLD must be a float, got {raw!r}.",
    ) from exc
  if not 0.0 <= value <= 1.0:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"DECISION_THRESHOLD must be between 0 and 1, got {value}.",
    )
  return value
