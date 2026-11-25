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
from src.models import extraction
from src.models import systemone
from src.models.base_model import BaseModel
from src.models.genai_model import GenaiModel
from src.models.openai_compatible_model import OpenAICompatibleModel

DECISION_ENDPOINT_SYSTEMONE = "systemone"
EXTRACT_ENDPOINT_CLAIRIC = "clairic"
_DEFAULT_DECISION_THRESHOLD = 0.5
# Keeps the default local Ollama model loaded between pipeline stages.
_OLLAMA_KEEP_ALIVE = "30m"


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
      DECISION_ENDPOINT_TYPE: 'systemone' to send judgment stages to a System
        One decision model. Unset keeps them on the generative model.
      DECISION_MODEL: Model name, e.g. 'clef'. Required when
        DECISION_ENDPOINT_TYPE is 'systemone'.
      DECISION_MODEL_HOST: Host, base URL, or full endpoint URL (optional;
        see systemone.endpoint_url). Unset means a local Ollama server.
      DECISION_MODEL_API_KEY: Bearer key for hosted endpoints (optional).
      DECISION_MODEL_KEEP_ALIVE: How long Ollama keeps the model loaded, sent
        as keep_alive (optional). Defaults to 30m for the default local Ollama
        server. With DECISION_MODEL_HOST set it is sent only when given,
        because other servers reject the field.
      DECISION_MAX_QUESTIONS: Most questions per request (optional, default
        64, Ollama's limit).
      DECISION_MAX_CONCURRENT: Maximum requests in flight (optional, default
        1, since a local runner has one slot).

  Returns:
      A SystemOneClient, or None when DECISION_ENDPOINT_TYPE is unset.

  Raises:
      SystemOneError: If DECISION_ENDPOINT_TYPE is unknown, DECISION_MODEL is
        missing, DECISION_MODEL_HOST is not a valid host, or a numeric
        setting is not a positive integer.
  """
  return _systemone_client_from_env("DECISION")


def get_extractor(prefix: str) -> extraction.Extractor | None:
  """Creates the extraction model configured under a prefix, if any.

  Extraction models mark exact spans of text, which judgment-only hosts do
  not do, so each stage that extracts has its own prefix: QUOTE for span
  quotes and CONTEXT for participant context.

  Environment Variables (shown for the QUOTE prefix):
      QUOTE_ENDPOINT_TYPE: 'systemone' for a System One host that answers
        extract questions, such as Inception Mercury Decide, or 'clairic' for
        Cloudflare's Clairic on Workers AI. Unset turns the stage's span path
        off.
      QUOTE_MODEL: Model name for 'systemone', e.g. 'mercury-decide'.
      QUOTE_MODEL_HOST: Host or endpoint URL. For 'clairic' it defaults to the
        Workers AI run URL for CLOUDFLARE_ACCOUNT_ID.
      QUOTE_MODEL_API_KEY: Bearer key (a Cloudflare API token for 'clairic').
      QUOTE_MAX_CONCURRENT: Maximum requests in flight (default 1).
      QUOTE_MODEL_KEEP_ALIVE, QUOTE_MAX_QUESTIONS: As for DECISION_*, with
        'systemone' only.

  Args:
      prefix: Variable prefix, e.g. 'QUOTE'.

  Returns:
      An Extractor, or None when {prefix}_ENDPOINT_TYPE is unset.

  Raises:
      SystemOneError: If the settings under the prefix are invalid.
  """
  endpoint_type = os.getenv(f"{prefix}_ENDPOINT_TYPE")
  if not endpoint_type:
    return None
  if endpoint_type == DECISION_ENDPOINT_SYSTEMONE:
    client = _systemone_client_from_env(prefix)
    return extraction.SystemOneExtractor(client) if client else None
  if endpoint_type != EXTRACT_ENDPOINT_CLAIRIC:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"Unknown {prefix}_ENDPOINT_TYPE: {endpoint_type}",
    )
  host = os.getenv(f"{prefix}_MODEL_HOST")
  if host and host.strip():
    endpoint = host.strip()
  else:
    account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    if not account:
      raise systemone.SystemOneError(
          systemone.FailureKind.CONFIG,
          f"{prefix}_MODEL_HOST or CLOUDFLARE_ACCOUNT_ID must be set when"
          f" {prefix}_ENDPOINT_TYPE is '{EXTRACT_ENDPOINT_CLAIRIC}'.",
      )
    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account}/ai/run/{extraction.CLAIRIC_MODEL}"
    )
  logging.info(f"Creating ClairicExtractor with endpoint: {endpoint}")
  return extraction.ClairicExtractor(
      endpoint=endpoint,
      api_key=os.getenv(f"{prefix}_MODEL_API_KEY") or None,
      max_concurrent=_positive_int_env(f"{prefix}_MAX_CONCURRENT", 1),
  )


def _systemone_client_from_env(
    prefix: str,
) -> systemone.SystemOneClient | None:
  """Builds a System One client from the variables under one prefix.

  Args:
    prefix: Variable prefix, e.g. 'DECISION' for DECISION_ENDPOINT_TYPE.

  Returns:
    A SystemOneClient, or None when {prefix}_ENDPOINT_TYPE is unset.

  Raises:
    SystemOneError: If a setting under the prefix is invalid.
  """
  endpoint_type = os.getenv(f"{prefix}_ENDPOINT_TYPE")
  if not endpoint_type:
    return None
  if endpoint_type != DECISION_ENDPOINT_SYSTEMONE:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"Unknown {prefix}_ENDPOINT_TYPE: {endpoint_type}",
    )
  model = os.getenv(f"{prefix}_MODEL")
  if not model:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"{prefix}_MODEL must be set when {prefix}_ENDPOINT_TYPE is"
        f" '{DECISION_ENDPOINT_SYSTEMONE}'.",
    )
  host = os.getenv(f"{prefix}_MODEL_HOST")
  endpoint = (
      systemone.endpoint_url(host)
      if host and host.strip()
      else systemone.DEFAULT_ENDPOINT
  )
  api_key = os.getenv(f"{prefix}_MODEL_API_KEY") or None
  keep_alive = os.getenv(f"{prefix}_MODEL_KEEP_ALIVE") or (
      None if host and host.strip() else _OLLAMA_KEEP_ALIVE
  )
  logging.info(
      f"Creating SystemOneClient with model: {model}, endpoint: {endpoint}"
  )
  return systemone.SystemOneClient(
      model=model,
      endpoint=endpoint,
      api_key=api_key,
      keep_alive=keep_alive,
      max_questions=_positive_int_env(
          f"{prefix}_MAX_QUESTIONS", systemone.MAX_QUESTIONS
      ),
      max_concurrent=_positive_int_env(f"{prefix}_MAX_CONCURRENT", 1),
  )


def _positive_int_env(name: str, default: int) -> int:
  """Reads a positive integer setting from the environment.

  Args:
    name: Environment variable name.
    default: Value used when the variable is unset or empty.

  Returns:
    The parsed value, or default.

  Raises:
    SystemOneError: If the value is not an integer or is below 1.
  """
  raw = os.getenv(name)
  if not raw:
    return default
  try:
    value = int(raw)
  except ValueError as exc:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG,
        f"{name} must be an integer, got {raw!r}.",
    ) from exc
  if value < 1:
    raise systemone.SystemOneError(
        systemone.FailureKind.CONFIG, f"{name} must be >= 1, got {value}."
    )
  return value


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
