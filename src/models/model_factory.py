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
from src.models.base_model import BaseModel
from src.models.genai_model import GenaiModel
from src.models.openai_compatible_model import OpenAICompatibleModel


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
