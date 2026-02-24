# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This namespace contains utilities for processing semantic embeddings

and semantic similarity/dissimilarity between pieces of text using
the Vertex AI Embeddings API. Embedding values are cached in memory to
avoid unnecessary computations.
"""

from google import genai
from google.genai.types import EmbedContentConfig
import numpy as np

# Create a cache for comment embeddings, so that we don't need to re-request or
# explicitly track string embeddings across multiple calls.
embeddings = {}


def get_embedding(comment_text: str) -> np.ndarray:
  """Gets the emedding for the comment text, memoizing the results."""
  vertex_client = genai.Client()
  # Return the cached comment embedding if present
  if comment_text in embeddings:
    return embeddings[comment_text]
  else:
    response = vertex_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[comment_text],
        config=EmbedContentConfig(task_type="CLUSTERING"),
    )
    result = np.array(response.embeddings[0].values)
    # Cache the result for later calls
    embeddings[comment_text] = result
    return result


def get_cosine_similarity(a: str | np.ndarray, b: str | np.ndarray) -> float:
  """Gets the cosine similarity between two vectors, or if an argument is a string,

  to it's embedding vector.
  """
  if isinstance(a, str):
    a = get_embedding(a)
  if isinstance(b, str):
    b = get_embedding(b)
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_cosine_distance(a: str | np.ndarray, b: str | np.ndarray) -> float:
  """Returns the cosine distance (1 - cosine_similarity) between two vectors,

  or if an argument is a string, to its embedding vector.
  """
  return 1 - get_cosine_similarity(a, b)
