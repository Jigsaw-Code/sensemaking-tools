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

"""Tests for embeddings_lib."""

import unittest
from unittest.mock import MagicMock, patch
# Module under test
import embeddings_lib
import numpy as np


class TestEmbeddingsLib(unittest.TestCase):

  def test_cosine_similarity_calculation(self):
    # Arrange
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    vec_c = np.array([1.0, 1.0])

    # Act & Assert
    self.assertAlmostEqual(
        embeddings_lib.get_cosine_similarity(vec_a, vec_a), 1.0
    )
    self.assertAlmostEqual(
        embeddings_lib.get_cosine_similarity(vec_a, vec_b), 0.0
    )
    # Using cos(pi/4) here since pi/4 is the angle between these vectors, and is
    # therefore by definition what the "cosine similarity" should equal
    self.assertAlmostEqual(
        embeddings_lib.get_cosine_similarity(vec_a, vec_c), np.cos(np.pi / 4)
    )  # ~0.707

  def test_cosine_distance_calculation(self):
    # Arrange
    vec_a = np.array([1.0, 0.0])
    vec_b = np.array([0.0, 1.0])
    vec_c = np.array([1.0, 1.0])

    # Act & Assert
    self.assertAlmostEqual(
        embeddings_lib.get_cosine_distance(vec_a, vec_a), 0.0
    )
    self.assertAlmostEqual(
        embeddings_lib.get_cosine_distance(vec_a, vec_b), 1.0
    )
    # As above, using pi/4 for the angle between these vectors
    self.assertAlmostEqual(
        embeddings_lib.get_cosine_distance(vec_a, vec_c),
        1.0 - np.cos(np.pi / 4),
    )  # ~0.293
