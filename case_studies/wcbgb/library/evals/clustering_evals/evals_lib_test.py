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

"""Tests for evals_lib."""

import math
import unittest
from unittest.mock import patch
import evals_lib
import numpy as np
import pandas as pd


def assert_topic_lists_equal(
    all_expected_topics: list[list[str]], all_result_topics: list[list[str]]
) -> None:
  """Asserts that two sets of topics lists are equal regardless of order."""
  for expected_topics, result_topics in zip(
      all_expected_topics, all_result_topics
  ):
    expected_topics = expected_topics.sort()
    result_topics = result_topics.sort()
    assert expected_topics == result_topics


class TestEvalsLib(unittest.TestCase):

  def test_convert_topics_col_to_list(self):
    data = {
        "topics": [
            "topic1:subtopic1;topic2:subtopic2",
            "topic3:subtopic3;topic4:subtopic4",
            "topic5:subtopic5;topic6:subtopic6",
        ]
    }
    all_expected_topics = [
        ["topic2", "topic1"],
        ["topic3", "topic4"],
        ["topic6", "topic5"],
    ]
    result_df = evals_lib.convert_topics_col_to_list(pd.DataFrame(data))
    all_result_topics = result_df["topics"].tolist()
    assert_topic_lists_equal(all_expected_topics, all_result_topics)

  def test_convert_topics_col_to_list_duplicate_topics(self):
    data = {
        "topics": [
            "topic1:subtopic1;topic1:subtopic2",
            "topic2:subtopic3;topic2:subtopic4",
            "topic3:subtopic5;topic4:subtopic6",
        ]
    }
    expected_topics = [
        ["topic1"],
        ["topic2"],
        ["topic3", "topic4"],
    ]
    result_df = evals_lib.convert_topics_col_to_list(pd.DataFrame(data))
    assert_topic_lists_equal(expected_topics, result_df["topics"].tolist())

  def test_analyze_categorization_diffs_no_diffs(self):
    """Test with two identical DataFrames."""
    df1 = pd.DataFrame({
        "comment-id": [1, 2],
        "comment_text": ["a", "b"],
        "topics": [["topic1"], ["topic2"]],
    })
    data = [df1, df1]
    result = evals_lib.analyze_categorization_diffs(data)
    self.assertEqual(result.mean, 0.0)

  def test_analyze_categorization_diffs_some_diffs(self):
    """Test with two DataFrames with some differences."""
    df1 = pd.DataFrame({
        "comment-id": [1, 2],
        "comment_text": ["a", "b"],
        "topics": [["topic1"], ["topic2"]],
    })
    df2 = pd.DataFrame({
        "comment-id": [1, 2],
        "comment_text": ["a", "b"],
        "topics": [["topic1"], ["topic3"]],
    })
    data = [df1, df2]
    result = evals_lib.analyze_categorization_diffs(data)
    self.assertEqual(result.mean, 0.5)

  @patch("evals_lib.embeddings.get_cosine_similarity")
  def test_get_topic_set_similarity_identical_sets(
      self, mock_get_cosine_similarity
  ):
    """Test with two identical sets of topics."""
    mock_get_cosine_similarity.return_value = 1.0
    topic_set_1 = {"topic1", "topic2", "topic3"}
    topic_set_2 = {"topic1", "topic2", "topic3"}
    result = evals_lib.get_topic_set_similarity(topic_set_1, topic_set_2)
    self.assertEqual(result, 1.0)
    self.assertEqual(mock_get_cosine_similarity.call_count, 18)

  @patch("evals_lib.embeddings.get_cosine_similarity")
  def test_get_topic_set_similarity_partial_overlap(
      self, mock_get_cosine_similarity
  ):
    """Test with two sets of topics that have some overlap."""
    mock_get_cosine_similarity.side_effect = lambda x, y: 1.0 if x == y else 0
    topic_set_1 = {"topic1", "topic2", "topic3"}
    topic_set_2 = {"topic2", "topic3", "topic4"}
    result = evals_lib.get_topic_set_similarity(topic_set_1, topic_set_2)
    self.assertEqual(result, 2 / 3)
    self.assertEqual(mock_get_cosine_similarity.call_count, 18)

  @patch("evals_lib.get_topic_set_similarity")
  def test_analyze_topic_set_similarity_identical_dataframes(
      self, mock_get_topic_set_similarity
  ):
    """Test with two identical DataFrames."""
    mock_get_topic_set_similarity.return_value = 1.0
    df1 = pd.DataFrame({"topics": [["topic1"], ["topic2"]]})
    df2 = pd.DataFrame({"topics": [["topic1"], ["topic2"]]})
    data = [df1, df2]
    result = evals_lib.analyze_topic_set_similarity(data)
    self.assertEqual(result.mean, 1.0)
    self.assertEqual(mock_get_topic_set_similarity.call_count, 1)

  @patch("evals_lib.get_topic_set_similarity")
  def test_analyze_topic_set_similarity_multiple_dataframes(
      self, mock_get_topic_set_similarity
  ):
    """Test with multiple DataFrames."""
    mock_get_topic_set_similarity.side_effect = [0.5, 0.75, 0.25]
    df1 = pd.DataFrame({"topics": [["topic1"], ["topic2"]]})
    df2 = pd.DataFrame({"topics": [["topic3"], ["topic4"]]})
    df3 = pd.DataFrame({"topics": [["topic5"], ["topic6"]]})
    data = [df1, df2, df3]
    result = evals_lib.analyze_topic_set_similarity(data)
    self.assertEqual(result.mean, 0.5)
    self.assertEqual(result.min, 0.25)
    self.assertEqual(result.max, 0.75)
    self.assertEqual(mock_get_topic_set_similarity.call_count, 3)

  @patch("evals_lib.embeddings.get_cosine_distance")
  def test_topic_centered_silhouette(self, mock_get_cosine_distance):
    """Test the topic_centered_silhouette function."""
    # Arrange
    mock_get_cosine_distance.side_effect = lambda x, y: {
        ("topic1", "comment1"): 0.1,
        ("topic1", "comment2"): 0.8,
        ("topic2", "comment1"): 0.7,
        ("topic2", "comment2"): 0.2,
    }[(x, y)]
    df = pd.DataFrame({
        "comment-id": [1, 2],
        "comment_text": ["comment1", "comment2"],
        "topics": [["topic1"], ["topic2"]],
    })
    # Here, constructing the individual silhouette scores for each topic
    silh1 = 0.6 / 0.7
    silh2 = 0.6 / 0.8
    # And the average of these
    silh = (silh1 + silh2) / 2

    # Act
    result = evals_lib.topic_centered_silhouette(df)

    # Assert
    self.assertAlmostEqual(result.mean, silh, places=3)
    self.assertAlmostEqual(result.min, silh2, places=3)
    self.assertAlmostEqual(result.max, silh1, places=3)

  @patch("evals_lib.embeddings.get_cosine_distance")
  def test_topic_centered_comment_separation_three_topics(
      self, mock_get_cosine_distance
  ):
    """Test the topic_centered_comment_separation function with three topics."""
    # Arrange so that topic3 is the closest separation topic, ensuring this gets
    # selected over topic2 (which is miminal by alphanumeric sort)
    mock_get_cosine_distance.side_effect = lambda x, y: {
        ("topic1", "comment1"): 0.1,
        ("topic2", "comment1"): 0.7,
        ("topic3", "comment1"): 0.5,
    }[(x, y)]
    comment1 = {"comment_text": "comment1", "topics": ["topic1"]}
    topics = ["topic1", "topic2", "topic3"]
    # Act & Assert
    result1 = evals_lib.topic_centered_comment_separation(comment1, topics)
    self.assertEqual(result1, (0.5, "topic3"))

  @patch("evals_lib.embeddings.get_embedding")
  def test_get_topic_centroid(self, mock_get_embedding):
    """Test the get_topic_centroid method."""
    # Arrange
    mock_get_embedding.side_effect = lambda x: {
        "comment1": np.array([0.1, 0.9]),
        "comment2": np.array([0.0, 0.9]),
    }[x]
    df = pd.DataFrame({
        "comment-id": [1, 2],
        "comment_text": ["comment1", "comment2"],
        "topics": [["topic1"], ["topic1"]],
    })
    expected_centroid = np.array([0.05, 0.9])

    # Act
    result = evals_lib.CentroidSilhouette(df).get_topic_centroid("topic1")

    # Assert
    np.testing.assert_array_almost_equal(result, expected_centroid)

  @patch("evals_lib.embeddings.get_embedding")
  def test_centroid_silhouette_for_single_topic(self, mock_get_embedding):
    """Test the centroid_silhouette function."""
    # Arrange
    mock_get_embedding.side_effect = lambda x: {
        "comment1": np.array([2, 7]),
        "comment2": np.array([0, 9]),
        "comment3": np.array([-2, -8]),
        "comment4": np.array([-4, -6]),
        "comment5": np.array([4, 5]),
        "comment6": np.array([6, 9]),
    }[x]
    df = pd.DataFrame([
        {"comment-id": 1, "comment_text": "comment1", "topics": ["topic1"]},
        {"comment-id": 2, "comment_text": "comment2", "topics": ["topic1"]},
        {"comment-id": 3, "comment_text": "comment3", "topics": ["topic2"]},
        {"comment-id": 4, "comment_text": "comment4", "topics": ["topic2"]},
        {"comment-id": 5, "comment_text": "comment5", "topics": ["topic3"]},
        {"comment-id": 6, "comment_text": "comment6", "topics": ["topic3"]},
    ])
    # Some intermediate results based on this setup (computed "by hand" with
    # with scikit-learn)
    coh1 = 0.00977411
    sep1 = 0.12208195
    silh1 = (sep1 - coh1) / sep1

    # Act
    silhouette_obj = evals_lib.CentroidSilhouette(df)
    topic1_cohesion = silhouette_obj.topic_cohesion("topic1")
    topic1_separation = silhouette_obj.topic_separation("topic1")
    topic1_result = silhouette_obj.topic_silhouette("topic1")

    # Assert
    self.assertAlmostEqual(topic1_cohesion, coh1, places=3)
    self.assertAlmostEqual(topic1_separation, sep1, places=3)
    self.assertAlmostEqual(topic1_result, silh1, places=3)

  @patch("evals_lib.CentroidSilhouette.topic_silhouette")
  def test_centroid_silhouette_for_all_topics(self, mock_topic_silhouette):
    """Test the centroid_silhouette function."""
    # Arrange
    mock_topic_silhouette.side_effect = [0.1, 0.2, 0.3]
    df = pd.DataFrame([
        {"comment-id": 1, "comment_text": "comment1", "topics": ["topic1"]},
        {"comment-id": 2, "comment_text": "comment2", "topics": ["topic2"]},
        {"comment-id": 3, "comment_text": "comment3", "topics": ["topic3"]},
    ])
    expected_mean = 0.2
    expected_min = 0.1
    expected_max = 0.3

    # Act
    silhouette_obj = evals_lib.CentroidSilhouette(df)
    result = silhouette_obj.silhouette()

    # Assert
    self.assertAlmostEqual(result.mean, expected_mean, places=3)
    self.assertAlmostEqual(result.min, expected_min, places=3)
    self.assertAlmostEqual(result.max, expected_max, places=3)
