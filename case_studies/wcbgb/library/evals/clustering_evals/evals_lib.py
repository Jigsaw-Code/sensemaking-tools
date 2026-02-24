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

"""Library for running performance and stability evals on Topic Identification

and Categorization.
"""

import embeddings_lib as embeddings
import numpy as np
import pandas as pd

TOPICS_COL = "topics"
COMMENT_ID_COL = "comment-id"
COMMENT_TEXT_COL = "comment_text"


class AnalysisResults:
  """Holds summary statistics for an analysis."""

  mean: float
  stdev: float
  min: float
  max: float

  def __init__(self, values: list[float]):
    self.mean = np.mean(values)
    self.stdev = np.std(values)
    self.min = np.min(values)
    self.max = np.max(values)


def convert_topics_col_to_list(comments: pd.DataFrame) -> pd.DataFrame:
  """Converts the topics column values from strings of semicolon-separated

  topics to lists of topic strings.

  Args:
    comments: The comments dataframe.

  Returns:
    The same comments dataframe, with converted topics column.
  """
  comments[TOPICS_COL] = comments[TOPICS_COL].str.split(";")
  comments[TOPICS_COL] = comments[TOPICS_COL].apply(
      lambda x: list(set([i.split(":")[0] for i in x]))
  )
  return comments


def get_pairwise_categorization_diffs(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> float:
  """Returns the rate of comments with at least one topic difference between df1

  and df2.

  Args:
    df1: The first comments dataframe.
    df2: The second comments dataframe.
  """
  count_diffs = 0

  for _, row in df1.iterrows():
    matching_row = df2[df2[COMMENT_ID_COL].eq(row[COMMENT_ID_COL])]
    unique_diffs = set(row[TOPICS_COL]) ^ set(matching_row[TOPICS_COL].iloc[0])
    if len(unique_diffs) >= 1:
      # TODO: add additional metric that tracks degree of change, ie how many
      # assignments changed.
      count_diffs += 1

  return count_diffs / df1.shape[0]


def analyze_categorization_diffs(data: list[pd.DataFrame]) -> float:
  """Returns the average rate of comments with at least one topic difference

  between all pairs of comments dataframes.

  Args:
    data: A list of comments dataframes.
  """
  pairwise_diffs = []
  for index, df1 in enumerate(data):
    for df2 in data[index + 1 : len(data)]:
      pairwise_diffs.append(get_pairwise_categorization_diffs(df1, df2))

  return AnalysisResults(pairwise_diffs)


def get_topic_set_similarity(
    topic_set_1: set[str], topic_set_2: set[str]
) -> float:
  """Returns the average semantic similarity between the closest matching topic

  names between two sets of topic names.

  Args:
    topic_set_1: The first set of topic names.
    topic_set_2: The second set of topic names.
  """

  def get_similarities_for_first_topic_set(
      topic_set_1: set[str], topic_set_2: set[str]
  ) -> list[float]:
    similarities = []
    for topic in topic_set_1:
      other_topics_and_similarity = [
          (other_topic, embeddings.get_cosine_similarity(topic, other_topic))
          for other_topic in topic_set_2
      ]
      similarities.append(
          max(other_topics_and_similarity, key=lambda x: x[1])[1]
      )
    return similarities

  # For each topic set get the average similarity of each topic to its most
  # similar topic. This is macro-averaged at the topic set level.
  mean_similarity_1 = np.mean(
      get_similarities_for_first_topic_set(topic_set_1, topic_set_2)
  )
  mean_similarity_2 = np.mean(
      get_similarities_for_first_topic_set(topic_set_2, topic_set_1)
  )

  return np.mean([mean_similarity_1, mean_similarity_2])


def analyze_topic_set_similarity(data: list[pd.DataFrame]) -> AnalysisResults:
  """Return the average topic_set_similarity between all pairs of dataframes.

  Args:
    data: A list of comments dataframes.
  """
  topic_sets = []
  for df in data:
    exploded_topics = df[TOPICS_COL].explode()
    topic_sets.append(exploded_topics.unique())

  similarities = []
  for index, topic_set_1 in enumerate(topic_sets):
    for topic_set_2 in topic_sets[index + 1 : len(topic_sets)]:
      similarity = get_topic_set_similarity(topic_set_1, topic_set_2)
      similarities.append(similarity)
  return AnalysisResults(similarities)


# Next we define a variant of the center based silhouette coefficient which
# treats the topic name embedding as the center of the cluster induced by each
# topic.


def get_topic_comments(comments: pd.DataFrame, topic_name: str) -> pd.DataFrame:
  """Returns the comments assigned to the given topic."""
  return comments[comments[TOPICS_COL].apply(lambda x: topic_name in list(x))]


def topic_centered_cohesion(comments: pd.DataFrame, topic_name: str) -> float:
  """Returns cluster cohesion for the topic, treating the topic name embedding as the

  center of the cluster. This is computed as the average distance between the
  topic
  name embedding and embedding of each comment assigned that topic.
  """
  topic_comments = get_topic_comments(comments, topic_name)
  distances = [
      embeddings.get_cosine_distance(topic_name, ct)
      for ct in topic_comments[COMMENT_TEXT_COL]
  ]
  return np.mean(distances)


def topic_centered_comment_separation(
    comment: dict, topics: list[str]
) -> tuple[float, str | None]:
  """Computes cluster separation for the given comment, treating the topic name embedding

  as the center of each cluster. This is computed as the shortest distance
  between the
  comment embedding any the embedding of any topic name _not_ assigned said
  comment. The
  return value is a tuple of the separation distance, together with the closest
  topic
  name.
  """
  distances = [
      (embeddings.get_cosine_distance(t, comment[COMMENT_TEXT_COL]), t)
      for t in topics
      if t not in comment[TOPICS_COL]
  ]
  # This covers the case where a comment is assigned to every topic
  if not distances:
    return (float("nan"), None)
  min_distance = min(distances)
  return min_distance


def topic_centered_separation(comments: pd.DataFrame, topic_name: str) -> float:
  """Returns the cluster sepration for the given topic, treating the topic name

  embedding as the center of each cluster. This is computed as the average of
  the
  tpoic_centered_comment_separation scores for all comments assigned said topic.
  """
  topics = comments[TOPICS_COL].explode().unique()
  topic_comments = get_topic_comments(comments, topic_name)
  separations = [
      topic_centered_comment_separation(comment, topics)[0]
      for comment in topic_comments.to_dict("records")
  ]
  return np.mean(separations)


def topic_centered_silhouette_for_topic(
    comments: pd.DataFrame, topic_name: str
) -> float:
  """Returns the silhouette score for the given topic, treating the topic name

  embedding as the center of each cluster. This is computed as the normalized
  difference between topic_centered_separation and topic_centered_cohesion.
  """
  cohesion = topic_centered_cohesion(comments, topic_name)
  separation = topic_centered_separation(comments, topic_name)
  return (separation - cohesion) / max(cohesion, separation)


def topic_centered_silhouette(comments: pd.DataFrame) -> AnalysisResults:
  """Returns analysis of silhouette scores for the clustering induced by the topics,

  treating the topic name embedding as the center of each cluster. This is
  computed
  as the average of all the topic_centered_topic_silhouette scores for all
  topics in
  the dataset.
  """
  topics = comments[TOPICS_COL].explode().unique()
  topic_scores = [
      topic_centered_silhouette_for_topic(comments, topic) for topic in topics
  ]
  return AnalysisResults(topic_scores)


def analyze_topic_centered_silhouette_scores(
    data: list[pd.DataFrame],
) -> AnalysisResults:
  """Returns analysis of topic_centered_silhouette scores for a collection

  of comment dataframes.
  """
  scores = [topic_centered_silhouette(df).mean for df in data]
  return AnalysisResults(scores)


# Now we define a more traditional centroid-based silhouette.


class CentroidSilhouette:
  """Centroid-based silhouette analysis for multi-topic classifications.

  Whereas in traditional silhouette analysis, silhouette scores are
  micro-averaged across all data points, this approach first micro-averages
  across data points within each cluster, and then macro averages the cluster
  scores. See https://arxiv.org/abs/2401.05831 for rationale behind this
  approach.

  This approach adapts to multiple topic classifications by considering
  the cohesion score for each data point separately in relation to each assigned
  topic. The separation scores then are selected among all topics to which the
  comment is _not_ assigned.

  Data points here are considered as the embedding vectors of the comment text,
  and cluster centroids as the mean of these emedding vectors for every comment
  assigned to the corresponding topic. Dissimilairty between points is defined
  in terms of the cosine similarity metric.
  """

  def __init__(self, comments: pd.DataFrame):
    """Initialize the centroid-based silhouette analysis.

    Args:
      comments: Comment dataframe with topics column of string lists.
    """
    self.__comments = comments
    self.__topics = list(comments[TOPICS_COL].explode().unique())
    self.__topic_centroids = {}

  def get_topic_centroid(self, topic_name: str) -> np.ndarray:
    """Return centroid array for comment embeddings corresponding to the topic.

    Args:
      topic_name: The name of the topic.
    """
    if topic_name in self.__topic_centroids:
      return self.__topic_centroids[topic_name]
    else:
      topic_comments = get_topic_comments(self.__comments, topic_name)
      comment_texts = topic_comments[COMMENT_TEXT_COL]
      topic_centroid = np.mean(
          np.stack([embeddings.get_embedding(ct) for ct in comment_texts]),
          axis=0,
      )
      return topic_centroid

  def silhouette(self) -> AnalysisResults:
    """Returns AnalysisResults summary of silhouette scores for the clustering

    induced by the topics.
    """
    topic_scores = [self.topic_silhouette(topic) for topic in self.__topics]
    return AnalysisResults(topic_scores)

  def topic_silhouette(self, topic_name: str) -> float:
    """Returns the silhouette score for the given topic.

    Args:
      topic_name: The name of the topic.
    """
    cohesion = self.topic_cohesion(topic_name)
    separation = self.topic_separation(topic_name)
    return (separation - cohesion) / max(cohesion, separation)

  def topic_cohesion(self, topic_name: str) -> float:
    """Returns the cluster cohesion for the topic, representing how tightly

    packed the cluster's data points are.

    Args:
      topic_name: The name of the topic.
    """
    topic_centroid = self.get_topic_centroid(topic_name)
    topic_comments = get_topic_comments(self.__comments, topic_name)
    distances = [
        embeddings.get_cosine_distance(topic_centroid, ct)
        for ct in topic_comments[COMMENT_TEXT_COL]
    ]
    return np.mean(distances)

  def topic_separation(self, topic_name: str) -> float:
    """Returns the cluster separation for the given topic, representing how

    well separated the cluster's data points are from other topics.

    Args:
      topic_name: The name of the topic.
    """
    topic_comments = get_topic_comments(self.__comments, topic_name)
    separations = [
        self.comment_separation(comment)[0]
        for comment in topic_comments.to_dict("records")
    ]
    return np.mean(separations)

  def comment_separation(self, comment: dict) -> tuple[float, str | None]:
    """Computes cluster separation for the given comment.

    Args:
      comment: A dict representation of a row from the comments dataframe,
        including the text and topics columns.

    Returns:
      A tuple (separation, closest_topic_name) where separation is the distance
      to the closest non-assigned topic, and closest_topic_name is the name of
      that topic. If a comment is assigned to all topics, returns (`nan`, None).
    """
    distances = []
    for topic in self.__topics:
      if topic not in comment[TOPICS_COL]:
        centroid = self.get_topic_centroid(topic)
        comment_text = comment[COMMENT_TEXT_COL]
        distances.append(
            (embeddings.get_cosine_distance(centroid, comment_text), topic)
        )
    # This covers the case where a comment is assigned to every topic
    if not distances:
      return (float("nan"), None)
    return min(distances)


def analyze_centroid_silhouette_scores(
    data: list[pd.DataFrame],
) -> AnalysisResults:
  """Returns analysis of centroid_silhouette scores for a collection

  of comment dataframes.
  """
  scores = [CentroidSilhouette(df).silhouette().mean for df in data]
  return AnalysisResults(scores)
