"""
Implementation of the Schulze social choice method.
"""

import itertools
import numpy as np
from typing import List, Dict


def get_preference_matrix(
    preferences: List[List[str]], statements=List[str]
) -> np.ndarray:
  """Computes the preference matrix from a list of preference rankings."""
  # Get a unique, sorted list of all candidates from the preferences
  num_candidates = len(statements)
  candidate_to_index = {statement: i for i, statement in enumerate(statements)}

  d = np.zeros((num_candidates, num_candidates), dtype=int)
  for ranking in preferences:
    for i in range(len(ranking)):
      for j in range(i + 1, len(ranking)):
        d[candidate_to_index[ranking[i]], candidate_to_index[ranking[j]]] += 1
  return d


def get_beatpaths(d: np.ndarray) -> np.ndarray:
  """
  Computes the strongest path strengths in a preference graph.

  Args:
      d: A 2D numpy array where d[i,j] is the number of voters who prefer candidate i to candidate j.

  Returns:
      A 2D numpy array where p[i,j] is the strength of the strongest path from candidate i to candidate j.
  """
  C = d.shape[0]
  p = np.zeros((C, C))

  # Initialize the strength of the strongest path
  for i in range(C):
    for j in range(C):
      if i != j:
        p[i, j] = d[i, j] - d[j, i]

  # Compute the strength of the strongest path using Floyd-Warshall algorithm
  for k in range(C):
    for i in range(C):
      if i != k:
        for j in range(C):
          if j != k and j != i:
            p[i, j] = max(p[i, j], min(p[i, k], p[k, j]))

  return p


def get_ranked_options(
    schulze_matrix: np.ndarray, statements: List[str], k: int = 3
) -> List[str]:
  """Gets the top k propositions from a Schulze matrix."""
  scores = np.sum(schulze_matrix > np.transpose(schulze_matrix), axis=1)
  top_k_indices = np.argsort(scores)[::-1][:k]
  return [statements[i] for i in top_k_indices]


def get_schulze_ranking(
    preferences: List[List[str]], k: int | None = None
) -> Dict[str, any]:
  """
  Computes the Schulze ranking from a list of preference rankings.
  """
  # Get all unique statements from the preferences
  statements = sorted(set(itertools.chain.from_iterable(preferences)))

  # If k is not provided, default to the total number of unique statements
  # to get the full ranking.
  if k is None:
    k = len(statements)

  d = get_preference_matrix(preferences, statements)
  beatpaths = get_beatpaths(d)
  top_propositions = get_ranked_options(beatpaths, statements, k)

  return {
      "preference_matrix": d,
      "schulze_matrix": beatpaths,
      "top_propositions": top_propositions,
      "orig_statements": statements,
  }
