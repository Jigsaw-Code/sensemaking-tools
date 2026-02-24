"""
Implementation of the Bridging-Based Representation social choice algorithm.
"""

import pandas as pd
import numpy as np
from src.social_choice import schulze


def _calculate_pav_score(
    candidate: str,
    representation_counts: pd.Series,
    approval_matrix: pd.DataFrame,
) -> float:
  """Calculates the PAV score for a single candidate proposition."""
  # Get the set of participants who approve of this candidate.
  # We cast to bool to ensure boolean indexing, even if the matrix contains floats.
  approvers = approval_matrix[approval_matrix[candidate].astype(bool)].index
  current_depths = representation_counts.loc[approvers]

  # The score is the sum of the marginal utility for each approver.
  # For an approver at depth `d`, the marginal utility of representing them
  # one more time is 1 / (d + 1).
  score = (1 / (current_depths + 1)).sum()
  return score


def _find_best_next_candidate(
    candidate_pool: list[str],
    representation_counts: pd.Series,
    approval_matrix: pd.DataFrame,
) -> str | None:
  """Iterates through candidates to find the one with the highest PAV score."""
  best_candidate = None
  max_score = -1

  for candidate in candidate_pool:
    score = _calculate_pav_score(
        candidate, representation_counts, approval_matrix
    )

    if score > max_score:
      max_score = score
      best_candidate = candidate

  return best_candidate


def get_pav_slate(
    candidate_propositions: list[str],
    approval_matrix: pd.DataFrame,
    k: int,
    selected_slate: list[str] = None,
    # similarity_penalty args will go here in Phase 2
) -> list[str]:
  """
  Selects a slate of k propositions using a recursive, greedy PAV algorithm.
  """
  # Initialization for the first call.
  if selected_slate is None:
    selected_slate = []

  # Base Case: recursion stops when the slate is full.
  if len(selected_slate) >= k:
    return selected_slate

  # --- Calculate current representation state from scratch ---
  if not selected_slate:
    representation_counts = pd.Series(0, index=approval_matrix.index)
  else:
    representation_counts = approval_matrix[selected_slate].sum(axis=1)

  # --- Recursive Step: Find and add the best next candidate ---
  candidate_pool = [
      p for p in candidate_propositions if p not in selected_slate
  ]
  if not candidate_pool:
    return selected_slate

  best_candidate = _find_best_next_candidate(
      candidate_pool, representation_counts, approval_matrix
  )

  if best_candidate:
    # Recursive call with the newly selected candidate added to the slate.
    return get_pav_slate(
        candidate_propositions,
        approval_matrix,
        k,
        selected_slate + [best_candidate],
    )
  else:
    # No more candidates could be selected.
    return selected_slate


def run_schulze_pav_selection(
    ranked_choice_results: list[list[str]],
    approval_matrix: pd.DataFrame,
    k: int,
    # ... other args for embeddings etc.
) -> list[str]:
  """
  Runs the hybrid Schulze-PAV selection method.

  First, it determines the Schulze winner from the ranked-choice results.
  Then, it calls the core recursive function to select the rest of the slate.
  """
  # Find the Schulze winner (top proposition)
  schulze_ranking = schulze.get_schulze_ranking(ranked_choice_results)
  schulze_winner = schulze_ranking["top_propositions"][0]

  candidate_propositions = approval_matrix.columns.tolist()

  # Initialize the recursion with the Schulze winner as the initial slate.
  return get_pav_slate(
      candidate_propositions=candidate_propositions,
      approval_matrix=approval_matrix,
      k=k,
      selected_slate=[schulze_winner],
  )
