"""
Implementation of the representative opinion ranking algorithm.
"""

import pandas as pd


def run_greedy_selection(topic_df, all_voter_ids, approval_matrix, k=3):
  """
  Applies a greedy representation optimization algorithm

  Args:
      topic_df: DataFrame filtered for a single topic.
      all_voter_ids: A list of all participant IDs.
      approval_matrix: The full approval matrix DataFrame.
      k: The number of items to select.

  Returns:
      A list of the top k selected statements.
  """

  # Prepare data for the current topic
  # Get unique statements for the current topic to avoid duplicate column selection
  # Modify this line to exclude statements named "Other"
  statements = [
      stmt for stmt in topic_df["opinion"].unique().tolist() if stmt != "Other"
  ]
  # Subset the approval matrix for the statements in this topic
  topic_approvals = approval_matrix[statements]

  selected_statements = []
  unrepresented_voters = set(all_voter_ids)

  for _ in range(k):
    # If all voters are represented...
    if not unrepresented_voters:
      # then reset to all voters, so that the algorithm has to start from
      # scratch attempting now to cover everyone 2 (or more) times.
      unrepresented_voters = set(all_voter_ids)

    # Define the set of statements not yet selected
    candidate_statements = [
        s for s in statements if s not in selected_statements
    ]
    if not candidate_statements:
      break

    # Calculate n/k threshold based on currently unrepresented voters
    n_unrep = len(unrepresented_voters)

    # Identify items approved by at least n/k voters (Stage 1 candidates)
    stage1_candidates = {}
    for stmt in candidate_statements:
      # Get voters who approve this statement AND are unrepresented
      approvers = frozenset(topic_approvals[topic_approvals[stmt]].index)

      if n_unrep == 0:
        stage1_candidates[stmt] = len(approvers)
      else:
        unrep_approvers = unrepresented_voters.intersection(approvers)
        coverage = len(unrep_approvers)
        stage1_candidates[stmt] = coverage

    # Get the best statement
    best_statement = max(stage1_candidates, key=stage1_candidates.get)

    if best_statement:
      selected_statements.append(best_statement)
      # Update the set of unrepresented voters
      approvers_of_best = set(
          topic_approvals[topic_approvals[best_statement]].index
      )
      unrepresented_voters -= approvers_of_best

  return selected_statements
