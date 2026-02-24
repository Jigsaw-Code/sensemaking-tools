import unittest
import pandas as pd
from case_studies.wtp.social_choice import proportional_approval_voting


class ProportionalApprovalVotingTest(unittest.TestCase):

  def setUp(self):
    """Set up a common approval matrix for testing."""
    self.voter_ids = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7']
    self.approval_matrix = pd.DataFrame(
        {
            'P1': [True, True, True, False, False, False, False],  # 3 approvals
            'P2': [
                True,
                True,
                False,
                False,
                False,
                False,
                False,
            ],  # 2 approvals
            'P3': [False, False, False, True, True, True, False],  # 3 approvals
            'P4': [
                False,
                False,
                True,
                True,
                False,
                False,
                True,
            ],  # 3 approvals (bridges P1, P3, and v7)
            'P5': [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ],  # 1 approval
        },
        index=self.voter_ids,
    )
    self.candidate_propositions = self.approval_matrix.columns.tolist()

  def test_calculate_pav_score_initial_pick(self):
    """
    Tests that the score for a candidate when no one is represented is
    simply its number of approvals.
    """
    representation_counts = pd.Series(0, index=self.voter_ids)

    # Score should be the raw number of approvals (utility is 1 / (0+1) = 1 for each)
    score_p1 = proportional_approval_voting._calculate_pav_score(
        'P1', representation_counts, self.approval_matrix
    )
    self.assertEqual(score_p1, 3.0)

    score_p4 = proportional_approval_voting._calculate_pav_score(
        'P4', representation_counts, self.approval_matrix
    )
    self.assertEqual(score_p4, 3.0)

  def test_calculate_pav_score_with_depth(self):
    """
    Tests that the score correctly calculates marginal utility based on depth.
    """
    # v1, v2 are represented once. Others are unrepresented.
    representation_counts = pd.Series(
        [1, 1, 0, 0, 0, 0, 0], index=self.voter_ids
    )

    # Candidate P2 is approved by v1 and v2.
    # Both are at depth 1. Marginal utility for each is 1 / (1+1) = 0.5
    # Total score should be: 0.5 + 0.5 = 1.0
    score_p2 = proportional_approval_voting._calculate_pav_score(
        'P2', representation_counts, self.approval_matrix
    )
    self.assertEqual(score_p2, 1.0)

    # Candidate P4 is approved by v3, v4, and v7.
    # All are at depth 0. Marginal utility is 1 / (0+1) = 1 for each.
    # Total score should be: 1.0 + 1.0 + 1.0 = 3.0
    score_p4 = proportional_approval_voting._calculate_pav_score(
        'P4', representation_counts, self.approval_matrix
    )
    self.assertEqual(score_p4, 3.0)

  def test_find_best_next_candidate(self):
    """
    Tests the helper function that selects the highest-scoring candidate.
    """
    # P1 has been selected. Voters v1, v2, v3 are represented once.
    representation_counts = pd.Series(
        [1, 1, 1, 0, 0, 0, 0], index=self.voter_ids
    )
    candidate_pool = ['P2', 'P3', 'P4', 'P5']

    # Expected scores:
    # P2: v1, v2 approve. Both at depth 1. Score = 0.5 + 0.5 = 1.0
    # P3: v4, v5, v6 approve. All at depth 0. Score = 1.0 + 1.0 + 1.0 = 3.0
    # P4: v3, v4, v7 approve. v3 is depth 1, v4/v7 are depth 0. Score = 0.5 + 1.0 + 1.0 = 2.5
    # P5: v6 approves. At depth 0. Score = 1.0

    best_candidate = proportional_approval_voting._find_best_next_candidate(
        candidate_pool, representation_counts, self.approval_matrix
    )
    self.assertEqual(best_candidate, 'P3')

  def test_get_pav_slate_pure_approval(self):
    """
    Tests the main recursive function starting from scratch (no initial slate).
    """
    # k=3
    # 1st pick: P1, P3, or P4 (tie, score 3.0). idxmax() picks P1. Slate: [P1]
    # 2nd pick: P3 (score 3.0). Slate: [P1, P3]
    # 3rd pick: P4.
    #   - After P1, P3 are picked, all voters v1-v6 are at depth 1. v7 is at depth 0.
    #   - P4 approvers: v3, v4, v7. Depths: 1, 1, 0.
    #   - P4 score = (1/(1+1)) + (1/(1+1)) + (1/(0+1)) = 0.5 + 0.5 + 1.0 = 2.0
    #   - P2 approvers: v1, v2. Depths: 1, 1.
    #   - P2 score = (1/(1+1)) + (1/(1+1)) = 0.5 + 0.5 = 1.0
    #   - P4 is the clear winner.

    slate = proportional_approval_voting.get_pav_slate(
        self.candidate_propositions, self.approval_matrix, k=3
    )
    self.assertEqual(slate, ['P1', 'P3', 'P4'])

  def test_get_pav_slate_with_initial_slate(self):
    """
    Tests the main function when it's seeded with an initial proposition.
    """
    # Start with P4. k=3
    # 1st pick (given): [P4]. Rep counts: v3, v4 are at 1.
    # 2nd pick: P1 (approvers v1,v2,v3. depths 0,0,1. score = 1+1+0.5=2.5)
    #           P3 (approvers v4,v5,v6. depths 1,0,0. score = 0.5+1+1=2.5)
    #           Tie, P1 wins. Slate: [P4, P1]
    # 3rd pick: P3. Slate: [P4, P1, P3]

    slate = proportional_approval_voting.get_pav_slate(
        self.candidate_propositions,
        self.approval_matrix,
        k=3,
        selected_slate=['P4'],
    )
    self.assertEqual(slate, ['P4', 'P1', 'P3'])


if __name__ == '__main__':
  unittest.main()
