import unittest
import pandas as pd
from src.social_choice import representation


class RepresentationTest(unittest.TestCase):

  def test_run_greedy_selection(self):
    topic_df = pd.DataFrame({
        'opinion': ['A', 'B', 'C', 'D'],
    })
    all_voter_ids = [1, 2, 3, 4, 5, 6]
    approval_matrix = pd.DataFrame(
        {
            # Should be first, because it has the most True approvals
            'A': [True, True, True, False, False, False],
            'B': [False, True, True, False, False, False],
            # Should be third since it's 6 is the only possible approver left ?
            'C': [True, False, False, False, False, True],
            # Should be second since this is the only option that gives you two
            # new True approvals
            'D': [False, False, False, True, True, False],
        },
        index=all_voter_ids,
    )

    top_3 = representation.run_greedy_selection(
        topic_df, all_voter_ids, approval_matrix, k=3
    )
    # Assert that the function returns exactly k=3 elements.
    self.assertEqual(len(top_3), 3)
    self.assertEqual(top_3, ['A', 'D', 'C'])


if __name__ == '__main__':
  unittest.main()
