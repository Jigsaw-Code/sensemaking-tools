import unittest
import numpy as np
from case_studies.wtp.social_choice import schulze
from typing import List


class SchulzeTest(unittest.TestCase):

  def test_wikipedia_example_ranking(self):
    """
    Validates the Schulze implementation against the standard example from
    Wikipedia, ensuring the final ranking is correct.
    """
    # Data from: https://en.wikipedia.org/wiki/Schulze_method
    voter_preferences_raw = {
        'ACBED': 5,
        'ADECB': 5,
        'BEDAC': 8,
        'CABED': 3,
        'CAEBD': 7,
        'CBADE': 2,
        'DCEBA': 7,
        'EBADC': 8,
    }

    all_preferences: List[List[str]] = []
    for ranking_str, count in voter_preferences_raw.items():
      all_preferences.extend([list(ranking_str)] * count)

    # Expected outcome
    expected_ranking = ['E', 'A', 'C', 'B', 'D']

    # Run the Schulze calculation
    result = schulze.get_schulze_ranking(all_preferences)
    actual_ranking = result.get('top_propositions', [])

    # The primary assertion: does our implementation produce the correct ranking?
    self.assertEqual(actual_ranking, expected_ranking)

    # Optional: We can also assert the strength of the beatpaths for completeness
    # These values are taken from our previous validation script's output
    p_matrix = result.get('schulze_matrix')
    statements = result.get('orig_statements')
    statement_to_index = {s: i for i, s in enumerate(statements)}

    # Strength of E > A should be 5
    self.assertEqual(
        p_matrix[statement_to_index['E'], statement_to_index['A']], 5
    )
    # Strength of A > C should be 11
    self.assertEqual(
        p_matrix[statement_to_index['A'], statement_to_index['C']], 11
    )


if __name__ == '__main__':
  unittest.main()
