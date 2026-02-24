import unittest
import pandas as pd
import sys
import os

# Ensure we can import from the directory
sys.path.append(os.getcwd())

from case_studies.wtp.simulated_jury import sampling_utils


class TestApplyJurySizeSampling(unittest.TestCase):

  def setUp(self):
    self.df = pd.DataFrame({'id': range(100)})

  def test_no_sampling(self):
    # Verify no sampling (1.0).
    result = sampling_utils.apply_jury_size_sampling(self.df, 1.0)
    self.assertEqual(len(result), 100)
    # Verify no sampling (None).
    result = sampling_utils.apply_jury_size_sampling(self.df, None)
    self.assertEqual(len(result), 100)

  def test_fractional_sampling(self):
    result = sampling_utils.apply_jury_size_sampling(self.df, 0.5)
    self.assertEqual(len(result), 50)

  def test_integer_sampling(self):
    result = sampling_utils.apply_jury_size_sampling(self.df, 10.0)
    self.assertEqual(len(result), 10)

  def test_integer_sampling_overflow(self):
    # Verify overflow is handled gracefully (returns full pool).
    result = sampling_utils.apply_jury_size_sampling(self.df, 150.0)
    self.assertEqual(len(result), 100)

  def test_invalid_input(self):
    with self.assertRaises(ValueError):
      sampling_utils.apply_jury_size_sampling(self.df, 0.0)
    with self.assertRaises(ValueError):
      sampling_utils.apply_jury_size_sampling(self.df, -5.0)


if __name__ == '__main__':
  unittest.main()
