# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for base_model."""

import unittest

import numpy as np

from src.models import base_model

_DEFAULT = 100


class ValidateMaxConcurrentCallsTest(unittest.TestCase):
  """Tests for base_model.validate_max_concurrent_calls."""

  def test_valid_values_are_returned_as_int(self):
    """Integral values >= 1, including numpy ints, are accepted."""
    for value in (1, 5, 2**31, np.int64(5)):
      with self.subTest(value=value):
        result = base_model.validate_max_concurrent_calls(value)
        self.assertEqual(result, value)
        self.assertIs(type(result), int)

  def test_non_positive_values_raise_value_error(self):
    """Zero and negative values are rejected."""
    for value in (0, -1, -100):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          base_model.validate_max_concurrent_calls(value)

  def test_non_integer_values_raise_type_error(self):
    """Bools, floats, and strings are rejected."""
    for value in (True, False, 2.5, "5", None):
      with self.subTest(value=value):
        with self.assertRaises(TypeError):
          base_model.validate_max_concurrent_calls(value)


class ResolveMaxConcurrentCallsTest(unittest.TestCase):
  """Tests for base_model.resolve_max_concurrent_calls."""

  def test_none_returns_default(self):
    """None falls back to the provided default."""
    self.assertEqual(
        base_model.resolve_max_concurrent_calls(None, default=_DEFAULT),
        _DEFAULT,
    )

  def test_valid_value_is_returned(self):
    """A valid value overrides the default."""
    self.assertEqual(
        base_model.resolve_max_concurrent_calls(5, default=_DEFAULT), 5
    )

  def test_invalid_values_are_rejected(self):
    """Invalid values raise rather than silently using the default."""
    for value, error in ((0, ValueError), (-1, ValueError), (2.5, TypeError)):
      with self.subTest(value=value):
        with self.assertRaises(error):
          base_model.resolve_max_concurrent_calls(value, default=_DEFAULT)


if __name__ == "__main__":
  unittest.main()
