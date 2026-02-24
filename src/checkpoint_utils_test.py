# Copyright 2025 Google LLC
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

import unittest
import os
import pickle
import tempfile
import shutil
from src import checkpoint_utils


class CheckpointUtilsTest(unittest.TestCase):

  def setUp(self):
    """Set up a temporary directory for tests."""
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    """Clean up the temporary directory."""
    shutil.rmtree(self.test_dir)

  def test_save_and_load_checkpoint(self):
    """Test saving and then loading a checkpoint."""
    step_name = "test_step"
    data_to_save = {"key": "value", "number": 123}

    # Save the checkpoint
    checkpoint_utils.save_checkpoint(data_to_save, step_name, self.test_dir)

    # Check that the file was created
    checkpoint_dir = os.path.join(self.test_dir, ".checkpoints")
    checkpoint_file = os.path.join(checkpoint_dir, f"{step_name}.pkl")
    self.assertTrue(os.path.exists(checkpoint_file))

    # Load the checkpoint
    loaded_data = checkpoint_utils.load_checkpoint(step_name, self.test_dir)

    # Verify the loaded data
    self.assertEqual(data_to_save, loaded_data)

  def test_load_nonexistent_checkpoint(self):
    """Test loading a checkpoint that does not exist."""
    step_name = "nonexistent_step"
    loaded_data = checkpoint_utils.load_checkpoint(step_name, self.test_dir)
    self.assertIsNone(loaded_data)

  def test_save_checkpoint_no_output_dir(self):
    """Test that saving does nothing if output_dir is None."""
    step_name = "test_step"
    data_to_save = {"key": "value"}
    checkpoint_utils.save_checkpoint(data_to_save, step_name, None)

    # Check that no .checkpoints directory was created in the current directory
    self.assertFalse(os.path.exists(".checkpoints"))

  def test_load_checkpoint_no_output_dir(self):
    """Test that loading returns None if output_dir is None."""
    step_name = "test_step"
    loaded_data = checkpoint_utils.load_checkpoint(step_name, None)
    self.assertIsNone(loaded_data)

  def test_checkpoint_content(self):
    """Test the raw content of the checkpoint file."""
    step_name = "content_test"
    data_to_save = ["a", "b", "c"]
    checkpoint_utils.save_checkpoint(data_to_save, step_name, self.test_dir)

    checkpoint_path = os.path.join(
        self.test_dir, ".checkpoints", f"{step_name}.pkl"
    )
    with open(checkpoint_path, "rb") as f:
      unpickled_data = pickle.load(f)

    self.assertEqual(data_to_save, unpickled_data)


if __name__ == "__main__":
  unittest.main()
