import os
import pickle
import tempfile
import unittest
import pandas as pd

from src.propositions import world_model_util


class WorldModelUtilTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.output_dir = self.temp_dir.name

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_save_dataframe_to_pickle(self):
    """Tests that a DataFrame is saved to a pickle file."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    file_path = os.path.join(self.output_dir, "test.pkl")
    world_model_util.save_dataframe_to_pickle(df, file_path)
    with open(file_path, "rb") as f:
      loaded_df = pickle.load(f)
    pd.testing.assert_frame_equal(df, loaded_df)

  def test_save_propositions_as_csv(self):
    """Tests that propositions are saved to a CSV file."""
    df = pd.DataFrame({
        "topic": ["Topic A"],
        "opinion": ["Opinion A1"],
        "propositions": [
            pd.DataFrame({
                "proposition": ["prop 1", "prop 2"],
                "reasoning": ["reason 1", "reason 2"],
                "topic_score": [4.0, 3.0],
                "opinion_score": [1.0, 2.0],
            })
        ],
    })
    file_path = os.path.join(self.output_dir, "test.csv")
    world_model_util.save_propositions_as_csv(
        df, file_path, reasoning=True, has_eval_data=True
    )

    saved_df = pd.read_csv(file_path)
    self.assertEqual(len(saved_df), 2)
    self.assertEqual(saved_df["proposition"].tolist(), ["prop 1", "prop 2"])
    self.assertEqual(saved_df["reasoning"].tolist(), ["reason 1", "reason 2"])
    self.assertEqual(saved_df["topic_score"].tolist(), [4.0, 3.0])


if __name__ == "__main__":
  unittest.main()
