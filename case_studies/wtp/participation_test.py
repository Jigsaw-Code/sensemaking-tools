import unittest
import pandas as pd
from case_studies.wtp import participation
from unittest.mock import patch


class ParticipantTest(unittest.TestCase):

  def test_get_prompt_representation_with_all_data(self):
    data = {
        'rid': '1',
        'question_1': 'What is your favorite color?',
        'answer_1': 'Blue',
        'question_2': 'Some quote',
        'answer_2': 'Some answer',
        'ranking_1_topic': 'Topic 1',
        'ranking_1_q_1': 'Statement A',
        'ranking_1_a_1': '1',
    }
    row = pd.Series(data)
    prompt = participation.get_prompt_representation(row)
    self.assertIn('<participant id="1">', prompt)
    self.assertIn('What is your favorite color?', prompt)
    self.assertIn('Blue', prompt)
    self.assertIn('Some quote', prompt)
    self.assertIn('Some answer', prompt)
    self.assertIn('Statement A', prompt)

  def test_get_prompt_representation_with_missing_r1_data(self):
    data = {
        'rid': '2',
        # question_1 and answer_1 are missing
        'question_2': 'Some quote',
        'answer_2': 'Some answer',
        'ranking_1_topic': 'Topic 1',
        'ranking_1_q_1': 'Statement A',
        'ranking_1_a_1': '1',
    }
    row = pd.Series(data)
    prompt = participation.get_prompt_representation(row)
    self.assertIn('<participant id="2">', prompt)
    self.assertNotIn('What is your favorite color?', prompt)
    self.assertIn('Some quote', prompt)
    self.assertIn('Some answer', prompt)
    self.assertIn('Statement A', prompt)

  def test_get_prompt_representation_with_missing_r2_data(self):
    data = {
        'rid': '3',
        'question_1': 'What is your favorite color?',
        'answer_1': 'Blue',
        'question_2': 'Some quote',
        'answer_2': 'Some answer',
        # ranking data is missing
    }
    row = pd.Series(data)
    prompt = participation.get_prompt_representation(row)
    self.assertIn('<participant id="3">', prompt)
    self.assertIn('What is your favorite color?', prompt)
    self.assertIn('Blue', prompt)
    self.assertIn('Some quote', prompt)
    self.assertIn('Some answer', prompt)
    self.assertNotIn('Statement A', prompt)

  def test_get_prompt_representation_with_only_base_data(self):
    data = {
        'rid': '4',
    }
    row = pd.Series(data)
    prompt = participation.get_prompt_representation(row)
    self.assertIn('<participant id="4">', prompt)

  def test_dynamic_column_discovery(self):
    data = {
        'rid': '5',
        'question_1': 'Intro question',
        'answer_1': 'Intro answer',
        'question_3': 'Gov 3',
        'answer_3': 'Gov answer 3',
        'question_5': 'Gov 5',
        'answer_5': 'Gov answer 5',
        'ranking_2_topic': 'Topic 2',
        'ranking_2_q_1': 'Rank 2 Q1',
        'ranking_2_a_1': '1',
        'ranking_2_q_2': 'Rank 2 Q2',
        'ranking_2_a_2': '2',
        'ranking_4_topic': 'Topic 4',
        'ranking_4_q_1': 'Rank 4 Q1',
        'ranking_4_a_1': '1',
    }
    row = pd.Series(data)
    prompt = participation.get_prompt_representation(row)
    self.assertIn('<question>Intro question</question>', prompt)
    self.assertNotIn('question_2', prompt)
    self.assertIn('Gov 3', prompt)
    self.assertIn('Gov 5', prompt)
    self.assertIn('<ranking_set topic="Topic 2">', prompt)
    self.assertIn('<topic>Topic 2</topic>', prompt)
    self.assertIn('Rank 2 Q1', prompt)
    self.assertNotIn('<ranking_set topic="Topic 3">', prompt)
    self.assertIn('<ranking_set topic="Topic 4">', prompt)

  @patch('case_studies.wtp.participation.google.auth.default')
  @patch('case_studies.wtp.participation.load_sheet_as_df')
  @patch('case_studies.wtp.participation.get_sheet_name_from_gid')
  def test_load_and_merge_participant_data(
      self,
      mock_get_sheet_name_from_gid,
      mock_load_sheet_as_df,
      mock_google_auth,
  ):
    # Mock the return value of google.auth.default
    mock_google_auth.return_value = (None, None)
    # Mock the return value of load_sheet_as_df
    df_r1 = pd.DataFrame({'rid': ['1', '2'], 'r1_col': ['a', 'b']})
    df_r2 = pd.DataFrame({'rid': ['1', '3'], 'r2_col': ['c', 'd']})
    mock_load_sheet_as_df.side_effect = [df_r1, df_r2]
    mock_get_sheet_name_from_gid.return_value = 'sheet_name'

    # Test with both R1 and R2 URLs
    merged_df = participation.load_and_merge_participant_data(
        r1_url='fake_r1_url?gid=1', r2_url='fake_r2_url?gid=1'
    )
    self.assertEqual(len(merged_df), 3)
    self.assertIn('r1_col', merged_df.columns)
    self.assertIn('r2_col', merged_df.columns)

    # Test with only R1 URL
    mock_load_sheet_as_df.side_effect = [df_r1]
    merged_df = participation.load_and_merge_participant_data(
        r1_url='fake_r1_url?gid=1'
    )
    self.assertEqual(len(merged_df), 2)
    self.assertIn('r1_col', merged_df.columns)
    self.assertNotIn('r2_col', merged_df.columns)

    # Test with only R2 URL
    mock_load_sheet_as_df.side_effect = [df_r2]
    merged_df = participation.load_and_merge_participant_data(
        r2_url='fake_r2_url?gid=1'
    )
    self.assertEqual(len(merged_df), 2)
    self.assertNotIn('r1_col', merged_df.columns)
    self.assertIn('r2_col', merged_df.columns)

    # Test with no URLs
    merged_df = participation.load_and_merge_participant_data()
    self.assertTrue(merged_df.empty)

  def test_get_r2_preferences_from_dataframe(self):
    data = {
        'ranking_1_topic': ['Topic 1', 'Topic 1'],
        'ranking_1_q_1': ['A', 'A'],
        'ranking_1_a_1': ['1', '2'],
        'ranking_1_q_2': ['B', 'B'],
        'ranking_1_a_2': ['2', '1'],
        'ranking_2_topic': ['Topic 2', 'Topic 2'],
        'ranking_2_q_1': ['C', 'C'],
        'ranking_2_a_1': ['1', '1'],
    }
    df = pd.DataFrame(data)
    preferences = participation.get_r2_preferences_from_dataframe(df)
    self.assertIn('Topic 1', preferences)
    self.assertIn('Topic 2', preferences)
    self.assertEqual(len(preferences['Topic 1']), 2)
    self.assertEqual(preferences['Topic 1'][0], ['A', 'B'])
    self.assertEqual(preferences['Topic 1'][1], ['B', 'A'])
    self.assertEqual(len(preferences['Topic 2']), 2)
    self.assertEqual(preferences['Topic 2'][0], ['C'])

  def test_get_r2_preferences_from_dataframe_dynamic_cols(self):
    data = {
        'ranking_3_topic': ['Topic 3', 'Topic 3'],
        'ranking_3_q_1': ['D', 'D'],
        'ranking_3_a_1': ['2', '1'],
        'ranking_3_q_2': ['E', 'E'],
        'ranking_3_a_2': ['1', '2'],
        'ranking_5_topic': ['Topic 5', 'Topic 5'],
        'ranking_5_q_1': ['F', 'F'],
        'ranking_5_a_1': ['1', '1'],
    }
    df = pd.DataFrame(data)
    preferences = participation.get_r2_preferences_from_dataframe(df)
    self.assertIn('Topic 3', preferences)
    self.assertIn('Topic 5', preferences)
    self.assertEqual(len(preferences['Topic 3']), 2)
    self.assertEqual(preferences['Topic 3'][0], ['E', 'D'])
    self.assertEqual(preferences['Topic 3'][1], ['D', 'E'])
    self.assertEqual(len(preferences['Topic 5']), 2)
    self.assertEqual(preferences['Topic 5'][0], ['F'])


if __name__ == '__main__':
  unittest.main()
