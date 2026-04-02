import unittest
from unittest import mock
from unittest.mock import AsyncMock
import pandas as pd
import os
import tempfile
import sys
from src.simulated_jury import main
from src.simulated_jury import simulated_jury


class MainTest(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.participants_csv = os.path.join(self.temp_dir.name, 'participants.csv')
    self.statements_csv = os.path.join(self.temp_dir.name, 'statements.csv')
    self.output_csv = os.path.join(self.temp_dir.name, 'output.csv')

    pd.DataFrame({'participant_id': ['p1', 'p2']}).to_csv(
        self.participants_csv, index=False
    )
    pd.DataFrame({'statement': ['A', 'B']}).to_csv(
        self.statements_csv, index=False
    )

  def tearDown(self):
    self.temp_dir.cleanup()

  @mock.patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=AsyncMock,
  )
  def test_main_approval(self, mock_run):
    mock_run.return_value = (
        pd.DataFrame([
            {
                'data_row': {'participant_id': 'p1'},
                'result': {'A': True, 'B': False},
            },
            {
                'data_row': {'participant_id': 'p2'},
                'result': {'A': True, 'B': True},
            },
        ]),
        {},
    )

    test_args = [
        'main.py',
        '--participants_csv',
        self.participants_csv,
        '--statements_csv',
        self.statements_csv,
        '--output_csv',
        self.output_csv,
        '--percent',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    self.assertTrue(os.path.exists(self.output_csv))
    df = pd.read_csv(self.output_csv)
    self.assertEqual(len(df), 2)
    self.assertIn('agree_rate', df.columns)

    row_a = df[df['statement'] == 'A'].iloc[0]
    row_b = df[df['statement'] == 'B'].iloc[0]
    self.assertEqual(row_a['agree_rate'], '100.0%')
    self.assertEqual(row_b['agree_rate'], '50.0%')

  @mock.patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=AsyncMock,
  )
  def test_main_ranking(self, mock_run):
    pd.DataFrame({'statement': ['A', 'B'], 'topic': ['t1', 't1']}).to_csv(
        self.statements_csv, index=False
    )

    mock_run.return_value = (
        pd.DataFrame([
            {
                'data_row': {'participant_id': 'p1'},
                'result': {'ranking': ['A', 'B']},
            },
            {
                'data_row': {'participant_id': 'p2'},
                'result': {'ranking': ['A', 'B']},
            },
        ]),
        {},
    )

    test_args = [
        'main.py',
        '--participants_csv',
        self.participants_csv,
        '--statements_csv',
        self.statements_csv,
        '--output_csv',
        self.output_csv,
        '--voting_mode',
        'ranking',
        '--group_by',
        'topic',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    df = pd.read_csv(self.output_csv)
    self.assertIn('schulze_rank', df.columns)
    self.assertNotIn('pav_rank', df.columns)
    self.assertNotIn('agree_rate', df.columns)

    row_a = df[df['statement'] == 'A'].iloc[0]
    row_b = df[df['statement'] == 'B'].iloc[0]
    self.assertEqual(row_a['schulze_rank'], 1.0)
    self.assertEqual(row_b['schulze_rank'], 2.0)

  @mock.patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=AsyncMock,
  )
  def test_main_schulze_pav(self, mock_run):
    pd.DataFrame({'statement': ['A', 'B'], 'topic': ['t1', 't1']}).to_csv(
        self.statements_csv, index=False
    )

    def mock_run_side_effect(*args, **kwargs):
      voting_mode = kwargs.get('voting_mode')
      if voting_mode == simulated_jury.VotingMode.APPROVAL:
        return (
            pd.DataFrame([
                {
                    'data_row': {'participant_id': 'p1'},
                    'result': {'A': True, 'B': False},
                },
                {
                    'data_row': {'participant_id': 'p2'},
                    'result': {'A': True, 'B': True},
                },
            ]),
            {},
        )
      elif voting_mode == simulated_jury.VotingMode.RANK:
        return (
            pd.DataFrame([
                {
                    'data_row': {'participant_id': 'p1'},
                    'result': {'ranking': ['A', 'B']},
                },
                {
                    'data_row': {'participant_id': 'p2'},
                    'result': {'ranking': ['A', 'B']},
                },
            ]),
            {},
        )

    mock_run.side_effect = mock_run_side_effect

    test_args = [
        'main.py',
        '--participants_csv',
        self.participants_csv,
        '--statements_csv',
        self.statements_csv,
        '--output_csv',
        self.output_csv,
        '--voting_mode',
        'schulze_pav',
        '--group_by',
        'topic',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    df = pd.read_csv(self.output_csv)
    self.assertIn('schulze_rank', df.columns)
    self.assertIn('pav_rank', df.columns)
    self.assertIn('agree_rate', df.columns)

    row_a = df[df['statement'] == 'A'].iloc[0]
    self.assertEqual(row_a['schulze_rank'], 1.0)
    self.assertEqual(row_a['pav_rank'], 1.0)


if __name__ == '__main__':
  unittest.main()
