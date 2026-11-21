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

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
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

    # Verify that max_concurrent_calls was passed as None by default
    _, kwargs = mock_run.call_args
    self.assertIsNone(kwargs.get('max_concurrent_calls'))

    self.assertTrue(os.path.exists(self.output_csv))
    df = pd.read_csv(self.output_csv)
    self.assertEqual(len(df), 2)
    self.assertIn('agree_rate', df.columns)

    row_a = df[df['statement'] == 'A'].iloc[0]
    row_b = df[df['statement'] == 'B'].iloc[0]
    self.assertEqual(row_a['agree_rate'], '100.0%')
    self.assertEqual(row_b['agree_rate'], '50.0%')

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
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

    # Verify that max_concurrent_calls was passed as None by default
    _, kwargs = mock_run.call_args
    self.assertIsNone(kwargs.get('max_concurrent_calls'))

    df = pd.read_csv(self.output_csv)
    self.assertIn('schulze_rank', df.columns)
    self.assertNotIn('pav_rank', df.columns)
    self.assertNotIn('agree_rate', df.columns)

    row_a = df[df['statement'] == 'A'].iloc[0]
    row_b = df[df['statement'] == 'B'].iloc[0]
    self.assertEqual(row_a['schulze_rank'], 1.0)
    self.assertEqual(row_b['schulze_rank'], 2.0)

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
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

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=AsyncMock,
  )
  def test_main_with_concurrency_flag(self, mock_run):
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
        '--max_concurrent_calls',
        '42',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    self.assertTrue(os.path.exists(self.output_csv))
    # Verify that max_concurrent_calls was passed correctly to the runner
    _, kwargs = mock_run.call_args
    self.assertEqual(kwargs.get('max_concurrent_calls'), 42)

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=AsyncMock,
  )
  def test_main_preserves_none_in_participants_csv(self, mock_run):
    with open(self.participants_csv, 'w', encoding='utf-8') as f:
      f.write('participant_id,survey_text\np1,None\np2,N/A\n')

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
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    _, kwargs = mock_run.call_args
    passed_df = kwargs['participants_df']
    self.assertEqual(passed_df['survey_text'].tolist(), ['None', 'N/A'])

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=mock.AsyncMock,
  )
  def test_main_forwards_additional_context(self, mock_run):
    context_file = os.path.join(self.temp_dir.name, 'context.txt')
    with open(context_file, 'w') as context_fp:
      context_fp.write('Evaluate as municipal proposals.')

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
        '--additional_context_file',
        context_file,
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    _, kwargs = mock_run.call_args
    self.assertEqual(
        kwargs['additional_context'], 'Evaluate as municipal proposals.'
    )

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=mock.AsyncMock,
  )
  def test_main_approval_with_segment_column(self, mock_run):
    pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3', 'p4'],
        'region': ['North', 'North', 'South', 'South'],
    }).to_csv(self.participants_csv, index=False)

    # Statement A: North=100%, South=0% -> overall=0.5, minmax=0.0, rank=2.
    # Statement B: North=50%, South=50% -> overall=0.5, minmax=0.5, rank=1.
    mock_run.return_value = (
        pd.DataFrame([
            {
                'data_row': {'participant_id': 'p1'},
                'result': {'A': True, 'B': True},
            },
            {
                'data_row': {'participant_id': 'p2'},
                'result': {'A': True, 'B': False},
            },
            {
                'data_row': {'participant_id': 'p3'},
                'result': {'A': False, 'B': True},
            },
            {
                'data_row': {'participant_id': 'p4'},
                'result': {'A': False, 'B': False},
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
        '--segment_column',
        'region',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    df = pd.read_csv(self.output_csv)
    row_a = df[df['statement'] == 'A'].iloc[0]
    row_b = df[df['statement'] == 'B'].iloc[0]
    self.assertAlmostEqual(row_a['minmax_approval'], 0.0)
    self.assertEqual(row_a['bottleneck_segment'], 'South')
    self.assertAlmostEqual(row_a['gic_product'], 0.0)
    self.assertAlmostEqual(row_a['gic_geomean'], 0.0)
    self.assertEqual(row_a['bridging_rank'], 2)

    self.assertAlmostEqual(row_b['minmax_approval'], 0.5)
    self.assertEqual(row_b['bottleneck_segment'], 'North')
    self.assertAlmostEqual(row_b['gic_product'], 0.25)
    self.assertAlmostEqual(row_b['gic_geomean'], 0.5)
    self.assertEqual(row_b['bridging_rank'], 1)

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=mock.AsyncMock,
  )
  def test_main_approval_with_overlapping_segment_columns_and_percent(
      self, mock_run
  ):
    pd.DataFrame({
        'participant_id': ['p1', 'p2', 'p3', 'p4'],
        'seg_a': [1, 1, 0, 0],
        'seg_b': [1, 0, 1, 0],
    }).to_csv(self.participants_csv, index=False)

    # Statement A: seg_a=1.0 (p1,p2), seg_b=0.5 (p1=1,p3=0) -> minmax=0.5.
    # Statement B: seg_a=0.5 (p1=1,p2=0), seg_b=1.0 (p1,p3) -> minmax=0.5.
    # Overall: A=0.5 (2/4), B=0.75 (3/4) -> B wins tie-break on gic_geomean.
    mock_run.return_value = (
        pd.DataFrame([
            {
                'data_row': {'participant_id': 'p1'},
                'result': {'A': True, 'B': True},
            },
            {
                'data_row': {'participant_id': 'p2'},
                'result': {'A': True, 'B': False},
            },
            {
                'data_row': {'participant_id': 'p3'},
                'result': {'A': False, 'B': True},
            },
            {
                'data_row': {'participant_id': 'p4'},
                'result': {'A': False, 'B': True},
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
        '--segment_columns',
        'seg_a,seg_b',
        '--bridging_metric',
        'gic_geomean',
        '--percent',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    df = pd.read_csv(self.output_csv)
    row_a = df[df['statement'] == 'A'].iloc[0]
    row_b = df[df['statement'] == 'B'].iloc[0]
    self.assertEqual(row_a['minmax_approval'], '50.0%')
    self.assertEqual(row_a['gic_geomean'], '70.7%')
    self.assertAlmostEqual(row_a['gic_product'], 0.5)
    self.assertEqual(row_a['bottleneck_segment'], 'seg_b')
    self.assertEqual(row_a['bridging_rank'], 2)

    self.assertEqual(row_b['minmax_approval'], '50.0%')
    self.assertEqual(row_b['gic_geomean'], '70.7%')
    self.assertAlmostEqual(row_b['gic_product'], 0.5)
    self.assertEqual(row_b['bottleneck_segment'], 'seg_a')
    self.assertEqual(row_b['bridging_rank'], 1)

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=mock.AsyncMock,
  )
  def test_main_schulze_pav_with_group_by_and_bridging(self, mock_run):
    pd.DataFrame({
        'participant_id': ['p1', 'p2'],
        'seg_a': [1, 0],
        'seg_b': [0, 1],
    }).to_csv(self.participants_csv, index=False)
    pd.DataFrame({
        'statement': ['A', 'B', 'C', 'D'],
        'topic': ['t1', 't1', 't2', 't2'],
    }).to_csv(self.statements_csv, index=False)

    def mock_run_side_effect(*args, **kwargs):
      del args
      voting_mode = kwargs.get('voting_mode')
      if voting_mode == simulated_jury.VotingMode.APPROVAL:
        return (
            pd.DataFrame([
                {
                    'data_row': {'participant_id': 'p1'},
                    'result': {'A': True, 'B': True, 'C': False, 'D': True},
                },
                {
                    'data_row': {'participant_id': 'p2'},
                    'result': {'A': False, 'B': True, 'C': True, 'D': True},
                },
            ]),
            {},
        )
      topic_name = kwargs.get('topic_name')
      ranking = ['A', 'B'] if topic_name == 't1' else ['C', 'D']
      return (
          pd.DataFrame([
              {
                  'data_row': {'participant_id': 'p1'},
                  'result': {'ranking': ranking},
              },
              {
                  'data_row': {'participant_id': 'p2'},
                  'result': {'ranking': ranking},
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
        '--segment_columns',
        'seg_a',
        'seg_b',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      main.main()

    df = pd.read_csv(self.output_csv)
    by_stmt = df.set_index('statement')
    # Within t1, B bridges both segments (rank 1) while A has minmax=0 (rank 2).
    self.assertEqual(by_stmt.loc['B', 'bridging_rank'], 1)
    self.assertEqual(by_stmt.loc['A', 'bridging_rank'], 2)
    # Within t2, D bridges both segments (rank 1) while C has minmax=0 (rank 2).
    self.assertEqual(by_stmt.loc['D', 'bridging_rank'], 1)
    self.assertEqual(by_stmt.loc['C', 'bridging_rank'], 2)
    self.assertIn('schulze_rank', df.columns)
    self.assertIn('pav_rank', df.columns)

  @mock.patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
  @mock.patch(
      'src.simulated_jury.simulated_jury.run_simulated_jury',
      new_callable=mock.AsyncMock,
  )
  def test_main_bridging_fails_fast_on_missing_segment_column(self, mock_run):
    test_args = [
        'main.py',
        '--participants_csv',
        self.participants_csv,
        '--statements_csv',
        self.statements_csv,
        '--output_csv',
        self.output_csv,
        '--segment_column',
        'missing_col',
    ]
    with mock.patch.object(sys, 'argv', test_args):
      with self.assertRaisesRegex(ValueError, 'missing_col'):
        main.main()

    mock_run.assert_not_called()


if __name__ == '__main__':
  unittest.main()
