"""
Tests for the simulated jury.
"""

import asyncio
import logging
import unittest
from unittest import mock

import pandas as pd

from src.models import decision
from src.models import fake_base_model
from src.simulated_jury import simulated_jury


class _FakeModel(fake_base_model.FakeBaseModel):
  """Minimal async model stub for simulated_jury unit tests."""

  async def process_prompts_concurrently(
      self,
      jobs,
      response_parser=None,
      max_concurrent_calls=None,
      retry_attempts=0,
  ):
    del response_parser, max_concurrent_calls, retry_attempts
    rows = [
        {
            'job_id': job['job_id'],
            'participant_rid': job['participant_rid'],
            'data_row': job['data_row'],
            'result': {stmt: True for stmt in job['shuffled_statements']},
            'topic': job['topic'],
            'opinion': job['opinion'],
        }
        for job in jobs
    ]
    stats = pd.DataFrame([
        {
            'job_id': job['job_id'],
            'is_complete_failure': False,
            'non_quota_failures': 0,
        }
        for job in jobs
    ])
    return pd.DataFrame(rows), stats, None, None


class SimulatedJuryTest(unittest.TestCase):

  def test_generate_rank_vote_prompt(self):
    participant_id = '123'
    participant_comment = 'I think we should focus on renewable energy.'
    statements = ['Statement A', 'Statement B', 'Statement C']
    prompt = simulated_jury.generate_vote_prompt(
        participant_id,
        participant_comment,
        statements,
        simulated_jury.VotingMode.RANK,
    )
    self.assertIn(
        'A participant in a public deliberation expressed the following'
        ' opinions:',
        prompt,
    )
    self.assertIn('<participant id="123">', prompt)
    self.assertIn('I think we should focus on renewable energy.', prompt)
    self.assertIn('Statements:', prompt)
    self.assertIn('A. Statement A', prompt)
    self.assertIn('B. Statement B', prompt)
    self.assertIn('C. Statement C', prompt)
    self.assertIn(
        'rank these statements in the order that the participant would most'
        ' likely agree with them',
        prompt,
    )

  def test_generate_approval_vote_prompt(self):
    participant_id = '123'
    participant_comment = 'I think we should focus on renewable energy.'
    statements = ['Statement A', 'Statement B', 'Statement C']
    prompt = simulated_jury.generate_vote_prompt(
        participant_id,
        participant_comment,
        statements,
        simulated_jury.VotingMode.APPROVAL,
    )
    self.assertIn(
        'A participant in a public deliberation expressed the following'
        ' opinions:',
        prompt,
    )
    self.assertIn('<participant id="123">', prompt)
    self.assertIn('I think we should focus on renewable energy.', prompt)
    self.assertIn('Statements:', prompt)
    self.assertIn('A. Statement A', prompt)
    self.assertIn('B. Statement B', prompt)
    self.assertIn('C. Statement C', prompt)
    self.assertIn(
        "Task: As an AI assistant, your job is to predict the participant's"
        ' vote on each of the following statements based on their opinion.',
        prompt,
    )
    self.assertIn('"Neither Agree nor Disagree"', prompt)

  def test_parse_llm_ranking_responses(self):
    results = [
        '{"ranking": ["A", "B", "C"], "reasoning": "reasoning"}',
        '{"ranking": ["B", "A", "C"], "reasoning": "reasoning"}',
        '{"ranking": ["C", "A", "B"], "reasoning": "reasoning"}',
    ]
    job = {'shuffled_statements': ['Statement A', 'Statement B', 'Statement C']}
    preferences = []
    for result in results:
      preferences.append(
          simulated_jury.parse_llm_ranking_response({'text': result}, job)
      )

    self.assertEqual(len(preferences), 3)
    self.assertEqual(
        preferences[0]['ranking'], ['Statement A', 'Statement B', 'Statement C']
    )
    self.assertEqual(
        preferences[1]['ranking'], ['Statement B', 'Statement A', 'Statement C']
    )
    self.assertEqual(
        preferences[2]['ranking'], ['Statement C', 'Statement A', 'Statement B']
    )

  def test_parse_llm_ranking_responses_with_extra_text(self):
    results = [
        '{"ranking": ["G", "F", "J", "E", "D", "K", "N", "A", "B", "C", "H",'
        ' "I", "L", "M", "O"], "reasoning": "Blah blah"}'
    ]
    job = {
        'shuffled_statements': [
            'Statement A',
            'Statement B',
            'Statement C',
            'Statement D',
            'Statement E',
            'Statement F',
            'Statement G',
            'Statement H',
            'Statement I',
            'Statement J',
            'Statement K',
            'Statement L',
            'Statement M',
            'Statement N',
            'Statement O',
        ]
    }
    preferences = []
    for result in results:
      preferences.append(
          simulated_jury.parse_llm_ranking_response({'text': result}, job)
      )

    self.assertEqual(len(preferences), 1)
    self.assertEqual(
        preferences[0]['ranking'],
        [
            'Statement G',
            'Statement F',
            'Statement J',
            'Statement E',
            'Statement D',
            'Statement K',
            'Statement N',
            'Statement A',
            'Statement B',
            'Statement C',
            'Statement H',
            'Statement I',
            'Statement L',
            'Statement M',
            'Statement O',
        ],
    )

  def test_parse_llm_ranking_response_missing_closing_tag(self):
    result = '<answer>reasoning<sep>A > B > C'
    job = {'shuffled_statements': ['Statement A', 'Statement B', 'Statement C']}
    with self.assertRaises(ValueError):
      simulated_jury.parse_llm_ranking_response({'text': result}, job)

  def test_parse_llm_ranking_response_empty_thinking(self):
    result = '{"ranking": ["A", "B", "C"], "reasoning": ""}'
    job = {'shuffled_statements': ['Statement A', 'Statement B', 'Statement C']}
    preference = simulated_jury.parse_llm_ranking_response(
        {'text': result}, job
    )
    self.assertEqual(
        preference['ranking'], ['Statement A', 'Statement B', 'Statement C']
    )

  def test_parse_llm_ranking_response_missing_thinking(self):
    result = '{"ranking": ["A", "B", "C"]}'
    job = {'shuffled_statements': ['Statement A', 'Statement B', 'Statement C']}
    preference = simulated_jury.parse_llm_ranking_response(
        {'text': result}, job
    )
    self.assertEqual(
        preference['ranking'], ['Statement A', 'Statement B', 'Statement C']
    )

  def test_parse_llm_ranking_response_empty_answer(self):
    result = '<thinking>reasoning</thinking><answer></answer>'
    job = {'shuffled_statements': ['Statement A', 'Statement B', 'Statement C']}
    with self.assertRaises(ValueError):
      simulated_jury.parse_llm_ranking_response({'text': result}, job)

  def test_batched_implementation(self):
    # Test with an iterable that is a multiple of the batch size
    self.assertEqual(
        list(simulated_jury.batched('ABCDEF', 3)),
        [('A', 'B', 'C'), ('D', 'E', 'F')],
    )
    # Test with an iterable that is not a multiple of the batch size
    self.assertEqual(
        list(simulated_jury.batched('ABCDEFG', 3)),
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)],
    )
    # Test with an empty iterable
    self.assertEqual(list(simulated_jury.batched('', 3)), [])
    # Test with a batch size of 1
    self.assertEqual(
        list(simulated_jury.batched('ABC', 1)), [('A',), ('B',), ('C',)]
    )
    # Test with a batch size larger than the iterable
    self.assertEqual(list(simulated_jury.batched('ABC', 5)), [('A', 'B', 'C')])
    # Test that it raises a ValueError for n < 1
    with self.assertRaises(ValueError):
      list(simulated_jury.batched('ABC', 0))

  def test_parse_llm_ranking_response_missing_answer_tags(self):
    result = '<thinking>reasoning</thinking>A > B > C'
    job = {'shuffled_statements': ['Statement A', 'Statement B', 'Statement C']}
    with self.assertRaises(ValueError):
      simulated_jury.parse_llm_ranking_response({'text': result}, job)

  def test_parse_llm_ranking_response_partial_on_last_attempt(self):
    result = '{"ranking": ["A", "A", "B"], "reasoning": "dup letter"}'
    job_early = {
        'shuffled_statements': ['Statement A', 'Statement B', 'Statement C'],
        'current_attempt': 0,
        'retry_attempts': 3,
    }
    with self.assertRaises(ValueError):
      simulated_jury.parse_llm_ranking_response({'text': result}, job_early)
    job_last = {
        'shuffled_statements': ['Statement A', 'Statement B', 'Statement C'],
        'current_attempt': 2,
        'retry_attempts': 3,
    }
    with mock.patch.object(simulated_jury.logging, 'warning') as mock_warn:
      preference = simulated_jury.parse_llm_ranking_response(
          {'text': result}, job_last
      )
    self.assertEqual(preference['ranking'], ['Statement A', 'Statement B'])
    mock_warn.assert_called_once()

  def test_parse_llm_approval_response_incomplete_retries_and_defaults(self):
    result = (
        '{"votes": [{"statement_letter": "A", "vote": "Agree"},'
        ' {"statement_letter": "A", "vote": "Agree"}], "reasoning": "dup"}'
    )
    job_early = {
        'shuffled_statements': ['Statement A', 'Statement B'],
        'current_attempt': 0,
        'retry_attempts': 3,
    }
    with self.assertRaises(ValueError):
      simulated_jury.parse_llm_approval_response({'text': result}, job_early)
    job_last = {
        'shuffled_statements': ['Statement A', 'Statement B'],
        'current_attempt': 2,
        'retry_attempts': 3,
    }
    with mock.patch.object(simulated_jury.logging, 'warning') as mock_warn:
      approvals = simulated_jury.parse_llm_approval_response(
          {'text': result}, job_last
      )
    self.assertEqual(approvals, {'Statement A': True, 'Statement B': False})
    mock_warn.assert_called_once()

  def test_generate_approval_vote_prompt_with_additional_context(self):
    prompt = simulated_jury.generate_vote_prompt(
        '123',
        'I want better bike lanes.',
        ['Statement A'],
        simulated_jury.VotingMode.APPROVAL,
        additional_context='Evaluate these as municipal capital proposals.',
    )
    self.assertIn(
        '<additionalContext>\n'
        'Evaluate these as municipal capital proposals.\n'
        '</additionalContext>',
        prompt,
    )

  def test_generate_vote_prompt_omits_empty_additional_context(self):
    prompt = simulated_jury.generate_vote_prompt(
        '123',
        'I want better bike lanes.',
        ['Statement A'],
        simulated_jury.VotingMode.APPROVAL,
        additional_context='',
    )
    self.assertNotIn('<additionalContext>', prompt)

  def test_generate_rank_vote_prompt_with_additional_context(self):
    prompt = simulated_jury.generate_vote_prompt(
        '123',
        'I want better bike lanes.',
        ['Statement A', 'Statement B'],
        simulated_jury.VotingMode.RANK,
        additional_context='Rank these by urgency.',
    )
    self.assertIn(
        '<additionalContext>\nRank these by urgency.\n</additionalContext>',
        prompt,
    )

  def test_run_simulated_jury_warns_on_non_canonical_scale(self):
    previous_level = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    self.addCleanup(logging.disable, previous_level)

    participants_df = pd.DataFrame(
        [{'participant_id': 'p1', 'survey_text': 'I support transit.'}]
    )
    with decision.use_decision_client(None):
      with self.assertLogs(level='WARNING') as captured_logs:
        asyncio.run(
            simulated_jury.run_simulated_jury(
                participants_df=participants_df,
                statements=['Expand bus routes.'],
                voting_mode=simulated_jury.VotingMode.APPROVAL,
                model=_FakeModel(),
                approval_scale=simulated_jury.ApprovalScale.AGREE_DISAGREE,
            )
        )

    self.assertTrue(
        any('non-canonical scale' in entry for entry in captured_logs.output)
    )

  def test_approval_scale_values_and_options(self):
    scale_values = simulated_jury.ApprovalScale.values()
    self.assertTrue(all(isinstance(v, str) for v in scale_values))
    self.assertEqual(
        scale_values,
        [
            'agree_disagree',
            'agree_disagree_neither',
            'likert_5',
            'likert_5_somewhat',
            'likert_4',
            'likert_4_somewhat',
        ],
    )
    for scale in simulated_jury.ApprovalScale:
      options = simulated_jury.ApprovalScale.get_options(scale)
      self.assertIsInstance(options, list)
      self.assertGreaterEqual(len(options), 2)

  def test_build_approval_matrix_with_errors(self):
    import pandas as pd

    data = [
        {
            'data_row': {'participant_id': 'p1'},
            'result': {'error': 'Failed after 4 attempts'},
        },
        {
            'data_row': {'participant_id': 'p2'},
            'result': {'Prop1': True, 'Prop2': False},
        },
        {
            'data_row': {'participant_id': 'p3'},
            'result': {
                'Prop1': False,
                'Prop2': True,
                'error': 'Partial failure',
            },
        },
    ]
    df = pd.DataFrame(data)
    matrix = simulated_jury.build_approval_matrix(df)

    self.assertEqual(matrix.shape, (2, 2))
    self.assertIn('p2', matrix.index)
    self.assertIn('p3', matrix.index)
    self.assertNotIn('p1', matrix.index)
    self.assertEqual(matrix.loc['p2', 'Prop1'], 1)
    self.assertEqual(matrix.loc['p2', 'Prop2'], 0)
    self.assertEqual(matrix.loc['p3', 'Prop1'], 0)
    self.assertEqual(matrix.loc['p3', 'Prop2'], 1)


if __name__ == '__main__':
  unittest.main()
