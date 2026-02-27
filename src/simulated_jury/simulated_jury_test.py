"""
Tests for the simulated jury.
"""

import unittest
from src.simulated_jury import simulated_jury


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

  def test_build_approval_matrix_with_errors(self):
    import pandas as pd

    data = [
        {
            'data_row': {'participant_id': 'p1'},
            'result': {'error': 'Failed after 4 attempts'},
        },
        {'data_row': {'participant_id': 'p2'}, 'result': {'Prop1': True, 'Prop2': False}},
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
