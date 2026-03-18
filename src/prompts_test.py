import unittest
from unittest.mock import patch

import pandas as pd
from src import prompts


class PromptsTest(unittest.TestCase):

  def setUp(self):
    self.df_r1 = pd.DataFrame({
        'participant_id': [1, 2],
        'topic': ['Topic A', 'Topic A'],
        'opinion': ['Opinion A1', 'Opinion A2'],
        'quote': ['Rep text 1', 'Rep text 2'],
        'Q1_Text': ['Full Q1', 'Full Q1'],
        'Q1': ['Full A1', 'Full A1_2'],
    })
    self.df_r2 = pd.DataFrame({
        'participant_id': [1, 2]
        # Other columns will be mocked by prompts_util
    })
    self.opinion_list = ['Opinion A1', 'Opinion B1', 'Opinion C1']

  def test_generate_preamble_prompt(self):
    prompt = prompts.proposition_generation_generate_preamble_prompt(self.opinion_list)
    self.assertNotIn('<additionalContext>', prompt)
    self.assertIn('# Role and Objective', prompt)
    self.assertIn('<full_opinion_list>', prompt)
    self.assertIn('<opinion>Opinion A1</opinion>', prompt)
    self.assertIn('<opinion>Opinion B1</opinion>', prompt)
    self.assertIn('<opinion>Opinion C1</opinion>', prompt)
    self.assertIn('</full_opinion_list>', prompt)
    self.assertIn('<R1_DATA>', prompt)
    self.assertIn('<R2_DATA>', prompt)

  def test_generate_preamble_prompt_with_context(self):
    prompt = prompts.proposition_generation_generate_preamble_prompt(
        self.opinion_list, additional_context='This is a test context.'
    )
    self.assertIn('# Role and Objective', prompt)
    self.assertIn('<additionalContext>', prompt)
    self.assertIn('This is a test context.', prompt)

  def test_generate_instructions_prompt_with_reasoning(self):
    prompt = prompts.proposition_generation_generate_instructions_prompt(5, reasoning=True)
    self.assertIn('maximum of `5` **new** statements', prompt)
    self.assertIn('Provide Concise Reasoning', prompt)
    self.assertIn(
        '"reasoning": "This new statement was created because', prompt
    )

  def test_generate_instructions_prompt_without_reasoning(self):
    prompt = prompts.proposition_generation_generate_instructions_prompt(10, reasoning=False)
    self.assertIn('maximum of `10` **new** statements', prompt)
    self.assertNotIn('Provide Concise Reasoning', prompt)
    self.assertNotIn('reasoning', prompt)
    self.assertIn(
        '"A new, declarative statement capturing a missed theme."', prompt
    )

  def test_generate_instructions_prompt_with_include_opinion_false(self):
    prompt = prompts.proposition_generation_generate_instructions_prompt(5, include_opinion=False)
    self.assertIn('maximum of `5` statements', prompt)
    self.assertNotIn('**Keep the Original:**', prompt)
    self.assertNotIn('maximum of `5` **new** statements', prompt)
    self.assertIn(
        'You may return the original opinion in unedited or lightly edited'
        ' form',
        prompt,
    )

  def test_generate_instructions_prompt_with_include_opinion_true(self):
    prompt = prompts.proposition_generation_generate_instructions_prompt(5, include_opinion=True)
    self.assertIn('maximum of `5` **new** statements', prompt)
    self.assertIn('**Keep the Original:**', prompt)
    self.assertIn(
        'Return only the original opinion (in unedited form) as your statement',
        prompt,
    )

  def test_generate_r1_prompt_string_validation(self):
    with self.assertRaisesRegex(ValueError, 'user_id_column_name'):
      prompts.proposition_generation_generate_r1_prompt_string(self.df_r1, None, 'topic', 'opinion')
    with self.assertRaisesRegex(ValueError, 'topic'):
      prompts.proposition_generation_generate_r1_prompt_string(self.df_r1, 'participant_id', None, 'opinion')
    with self.assertRaisesRegex(ValueError, 'opinion'):
      prompts.proposition_generation_generate_r1_prompt_string(self.df_r1, 'participant_id', 'topic', None)
    with self.assertRaisesRegex(ValueError, 'quote'):
      prompts.proposition_generation_generate_r1_prompt_string(
          self.df_r1,
          'participant_id',
          'topic',
          'opinion',
          quote_column_name=None,
      )

  def test_generate_r1_prompt_string_defaults(self):
    prompt = prompts.proposition_generation_generate_r1_prompt_string(
        self.df_r1, 'participant_id', 'topic', 'opinion'
    )
    self.assertIn('<R1_DATA>', prompt)
    self.assertIn('<topic>Topic A</topic>', prompt)
    self.assertIn('<opinion>Opinion A1</opinion>', prompt)
    self.assertIn('<participant id=1>Rep text 1</participant>', prompt)
    self.assertIn('</R1_DATA>', prompt)
    self.assertNotIn(
        '<topic>Topic A</topic>\n<opinion>Opinion A1</opinion>\nRep text 1',
        prompt,
    )

  def test_generate_r1_prompt_string_no_sharding(self):
    prompt = prompts.proposition_generation_generate_r1_prompt_string(
        self.df_r1, 'participant_id', 'topic', 'opinion', should_use_opinion_sharding=False
    )
    self.assertEqual(prompt.count('<topic>'), 2)
    self.assertEqual(prompt.count('<opinion>'), 2)

  def test_generate_r1_prompt_string_full_text(self):
    prompt = prompts.proposition_generation_generate_r1_prompt_string(
        self.df_r1,
        'participant_id',
        'topic',
        'opinion',
        should_use_quote=False,
    )
    self.assertIn('<question_1>Full Q1</question_1>', prompt)
    self.assertIn('<answer_1>Full A1</answer_1>', prompt)
    self.assertNotIn('Rep text 1', prompt)

  @patch('src.propositions.prompts_util.extract_reusable_strings')
  @patch('src.propositions.prompts_util.build_free_text_response_prompt')
  def test_generate_r2_prompt_string_freetext_only(
      self, mock_build_freetext, mock_extract
  ):
    mock_extract.return_value = ('<definitions>...</definitions>', {'map': 'A'})
    mock_build_freetext.return_value = '<freetext_response>'

    prompt = prompts.proposition_generation_generate_r2_prompt_string(self.df_r2)

    mock_extract.assert_called_once()
    mock_build_freetext.assert_called()
    self.assertIn('<R2_DATA>', prompt)
    self.assertIn('<definitions>...</definitions>', prompt)
    self.assertIn('<freetext_response>', prompt)
    self.assertIn('</R2_DATA>', prompt)
    self.assertNotIn("<response type='freetext'>", prompt)

  @patch('src.propositions.prompts_util.extract_reusable_strings')
  @patch('src.propositions.prompts_util.build_free_text_response_prompt')
  @patch('src.propositions.prompts_util.build_ranking_response_prompt')
  def test_generate_r2_prompt_string_with_ranking(
      self, mock_build_ranking, mock_build_freetext, mock_extract
  ):
    # Let extract be called twice, once for freetext, once for ranking
    mock_extract.side_effect = [
        ('<freetext_defs>', {'map_f': 'A'}),
        ('<ranking_defs>', {'map_r': 'B'}),
    ]
    mock_build_freetext.return_value = '<freetext_response>'
    mock_build_ranking.return_value = '<ranking_response>'

    prompt = prompts.proposition_generation_generate_r2_prompt_string(
        self.df_r2, include_non_gov_sections=True
    )

    self.assertEqual(mock_extract.call_count, 2)
    mock_build_freetext.assert_called()
    mock_build_ranking.assert_called()
    self.assertIn('<R2_DATA>', prompt)
    self.assertIn("<response type='freetext'>", prompt)
    self.assertIn('<ranking_response>', prompt)
    self.assertIn('</R2_DATA>', prompt)

  def test_generate_r2_prompt_string_empty_df(self):
    prompt = prompts.proposition_generation_generate_r2_prompt_string(pd.DataFrame())
    self.assertEqual(prompt, '')


if __name__ == '__main__':
  unittest.main()
