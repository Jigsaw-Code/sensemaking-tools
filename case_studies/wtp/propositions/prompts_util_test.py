import unittest
import pandas as pd
from case_studies.wtp.propositions import prompts_util


class PromptsUtilTest(unittest.TestCase):

  def test_find_prefix_num(self):
    row = pd.Series(index=['ranking_1_q_1', 'ranking_1_a_1', 'other_col'])
    self.assertEqual(prompts_util.find_prefix_num(row, 'ranking'), '1')
    self.assertIsNone(prompts_util.find_prefix_num(row, 'nonexistent'))

  def test_parse_proposition_response_json_reasoning(self):
    # Test valid JSON with reasoning
    json_string = (
        '[{"statement": "prop 1", "reasoning": "reason 1"}, {"statement": "prop'
        ' 2", "reasoning": "reason 2"}]'
    )
    expected_df = pd.DataFrame({
        'proposition': ['prop 1', 'prop 2'],
        'reasoning': ['reason 1', 'reason 2'],
    })
    pd.testing.assert_frame_equal(
        prompts_util.parse_proposition_response_json_reasoning(
            json_string, None
        ),
        expected_df,
    )

    # Test dictionary input
    resp = {
        'text': (
            '[{"statement": "prop 1", "reasoning": "reason 1"},'
            ' {"statement": "prop 2", "reasoning": "reason 2"}]'
        )
    }
    pd.testing.assert_frame_equal(
        prompts_util.parse_proposition_response_json_reasoning(resp, None),
        expected_df,
    )

    # Test invalid JSON
    invalid_json = '{"key": "value"'
    self.assertTrue(
        prompts_util.parse_proposition_response_json_reasoning(
            invalid_json, None
        ).empty
    )

    # Test empty list
    empty_list_json = '[]'
    self.assertTrue(
        prompts_util.parse_proposition_response_json_reasoning(
            empty_list_json, None
        ).empty
    )

  def test_parse_proposition_response_json(self):
    # Test valid JSON array of strings
    json_string = '["prop 1", "prop 2"]'
    expected_df = pd.DataFrame({'proposition': ['prop 1', 'prop 2']})
    pd.testing.assert_frame_equal(
        prompts_util.parse_proposition_response_json(
            {'text': json_string}, None
        ),
        expected_df,
    )

    # Test invalid JSON
    invalid_json = '{"key": "value"'
    self.assertTrue(
        prompts_util.parse_proposition_response_json(
            {'text': invalid_json}, None
        ).empty
    )

    # Test empty list
    empty_list_json = '[]'
    self.assertTrue(
        prompts_util.parse_proposition_response_json(
            {'text': empty_list_json}, None
        ).empty
    )

    # Test non-list JSON
    non_list_json = '{"key": "value"}'
    self.assertTrue(
        prompts_util.parse_proposition_response_json(
            {'text': non_list_json}, None
        ).empty
    )

  def test_extract_reusable_strings_ranking(self):
    df = pd.DataFrame({
        'ranking_1_q_1': ['Opinion A'],
        'ranking_1_q_2': ['Opinion B'],
    })
    header, mapping = prompts_util.extract_reusable_strings(
        df, prompts_util.QuestionType.RANKING
    )
    self.assertIn("<definitions type='ranking'>", header)
    self.assertIn('<A>Opinion A</A>', header)
    self.assertIn('<B>Opinion B</B>', header)
    self.assertEqual(mapping, {'Opinion A': 'A', 'Opinion B': 'B'})

  def test_extract_reusable_strings_freetext(self):
    df = pd.DataFrame({
        'question_2': ['Question A'],
        'question_3': ['Question B'],
    })
    header, mapping = prompts_util.extract_reusable_strings(
        df, prompts_util.QuestionType.FREE_TEXT
    )
    self.assertIn("<definitions type='freetext'>", header)
    self.assertIn('<1>Question A</1>', header)
    self.assertIn('<2>Question B</2>', header)
    self.assertEqual(mapping, {'Question A': '1', 'Question B': '2'})

  def test_extract_reusable_strings_single_statement(self):
    df = pd.DataFrame({'question_2': ['Question A']})
    header, _ = prompts_util.extract_reusable_strings(
        df, prompts_util.QuestionType.FREE_TEXT
    )
    self.assertEqual(header, '<statement>Question A</statement>\n')

  def test_extract_reusable_strings_empty(self):
    df = pd.DataFrame({'col1': [1]})
    header, mapping = prompts_util.extract_reusable_strings(
        df, prompts_util.QuestionType.RANKING
    )
    self.assertEqual(header, '')
    self.assertEqual(mapping, {})

  def test_build_free_text_response_prompt(self):
    row = pd.Series({
        'question_1': 'Q1',
        'answer_1': 'A1',
        'question_2': 'Q2',
        'answer_2': 'A2',
    })
    opinions_map = {'Q1': '1', 'Q2': '2'}

    # Standard case
    prompt = prompts_util.build_free_text_response_prompt(row, opinions_map)
    self.assertIn('<statement_id>1</statement_id>\n<answer>A1</answer>', prompt)
    self.assertIn('<statement_id>2</statement_id>\n<answer>A2</answer>', prompt)

  def test_build_free_text_response_prompt_single(self):
    row = pd.Series({'question_2': 'Q2', 'answer_2': 'A2'})
    prompt = prompts_util.build_free_text_response_prompt(row, {'Q2': '2'})
    self.assertEqual(prompt, 'A2')

  def test_build_ranking_response_prompt(self):
    row = pd.Series({
        'ranking_1_q_1': 'Opinion A',
        'ranking_1_a_1': 2,
        'ranking_1_q_2': 'Opinion B',
        'ranking_1_a_2': 1,
        'ranking_1_q_4': 'Follow-up',
        'ranking_1_a_4': 'Answer',
    })
    opinions_map = {'Opinion A': 'A', 'Opinion B': 'B'}
    prompt = prompts_util.build_ranking_response_prompt(row, opinions_map)
    self.assertIn('<ranking>B>A</ranking>', prompt)
    self.assertIn('<followup_question>Follow-up</followup_question>', prompt)
    self.assertIn('<answer>Answer</answer>', prompt)

  def test_build_ranking_response_prompt_no_followup(self):
    row = pd.Series({
        'ranking_1_q_1': 'Opinion A',
        'ranking_1_a_1': 2,
        'ranking_1_q_2': 'Opinion B',
        'ranking_1_a_2': 1,
    })
    prompt = prompts_util.build_ranking_response_prompt(
        row, {'Opinion A': 'A', 'Opinion B': 'B'}
    )
    self.assertIn('<ranking>B>A</ranking>', prompt)
    self.assertNotIn('followup', prompt)

  def test_regex_word_boundary(self):
    """Tests that the regex word boundary correctly distinguishes columns."""
    df = pd.DataFrame({
        'question_1': ['Question A'],
        'question_1_topic': ['Some other text'],
        'question_2': ['Question B'],
    })
    header, mapping = prompts_util.extract_reusable_strings(
        df, prompts_util.QuestionType.FREE_TEXT
    )
    self.assertIn('<1>Question A</1>', header)
    self.assertIn('<2>Question B</2>', header)
    self.assertNotIn('Some other text', header)
    self.assertEqual(mapping, {'Question A': '1', 'Question B': '2'})

    row = pd.Series({
        'question_1': 'Q1',
        'answer_1': 'A1',
        'question_1_topic': 'Some other text',
        'question_2': 'Q2',
        'answer_2': 'A2',
    })
    opinions_map = {'Q1': '1', 'Q2': '2'}
    prompt = prompts_util.build_free_text_response_prompt(row, opinions_map)
    self.assertIn('<statement_id>1</statement_id>\n<answer>A1</answer>', prompt)
    self.assertIn('<statement_id>2</statement_id>\n<answer>A2</answer>', prompt)
    self.assertNotIn('Some other text', prompt)


if __name__ == '__main__':
  unittest.main()
