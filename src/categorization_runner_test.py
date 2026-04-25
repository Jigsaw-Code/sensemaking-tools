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

import argparse
import asyncio
import io
import unittest
from unittest import mock

from src import categorization_runner
from src.models import custom_types


class CategorizationRunnerTest(unittest.TestCase):

  @mock.patch(
      'os.path.expanduser',
      return_value=io.StringIO(
          """participant_id,survey_text\n1,statement 1\n2,statement 2"""
      ),
  )
  @mock.patch('os.path.exists', return_value=True)
  def test_read_csv_to_dicts(self, mock_expanduser, mock_exists):
    # pylint: disable=protected-access
    output = categorization_runner._read_csv_to_dicts('dummy_path')
    mock_exists.assert_called()
    mock_expanduser.assert_called()
    self.assertEqual(
        output,
        [
            {
                'participant_id': '1',
                'survey_text': 'statement 1',
            },
            {
                'participant_id': '2',
                'survey_text': 'statement 2',
            },
        ],
    )

  def test_convert_csv_rows_to_statements(self):
    csv_rows = [
        {
            'participant_id': '1',
            'survey_text': 'statement 1',
        },
        {
            'participant_id': '2',
            'survey_text': 'statement 2',
        },
    ]
    statements = categorization_runner._convert_csv_rows_to_statements(csv_rows)
    self.assertEqual(
        statements,
        [
            custom_types.Statement(
                id='1',
                text='statement 1',
            ),
            custom_types.Statement(
                id='2',
                text='statement 2',
            ),
        ],
    )

  def test_convert_csv_rows_to_statements_missing_participant_id(self):
    csv_rows = [
        {
            'participant_id': '1',
            'survey_text': 'statement 1',
        },
        {
            'participant_id': '',
            'survey_text': 'empty participant id should be dropped',
        },
        {
            'participant_id': '2',
            'survey_text': 'statement 2',
        },
    ]
    with self.assertRaisesRegex(
        ValueError, "Row 2 is missing 'participant_id'"
    ):
      categorization_runner._convert_csv_rows_to_statements(csv_rows)

  @mock.patch('src.runner_utils.generate_and_save_topic_tree')
  def test_process_and_print_topic_tree(self, mock_generate_and_save):
    output_csv_rows = [
        {
            'participant_id': 's1',
            'topic': 'Topic 1',
            'opinion': 'Opinion 1',
            'quote': 'quote 1',
        },
        {
            'participant_id': 's2',
            'topic': 'Topic 1',
            'opinion': 'Opinion 1',
            'quote': 'quote 2',
        },
    ]
    output_file_base = '/tmp/test'

    categorization_runner._process_and_print_topic_tree(
        output_csv_rows, output_file_base
    )

    # Check that the save function was called with the correct data structure
    self.assertEqual(mock_generate_and_save.call_count, 1)
    args, _ = mock_generate_and_save.call_args
    json_data = args[0]
    self.assertEqual(len(json_data), 1)
    topic = json_data[0]
    self.assertEqual(topic['topic_name'], 'Topic 1')
    self.assertEqual(len(topic['opinions']), 1)
    opinion = topic['opinions'][0]
    self.assertEqual(opinion['opinion_text'], 'Opinion 1')
    self.assertEqual(len(opinion['quotes']), 2)
    self.assertIn('quote 1', opinion['quotes'])
    self.assertIn('quote 2', opinion['quotes'])

  def test_set_topics_on_csv_rows_opinion_categorization(self):
    original_csv_rows = [
        {'participant_id': '1', 'survey_text': 'Statement 1'},
    ]
    categorized_statements = [
        custom_types.Statement(
            id='1',
            text='Statement 1',
            topics=[
                custom_types.NestedTopic(
                    name='Topic A',
                    subtopics=[custom_types.FlatTopic(name='Opinion A')],
                )
            ],
            quotes=[
                custom_types.Quote(
                    id='1-Topic A',
                    text='Quote 1',
                    topic=custom_types.NestedTopic(
                        name='Topic A',
                        subtopics=[custom_types.FlatTopic(name='Opinion A')],
                    ),
                )
            ],
        )
    ]
    output_rows = categorization_runner._set_topics_on_csv_rows(
        original_csv_rows, categorized_statements
    )
    self.assertEqual(len(output_rows), 1)
    row = output_rows[0]
    self.assertIn('quote', row)
    self.assertEqual(row['quote'], 'Quote 1')
    self.assertIn('topic', row)
    self.assertEqual(row['topic'], 'Topic A')
    self.assertIn('opinion', row)
    self.assertEqual(row['opinion'], 'Opinion A')

  def test_set_topics_on_csv_rows_multiple_opinions(self):
    original_csv_rows = [
        {'participant_id': '1', 'survey_text': 'Statement 1'},
    ]
    categorized_statements = [
        custom_types.Statement(
            id='1',
            text='Statement 1',
            topics=[
                custom_types.NestedTopic(
                    name='Topic A',
                    subtopics=[
                        custom_types.FlatTopic(name='Opinion A'),
                        custom_types.FlatTopic(name='Opinion B'),
                    ],
                )
            ],
            quotes=[
                custom_types.Quote(
                    id='1-Topic A',
                    text='Quote 1',
                    topic=custom_types.NestedTopic(
                        name='Topic A',
                        subtopics=[
                            custom_types.FlatTopic(name='Opinion A'),
                            custom_types.FlatTopic(name='Opinion B'),
                        ],
                    ),
                )
            ],
        )
    ]
    output_rows = categorization_runner._set_topics_on_csv_rows(
        original_csv_rows, categorized_statements
    )
    self.assertEqual(len(output_rows), 2)

    opinions = {row['opinion'] for row in output_rows}
    self.assertEqual(opinions, {'Opinion A', 'Opinion B'})

    for row in output_rows:
      self.assertEqual(row['participant_id'], '1')
      self.assertEqual(row['survey_text'], 'Statement 1')
      self.assertEqual(row['quote'], 'Quote 1')
      self.assertEqual(row['topic'], 'Topic A')

  def test_set_topics_on_csv_rows_with_brackets_in_quote(self):
    original_csv_rows = [
        {'participant_id': '1', 'survey_text': 'Statement 1'},
    ]
    categorized_statements = [
        custom_types.Statement(
            id='1',
            text='Statement 1',
            topics=[],
            quotes=[
                custom_types.Quote(
                    id='1-Topic A',
                    text='[a quote with brackets] and [...] some [s---]',
                    topic=custom_types.NestedTopic(
                        name='Topic A',
                        subtopics=[custom_types.FlatTopic(name='Opinion A')],
                    ),
                )
            ],
        )
    ]
    output_rows = categorization_runner._set_topics_on_csv_rows(
        original_csv_rows, categorized_statements
    )
    self.assertEqual(
        output_rows,
        [{
            'participant_id': '1',
            'survey_text': 'Statement 1',
            'quote': 'a quote with brackets and ... some s---',
            'quote_with_brackets': (
                '[a quote with brackets] and [...] some [s---]'
            ),
            'topic': 'Topic A',
            'opinion': 'Opinion A',
        }],
    )

  def test_set_topics_on_csv_rows_opinion_categorization_flat_topic_fallback(
      self,
  ):
    """Tests that we don't crash if a quote has a FlatTopic."""
    original_csv_rows = [
        {'participant_id': '1', 'survey_text': 'Statement 1'},
    ]
    categorized_statements = [
        custom_types.Statement(
            id='1',
            text='Statement 1',
            topics=[],
            quotes=[
                custom_types.Quote(
                    id='1-Topic A',
                    text='Quote 1',
                    topic=custom_types.FlatTopic(name='Topic A'),
                )
            ],
        )
    ]
    output_rows = categorization_runner._set_topics_on_csv_rows(
        original_csv_rows, categorized_statements
    )
    # Since FlatTopic has no opinions, and we iterate over opinions to make rows,
    # we expect 0 rows for this quote.
    self.assertEqual(len(output_rows), 0)

  @mock.patch('src.categorization_runner.genai_model.GenaiModel')
  @mock.patch('src.categorization_runner.sensemaker.Sensemaker')
  @mock.patch('src.categorization_runner.runner_utils')
  @mock.patch('src.categorization_runner._convert_csv_rows_to_statements')
  @mock.patch('src.categorization_runner._read_csv_to_dicts')
  @mock.patch('argparse.ArgumentParser.parse_args')
  def test_main_stops_on_skipped_statements(
      self,
      mock_parse_args,
      mock_read_csv,
      mock_convert,
      mock_runner_utils,
      mock_sensemaker_cls,
      _mock_genai_model_cls,
  ):
    # Setup mocks
    mock_parse_args.return_value = argparse.Namespace(
        output_dir='/tmp/output',
        input_file='/tmp/input.csv',
        topics=None,
        topic_and_opinion_csv=None,
        model_name='gemini-pro',
        force_rerun=False,
        log_level='INFO',
        skip_autoraters=False,
        max_llm_retries=None,
    )
    mock_read_csv.return_value = [
        {'participant_id': '1', 'survey_text': 'test'}
    ]
    mock_convert.return_value = [
        custom_types.Statement(id='1', text='test', topics=[], quotes=[])
    ]

    # Simulate skipping statements
    mock_statement = custom_types.Statement(
        id='1', text='test', topics=[], quotes=[]
    )
    mock_runner_utils.filter_large_statements.return_value = (
        [],  # valid statements
        [mock_statement],  # skipped statements
    )

    asyncio.run(categorization_runner.main())

    # Verify that we tried to write the skipped rows
    mock_runner_utils.write_dicts_to_csv.assert_called()
    call_args = mock_runner_utils.write_dicts_to_csv.call_args
    self.assertIn('skipped_rows.csv', call_args[0][1])

    # Verify that Sensemaker was NOT initialized or used (process stopped)
    mock_sensemaker_cls.assert_not_called()


if __name__ == '__main__':
  unittest.main()
