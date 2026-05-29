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

import asyncio
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

from src.generate_report_text import generate_report_text


class GenerateReportTextTest(unittest.TestCase):

  def get_mock_model(self, mock_process_prompts):
    # Mock the call to GenaiModel.process_prompts_concurrently
    # so that it constructs summaries based on the input topic and opinion
    async def mock_async_function(*args, **kwargs):
      prompts = args[0]
      responses = []
      for i, p in enumerate(prompts):
        response = {
            'job_id': i,
            'topic': p['topic'],
        }
        if 'opinion' in p:
          response['opinion'] = p['opinion']
          response['result'] = f"{p['topic']}-{p['opinion']} summary"
        else:
          response['result'] = f"{p['topic']} summary"
        responses.append(response)
      return (pd.DataFrame(responses), pd.DataFrame(), 0.0, 1.0)

    mock_process_prompts.side_effect = mock_async_function
    mock_model = MagicMock()
    mock_model.process_prompts_concurrently = mock_process_prompts
    return mock_model

  def test_get_combined_report_data(self):
    overview_text = 'an overview'
    topic_summaries_df = pd.DataFrame({
        'topic': ['Topic1', 'Topic2'],
        'result': ['Topic1 Summary', 'Topic2 Summary'],
    })
    opinion_summaries_df = pd.DataFrame({
        'topic': ['Topic1', 'Topic2', 'Topic2'],
        'opinion': ['T1_Opinion1', 'T2_Opinion1', 'T2_Opinion2'],
        'result': [
            'T1_Opinion1 Summary',
            'T2_Opinion1 Summary',
            'T2_Opinion2 Summary',
        ],
    })
    data_with_opinions, data_without_opinions = (
        generate_report_text.get_combined_report_data(
            overview_text, topic_summaries_df, opinion_summaries_df
        )
    )
    self.assertEqual(
        data_with_opinions,
        {
            'text': 'an overview',
            'sub_contents': [
                {
                    'title': 'Topic1',
                    'text': 'Topic1 Summary',
                    'sub_contents': [
                        {'title': 'T1_Opinion1', 'text': 'T1_Opinion1 Summary'}
                    ],
                },
                {
                    'title': 'Topic2',
                    'text': 'Topic2 Summary',
                    'sub_contents': [
                        {'title': 'T2_Opinion1', 'text': 'T2_Opinion1 Summary'},
                        {'title': 'T2_Opinion2', 'text': 'T2_Opinion2 Summary'},
                    ],
                },
            ],
        },
    )
    self.assertEqual(
        data_without_opinions,
        {
            'text': 'an overview',
            'sub_contents': [
                {'title': 'Topic1', 'text': 'Topic1 Summary'},
                {'title': 'Topic2', 'text': 'Topic2 Summary'},
            ],
        },
    )

  @patch('src.models.genai_model.GenaiModel.process_prompts_concurrently')
  def test_generate_opinion_summaries(self, mock_process_prompts):
    mock_model = self.get_mock_model(mock_process_prompts)

    # input contains 4 quotes across 3 opinions
    categorized_quotes_df = pd.DataFrame({
        'topic': ['t1', 't2', 't2', 't2'],
        'opinion': ['o1', 'o2', 'o3', 'o3'],
        'quote': ['q1', 'q2', 'q3', 'q4'],
    })
    opinion_summaries_df = asyncio.run(
        generate_report_text.generate_opinion_summaries(
            mock_model, categorized_quotes_df, ''
        )
    )
    # output dataframe should have 1 row per opinion
    pd.testing.assert_frame_equal(
        opinion_summaries_df,
        pd.DataFrame({
            'job_id': [0, 1, 2],
            'topic': ['t1', 't2', 't2'],
            'opinion': ['o1', 'o2', 'o3'],
            'result': ['t1-o1 summary', 't2-o2 summary', 't2-o3 summary'],
        }),
    )

  @patch('src.models.genai_model.GenaiModel.process_prompts_concurrently')
  def test_generate_topic_summaries(self, mock_process_prompts):
    mock_model = self.get_mock_model(mock_process_prompts)

    # input contains 4 quotes across 3 opinions
    categorized_quotes_df = pd.DataFrame({
        'topic': ['t1', 't2', 't2', 't2'],
        'opinion': ['o1', 'o2', 'o3', 'o3'],
        'quote': ['q1', 'q2', 'q3', 'q4'],
    })
    # and opinion summaries for each opinion
    opinion_summaries_df = pd.DataFrame({
        'topic': ['t1', 't2', 't2'],
        'opinion': ['o1', 'o2', 'o3'],
        'result': ['t1-o1 summary', 't2-o2 summary', 't2-o3 summary'],
    })
    topic_summaries_df = asyncio.run(
        generate_report_text.generate_topic_summaries(
            mock_model, categorized_quotes_df, opinion_summaries_df, ''
        )
    )
    # output dataframe should have 1 row per opinion
    pd.testing.assert_frame_equal(
        topic_summaries_df,
        pd.DataFrame({
            'job_id': [0, 1],
            'topic': ['t1', 't2'],
            'result': ['t1 summary', 't2 summary'],
        }),
    )


if __name__ == '__main__':
  unittest.main()
