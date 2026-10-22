# Copyright 2026 Google LLC
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

"""Tests for OpenAICompatibleModel."""

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from src.models.openai_compatible_model import OpenAICompatibleModel
import pandas as pd
from pydantic import BaseModel

# Disable logging for tests
logging.disable(logging.CRITICAL)


class DummySchema(BaseModel):
  field1: str
  field2: int


class OpenAICompatibleModelTest(unittest.TestCase):

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_init(self, mock_async_openai):
    model = OpenAICompatibleModel(
        model_name='test_model', endpoint_url='http://localhost:1234'
    )
    self.assertEqual(model.model, 'test_model')
    self.assertEqual(model.endpoint_url, 'http://localhost:1234')
    mock_async_openai.assert_called_once_with(
        api_key='dummy_key', base_url='http://localhost:1234'
    )

  @patch('src.models.openai_compatible_model.tqdm.asyncio.tqdm')
  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_process_prompts_concurrently_rejects_invalid_max_concurrent_calls(
      self, mock_async_openai, mock_tqdm
  ):
    """Invalid limits raise before any progress bar or workers are created."""
    model = OpenAICompatibleModel(
        model_name='test_model', endpoint_url='http://localhost:1234'
    )
    for value, error in ((0, ValueError), (-1, ValueError), (2.5, TypeError)):
      with self.subTest(value=value):
        with self.assertRaises(error):
          asyncio.run(
              model.process_prompts_concurrently(
                  [{'prompt': 'p'}],
                  response_parser=lambda resp, job: resp,
                  max_concurrent_calls=value,
              )
          )
    mock_tqdm.assert_not_called()

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_generate_content_success(self, mock_async_openai):
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create = AsyncMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = 'Success response'
    mock_response.choices = [mock_choice]
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10
    mock_usage.prompt_tokens = 4
    mock_usage.completion_tokens = 6
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    model = OpenAICompatibleModel(
        model_name='test_model', endpoint_url='http://localhost:1234'
    )
    result = asyncio.run(
        model.generate_content(prompt='test prompt', run_name='test_run')
    )

    self.assertEqual(result['text'], 'Success response')
    self.assertEqual(result['total_token_count'], 10)
    self.assertEqual(result['prompt_token_count'], 4)
    self.assertEqual(result['candidates_token_count'], 6)
    self.assertIsNone(result['error'])

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_generate_content_with_schema(self, mock_async_openai):
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create = AsyncMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"field1": "val", "field2": 1}'
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    model = OpenAICompatibleModel(
        model_name='test_model', endpoint_url='http://localhost:1234'
    )
    result = asyncio.run(
        model.generate_content(
            prompt='test prompt',
            run_name='test_run',
            response_schema=DummySchema,
        )
    )

    self.assertEqual(result['text'], '{"field1": "val", "field2": 1}')

    # Verify response_format was passed correctly
    args, kwargs = mock_client.chat.completions.create.call_args
    self.assertIn('response_format', kwargs)
    self.assertEqual(kwargs['response_format']['type'], 'json_schema')
    self.assertEqual(
        kwargs['response_format']['json_schema']['name'], 'DummySchema'
    )

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_generate_content_failure(self, mock_async_openai):
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception('API Error')
    )

    model = OpenAICompatibleModel(
        model_name='test_model', endpoint_url='http://localhost:1234'
    )
    result = asyncio.run(
        model.generate_content(prompt='test prompt', run_name='test_run')
    )

    self.assertIn('API Error', result['error'])

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_process_prompts_concurrently(self, mock_async_openai):
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create = AsyncMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = 'result'
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    model = OpenAICompatibleModel(
        model_name='test_model', endpoint_url='http://localhost:1234'
    )

    prompts = [{'prompt': 'p1'}, {'prompt': 'p2'}]

    def parser(resp, job):
      return resp['text']

    llm_response, llm_response_stats, wall_delay, duration = asyncio.run(
        model.process_prompts_concurrently(
            prompts, parser, max_concurrent_calls=2
        )
    )

    self.assertEqual(len(llm_response), 2)
    self.assertEqual(llm_response['result'].tolist(), ['result', 'result'])
    self.assertEqual(mock_client.chat.completions.create.call_count, 2)

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_generate_content_with_dict_schema_normalizes_uppercase_types(
      self, mock_async_openai
  ):
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create = AsyncMock()

    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"propositions": []}'
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_client.chat.completions.create.return_value = mock_response

    model = OpenAICompatibleModel(
        model_name='gemma3:270m', endpoint_url='http://localhost:11434/v1'
    )
    dict_schema = {
        'type': 'OBJECT',
        'properties': {
            'propositions': {
                'type': 'ARRAY',
                'items': {
                    'type': 'OBJECT',
                    'properties': {
                        'statement': {'type': 'STRING'},
                    },
                },
            }
        },
        'required': ['propositions'],
    }
    asyncio.run(
        model.generate_content(
            prompt='test prompt',
            run_name='test_run',
            response_schema=dict_schema,
        )
    )

    _, kwargs = mock_client.chat.completions.create.call_args
    passed_schema = kwargs['response_format']['json_schema']['schema']
    self.assertEqual(passed_schema['type'], 'object')
    self.assertEqual(
        passed_schema['properties']['propositions']['type'], 'array'
    )
    self.assertEqual(
        passed_schema['properties']['propositions']['items']['type'], 'object'
    )
    self.assertEqual(
        passed_schema['properties']['propositions']['items']['properties'][
            'statement'
        ]['type'],
        'string',
    )

  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_estimate_token_count_needed(self, mock_async_openai):
    model = OpenAICompatibleModel(
        model_name='gemma3:270m', endpoint_url='http://localhost:11434/v1'
    )
    token_count = model.estimate_token_count_needed('12345678')
    self.assertEqual(token_count, 2)

  @patch('src.models.openai_compatible_model.asyncio.sleep', new_callable=AsyncMock)
  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_process_prompts_concurrently_plain_text_failure_returns_empty_str(
      self, mock_async_openai, mock_sleep
  ):
    mock_client = mock_async_openai.return_value
    mock_client.chat.completions.create = AsyncMock(
        side_effect=Exception('API Error')
    )
    model = OpenAICompatibleModel(
        model_name='gemma3:270m',
        endpoint_url='http://localhost:11434/v1',
        max_llm_retries=1,
    )
    llm_response, _, _, _ = asyncio.run(
        model.process_prompts_concurrently(
            [{'prompt': 'p1'}],
            lambda resp, _: resp['text'],
            max_concurrent_calls=1,
        )
    )
    self.assertEqual(llm_response.iloc[0]['result'], '')
    self.assertNotIn('propositions', llm_response.columns)
    self.assertNotIn('allocations', llm_response.columns)

  @patch('src.models.openai_compatible_model.asyncio.sleep', new_callable=AsyncMock)
  @patch('src.models.openai_compatible_model.AsyncOpenAI')
  def test_retry_increments_temperature_by_002_and_tracks_quota_vs_non_quota(
      self, mock_async_openai, mock_sleep
  ):
    """Tests +0.02 temperature increment on retry and quota error exclusion from non_quota_failures."""
    mock_client = mock_async_openai.return_value
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = 'Success after retries'
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    quota_err = Exception('Error code: 429 - Rate limit reached')
    generic_err = Exception('Generic parsing or server error')
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[quota_err, generic_err, mock_response]
    )

    model = OpenAICompatibleModel(
        model_name='gemma3:270m',
        endpoint_url='http://localhost:11434/v1',
        max_llm_retries=3,
    )
    llm_response, llm_stats, _, _ = asyncio.run(
        model.process_prompts_concurrently(
            [{'prompt': 'p1', 'temperature': 0.0}],
            lambda resp, _: resp['text'],
            max_concurrent_calls=1,
        )
    )

    self.assertEqual(llm_response.iloc[0]['result'], 'Success after retries')
    self.assertEqual(llm_response.iloc[0]['temperature'], 0.04)
    self.assertEqual(llm_stats.iloc[0]['api_calls_made'], 3)
    self.assertEqual(llm_stats.iloc[0]['non_quota_failures'], 1)


if __name__ == '__main__':
  unittest.main()
