import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import pandas as pd
import logging

from models import genai_model

# Disable logging for tests
logging.disable(logging.CRITICAL)


# Mock the entire google.genai library
@patch('google.genai.Client')
class GenaiModelInitTest(unittest.TestCase):

  def test_init_safety_filters_off(self, mock_genai_client):
    """Tests that safety filters are set to BLOCK_NONE when safety_filters_on=False."""
    model = genai_model.GenaiModel(
        api_key='test_key', model_name='test_model', safety_filters_on=False
    )

    settings = model.safety_settings
    self.assertEqual(len(settings), 4)
    for setting in settings:
      self.assertEqual(setting.threshold.name, 'BLOCK_NONE')

  def test_init_safety_filters_on(self, mock_genai_client):
    """Tests that safety filters are set to BLOCK_ONLY_HIGH when safety_filters_on=True."""
    model = genai_model.GenaiModel(
        api_key='test_key', model_name='test_model', safety_filters_on=True
    )

    settings = model.safety_settings
    self.assertEqual(len(settings), 4)
    for setting in settings:
      self.assertEqual(setting.threshold.name, 'BLOCK_ONLY_HIGH')


@patch('google.genai.Client')
class GenaiModelAsyncMethodsTest(unittest.TestCase):

  def setUp(self):
    self.prompts = [
        {'job_id': 0, 'opinion': 'Opinion 1', 'prompt': 'p1'},
        {'job_id': 1, 'opinion': 'Opinion 2', 'prompt': 'p2'},
        {'job_id': 2, 'opinion': 'Opinion 3', 'prompt': 'p3'},
    ]

  def test_calculate_token_count_needed(self, mock_genai_client):
    """Tests that the token count is correctly returned."""
    mock_client_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.total_tokens = 123
    mock_client_instance.models.count_tokens.return_value = mock_response

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    token_count = model.calculate_token_count_needed(prompt='test prompt')

    self.assertEqual(token_count, 123)
    mock_client_instance.models.count_tokens.assert_called_once()

  def test_call_gemini_success(self, mock_genai_client):
    """Tests a successful call to the Gemini API."""
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content = AsyncMock()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason.name = 'STOP'
    mock_candidate.content.parts = [MagicMock()]
    mock_candidate.content.parts[0].text = 'Success response'
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.total_token_count = 456
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    result = asyncio.run(model._call_gemini(prompt='test', run_name='test_run'))

    self.assertEqual(result['text'], 'Success response')
    self.assertEqual(result['input_token_count'], 456)
    self.assertIsNone(result['error'])

  def test_call_gemini_safety_failure(self, mock_genai_client):
    """Tests an API call blocked due to safety reasons."""
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content = AsyncMock()

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.finish_reason.name = 'SAFETY'
    mock_candidate.finish_message = 'Blocked for safety'
    mock_candidate.token_count = 0
    mock_response.candidates = [mock_candidate]
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    result = asyncio.run(model._call_gemini(prompt='test', run_name='test_run'))

    self.assertEqual(result['error'], 'SAFETY')
    self.assertEqual(result['finish_message'], 'Blocked for safety')

  def test_call_gemini_empty_prompt(self, mock_genai_client):
    """Tests a call to the Gemini API with an empty prompt."""
    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    with self.assertRaises(ValueError):
      asyncio.run(model._call_gemini(prompt=None, run_name='test_run'))

  def test_call_gemini_no_candidate(self, mock_genai_client):
    """Tests a call to the Gemini API with no candidates."""
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content = AsyncMock()

    mock_response = MagicMock()
    mock_response.candidates = []
    mock_response.prompt_feedback = '<test> No candidates found'
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )
    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    result = asyncio.run(model._call_gemini(prompt='test', run_name='test_run'))
    self.assertEqual(result['error'], '<test> No candidates found')

  @patch('models.genai_model.GenaiModel._call_gemini')
  def test_process_prompts_all_succeed(
      self, mock_call_gemini, mock_genai_client
  ):
    """Tests when all jobs succeed on the first attempt."""
    mock_call_gemini.return_value = {
        'text': 'parsed',
        'input_token_count': 10,
        'error': None,
    }

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')

    def simple_parser(text, job):
      return f"parsed_{job['opinion']}"

    results_df, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts, simple_parser, delay_between_calls_seconds=0
        )
    )

    self.assertEqual(len(results_df), 3)
    self.assertEqual(mock_call_gemini.call_count, 3)
    self.assertTrue(all(results_df['failed_tries'].apply(lambda d: d.empty)))
    self.assertIn('parsed_Opinion 1', results_df['result'].tolist())

  @patch('models.genai_model.GenaiModel._call_gemini')
  def test_process_prompts_with_retry(
      self, mock_call_gemini, mock_genai_client
  ):
    """Tests when one job succeeds after a retry."""
    # Fail for Opinion 2 on the first call, then succeed
    mock_call_gemini.side_effect = [
        {'text': 'parsed', 'input_token_count': 10, 'error': None},  # Job 1
        Exception('API Error'),  # Job 2, Attempt 1
        {'text': 'parsed', 'input_token_count': 10, 'error': None},  # Job 3
        {
            'text': 'parsed_retry',
            'input_token_count': 10,
            'error': None,
        },  # Job 2, Attempt 2
    ]

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')

    def simple_parser(text, job):
      return f"{text}_{job['opinion']}"

    results_df, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts,
            simple_parser,
            initial_retry_delay=0.01,
            delay_between_calls_seconds=0,
        )
    )

    self.assertEqual(len(results_df), 3)
    self.assertEqual(mock_call_gemini.call_count, 4)

    failed_job_row = results_df[results_df['opinion'] == 'Opinion 2'].iloc[0]
    self.assertEqual(len(failed_job_row['failed_tries']), 1)
    self.assertEqual(failed_job_row['result'], 'parsed_retry_Opinion 2')

  @patch('models.genai_model.GenaiModel._call_gemini')
  def test_process_prompts_permanent_failure(
      self, mock_call_gemini, mock_genai_client
  ):
    """Tests when one job fails all retry attempts."""
    mock_call_gemini.side_effect = [
        {'text': 'parsed', 'input_token_count': 10, 'error': None},  # Job 1
        Exception('API Error 1'),  # Job 2, Attempt 1
        {'text': 'parsed', 'input_token_count': 10, 'error': None},  # Job 3
        Exception('API Error 2'),  # Job 2, Attempt 2
    ]

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    results_df, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts,
            lambda t, j: t,
            retry_attempts=2,
            initial_retry_delay=0.01,
            delay_between_calls_seconds=0,
        )
    )

    self.assertEqual(len(results_df), 2)
    self.assertEqual(mock_call_gemini.call_count, 4)
    self.assertNotIn('Opinion 2', results_df['opinion'].tolist())

  @patch('models.genai_model.GenaiModel._log_retry_summary')
  def test_log_retry_summary(self, mock_log, mock_genai_client):
    """Tests that the retry summary is logged correctly."""
    failed_tries_data = [
        pd.DataFrame(),
        pd.DataFrame([{'attempt_index': 0}]),
    ]
    results_df = pd.DataFrame({'failed_tries': failed_tries_data})

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    model._log_retry_summary(results_df)
    mock_log.assert_called_once()


if __name__ == '__main__':
  unittest.main()
