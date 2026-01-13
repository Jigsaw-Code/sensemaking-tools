import asyncio
from asyncio import sleep as real_sleep
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from google.api_core import exceptions as google_exceptions
from models import genai_model
import pandas as pd

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

  def test_parse_duration(self, mock_genai_client):
    """Tests that duration strings are correctly parsed into seconds."""
    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    self.assertEqual(model._parse_duration('18s'), 18)
    self.assertEqual(model._parse_duration('60s'), 60)
    self.assertEqual(model._parse_duration('0s'), 0)
    # Test with fractional seconds, which should be truncated
    self.assertEqual(model._parse_duration('12.345s'), 12)


@patch('google.genai.Client')
class GenaiModelAsyncMethodsTest(unittest.TestCase):

  def setUp(self):
    self.prompts = [
        {'job_id': 0, 'opinion': 'Opinion 1', 'prompt': 'p1'},
        {'job_id': 1, 'opinion': 'Opinion 2', 'prompt': 'p2'},
        {'job_id': 2, 'opinion': 'Opinion 3', 'prompt': 'p3'},
    ]
    # Patch constants to avoid long sleeps during tests
    self.patcher1 = patch(
        'models.genai_model.WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS', 0.01
    )
    self.patcher2 = patch('models.genai_model.FAIL_RETRY_DELAY_SECONDS', 0.01)
    self.patcher1.start()
    self.patcher2.start()
    self.addCleanup(self.patcher1.stop)
    self.addCleanup(self.patcher2.stop)

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
    mock_candidate.content.parts[0].function_call = None
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata.total_token_count = 456
    mock_response.usage_metadata.prompt_token_count = 7
    mock_response.usage_metadata.candidates_token_count = 1
    mock_response.usage_metadata.tool_use_prompt_token_count = 1
    mock_response.usage_metadata.thoughts_token_count = 1
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    result = asyncio.run(model.call_gemini(prompt='test', run_name='test_run'))

    self.assertEqual(result['text'], 'Success response')
    self.assertEqual(result['total_token_count'], 456)
    self.assertEqual(result['prompt_token_count'], 7)
    self.assertEqual(result['candidates_token_count'], 1)
    self.assertEqual(result['tool_use_prompt_token_count'], 1)
    self.assertEqual(result['thoughts_token_count'], 1)
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
    mock_candidate.content.parts = []
    mock_response.candidates = [mock_candidate]
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    result = asyncio.run(model.call_gemini(prompt='test', run_name='test_run'))

    self.assertEqual(result['error'], 'SAFETY')
    self.assertEqual(result['finish_message'], 'Blocked for safety')

  def test_call_gemini_empty_prompt(self, mock_genai_client):
    """Tests a call to the Gemini API with an empty prompt."""
    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    with self.assertRaises(ValueError):
      asyncio.run(model.call_gemini(prompt=None, run_name='test_run'))

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
    result = asyncio.run(model.call_gemini(prompt='test', run_name='test_run'))
    self.assertEqual(result['error'], '<test> No candidates found')

  @patch('models.genai_model.GenaiModel.call_gemini')
  def test_process_prompts_all_succeed(
      self, mock_call_gemini, mock_genai_client
  ):
    """Tests when all jobs succeed on the first attempt."""
    mock_call_gemini.return_value = {
        'text': 'parsed',
        'total_token_count': 10,
        'prompt_token_count': 7,
        'candidates_token_count': 1,
        'tool_use_prompt_token_count': 1,
        'thoughts_token_count': 1,
        'error': None,
    }

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')

    def simple_parser(text, job):
      return f"parsed_{job['opinion']}"

    results_df, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts,
            simple_parser,
        )
    )

    self.assertEqual(len(results_df), 3)
    self.assertEqual(mock_call_gemini.call_count, 3)
    self.assertTrue(all(results_df['failed_tries'].apply(lambda d: d.empty)))
    self.assertIn('parsed_Opinion 1', results_df['result'].tolist())

  @patch('models.genai_model.GenaiModel.call_gemini')
  def test_process_prompts_with_retry(
      self, mock_call_gemini, mock_genai_client
  ):
    """Tests when one job succeeds after a retry."""
    # Fail for Opinion 2 on the first call, then succeed
    mock_call_gemini.side_effect = [
        {
            'text': 'parsed',
            'total_token_count': 10,
            'prompt_token_count': 7,
            'candidates_token_count': 1,
            'tool_use_prompt_token_count': 1,
            'thoughts_token_count': 1,
            'error': None,
        },  # Job 1
        Exception('API Error'),  # Job 2, Attempt 1
        {
            'text': 'parsed',
            'total_token_count': 10,
            'prompt_token_count': 7,
            'candidates_token_count': 1,
            'tool_use_prompt_token_count': 1,
            'thoughts_token_count': 1,
            'error': None,
        },  # Job 3
        {
            'text': 'parsed_retry',
            'total_token_count': 10,
            'prompt_token_count': 7,
            'candidates_token_count': 1,
            'tool_use_prompt_token_count': 1,
            'thoughts_token_count': 1,
            'error': None,
        },  # Job 2, Attempt 2
    ]

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')

    def simple_parser(resp, job):
      return f"{resp['text']}_{job['opinion']}"

    results_df, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts,
            simple_parser,
        )
    )

    self.assertEqual(len(results_df), 3)
    self.assertEqual(mock_call_gemini.call_count, 4)

    failed_job_row = results_df[results_df['opinion'] == 'Opinion 2'].iloc[0]
    self.assertEqual(len(failed_job_row['failed_tries']), 1)
    self.assertEqual(failed_job_row['result'], 'parsed_retry_Opinion 2')

  @patch('models.genai_model.GenaiModel.call_gemini')
  def test_process_prompts_permanent_failure(
      self, mock_call_gemini, mock_genai_client
  ):
    """Tests when one job fails all retry attempts."""
    mock_call_gemini.side_effect = [
        {
            'text': 'parsed',
            'total_token_count': 10,
            'prompt_token_count': 7,
            'candidates_token_count': 1,
            'tool_use_prompt_token_count': 1,
            'thoughts_token_count': 1,
            'error': None,
        },  # Job 1
        Exception('API Error 1'),  # Job 2, Attempt 1
        {
            'text': 'parsed',
            'total_token_count': 10,
            'prompt_token_count': 7,
            'candidates_token_count': 1,
            'tool_use_prompt_token_count': 1,
            'thoughts_token_count': 1,
            'error': None,
        },  # Job 3
        Exception('API Error 2'),  # Job 2, Attempt 2
    ]

    model = genai_model.GenaiModel(api_key='test_key', model_name='test_model')
    results_df, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts,
            lambda resp, j: resp['text'],
            retry_attempts=2,
        )
    )

    self.assertEqual(len(results_df), 3)
    self.assertEqual(mock_call_gemini.call_count, 4)
    # Verify the failed job is present and has an error
    failed_job = results_df[results_df['job_id'] == 1].iloc[0]
    self.assertIn('Failed after 2 attempts', str(failed_job['result']))

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


class GenaiModelBackoffTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.model = genai_model.GenaiModel(
        api_key='test_key', model_name='test_model'
    )
    self.prompts = [{'prompt': 'test prompt'}]

  @patch('asyncio.sleep', new_callable=AsyncMock)
  @patch('models.genai_model.GenaiModel.call_gemini')
  def test_service_unavailable_triggers_backoff_and_recovers(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a 503 ServiceUnavailable error triggers an exponential backoff

    and that the system recovers and succeeds on the next attempt.
    """
    # Setup: Fail first, then succeed
    mock_call_gemini.side_effect = [
        {'error': google_exceptions.ServiceUnavailable('Service is down')},
        {
            'text': 'Success',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
    ]
    # Run the process
    results_df, _ = asyncio.run(
        self.model.process_prompts_concurrently(
            self.prompts,
            lambda resp, j: resp['text'],
            retry_attempts=3,
        )
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that backoff was triggered
    # New logic: retry_attempts=3, half=1.
    # Attempt 1: attempt=1, half=1. 1>=1. delay = 60 * 2^(0) = 60.
    self.assertIn(unittest.mock.call(60.0), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

    # Check that the per-job retry counter WAS incremented (since we had 1 failure)
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 1)

  @patch('asyncio.sleep', new_callable=AsyncMock)
  @patch('models.genai_model.GenaiModel.call_gemini')
  def test_backoff_delay_increases_exponentially(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that the backoff delay squares on repeated failures."""
    # Setup: Fail twice, then succeed
    mock_call_gemini.side_effect = [
        {'error': google_exceptions.ServiceUnavailable('Service is down')},
        {
            'error': google_exceptions.ServiceUnavailable(
                'Service is still down'
            )
        },
        {
            'text': 'Success',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
    ]

    # Run the process
    asyncio.run(
        self.model.process_prompts_concurrently(
            self.prompts,
            lambda t, j: t,
            retry_attempts=3,
        )
    )

    # Assertions
    # New logic: retry_attempts=3, half=1.
    # Attempt 1: delay=60
    # Attempt 2: delay=120
    self.assertIn(unittest.mock.call(60.0), mock_sleep.call_args_list)
    self.assertIn(unittest.mock.call(120.0), mock_sleep.call_args_list)

  @patch('asyncio.sleep', new_callable=AsyncMock)
  @patch('models.genai_model.GenaiModel.call_gemini')
  def test_backoff_delay_is_capped(self, mock_call_gemini, mock_sleep):
    """Tests that the backoff delay does not exceed the maximum."""

    # Setup: Fail 9 times (hitting the cap), then succeed
    # We need attempts until we hit cap.
    # retry_attempts=10, half=5.
    # 1..4: 60
    # 5: 60
    # 6: 120
    # 7: 240
    # 8: 480
    # 9: 960 -> 600
    mock_call_gemini.side_effect = [
        {'error': google_exceptions.ServiceUnavailable('Service is down')}
        for _ in range(9)
    ] + [
        {
            'text': 'Success',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
    ]

    # Run the process
    asyncio.run(
        self.model.process_prompts_concurrently(
            self.prompts,
            lambda t, j: t,
            retry_attempts=10,
        )
    )

    # Assertions
    self.assertIn(unittest.mock.call(600.0), mock_sleep.call_args_list)

  @patch('asyncio.sleep', new_callable=AsyncMock)
  @patch('models.genai_model.GenaiModel.call_gemini')
  async def test_quota_error_triggers_global_pause(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a 429 Quota error triggers a global pause and the job retries."""

    # a mock, which fails the `isinstance` check.
    class MockClientError(genai_model.google_genai_errors.ClientError):

      def __init__(self, message, response, response_json=None):
        super().__init__(message, response_json)
        self.response = response

    mock_response = unittest.mock.MagicMock()
    mock_response.status = 429

    # Mock the async json() method on the response
    async def mock_json():
      return {
          'error': {
              'details': [{
                  '@type': 'type.googleapis.com/google.rpc.RetryInfo',
                  'retryDelay': '10s',
              }]
          }
      }

    mock_response.json = mock_json

    mock_error = MockClientError(
        'Quota exceeded', mock_response, response_json=await mock_json()
    )

    # Setup: Fail first with quota error, then succeed
    mock_call_gemini.side_effect = [
        {'error': mock_error},
        {
            'text': 'Success',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
    ]

    # Run the process
    results_df, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that global pause was triggered with the correct delay (60s)
    self.assertIn(unittest.mock.call(60), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

    # Check that the per-job retry counter WAS incremented (since we had 1 failure)
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 1)

  @patch('asyncio.sleep', new_callable=AsyncMock)
  @patch('models.genai_model.GenaiModel.call_gemini')
  async def test_service_unavailable_triggers_global_pause(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a 503 Service Unavailable error triggers a global pause."""

    # Mock ServerError
    class MockServerError(genai_model.google_genai_errors.ServerError):

      def __init__(self, message, response):
        super().__init__(message, response_json={})
        self.response = response

    mock_response = unittest.mock.MagicMock()
    mock_response.status = 503

    mock_error = MockServerError('Service Unavailable', mock_response)

    # Setup: Fail first with 503, then succeed
    mock_call_gemini.side_effect = [
        {'error': mock_error},
        {
            'text': 'Success',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
    ]

    # Run the process
    results_df, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that global pause was triggered with the correct delay (60s default)
    self.assertIn(unittest.mock.call(60), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)


if __name__ == '__main__':
  unittest.main()
