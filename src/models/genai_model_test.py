import asyncio
from asyncio import sleep as real_sleep
import logging
import unittest
from unittest import mock
from google.api_core import exceptions as google_exceptions
from src.models import genai_model
import pandas as pd

# Disable logging for tests
logging.disable(logging.CRITICAL)


class MockServerError(genai_model.google_genai_errors.ServerError):

  def __init__(self, message, response):
    super().__init__(message, response_json={})
    self.response = response


class MockClientError(genai_model.google_genai_errors.ClientError):

  def __init__(self, message, response=None, response_json=None):
    super().__init__(message, response_json)
    if response is not None:
      self.response = response
    elif hasattr(self, 'response'):
      delattr(self, 'response')


# Mock the entire google.genai library
@mock.patch('google.genai.Client')
class GenaiModelInitTest(unittest.TestCase):

  def test_init_safety_filters_off(self, mock_genai_client):
    """Tests that safety filters are set to BLOCK_NONE when safety_filters_on=False."""
    model = genai_model.GenaiModel(
        gemini_api_key='test_key', model_name='test_model', safety_filters_on=False
    )

    settings = model.safety_settings
    self.assertEqual(len(settings), 4)
    for setting in settings:
      self.assertEqual(setting.threshold.name, 'BLOCK_NONE')

  def test_init_safety_filters_on(self, mock_genai_client):
    """Tests that safety filters are set to BLOCK_ONLY_HIGH when safety_filters_on=True."""
    model = genai_model.GenaiModel(
        gemini_api_key='test_key', model_name='test_model', safety_filters_on=True
    )

    settings = model.safety_settings
    self.assertEqual(len(settings), 4)
    for setting in settings:
      self.assertEqual(setting.threshold.name, 'BLOCK_ONLY_HIGH')

  def test_parse_duration(self, mock_genai_client):
    """Tests that duration strings are correctly parsed into seconds."""
    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    self.assertEqual(model._parse_duration('18s'), 18)
    self.assertEqual(model._parse_duration('60s'), 60)
    self.assertEqual(model._parse_duration('0s'), 0)
    # Test with fractional seconds, which should be truncated
    self.assertEqual(model._parse_duration('12.345s'), 12)


class GenaiModelMaxConcurrentCallsTest(unittest.TestCase):
  """Invalid max_concurrent_calls is rejected at each public entry point."""

  def setUp(self):
    self.fake_client = mock.MagicMock()
    self.model = genai_model.GenaiModel(model_name=self.fake_client)

  def test_process_prompts_concurrently_rejects_invalid(self):
    """Rejects before creating a progress bar or any workers."""
    for value, error in ((0, ValueError), (-1, ValueError), (2.5, TypeError)):
      with self.subTest(value=value):
        with mock.patch.object(genai_model.tqdm.asyncio, 'tqdm') as mock_tqdm:
          with self.assertRaises(error):
            asyncio.run(
                self.model.process_prompts_concurrently(
                    [{'prompt': 'p'}],
                    response_parser=lambda resp, job: resp,
                    max_concurrent_calls=value,
                )
            )
          mock_tqdm.assert_not_called()

  def test_start_concurrent_workers_rejects_invalid(self):
    """Rejects before scheduling any worker tasks."""
    with mock.patch.object(genai_model.asyncio, 'create_task') as mock_create:
      with self.assertRaises(ValueError):
        self.model.start_concurrent_workers(
            response_parser=lambda resp, job: resp, max_concurrent_calls=0
        )
      mock_create.assert_not_called()

  def test_call_gemini_rejects_invalid(self):
    """Rejects before calling the API."""
    with self.assertRaises(ValueError):
      asyncio.run(
          self.model.call_gemini(
              prompt='p', run_name='test', max_concurrent_calls=0
          )
      )
    self.fake_client.aio.models.generate_content.assert_not_called()


@mock.patch('google.genai.Client')
class GenaiModelAsyncMethodsTest(unittest.TestCase):

  def setUp(self):
    self.prompts = [
        {'job_id': 0, 'opinion': 'Opinion 1', 'prompt': 'p1'},
        {'job_id': 1, 'opinion': 'Opinion 2', 'prompt': 'p2'},
        {'job_id': 2, 'opinion': 'Opinion 3', 'prompt': 'p3'},
    ]
    # Patch constants to avoid long sleeps during tests
    self.patcher1 = mock.patch(
        'src.models.genai_model.WAIT_BETWEEN_SUCCESSFUL_CALLS_SECONDS', 0.01
    )
    self.patcher2 = mock.patch(
        'src.models.genai_model.FAIL_RETRY_DELAY_SECONDS', 0.01
    )
    self.patcher1.start()
    self.patcher2.start()
    self.addCleanup(self.patcher1.stop)
    self.addCleanup(self.patcher2.stop)

  def test_estimate_token_count_needed(self, mock_genai_client):
    """Tests that the token count is correctly returned."""
    mock_client_instance = mock_genai_client.return_value
    mock_response = mock.MagicMock()
    mock_response.total_tokens = 123
    mock_client_instance.models.count_tokens.return_value = mock_response

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    token_count = model.estimate_token_count_needed(prompt='test prompt')

    self.assertEqual(token_count, 123)
    mock_client_instance.models.count_tokens.assert_called_once()

  def test_call_gemini_with_fake_model(self, mock_genai_client):
    """Tests injecting a FakeModel to handle generation requests."""
    from src.models.fake_genai_model import FakeModel
    
    fake_model = FakeModel()
    fake_model.set_responses({'test prompt': 'Injected Fake Response'})
    
    model = genai_model.GenaiModel(model_name=fake_model)
    result = asyncio.run(model.call_gemini(prompt='test prompt', run_name='test_run'))
    
    self.assertEqual(result['text'], 'Injected Fake Response')
    self.assertIsNone(result['error'])

  def test_call_gemini_with_fake_model_unexpected_prompt(self, mock_genai_client):
    """Tests injecting a FakeModel asserts on unexpected prompt."""
    from src.models.fake_genai_model import FakeModel
    
    fake_model = FakeModel()
    fake_model.set_responses({'known prompt': 'Response'})
    
    model = genai_model.GenaiModel(model_name=fake_model)
    result = asyncio.run(model.call_gemini(prompt='unknown prompt', run_name='test_run'))
    
    self.assertIsInstance(result['error'], AssertionError)
    self.assertIn('Unexpected prompt received: unknown prompt', str(result['error']))

  def test_call_gemini_success(self, mock_genai_client):
    """Tests a successful call to the Gemini API."""
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content = mock.AsyncMock()

    mock_response = mock.MagicMock()
    mock_candidate = mock.MagicMock()
    mock_candidate.finish_reason.name = 'STOP'
    mock_candidate.content.parts = [mock.MagicMock()]
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

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
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
    mock_client_instance.aio.models.generate_content = mock.AsyncMock()

    mock_response = mock.MagicMock()
    mock_candidate = mock.MagicMock()
    mock_candidate.finish_reason.name = 'SAFETY'
    mock_candidate.finish_message = 'Blocked for safety'
    mock_candidate.token_count = 0
    mock_candidate.content.parts = []
    mock_response.candidates = [mock_candidate]
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    result = asyncio.run(model.call_gemini(prompt='test', run_name='test_run'))

    self.assertEqual(result['error'], 'SAFETY')
    self.assertEqual(result['finish_message'], 'Blocked for safety')

  def test_call_gemini_empty_prompt(self, mock_genai_client):
    """Tests a call to the Gemini API with an empty prompt."""
    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    with self.assertRaises(ValueError):
      asyncio.run(model.call_gemini(prompt=None, run_name='test_run'))

  def test_call_gemini_no_candidate(self, mock_genai_client):
    """Tests a call to the Gemini API with no candidates."""
    mock_client_instance = mock_genai_client.return_value
    mock_client_instance.aio.models.generate_content = mock.AsyncMock()

    mock_response = mock.MagicMock()
    mock_response.candidates = []
    mock_response.prompt_feedback = '<test> No candidates found'
    mock_client_instance.aio.models.generate_content.return_value = (
        mock_response
    )
    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    result = asyncio.run(model.call_gemini(prompt='test', run_name='test_run'))
    self.assertEqual(result['error'], '<test> No candidates found')

  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
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

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')

    def simple_parser(text, job):
      return f"parsed_{job['opinion']}"

    results_df, _, _, _ = asyncio.run(
        model.process_prompts_concurrently(
            self.prompts,
            simple_parser,
        )
    )

    self.assertEqual(len(results_df), 3)
    self.assertEqual(mock_call_gemini.call_count, 3)
    self.assertTrue(all(results_df['failed_tries'].apply(lambda d: d.empty)))
    self.assertIn('parsed_Opinion 1', results_df['result'].tolist())

  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
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

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')

    def simple_parser(resp, job):
      return f"{resp['text']}_{job['opinion']}"

    results_df, _, _, _ = asyncio.run(
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

  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
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

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    results_df, _, _, _ = asyncio.run(
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

  @mock.patch('src.models.genai_model.logging.error')
  @mock.patch('src.models.genai_model.logging.debug')
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  def test_process_prompts_logging_levels(
      self, mock_call_gemini, mock_debug, mock_error, mock_genai_client
  ):
    """Tests that intermediate failures log as debug and permanent failures log as error."""
    mock_call_gemini.side_effect = [
        Exception('API Error 1'),  # Job 1, Attempt 1 (intermediate failure)
        Exception('API Error 2'),  # Job 1, Attempt 2 (permanent failure)
    ]

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    results_df, _, _, _ = asyncio.run(
        model.process_prompts_concurrently(
            [{'job_id': 0, 'opinion': 'Opinion 1', 'prompt': 'p1'}],
            lambda resp, j: resp['text'],
            retry_attempts=2,
            skip_log=True,
        )
    )

    # Assert intermediate failures are logged at debug level
    debug_calls = [c[0][0] for c in mock_debug.call_args_list]
    self.assertTrue(
        any('attempt 1: Exception' in str(call) for call in debug_calls)
    )
    self.assertTrue(
        any('attempt 2: Exception' in str(call) for call in debug_calls)
    )

    # Assert permanent failures are logged at error level
    error_calls = [c[0][0] for c in mock_error.call_args_list]
    self.assertTrue(
        any(
            'Failed to process opinion' in str(call)
            and 'after 2 attempts' in str(call)
            for call in error_calls
        )
    )

  @mock.patch('src.models.genai_model.GenaiModel._log_retry_summary')
  def test_log_retry_summary(self, mock_log, mock_genai_client):
    """Tests that the retry summary is logged correctly."""
    failed_tries_data = [
        pd.DataFrame(),
        pd.DataFrame([{'attempt_index': 0}]),
    ]
    results_df = pd.DataFrame({'failed_tries': failed_tries_data})

    model = genai_model.GenaiModel(gemini_api_key='test_key', model_name='test_model')
    model._log_retry_summary(results_df)
    mock_log.assert_called_once()


class GenaiModelBackoffTest(unittest.IsolatedAsyncioTestCase):

  def setUp(self):
    self.model = genai_model.GenaiModel(
        gemini_api_key='test_key', model_name='test_model'
    )
    self.prompts = [{'prompt': 'test prompt'}]

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
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
    results_df, _, _, _ = asyncio.run(
        self.model.process_prompts_concurrently(
            self.prompts,
            lambda resp, j: resp['text'],
            retry_attempts=3,
        )
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that global pause was triggered (restored logic starts at 2s)
    self.assertIn(mock.call(2), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

    # Check that the per-job retry counter was NOT incremented for a 503 error
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 0)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
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
    # Restored logic: delay starts at 2s and squares: 2, 4
    self.assertIn(mock.call(2), mock_sleep.call_args_list)
    self.assertIn(mock.call(4), mock_sleep.call_args_list)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  def test_backoff_delay_is_capped(self, mock_call_gemini, mock_sleep):
    """Tests that the backoff delay does not exceed the maximum."""

    # Setup: Fail enough times to hit the 60s cap, then succeed
    # 1..5: 2, 4, 8, 16, 32
    # 6..16: 60.0s delay (capped)
    mock_call_gemini.side_effect = [
        {'error': google_exceptions.ServiceUnavailable('Service is down')}
        for _ in range(16)
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
            retry_attempts=20,
        )
    )

    # Assertions
    self.assertIn(mock.call(60.0), mock_sleep.call_args_list)
    # Ensure it stays capped at 60 (attempts 6..16 = 11 times)
    self.assertEqual(
        mock_sleep.call_args_list.count(mock.call(60.0)), 11
    )

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_quota_error_triggers_global_pause(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a 429 Quota error triggers a global pause and the job retries."""

    # a mock, which fails the `isinstance` check.

    mock_response = mock.MagicMock()
    mock_response.status_code = 429

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
    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that global pause was triggered with the correct delay (RetryInfo 10s + 1s offset)
    self.assertIn(mock.call(11), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

    # Check that the per-job retry counter was NOT incremented for a quota error
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 0)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_service_unavailable_triggers_global_pause(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a 503 Service Unavailable error triggers a global pause."""

    # Mock ServerError

    mock_response = mock.MagicMock()
    mock_response.status_code = 503

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
    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that global pause was triggered with the correct delay (Restored logic starts at 2s)
    self.assertIn(mock.call(2), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_concurrent_infrastructure_errors_all_retry(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that multiple concurrent 503 errors only trigger ONE pause and ALL retry."""

    # Mock ServerError

    mock_response = mock.MagicMock()
    mock_response.status_code = 503
    mock_error = MockServerError('Service Unavailable', mock_response)

    # Setup: 3 jobs all fail once with 503, then all succeed
    # We expect 6 calls total (3 failures, 3 successes)
    mock_call_gemini.side_effect = [
        {'error': mock_error},
        {'error': mock_error},
        {'error': mock_error},
        {
            'text': 'Success 1',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
        {
            'text': 'Success 2',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
        {
            'text': 'Success 3',
            'total_token_count': 10,
            'prompt_token_count': 5,
            'candidates_token_count': 5,
            'tool_use_prompt_token_count': 0,
            'thoughts_token_count': 0,
            'error': None,
        },
    ]

    # Use 3 prompts to trigger concurrent workers
    prompts = [
        {'prompt': 'p1'},
        {'prompt': 'p2'},
        {'prompt': 'p3'},
    ]

    # Run the process with 3 concurrent calls
    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
        max_concurrent_calls=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 3)
    results_list = results_df['result'].tolist()
    self.assertIn('Success 1', results_list)
    self.assertIn('Success 2', results_list)
    self.assertIn('Success 3', results_list)

    # Check that global pause was triggered ONLY ONCE for the 3 concurrent errors
    # (The leader elections ensures this)
    self.assertEqual(mock_sleep.call_args_list.count(mock.call(2)), 1)

    # Total calls should be 6
    self.assertEqual(mock_call_gemini.call_count, 6)

    # Check that failed_tries is empty for all (infra errors don't count)
    for i in range(3):
      self.assertEqual(len(results_df.iloc[i]['failed_tries']), 0)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_token_limit_handled_as_job_error(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that an error containing 'limit' is handled as a job error (increments attempts)."""

    limit_error = Exception('Output length limit reached')

    # Setup: Fail first with limit error, then succeed
    mock_call_gemini.side_effect = [
        {'error': limit_error},
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
    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # The attempt counter SHOULD be incremented, so failed_tries should have 1 entry
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 1)
    self.assertEqual(
        results_df.iloc[0]['failed_tries'].iloc[0]['error_message'],
        'Output length limit reached',
    )

    # Temperature should have been nudged from 0.0 to 0.02
    self.assertAlmostEqual(results_df.iloc[0]['temperature'], 0.02)

    # Global pause should NOT have been triggered
    # (Checking that no sleep calls were for 60s/backoff)
    for call in mock_sleep.call_args_list:
      self.assertNotEqual(call, mock.call(60))

    # Should have been two calls: failure and success
    self.assertEqual(mock_call_gemini.call_count, 2)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_missing_status_code_does_not_crash(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a ClientError with no status_code is handled safely without crashing."""


    mock_response = mock.NonCallableMock(spec=['json'])
    mock_response.status_code = None

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
    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that generic pause was triggered with the correct delay
    self.assertIn(mock.call(0.1), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_concurrent_retries_client_error_no_response(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a ClientError with no response object is handled safely without crashing."""


    mock_error = MockClientError('Network or Parsing Failure')

    # Setup: Fail first with generic error, then succeed
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
    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        self.prompts,
        lambda resp, j: resp['text'],
        retry_attempts=3,
    )

    # Assertions
    self.assertEqual(len(results_df), 1)
    self.assertEqual(results_df.iloc[0]['result'], 'Success')

    # Check that generic pause was triggered with the correct delay
    self.assertIn(mock.call(0.1), mock_sleep.call_args_list)
    self.assertEqual(mock_call_gemini.call_count, 2)

  async def test_extract_error_details_quota_fallbacks_without_response(self):
    """Tests that 429 quota errors are detected via code, status_code, or message when response is None."""
    err_by_code = MockClientError(429)
    is_quota, is_unavail, _ = await self.model._extract_error_details(
        err_by_code, '[Test]'
    )
    self.assertTrue(is_quota)
    self.assertFalse(is_unavail)

    err_by_status_code = Exception('Rate limit')
    err_by_status_code.status_code = 429
    is_quota, is_unavail, _ = await self.model._extract_error_details(
        err_by_status_code, '[Test]'
    )
    self.assertTrue(is_quota)
    self.assertFalse(is_unavail)

    err_by_str = MockClientError('RESOURCE_EXHAUSTED: Quota exceeded')
    is_quota, is_unavail, _ = await self.model._extract_error_details(
        err_by_str, '[Test]'
    )
    self.assertTrue(is_quota)
    self.assertFalse(is_unavail)

    # Verify RetryInfo is extracted from e.details even when response is None
    err_with_details = MockClientError(
        429,
        response=None,
        response_json={
            'error': {
                'details': [{
                    '@type': 'type.googleapis.com/google.rpc.RetryInfo',
                    'retryDelay': '15s',
                }]
            }
        },
    )
    is_quota, _, delay = await self.model._extract_error_details(
        err_with_details, '[Test]'
    )
    self.assertTrue(is_quota)
    self.assertEqual(delay, 16)

    # Verify synchronous response.json() (e.g. httpx.Response) works without TypeError
    sync_response = mock.MagicMock()
    sync_response.status_code = 429
    sync_response.json.return_value = {
        'error': {
            'details': [{
                '@type': 'type.googleapis.com/google.rpc.RetryInfo',
                'retryDelay': '12s',
            }]
        }
    }
    err_sync_json = MockClientError(429, response=sync_response, response_json={})
    is_quota, _, delay = await self.model._extract_error_details(
        err_sync_json, '[Test]'
    )
    self.assertTrue(is_quota)
    self.assertEqual(delay, 13)

  async def test_extract_error_details_service_unavailable_fallbacks_without_response(
      self,
  ):
    """Tests that 503 service unavailable errors are detected via code or message when response is None."""
    err_by_code = MockServerError(503, None)
    is_quota, is_unavail, _ = await self.model._extract_error_details(
        err_by_code, '[Test]'
    )
    self.assertFalse(is_quota)
    self.assertTrue(is_unavail)

    err_by_str = MockServerError('UNAVAILABLE: backend overloaded', None)
    is_quota, is_unavail, _ = await self.model._extract_error_details(
        err_by_str, '[Test]'
    )
    self.assertFalse(is_quota)
    self.assertTrue(is_unavail)


_SUCCESS_RESP = {
    'text': 'Success',
    'total_token_count': 10,
    'prompt_token_count': 5,
    'candidates_token_count': 5,
    'tool_use_prompt_token_count': 0,
    'thoughts_token_count': 0,
    'error': None,
}


def _unavailable():
  """Returns a call_gemini response carrying a 503 error."""
  return {'error': google_exceptions.ServiceUnavailable('Service is down')}


class GenaiModelInfraToleranceTest(unittest.IsolatedAsyncioTestCase):
  """Tests time-based 429/503 tolerance, stall detection and retry delays."""

  def setUp(self):
    logging.disable(logging.NOTSET)
    self.addCleanup(lambda: logging.disable(logging.CRITICAL))
    self.model = genai_model.GenaiModel(
        gemini_api_key='test_key', model_name='test_model'
    )
    # Simulated monotonic clock (seconds), advanced explicitly by tests.
    self.now = 0.0
    self.model._clock = lambda: self.now

  def test_defaults(self):
    """Tests the default stall budget and backoff constants."""
    self.assertEqual(genai_model.DEFAULT_MAX_STALL_MINUTES, 20)
    self.assertEqual(self.model.max_stall_minutes, 20)
    self.assertEqual(genai_model.INFRA_INITIAL_BACKOFF_SECONDS, 2)
    self.assertEqual(self.model._backoff_delay, 2)
    self.assertEqual(self.model._max_backoff_delay, 60.0)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_many_infra_errors_do_not_consume_attempts(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that >30 infra errors never burn standard attempts."""
    mock_call_gemini.side_effect = [_unavailable() for _ in range(45)] + [
        _SUCCESS_RESP
    ]

    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        [{'prompt': 'p'}], lambda resp, j: resp['text'], retry_attempts=3
    )

    self.assertEqual(results_df.iloc[0]['result'], 'Success')
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 0)
    self.assertEqual(results_df.iloc[0]['infra_retries'], 45)
    self.assertEqual(mock_call_gemini.call_count, 46)
    # Success resets the streak and the backoff.
    self.assertIsNone(self.model._infra_streak_start)
    self.assertEqual(self.model._backoff_delay, 2)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_stall_raises_model_stalled_error(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that >= max_stall_minutes without success raises loudly."""

    async def always_unavailable(**kwargs):
      self.now += 61.0  # Each call takes ~1 simulated minute.
      return _unavailable()

    mock_call_gemini.side_effect = always_unavailable

    with self.assertRaises(genai_model.ModelStalledError) as ctx:
      await asyncio.wait_for(
          self.model.process_prompts_concurrently(
              [{'prompt': 'p1'}, {'prompt': 'p2'}],
              lambda resp, j: resp['text'],
              retry_attempts=3,
              max_concurrent_calls=2,
          ),
          timeout=30,
      )

    self.assertIn("'test_model'", str(ctx.exception))
    self.assertIn('x 503', str(ctx.exception))
    # 20 min / 61 s per call => stall detected on the 21st error (~20.3 min).
    self.assertGreaterEqual(mock_call_gemini.call_count, 20)
    self.assertTrue(self.model._stalled)
    # Sticky across subsequent attempts on the same model.
    with self.assertRaises(genai_model.ModelStalledError):
      self.model.raise_if_stalled()

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_stall_stops_all_workers_and_surfaces_one_error(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a stall halts the worker pool and raises once."""

    async def always_unavailable(**kwargs):
      self.now += 600.0  # 10 min each, trips in 3 calls.
      return _unavailable()

    mock_call_gemini.side_effect = always_unavailable
    queue, workers, results, _, stop_event = (
        self.model.start_concurrent_workers(
            lambda resp, j: resp['text'], max_concurrent_calls=3
        )
    )
    for i in range(3):
      queue.put_nowait({'prompt': f'p{i}', 'stats': {}, 'retry_attempts': 3})
    for _ in workers:
      queue.put_nowait(None)

    with self.assertRaises(genai_model.ModelStalledError):
      await asyncio.wait_for(asyncio.gather(*workers), timeout=30)

    outcomes = await asyncio.wait_for(
        asyncio.gather(*workers, return_exceptions=True), timeout=30
    )
    stalled = [
        o for o in outcomes if isinstance(o, genai_model.ModelStalledError)
    ]
    self.assertEqual(len(stalled), 1)
    self.assertTrue(stop_event.is_set())
    self.assertEqual(results, [])

  def test_infra_backoff_exact_sequence(self):
    """Tests the doubling sequence from 2s capped at 60s."""
    delays = [round(self.model._next_infra_backoff(), 4) for _ in range(7)]
    self.assertEqual(
        delays,
        [2, 4, 8, 16, 32, 60.0, 60.0],
    )

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_non_infra_errors_exhaust_attempts_with_fast_retry(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests generic errors still exhaust retry_attempts, with 0.1s fast retry."""
    mock_call_gemini.side_effect = lambda **kwargs: {
        'error': Exception('boom')
    }

    results_df, stats_df, _, _ = await self.model.process_prompts_concurrently(
        [{'prompt': 'p'}], lambda resp, j: resp['text'], retry_attempts=3
    )

    self.assertEqual(mock_call_gemini.call_count, 3)
    self.assertEqual(
        results_df.iloc[0]['result'], {'error': 'Failed after 3 attempts'}
    )
    self.assertEqual(len(results_df.iloc[0]['failed_tries']), 3)
    self.assertTrue(stats_df.iloc[0]['is_complete_failure'])
    sleeps = [c.args[0] for c in mock_sleep.call_args_list]
    self.assertIn(0.1, sleeps)
    # Non-infra errors never start an infra streak.
    self.assertIsNone(self.model._infra_streak_start)

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_partial_outage_reset_prevents_false_stall(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that a single mid-outage success resets the wall-clock stall timer."""
    # 15 minutes of 503s (3 calls * 300s = 600s elapsed from first error),
    # then 1 success (resets streak), then another 15 minutes of 503s,
    # then final success. Total elapsed time is 30 min (> 20 min limit),
    # but neither streak exceeds 20 minutes.
    events = [
        (0.0, _unavailable()),
        (300.0, _unavailable()),
        (900.0, _unavailable()),
        (910.0, _SUCCESS_RESP),
        (1000.0, _unavailable()),
        (1400.0, _unavailable()),
        (1850.0, _unavailable()),
        (1860.0, _SUCCESS_RESP),
    ]
    idx = 0

    async def scripted_call(**kwargs):
      nonlocal idx
      t, resp = events[idx]
      idx += 1
      self.now = t
      return resp

    mock_call_gemini.side_effect = scripted_call

    results_df, _, _, _ = await self.model.process_prompts_concurrently(
        [{'prompt': 'p1'}, {'prompt': 'p2'}],
        lambda resp, j: resp['text'],
        retry_attempts=3,
        max_concurrent_calls=1,
    )

    self.assertEqual(len(results_df), 2)
    self.assertEqual(list(results_df['result']), ['Success', 'Success'])
    self.assertFalse(self.model._stalled)
    self.assertIsNone(self.model._infra_streak_start)
    self.assertEqual(
        self.model._backoff_delay, genai_model.INFRA_INITIAL_BACKOFF_SECONDS
    )

  @mock.patch('asyncio.sleep', new_callable=mock.AsyncMock)
  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_429_without_retry_info_uses_exponential_backoff_and_stalls(
      self, mock_call_gemini, mock_sleep
  ):
    """Tests that 429s with response=None use exponential backoff and count toward stall."""

    async def always_429(**kwargs):
      self.now += 61.0
      return {'error': MockClientError(429)}

    mock_call_gemini.side_effect = always_429

    with self.assertRaises(genai_model.ModelStalledError) as ctx:
      await asyncio.wait_for(
          self.model.process_prompts_concurrently(
              [{'prompt': 'p1'}],
              lambda resp, j: resp['text'],
              retry_attempts=3,
              max_concurrent_calls=1,
          ),
          timeout=30,
      )

    self.assertIn('x 429', str(ctx.exception))
    self.assertIn('0 x 503', str(ctx.exception))
    sleeps = [c.args[0] for c in mock_sleep.call_args_list]
    self.assertIn(genai_model.INFRA_INITIAL_BACKOFF_SECONDS, sleeps)
    self.assertIn(genai_model.INFRA_INITIAL_BACKOFF_SECONDS * 2, sleeps)

  @mock.patch('src.models.genai_model.GenaiModel.call_gemini')
  async def test_stall_wakes_workers_blocked_on_global_pause(
      self, mock_call_gemini
  ):
    """Tests that a worker blocked on _global_pause_event.wait() is woken early on stall."""
    both_in_flight = asyncio.Event()
    pause_started = asyncio.Event()
    never_finish_pause = asyncio.Event()
    call_num = 0

    async def controlled_sleep(delay):
      # Initial worker startup jitter (0 <= delay <= 1, not equal to 60.0)
      if delay == 60.0:
        pause_started.set()
        # Simulate a long 60s global pause that never finishes on its own;
        # the worker must be woken by _record_infra_error setting
        # _global_pause_event directly.
        await never_finish_pause.wait()
      else:
        await real_sleep(0)

    async def coordinated_calls(**kwargs):
      nonlocal call_num
      call_num += 1
      if call_num == 1:
        # Wait until Worker 2 is also inside call_gemini before triggering
        # the global pause, so Worker 2 isn't blocked prior to calling.
        await both_in_flight.wait()
        self.now = 0.0
        self.model._backoff_delay = 60.0
        return _unavailable()
      # Worker 2 signals that both workers are in flight, then waits until
      # Worker 1 is blocked on _global_pause_event.wait() before tripping stall.
      both_in_flight.set()
      await pause_started.wait()
      await real_sleep(0)
      self.now = 21 * 60.0
      return _unavailable()

    mock_call_gemini.side_effect = coordinated_calls

    with mock.patch('asyncio.sleep', side_effect=controlled_sleep):
      with self.assertRaises(genai_model.ModelStalledError):
        await asyncio.wait_for(
            self.model.process_prompts_concurrently(
                [{'prompt': 'p1'}, {'prompt': 'p2'}],
                lambda resp, j: resp['text'],
                retry_attempts=3,
                max_concurrent_calls=2,
            ),
            timeout=5,
        )
      # Unblock the background _handle_global_pause task so it cleans up.
      never_finish_pause.set()
      await real_sleep(0)


if __name__ == '__main__':
  unittest.main()
