"""A fake GenAI model for testing purposes."""

from typing import Any


class FakeCountTokensResponse:
  """Fake response for count_tokens."""

  def __init__(self, total_tokens: int = 10):
    self.total_tokens = total_tokens


class FakeUsageMetadata:
  """Fake usage metadata."""

  def __init__(self):
    self.total_token_count = 10
    self.prompt_token_count = 5
    self.candidates_token_count = 5
    self.tool_use_prompt_token_count = 0
    self.thoughts_token_count = 0


class FakePart:
  """Fake part of a candidate's content."""

  def __init__(self, text: str):
    self.text = text
    self.function_call = None


class FakeContent:
  """Fake content of a candidate."""

  def __init__(self, text: str):
    self.parts = [FakePart(text)]


class FakeFinishReason:
  """Fake finish reason."""

  def __init__(self, name: str = "STOP"):
    self.name = name


class FakeCandidate:
  """Fake candidate from a generation response."""

  def __init__(self, text: str):
    self.content = FakeContent(text)
    self.finish_reason = FakeFinishReason()
    self.finish_message = ""
    self.token_count = 5
    self.safety_ratings = []


class FakeGenerateContentResponse:
  """Fake response for generate_content."""

  def __init__(self, text: str):
    self.candidates = [FakeCandidate(text)]
    self.usage_metadata = FakeUsageMetadata()


class FakeModels:
  """Fake models module."""

  def __init__(self, fake_model):
    self._fake_model = fake_model

  def count_tokens(self, model: str, contents: str, **kwargs) -> FakeCountTokensResponse:
    """Fakes the count_tokens API call."""
    return FakeCountTokensResponse(10)


class FakeAioModels:
  """Fake async models module."""

  def __init__(self, fake_model):
    self._fake_model = fake_model

  async def generate_content(
      self,
      model: str,
      contents: str,
      **kwargs
  ) -> FakeGenerateContentResponse:
    """Fakes the generate_content API call."""
    response_text = self._fake_model.get_response(contents)
    return FakeGenerateContentResponse(response_text)


class FakeAio:
  """Fake async module."""

  def __init__(self, fake_model):
    self.models = FakeAioModels(fake_model)


class FakeModel:
  """A fake GenAI client that mimics the interactions used in GenaiModel."""

  def __init__(self):
    self.models = FakeModels(self)
    self.aio = FakeAio(self)
    self._responses = {}

  def set_responses(self, responses: dict[str, str]):
    """Sets the dictionary of expected prompts and their responses."""
    self._responses = dict(responses)

  def get_response(self, prompt: str) -> str:
    """Gets the response for the given prompt, or asserts if unknown."""
    if prompt not in self._responses:
      raise AssertionError(f"Unexpected prompt received: {prompt}")
    return self._responses[prompt]
