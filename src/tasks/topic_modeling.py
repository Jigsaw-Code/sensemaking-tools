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

import logging
from typing import List, Optional
from src.models.genai_model import GenaiModel
from src import sensemaker_utils
from src import prompts
from src.tasks import topic_modeling_util
from src.models import custom_types


def _are_valid_topics(response: List[custom_types.FlatTopic], *args) -> bool:
  """Validates that the response is a list of FlatTopic."""
  if not isinstance(response, list):
    logging.warning(
        f"Validation failed: Response is not a list. Got: {type(response)}"
    )
    return False
  if not all(isinstance(t, custom_types.FlatTopic) for t in response):
    logging.warning(
        f"Validation failed: Not all topics are FlatTopic. Response: {response}"
    )
    return False
  return True


def _is_valid_opinion(
    response: Optional[custom_types.NestedTopic],
    parent_topic: custom_types.Topic,
) -> bool:
  """Validates the structure for a learned opinions response."""
  if not response or not isinstance(response, custom_types.NestedTopic):
    logging.warning(
        "Validation failed: Response is not a NestedTopic. Got:"
        f" {type(response)}"
    )
    return False
  # When learning opinions, expect a single nested topic with the parent's name.
  if response.name != parent_topic.name:
    logging.warning(
        f"Validation failed for sub-level topics of '{parent_topic.name}'."
        f" Response: {response}"
    )
    return False
  return True


async def learn_topics(
    statements: list[custom_types.Statement],
    model: GenaiModel,
    additional_context: Optional[str] = None,
) -> list[custom_types.FlatTopic]:
  """Learns top-level topics from a list of statements."""
  instructions = prompts.topic_modeling_learn_topics_prompt
  schema_to_expect = custom_types.FlatTopicList
  logging.debug("Using topic_modeling_learn_topics_prompt (expecting FlatTopicList)")

  prompt_input_data = [statement_item.text for statement_item in statements]

  if not prompt_input_data:
    logging.warning(
        "No statements provided to learn topics from. Returning empty list."
    )
    return []

  chunks = await topic_modeling_util.create_chunks(
      model, instructions, prompt_input_data, additional_context
  )

  async def _generate_topics(
      current_model: GenaiModel,
  ) -> List[custom_types.FlatTopic]:
    logging.info(
        f"Identifying topics for {len(prompt_input_data)} input items..."
    )
    result = await topic_modeling_util.generate_topics_with_chunking(
        model=current_model,
        instructions=instructions,
        prompt_input_data=prompt_input_data,
        schema_to_expect=schema_to_expect,
        additional_context=additional_context,
        chunks=chunks,
    )
    if isinstance(result, custom_types.FlatTopicList):
      return result.topics
    return result

  response_topics = await sensemaker_utils.retry_call(
      _generate_topics,
      _are_valid_topics,
      model.max_llm_retries,
      "Topic identification failed after multiple retries.",
      func_args=[model],
      is_valid_args=[],
  )

  if response_topics:
    return response_topics

  logging.error(
      "Could not generate topics after retries. Returning empty list."
  )
  return []


async def learn_opinions(
    statements: List[custom_types.Statement],
    model: GenaiModel,
    topic: custom_types.Topic,
    additional_context: Optional[str] = None,
) -> custom_types.NestedTopic:
  """Learns opinions (as subtopics) for a given parent topic."""
  prompt_input_data: List[str] = []
  for statement_item in statements:
    if statement_item.quotes:
      relevant_quotes_for_statement = [
          q for q in statement_item.quotes if q.topic.name == topic.name
      ]
      for quote_obj in relevant_quotes_for_statement:
        prompt_input_data.append(f"<quote>{quote_obj.text}</quote>")

  if not prompt_input_data:
    logging.warning(
        f"No relevant quotes for topic '{topic.name}' to generate opinions"
        " from. Returning parent topic with empty subtopics."
    )
    return custom_types.NestedTopic(name=topic.name, subtopics=[])

  instructions = prompts.get_topic_modeling_opinions_prompt(topic.name)
  # Use non-recursive schema to avoid RecursionError in SDK
  schema_to_expect = custom_types.OpinionResponseSchema
  logging.debug(
      f"Using get_topic_modeling_opinions_prompt for topic: {topic.name} (expecting"
      " OpinionResponseSchema)"
  )

  chunks = await topic_modeling_util.create_chunks(
      model, instructions, prompt_input_data, additional_context
  )

  async def _generate_opinions(
      current_model: GenaiModel,
  ) -> Optional[custom_types.NestedTopic]:
    logging.info(
        f"Identifying opinions for topic '{topic.name}' from"
        f" {len(prompt_input_data)} quotes..."
    )

    result = await topic_modeling_util.generate_opinions_with_chunking(
        model=current_model,
        instructions=instructions,
        prompt_input_data=prompt_input_data,
        schema_to_expect=schema_to_expect,
        parent_topic=topic,
        additional_context=additional_context,
        chunks=chunks,
    )

    if isinstance(result, custom_types.OpinionResponseSchema):
      # Convert back to NestedTopic
      return custom_types.NestedTopic(
          name=result.name, subtopics=result.subtopics
      )
    return result

  response_topic = await sensemaker_utils.retry_call(
      _generate_opinions,
      _is_valid_opinion,
      model.max_llm_retries,
      "Opinion identification or restructuring failed after multiple retries.",
      func_args=[model],
      is_valid_args=[topic],
  )

  if response_topic:
    return response_topic

  logging.error(
      f"Could not generate opinions for topic '{topic.name}' after"
      " retries. Returning parent with empty subtopics."
  )
  return custom_types.NestedTopic(name=topic.name, subtopics=[])
