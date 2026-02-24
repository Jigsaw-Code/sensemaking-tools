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
from case_studies.wtp.models.genai_model import GenaiModel, MAX_LLM_RETRIES
from case_studies.wtp import sensemaker_utils
from case_studies.wtp.tasks import topic_modeling_util
from case_studies.wtp.models import custom_types


LEARN_TOPICS_PROMPT = """
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
Your task is to analyze this entire data,
identify the topics according to the criteria below,
and then generate a JSON output with the identified topics.

Important Context:
There will be another round, where participants will be exploring topics, so it will be easier for them to tie everything together in their head, if they see topics being linked to the main subject.
To facilitate this connection, we want topic language to more explicitly connect to the main subject. Topics should be phrased like they are aspects of the main subject.

### **Criteria for Topics**

  * **Distinct:** Topics should be meaningfully different and cover separate conceptual areas.
  * **Substantive:** Topics should contain multiple, distinct opinions. Do not create single-opinion topics.
  * **Subject Linkage:** Topic names should be phrased as aspects of the main subject. E.g. "Freedom and Equality" subject could have the following topics: "Defining Freedom", "Defining Equality", "Barriers to Freedom and Equality", "Society's Role in Freedom and Equality", "The Individual's Role in Freedom and Equality", etc.
  * **Concise:** The topic name should be concise.
  * **Consistent Scope:** Topics should be at a similar level of abstraction (e.g., don't mix a very broad topic with a very narrow one).
  * **Efficiency:** Keep the number of topics as low as possible. Actively consolidate topics when their content can be logically grouped.


RESPONSE STRUCTURE:
Respond with a list of the identified topics only, nothing else.
The response should be in JSON format, that can be parse into the following class:
class FlatTopicList:
    topics: List[Topic]
class Topic:
    name: str

Do not include markdown code blocks around the JSON response, such as ```json or ```
Response example:
{"topics": [{"name": "Topic 1"}, {"name": "Topic 2"}]}
"""


def learn_opinions_prompt(parent_topic: custom_types.Topic) -> str:
  return f"""
You are an expert qualitative data analyst specializing in thematic analysis and data structuring.
Your task is to analyze this entire data,
identify the opinions on the following topic: "{parent_topic.name}" according to the criteria below,
and then generate a JSON output with the identified opinions.

Important Context:
On later stages extracted quotes will be categorized into the identified opinions.
There will be another round, where participants will be exploring topics and opinions, so it will be easier for them to tie everything together in their head, if they see opinions being linked to the corresponding topic.
To facilitate this connection, we want opinion language to more explicitly connect to the overarching topic of "{parent_topic.name}". Opinion should be phrased so that it's clear that they cover different opinions on that topic.

### **Criteria for Opinions**

  * **Distinct:** Within a single topic, opinions should represent unique viewpoints. Opinions should be meaningfully distinct and different within the topic.
  * **Substantive:** Opinions should be substantive. Do not create single-quote opinions.
  * **Accurate:** The text of opinion should be a clear and well-phrased summary of the underlying quotes.
  * **Concise:** The opinions should be concise and should not contain multiple claims.
  * **Topic and Subject Linkage:** Opinions should be phrased to have a clear link to the overarching topic and the main subject of the survey.
  * **Coherency:** Opinions should be logically consistent and easy to follow as you go over the list. E.g. for "Barriers to Freedom and Equality" topic, all the opinions could start with: "A key barrier to freedom and equality is ..."; for "Defining Equality" topic: "Equality is ...", etc.
  * **Simplicity:** Phrase opinions in plain language. Aim for a 5th-grade reading level to ensure broad accessibility
  * **No "Perception" terminology**: Do not use words like "perception" in the opinion text. For example, instead of "A key barrier to freedom and equality is the perception that...", write "A key barrier to freedom and equality is...".
  * **Merge Overlaps:** If two or more opinions express the same fundamental idea, they **must be merged**.
  * **Efficiency:** Keep the number of opinions as low as possible. Actively consolidate opinions when their content can be logically grouped while ensuring that opinions remain mutually disjoint and avoid making multiple claims.

When creating opinions, keep in mind that on later stages extracted quotes will be categorized into the identified opinions.
For that, the quote must holistically match the entire opinion. A partial match, where the quote only supports one piece of the opinion, is not sufficient.
To be a match, the quote must explicitly support every key concept within the opinion.
So avoid creating opinions that will be hard to completely match to quotes.

VERY IMPORTANT:
Keep the explanation to minimum to avoid exceeding max token limit for the output.

RESPONSE STRUCTURE:
Respond only with the identified opinions, where top level is the overarching topic, and opinions are subtopics.
The response should be in JSON format, that can be parse into the following class:
class Topic:
    name: str # This will be the overarching topic
    subtopics: List[Topic] # Where subtopic is class Topic {{ name: str }} (the opinions)

Do not include markdown code blocks around the JSON response, such as ```json or ```
For example:
{{
  "name": "{parent_topic.name}",
  "subtopics": [
      {{ "name": "Opinion 1" }},
      {{ "name": "Opinion 2" }}
  ]
}}
"""


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
  instructions = LEARN_TOPICS_PROMPT
  schema_to_expect = custom_types.FlatTopicList
  logging.debug("Using LEARN_TOPICS_PROMPT (expecting FlatTopicList)")

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
      MAX_LLM_RETRIES,
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

  instructions = learn_opinions_prompt(topic)
  # Use non-recursive schema to avoid RecursionError in SDK
  schema_to_expect = custom_types.OpinionResponseSchema
  logging.debug(
      f"Using learn_opinions_prompt for topic: {topic.name} (expecting"
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
      MAX_LLM_RETRIES,
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
