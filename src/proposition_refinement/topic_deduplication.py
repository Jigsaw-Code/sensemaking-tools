"""
This module handles the topic-based deduplication of propositions.
It identifies propositions with equivalent meaning across different topics,
selects a single "winning" topic for each set of duplicates, and then
chooses the top N propositions from the remaining non-duplicate set for each topic.
"""

import pandas as pd
import json
import re
from src.models.genai_model import GenaiModel
from . import deduplication


def _parse_equivalence_sets_with_logging(text, job):
  """Parses the LLM response for topic equivalence sets and logs the raw output."""
  print(
      '--- Raw LLM response for topic equivalence sets'
      f' ---\n{text}\n-------------------------------------------------'
  )
  return json.loads(text.strip().strip('`').strip('json').strip())


def _generate_topic_equivalence_prompt(propositions_map):
  """
  Generates a prompt for the LLM to find propositions with equivalent meaning
  ACROSS different topics.
  """
  prompt = 'You are an expert in semantic analysis and public deliberation.\n\n'
  prompt += (
      'Below is a list of propositions, each with a unique ID in the format'
      " 'topic_id:proposition_index'. Your task is to group propositions that"
      ' have an effectively equivalent meaning, with a focus on propositions'
      ' that are phrased very similarly.\n\n'
  )
  prompt += (
      'Crucially, each equivalence set should contain **at most one proposition'
      ' from any given topic**.\n\n'
  )
  prompt += 'Propositions:\n'
  for prop_id, prop_text in propositions_map.items():
    prompt += f'{prop_id}: "{prop_text}"\n'

  prompt += (
      '\nProvide your answer as a JSON object with a single key'
      " 'equivalence_sets' which is a list of lists of proposition IDs. Only"
      ' include propositions that are part of a set with two or more equivalent'
      ' members.\n\n'
  )
  prompt += 'Example:\n'
  prompt += '{"equivalence_sets": [["0:1", "3:5"], ["0:2", "1:2", "4:7"]]}'

  return prompt


async def _generate_topic_equivalence_sets(world_model_df, model):
  """
  Calls the LLM to generate the cross-topic equivalence sets.
  """
  print('---' + ' Generating cross-topic proposition equivalence sets' + ' ---')

  propositions_map = {}
  for _, row in world_model_df.iterrows():
    props_df = row.get('propositions')
    if isinstance(props_df, pd.DataFrame):
      for _, prop_row in props_df.iterrows():
        propositions_map[prop_row['proposition_id']] = prop_row['proposition']

  if not propositions_map:
    return []

  prompt = _generate_topic_equivalence_prompt(propositions_map)

  response, _ = await model.process_prompts_concurrently(
      [{'prompt': prompt, 'topic': 'topic_deduplication'}],
      response_parser=_parse_equivalence_sets_with_logging,
  )

  if not response.empty:
    result_json = response.iloc[0]['result']
    return result_json.get('equivalence_sets', [])
  return []


async def _resolve_winning_topic(prop_set, world_model_df, model):
  """
  For a given set of equivalent propositions, asks an LLM to determine which
  topic is the most relevant home for the core idea.
  """
  prompt = (
      'You are an expert in thematic analysis. Below is a set of propositions'
      ' that have been identified as having effectively equivalent meaning.'
      ' Which of the following topics is the most relevant and appropriate home'
      ' for this core idea?\n\n'
  )

  topics = []
  for prop_id in prop_set:
    topic_idx, _ = map(int, prop_id.split(':'))
    topic_name = world_model_df.iloc[topic_idx]['topic']
    if topic_name not in topics:
      topics.append(topic_name)

  prompt += 'Relevant Topics:\n'
  for topic in topics:
    prompt += f'- {topic}\n'

  prompt += '\nPropositions:\n'
  for prop_id in prop_set:
    topic_idx, prop_idx = map(int, prop_id.split(':'))
    prop_text = world_model_df.iloc[topic_idx]['propositions'].iloc[prop_idx][
        'proposition'
    ]
    prompt += f'- {prop_id}: "{prop_text}"\n'

  prompt += (
      '\nPlease respond with only the single, most relevant topic name from the'
      ' list provided.'
  )

  def parser(text, job):
    # Find the topic name in the response that is one of the candidates
    for topic in topics:
      if topic in text:
        return topic
    raise ValueError(f'Could not parse topic from response: {text}')

  response, _ = await model.process_prompts_concurrently(
      [{'prompt': prompt, 'topic': 'topic-tie-breaker'}],
      response_parser=parser,
      max_concurrent_calls=10,
  )

  if not response.empty:
    return response.iloc[0]['result']
  return None


async def run_topic_deduplication(
    by_topic_data,
    final_propositions_per_topic,
    model,
    ranking_column: str = 'full_schulze_ranking',
):
  """Main orchestrator for the topic-based deduplication process.

  This function identifies semantically equivalent propositions across different
  topics, selects a single "winning" topic for each equivalence set, and then
  chooses the top N propositions for each topic from the remaining non-duplicate
  propositions.

  Args:
    by_topic_data: A pandas DataFrame where each row represents a topic. It is
      expected to have a 'propositions' column, which contains another pandas
      DataFrame for each topic, where each row is a proposition. It should also
      have a column containing a ranked list of proposition strings.
    final_propositions_per_topic: The number of final propositions to select
      for each topic after deduplication.
    model: The GenaiModel instance to use for equivalence detection and
      tie-breaking.
    ranking_column: The name of the column in `by_topic_data` that contains
        the ranked list of propositions to use for selection.

  Returns:
    A tuple containing:
      - by_topic_data: The modified DataFrame (which includes new metadata like
        'duplicate' and 'selected' flags).
      - top_propositions_by_topic: A dictionary mapping each topic name to a
        list of the final, deduplicated proposition strings.
  """
  print('---' + ' Running topic-based deduplication' + ' ---')

  # This method requires the IDs to be present.
  by_topic_data = deduplication._preprocess_and_add_ids(by_topic_data)

  # 1. Generate the cross-topic equivalence sets
  equivalence_sets = await _generate_topic_equivalence_sets(
      by_topic_data, model
  )

  # 2. For each set, determine the winning topic and mark losers as duplicates
  for i, prop_set in enumerate(equivalence_sets):
    winning_topic = await _resolve_winning_topic(prop_set, by_topic_data, model)
    print(f"  - Equivalence set {i}: Winning topic is '{winning_topic}'")

    for prop_id in prop_set:
      topic_idx, prop_idx = map(int, prop_id.split(':'))
      props_df = by_topic_data.at[topic_idx, 'propositions']

      props_df.at[prop_idx, 'equivalence_set_id'] = i

      # If the proposition's topic is not the winner, mark it as a duplicate
      if by_topic_data.iloc[topic_idx]['topic'] != winning_topic:
        props_df.at[prop_idx, 'duplicate'] = True

  # 3. Select the top N propositions for each topic from the non-duplicates
  top_propositions_by_topic = {}
  for _, row in by_topic_data.iterrows():
    props_df = row['propositions']

    # Filter out any propositions that were marked as duplicates
    non_duplicates = props_df[~props_df['duplicate']]

    # Get the top N from the remaining ranked list
    full_ranking = row[ranking_column]

    selected_props_list = []
    for prop_text in full_ranking:
      if len(selected_props_list) >= final_propositions_per_topic:
        break
      if prop_text in non_duplicates['proposition'].values:
        selected_props_list.append(prop_text)

    # Mark the final selected propositions in the DataFrame. This 'selected'
    # flag is the key output of the deduplication process.
    selected_indices = props_df[
        props_df['proposition'].isin(selected_props_list)
    ].index
    props_df.loc[selected_indices, 'selected'] = True

    top_propositions_by_topic[row['topic']] = selected_props_list

  return by_topic_data, top_propositions_by_topic
