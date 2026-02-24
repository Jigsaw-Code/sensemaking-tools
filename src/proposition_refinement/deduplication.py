"""Handles the deduplication and refined selection of propositions based on semantic equivalence."""

import pandas as pd
import json
import re
import logging
from src.models.genai_model import GenaiModel


def _preprocess_and_add_ids(by_topic_df):
  """Prepares the DataFrame for deduplication by adding stable IDs.

  Expects a DataFrame where each row is a topic and there is a
  'propositions' column containing a nested DataFrame of propositions.
  """
  # Sort topics by r1_df_length to ensure stable topic_ids
  sorted_df = by_topic_df.sort_values(
      by='r1_quotes_by_topic', ascending=False
  ).reset_index(drop=True)
  sorted_df['topic_id'] = sorted_df.index

  # Add proposition_id to each proposition within each topic's DataFrame
  new_propositions_dfs = []
  for topic_id, row in sorted_df.iterrows():
    props_df = row['propositions'].copy()  # Work on an explicit copy
    if isinstance(props_df, pd.DataFrame) and not props_df.empty:
      # Reset index to ensure proposition IDs are sequential.
      props_df.reset_index(drop=True, inplace=True)
      props_df = props_df.assign(
          proposition_id=[f'{topic_id}:{i}' for i in props_df.index],
          duplicate=False,
          selected=False,
          equivalence_set_id=None,  # Default to None for singletons
      )
    new_propositions_dfs.append(props_df)

  sorted_df['propositions'] = new_propositions_dfs
  return sorted_df


def generate_equivalence_prompt(propositions_map):
  """Generates the prompt for the LLM to cluster propositions."""
  proposition_list = ''
  for prop_id, prop_text in propositions_map.items():
    proposition_list += f'{prop_id}: "{prop_text}"\n'
  prompt = f"""You are an expert in semantic analysis and public deliberation.

Below is a list of propositions, each with a unique ID in the format 'topic_id:proposition_index'. Your task is to group propositions into sets that have an effectively equivalent meaning—i.e. contain distinction without difference. The goal is to identify propositions that would feel very repetitive if presented together to a survey participant.

Examples of effectively equivalent meaning include:
* If two propositions only differ by synonyms (or related words), they are effectively equivalent, e.g.
  * Everyone should be treated with respect regardless of their race, background, or beliefs.
  * Equality is recognizing that all people have the same basic worth and deserve to be treated with dignity and respect.
* If two propositions only differ in degree of specificity, they are effectively equivalent, e.g.
  * Freedom is the right to express your thoughts and beliefs without fear of punishment. 
  * Freedom includes the right to express your own ideas without fear of punishment from the government.

Propositions:
{proposition_list}

Provide your answer as a JSON object with a single key 'equivalence_sets' which is a list of lists of proposition IDs. Only include propositions that are part of a set with two or more equivalent members.

Example:
"""
  prompt += '{"equivalence_sets": [["0:1", "3:5"], ["0:2", "1:2", "4:7"]]}'
  return prompt


async def generate_equivalence_sets(processed_by_topic_df, model):
  """
  Uses an LLM to cluster propositions with effectively equivalent meaning.
  """
  print('--- Generating proposition equivalence sets ---')

  # Flatten all propositions into a single map with unique IDs
  propositions_map = {}
  for _, row in processed_by_topic_df.iterrows():
    propositions_df = row['propositions']
    if isinstance(propositions_df, pd.DataFrame) and not propositions_df.empty:
      for _, prop_row in propositions_df.iterrows():
        propositions_map[prop_row['proposition_id']] = prop_row['proposition']
    elif propositions_df is not None:
      print(
          f"Warning: 'propositions' for topic '{row['topic']}' is not a"
          f' DataFrame or is empty. Type: {type(propositions_df)}'
      )

  if not propositions_map:
    return []

  prompt = generate_equivalence_prompt(propositions_map)
  logging.debug('Equivalence prompt:\n%s', prompt)

  def parser_with_logging(resp, _):
    logging.debug(
        '--- Raw LLM response for equivalence set generation ---\n%s', resp
    )
    # strip leading and tailing ```json markdown fencing
    x = resp['text'].strip().strip('```').strip().strip('json').strip()
    logging.debug('--- ready to jsonify ---\n%s', x)
    return json.loads(x)

  response, _ = await model.process_prompts_concurrently(
      [{'prompt': prompt, 'topic': 'deduplication'}],
      # response_parser=lambda x, job: json.loads(x)
      response_parser=parser_with_logging,
  )
  print('Done running equivalence set generation:', response)

  if not response.empty:
    result_json = response.iloc[0]['result']
    return result_json.get('equivalence_sets', [])

  return []


def _generate_collision_prompt(collision_group):
  """Generates the prompt for the LLM to resolve a collision."""
  prompt = (
      'From the following list of propositions with effectively equivalent'
      ' meaning, which one is the best-framed and most representative of the'
      ' core idea?\n'
  )
  prompt += (
      'Each proposition has a unique identifier in the format'
      " 'topic_id:proposition_index'.\n"
  )
  prompt += (
      'Please respond with only the single, full proposition_id of the winning'
      ' proposition.\n\n'
  )

  for item in collision_group:
    prompt += f"{item['prop_id']} - \"{item['text']}\"\n"
  return prompt


async def _resolve_collision(collision_group, model):
  """
  Uses an LLM to select the best-framed proposition from a set of duplicates.
  """
  prompt = _generate_collision_prompt(collision_group)

  def parser(resp, job):
    text = resp['text']
    match = re.search(r'(\d+:\d+)', text)
    if match:
      return match.group(1)
    raise ValueError(f'Could not parse proposition_id from response: {text}')

  response, _ = await model.process_prompts_concurrently(
      [{'prompt': prompt, 'topic': 'tie-breaker'}],
      response_parser=parser,
  )

  if not response.empty:
    return response.iloc[0]['result']
  return None


async def select_final_propositions(
    processed_by_topic_df,
    equivalence_sets,
    final_propositions_per_topic,
    model,
    ranking_column: str,
):
  """Selects the final, deduplicated set of top propositions for each topic using a rank-filling draft algorithm.

  This ensures that topics that lose a collision for a high-rank spot get a
  chance to fill that spot with their next-best candidate before the algorithm moves on to filling the next rank for all topics.
  """
  print('--- Selecting final propositions using rank-filling draft ---')

  # Create a map from a proposition_id to its equivalence set index for efficient lookup.
  equivalence_map = {
      prop_id: i for i, s in enumerate(equivalence_sets) for prop_id in s
  }

  # Mark all propositions that are part of a multi-member equivalence set as duplicates
  # and assign their equivalence set ID.
  for i, s in enumerate(equivalence_sets):
    if len(s) > 1:
      for prop_id in s:
        topic_idx, prop_idx = map(int, prop_id.split(':'))
        props_df = processed_by_topic_df.at[topic_idx, 'propositions']
        props_df.at[prop_idx, 'duplicate'] = True
        props_df.at[prop_idx, 'equivalence_set_id'] = i

  # Initialize data structures for the selection loop.
  topics = processed_by_topic_df['topic'].tolist()
  final_propositions = {topic: [] for topic in topics}
  seen_equivalence_sets = set()
  next_rank_to_consider = {topic: 0 for topic in topics}
  topic_name_to_id = {
      row.topic: row.topic_id for row in processed_by_topic_df.itertuples()
  }

  # --- Rank-Filling Draft Loop ---
  # This outer loop iterates from 0 to N-1. Its purpose is to ensure that we select
  # the #1 proposition for all topics before moving on to select the #2, and so on.
  for final_rank in range(final_propositions_per_topic):

    # This inner loop continues until all topics have filled the current `final_rank`.
    # If a topic loses a collision, it will re-enter this loop to submit its next-best candidate.
    topics_in_round = topics.copy()
    while topics_in_round:
      current_round_candidates = []

      # --- Candidate Gathering ---
      # For each topic that still needs a proposition for the current rank,
      # find its best available (i.e., not yet seen) candidate.
      for topic_name in topics_in_round:
        topic_idx = topic_name_to_id[topic_name]
        row = processed_by_topic_df.iloc[topic_idx]

        rank_idx = next_rank_to_consider[topic_name]
        full_ranking = row[ranking_column]

        # Find the next available proposition for this topic.
        while rank_idx < len(full_ranking):
          prop_text = full_ranking[rank_idx]
          props_df = row['propositions']
          prop_row = props_df[props_df['proposition'] == prop_text].iloc[0]
          prop_id = prop_row['proposition_id']

          # A proposition's equivalence set ID is either from the map or its own ID if it's a singleton.
          eq_set_id = equivalence_map.get(prop_id, prop_id)

          if eq_set_id not in seen_equivalence_sets:
            current_round_candidates.append({
                'prop_id': prop_id,
                'topic': topic_name,
                'text': prop_text,
                'rank': rank_idx,
            })
            next_rank_to_consider[topic_name] = rank_idx + 1
            break  # Found a candidate for this topic, move to the next topic.
          rank_idx += 1

      if not current_round_candidates:
        break  # No more available candidates for any topic, exit the while loop.

      # --- Collision Resolution ---
      # Group the candidates for this round by their equivalence set ID.
      groups = {}
      for cand in current_round_candidates:
        eq_set_id = equivalence_map.get(cand['prop_id'], cand['prop_id'])
        if eq_set_id not in groups:
          groups[eq_set_id] = []
        groups[eq_set_id].append(cand)

      # Process each group to select a winner for this round.
      for eq_set_id, items in groups.items():
        winner = None
        if len(items) == 1:
          # If there's only one candidate in an equivalence group, it wins automatically.
          winner = items[0]
        else:
          # If there's a collision, resolve it with an LLM tie-breaker.
          winner_id = await _resolve_collision(items, model)
          winner = None
          for item in items:
            if item['prop_id'] == winner_id:
              winner = item
              break

        if winner:
          # Add the winner to the final list and mark its equivalence set as seen for all subsequent rounds.
          final_propositions[winner['topic']].append(winner['text'])
          topic_idx, prop_idx = map(int, winner['prop_id'].split(':'))
          processed_by_topic_df.at[topic_idx, 'propositions'].at[
              prop_idx, 'selected'
          ] = True
          seen_equivalence_sets.add(eq_set_id)

      # Determine which topics still need to find a proposition for this final_rank.
      # A topic is "done" for this rank if its count of final propositions now exceeds the rank number.
      # A topic is also "done" if it has run out of candidates to propose.
      topics_in_round = []
      for topic_name in processed_by_topic_df['topic']:
        if len(final_propositions[topic_name]) <= final_rank:
          topic_idx = topic_name_to_id[topic_name]
          row = processed_by_topic_df.iloc[topic_idx]
          full_ranking = row[ranking_column]
          if next_rank_to_consider[topic_name] < len(full_ranking):
            topics_in_round.append(topic_name)

  return processed_by_topic_df, final_propositions


async def run_deduplication(
    by_topic_data,
    final_propositions_per_topic,
    model,
    ranking_column: str = 'full_schulze_ranking',
):
  """Orchestrates the entire proposition deduplication and selection process.

  This function takes the raw world model, identifies semantically equivalent
  propositions across different topics, and then uses a rank-filling draft
  algorithm to select the final, best-framed set of propositions for each topic,
  ensuring no duplicates are chosen.

  Args:
    by_topic_data: A pandas DataFrame where each row represents a topic. Expected
      to have at least 'r1_quotes_by_topic' and a column named 'propositions'. The
      'propositions' column contains a pandas DataFrame for each topic, where
      each row is a proposition.
    final_propositions_per_topic: The number of final propositions to select
      for each topic.
    model: The GenaiModel instance to use for equivalence detection and
      tie-breaking.
    ranking_column: The name of the column in `by_topic_data` that contains
        the ranked list of propositions to use for selection.

  Returns:
    A tuple containing:
      - by_topic_data: The modified DataFrame (which includes new IDs and
        selection metadata).
      - top_propositions_by_topic: A dictionary mapping each topic name to a
        list of the final, deduplicated proposition strings.
  """
  by_topic_data = _preprocess_and_add_ids(by_topic_data)

  equivalence_sets = await generate_equivalence_sets(by_topic_data, model)

  by_topic_data, top_propositions_by_topic = await select_final_propositions(
      by_topic_data,
      equivalence_sets,
      final_propositions_per_topic,
      model,
      ranking_column,
  )

  return by_topic_data, top_propositions_by_topic
