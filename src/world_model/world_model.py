import pickle
import os
import pandas as pd
from functools import reduce
import json
from typing import Union, Optional
import logging


def load_world_model(file_path: str) -> Union[pd.DataFrame, dict]:
  """
  Loads a world model from a pickle file.

  Args:
      file_path (str): The path to the .pkl file.

  Returns:
      The loaded world model object, expected to be a DataFrame or a dict.
  """
  if not os.path.exists(file_path):
    raise FileNotFoundError(f'Error: File not found at {file_path}')

  try:
    with open(file_path, 'rb') as f:
      return pickle.load(f)
  except Exception as e:
    raise IOError(f'An error occurred while loading the pickle file: {e}')


def get_nested_attribute(
    obj: Union[pd.DataFrame, dict], attr_string: str
) -> Optional[Union[pd.DataFrame, pd.Series, dict, list, str, int, float]]:
  """
  Retrieves a nested attribute from an object using a dot-separated string.

  Args:
      obj: The object to search within (typically a DataFrame or dict).
      attr_string (str): A dot-separated string representing the nested attribute path.

  Returns:
      The value of the nested attribute, or None if not found.
  """
  try:
    return reduce(
        lambda o, a: o.get(a) if isinstance(o, dict) else getattr(o, a),
        attr_string.split('.'),
        obj,
    )
  except (AttributeError, KeyError):
    return None


def get_selected_by_opinion_propositions(world_model_data, top_n=None):
  """
  Extracts the top-ranked propositions for each opinion group.
  """
  by_opinion_df = world_model_data.get('world_model')
  if by_opinion_df is None:
    return pd.DataFrame()

  # Pre-calculate approval rates if the matrix is available
  approval_rates = None
  if 'initial_approval_matrix' in world_model_data:
    approval_rates = world_model_data['initial_approval_matrix'].mean()

  results = []
  for _, row in by_opinion_df.iterrows():
    topic = row.get('topic')
    opinion = row.get('opinion')
    schulze_ranking = row.get('opinion_level_schulze_ranking', [])
    pav_ranking = row.get('opinion_level_pav_ranking', [])

    # Determine the primary ranking to use
    primary_ranking = pav_ranking if pav_ranking else schulze_ranking
    limit = top_n if top_n is not None and top_n > 0 else len(primary_ranking)

    for i, prop in enumerate(primary_ranking[:limit], 1):
      rate = approval_rates.get(prop) if approval_rates is not None else None
      schulze_rank = (
          schulze_ranking.index(prop) + 1 if prop in schulze_ranking else None
      )
      pav_rank = pav_ranking.index(prop) + 1 if prop in pav_ranking else None

      results.append({
          'topic': topic,
          'opinion': opinion,
          'rank': i,
          'proposition': prop,
          'approval_rate': rate,
          'schulze_rank': schulze_rank,
          'pav_rank': pav_rank,
      })
  return pd.DataFrame(results)


def get_selected_by_topic_propositions(world_model_data, top_n=None):
  """
  Extracts the final selected propositions for each topic, after the
  topic-level jury and deduplication.
  """
  topic_level_df = world_model_data.get('topic_level_results')
  if topic_level_df is None:
    return pd.DataFrame()

  results = []
  for _, row in topic_level_df.iterrows():
    topic = row.get('topic')
    propositions_df = row.get('propositions')

    if (
        isinstance(propositions_df, pd.DataFrame)
        and 'selected' in propositions_df.columns
        and propositions_df['selected'].any()
    ):
      selected_df = propositions_df[propositions_df['selected']].copy()
      schulze_ranking = row.get('full_schulze_ranking', [])
      pav_ranking = row.get('topic_level_pav_ranking', [])

      # Use PAV for ranking if it exists, otherwise fall back to Schulze
      primary_ranking = pav_ranking if pav_ranking else schulze_ranking
      if isinstance(primary_ranking, list) and primary_ranking:
        rank_map = {prop: i + 1 for i, prop in enumerate(primary_ranking)}
        selected_df['rank'] = selected_df['proposition'].map(rank_map)
        selected_df = selected_df.sort_values(by='rank')
      else:
        selected_df['rank'] = range(1, len(selected_df) + 1)

      limit = top_n if top_n is not None and top_n > 0 else len(selected_df)

      for _, prop_row in selected_df.head(limit).iterrows():
        prop_text = prop_row['proposition']
        schulze_rank = (
            schulze_ranking.index(prop_text) + 1
            if prop_text in schulze_ranking
            else None
        )
        pav_rank = (
            pav_ranking.index(prop_text) + 1
            if prop_text in pav_ranking
            else None
        )

        results.append({
            'topic': topic,
            'rank': prop_row['rank'],
            'proposition': prop_text,
            'opinion': prop_row.get('opinion'),
            'r1_quotes_by_topic': row.get('r1_quotes_by_topic'),
            'r1_quotes_by_opinion': prop_row.get('r1_quotes_by_opinion'),
            'approval_rate': prop_row.get('approval_rate'),
            'schulze_rank': schulze_rank,
            'pav_rank': pav_rank,
        })
  return pd.DataFrame(results)


def get_selected_nuanced_propositions(world_model_data, top_n=None):
  """
  Extracts the top-ranked nuanced propositions with full ranking details.
  """
  top_nuanced_df = world_model_data.get('top_nuanced_propositions')
  if top_nuanced_df is None or top_nuanced_df.empty:
    return pd.DataFrame(
        columns=['topic', 'rank', 'proposition', 'approval_rate']
    )

  # Pre-calculate approval rates if the matrix is available
  approval_rates = None
  if 'nuanced_approval_matrix' in world_model_data:
    approval_rates = world_model_data['nuanced_approval_matrix'].mean()

  schulze_ranking = world_model_data.get('nuanced_schulze_ranking', [])
  pav_ranking = world_model_data.get('nuanced_pav_ranking', [])

  limit = top_n if top_n and top_n > 0 else len(top_nuanced_df)
  propositions_to_process = top_nuanced_df.head(limit)

  results = []
  for i, row in propositions_to_process.iterrows():
    prop = row['proposition']
    rate = approval_rates.get(prop) if approval_rates is not None else None
    schulze_rank = (
        schulze_ranking.index(prop) + 1 if prop in schulze_ranking else None
    )
    pav_rank = pav_ranking.index(prop) + 1 if prop in pav_ranking else None

    results.append({
        'topic': 'Nuanced',
        'rank': i + 1,
        'proposition': prop,
        'approval_rate': rate,
        'schulze_rank': schulze_rank,
        'pav_rank': pav_rank,
    })

  return pd.DataFrame(results)


def get_selected_propositions(
    world_model_data, top_n_topic=None, top_n_nuanced=None
):
  """
  Extracts all top-ranked propositions (topic-level and nuanced).
  """
  topic_props = get_selected_by_topic_propositions(
      world_model_data, top_n=top_n_topic
  )
  nuanced_props = get_selected_nuanced_propositions(
      world_model_data, top_n=top_n_nuanced
  )

  return pd.concat([topic_props, nuanced_props], ignore_index=True)


def get_participant_data(world_model_data, data_source='both'):
  """
  Extracts participant data from the world model.
  """
  world_model_df = world_model_data.get('world_model')
  if world_model_df is None:
    return pd.DataFrame()

  r1_dfs = [row['r1_df'] for _, row in world_model_df.iterrows()]
  r2_dfs = [row['r2_df'] for _, row in world_model_df.iterrows()]

  if not r1_dfs and not r2_dfs:
    return pd.DataFrame()

  if data_source == 'r1':
    return pd.concat(r1_dfs, ignore_index=True).drop_duplicates(subset=['rid'])
  elif data_source == 'r2':
    return pd.concat(r2_dfs, ignore_index=True).drop_duplicates(subset=['rid'])
  else:  # both
    r1_full = pd.concat(r1_dfs, ignore_index=True).drop_duplicates(
        subset=['rid']
    )
    r2_full = pd.concat(r2_dfs, ignore_index=True).drop_duplicates(
        subset=['rid']
    )
    return pd.merge(r1_full, r2_full, on='rid', how='outer')


def get_simulation_results(world_model_data, aggregation='by_topic'):
  """
  Extracts simulation results from the world model.
  """
  world_model_df = world_model_data.get('world_model')
  if (
      world_model_df is None
      or 'simulation_results' not in world_model_df.columns
  ):
    return pd.DataFrame()

  all_results = []
  for _, row in world_model_df.iterrows():
    sim_results = row.get('simulation_results')
    if isinstance(sim_results, pd.DataFrame) and not sim_results.empty:
      sim_results['topic'] = row['topic']
      all_results.append(sim_results)

  if not all_results:
    return pd.DataFrame()

  combined = pd.concat(all_results, ignore_index=True)
  if aggregation == 'by_topic':
    return combined
  else:  # combined
    return combined.drop(columns=['topic'])


def get_failed_tries(world_model_data, aggregation='by_topic'):
  """
  Extracts failed tries from the simulation results.
  """
  sim_results = get_simulation_results(world_model_data, aggregation)
  if sim_results.empty or 'failed_tries' not in sim_results.columns:
    return pd.DataFrame()

  all_failures = []
  for _, row in sim_results.iterrows():
    failures = row.get('failed_tries')
    if isinstance(failures, pd.DataFrame) and not failures.empty:
      failures['rid'] = row.get('rid')
      if 'topic' in row:
        failures['topic'] = row['topic']
      all_failures.append(failures)

  if not all_failures:
    return pd.DataFrame()

  return pd.concat(all_failures, ignore_index=True)


def get_all_by_opinion_propositions(world_model_data):
  """
  Extracts all propositions ranked at the opinion level, before topic-level
  aggregation.
  """
  return get_selected_by_opinion_propositions(world_model_data, top_n=None)


def get_all_by_topic_propositions(world_model_data):
  """
  Extracts all propositions ranked at the topic level, before deduplication.
  """
  topic_level_df = world_model_data.get('topic_level_results')
  if topic_level_df is None:
    return pd.DataFrame()

  all_topic_props = []
  for _, row in topic_level_df.iterrows():
    propositions_df = row.get('propositions')
    if isinstance(propositions_df, pd.DataFrame) and not propositions_df.empty:
      # Create a copy to avoid modifying the original DataFrame
      ranked_df = propositions_df.copy()
      schulze_ranking = row.get('full_schulze_ranking', [])
      pav_ranking = row.get('topic_level_pav_ranking', [])

      # Add both rankings if they exist
      if isinstance(schulze_ranking, list) and schulze_ranking:
        s_rank_map = {p: i + 1 for i, p in enumerate(schulze_ranking)}
        ranked_df['schulze_rank'] = ranked_df['proposition'].map(s_rank_map)
      if isinstance(pav_ranking, list) and pav_ranking:
        p_rank_map = {p: i + 1 for i, p in enumerate(pav_ranking)}
        ranked_df['pav_rank'] = ranked_df['proposition'].map(p_rank_map)

      # Determine primary rank
      if 'pav_rank' in ranked_df.columns:
        ranked_df['rank'] = ranked_df['pav_rank']
      elif 'schulze_rank' in ranked_df.columns:
        ranked_df['rank'] = ranked_df['schulze_rank']

      all_topic_props.append(ranked_df)

  if not all_topic_props:
    return pd.DataFrame()

  return pd.concat(all_topic_props, ignore_index=True)


def get_all_nuanced_propositions(world_model_data):
  """
  Extracts all nuanced propositions that were generated, before selection or
  ranking, and enriches them with available data.
  """
  nuanced_df_raw = world_model_data.get('nuanced_propositions')
  if (
      nuanced_df_raw is None
      or nuanced_df_raw.empty
      or 'result' not in nuanced_df_raw.columns
  ):
    return pd.DataFrame(columns=['proposition'])

  all_propositions_list = nuanced_df_raw['result'].iloc[0]
  if not isinstance(all_propositions_list, list):
    return pd.DataFrame(columns=['proposition'])

  # Create the base DataFrame
  results_df = pd.DataFrame({'proposition': all_propositions_list})
  results_df['topic'] = 'Nuanced'

  # --- Enrich with Approval Rate ---
  if 'nuanced_approval_matrix' in world_model_data:
    approval_rates = world_model_data['nuanced_approval_matrix'].mean()
    results_df['approval_rate'] = results_df['proposition'].map(approval_rates)
  else:
    results_df['approval_rate'] = None

  # --- Enrich with Schulze Rank ---
  if 'nuanced_schulze_ranking' in world_model_data:
    schulze_ranking = world_model_data['nuanced_schulze_ranking']
    schulze_rank_map = {prop: i + 1 for i, prop in enumerate(schulze_ranking)}
    results_df['schulze_rank'] = results_df['proposition'].map(schulze_rank_map)
  else:
    results_df['schulze_rank'] = None

  # --- Enrich with PAV Rank ---
  if 'nuanced_pav_ranking' in world_model_data:
    pav_ranking = world_model_data['nuanced_pav_ranking']
    pav_rank_map = {prop: i + 1 for i, prop in enumerate(pav_ranking)}
    results_df['pav_rank'] = results_df['proposition'].map(pav_rank_map)
  else:
    results_df['pav_rank'] = None

  # --- Determine Primary Rank for Sorting (only if ranking data exists) ---
  if (
      results_df['schulze_rank'].notna().any()
      or results_df['pav_rank'].notna().any()
  ):
    # To be future-proof, we determine which ranking method was used for the final
    # 'rank' by inspecting the already-selected top propositions.
    selected_props = get_selected_nuanced_propositions(world_model_data)
    primary_rank_col = None

    if not selected_props.empty and 'rank' in selected_props.columns:
      # Check if PAV was the primary ranking method
      if 'pav_rank' in selected_props.columns and selected_props['rank'].equals(
          selected_props['pav_rank']
      ):
        primary_rank_col = 'pav_rank'
      # Check if Schulze was the primary ranking method
      elif 'schulze_rank' in selected_props.columns and selected_props[
          'rank'
      ].equals(selected_props['schulze_rank']):
        primary_rank_col = 'schulze_rank'
      else:
        # If neither matched, something is wrong with the upstream selection logic.
        raise ValueError(
            "Could not determine the primary ranking method. The 'rank' column"
            " in selected propositions does not match 'pav_rank' or"
            " 'schulze_rank'."
        )
    else:
      # Default to PAV if it exists, otherwise Schulze. This handles cases
      # where selection hasn't run but ranking has.
      if (
          'pav_rank' in results_df.columns
          and results_df['pav_rank'].notna().any()
      ):
        logging.warning(
            'Could not determine primary ranking method from selected'
            " propositions. Defaulting to 'pav_rank' for sorting."
        )
        primary_rank_col = 'pav_rank'
      else:
        logging.warning(
            'Could not determine primary ranking method from selected'
            " propositions. Defaulting to 'schulze_rank' for sorting."
        )
        primary_rank_col = 'schulze_rank'

    results_df['rank'] = results_df[primary_rank_col]
    results_df = results_df.sort_values(by='rank').reset_index(drop=True)

  return results_df


def get_simulated_jury_stats(world_model_data: dict) -> pd.DataFrame:
  """
  Extracts the simulated jury failure statistics from the world model.
  """
  if 'simulated_jury_stats' not in world_model_data:
    raise ValueError(
        'Simulated jury stats not found. This query can only be run on'
        ' world models generated with a compatible version of the'
        ' proposition_refinement pipeline.'
    )

  stats_list = world_model_data['simulated_jury_stats']
  if not stats_list:
    return pd.DataFrame()

  return pd.DataFrame(stats_list)
