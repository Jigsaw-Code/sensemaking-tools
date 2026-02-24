"""
Command line utility for representative opinion ranking.
"""

import argparse
import asyncio
import pandas as pd
import os
from case_studies.wtp.models import genai_model
from case_studies.wtp.simulated_jury import simulated_jury
from case_studies.wtp.social_choice.representation import run_greedy_selection
from case_studies.wtp.participation import load_and_merge_participant_data, get_prompt_representation


async def main():
  parser = argparse.ArgumentParser(
      description='Run representative opinion ranking.'
  )
  parser.add_argument(
      '--r1_url', required=True, help='Google Sheet URL for R1 data'
  )
  parser.add_argument(
      '--verbose', action='store_true', help='Print verbose output.'
  )
  args = parser.parse_args()

  # Load participant data
  df = load_and_merge_participant_data(
      r1_url=args.r1_url, r1_merge_col='comment-id'
  )

  if args.verbose:
    print('--- Verbose Debugging ---')
    print(f'Total rows: {len(df)}')
    print(f"Distinct opinions: {df['opinion'].nunique()}")
    print(f"Distinct rdud values: {df['rdud'].nunique()}")
    print(f"Distinct comment_text entries: {df['comment_text'].nunique()}")
    print('\n--- R1 Dataframe Head ---')
    print(df.head())

  # Get unique participants and statements
  unique_participants = (
      df.drop_duplicates(subset=['rdud']).copy().reset_index(drop=True)
  )
  unique_statements = df['opinion'].unique().tolist()

  if args.verbose:
    print('\n--- Prompt representation for first participant ---')
    print(get_prompt_representation(unique_participants.iloc[0]))

  gemini_api_key = os.environ.get('GEMINI_API_KEY')
  if not gemini_api_key:
    raise ValueError('GEMINI_API_KEY not found in environment variables.')

  model = genai_model.GenaiModel(
      api_key=gemini_api_key,
      model_name='gemini-2.5-flash',
      embedding_model_name='text-embedding-004',
  )

  # Run the simulation and get the approval matrix
  results = await simulated_jury.run_simulated_jury(
      unique_participants,
      unique_statements,
      simulated_jury.VotingMode.APPROVAL,
      model,
  )

  # Parse the results into an approval matrix
  approval_matrix = pd.DataFrame(
      index=[p['rdud'] for _, p in unique_participants.iterrows()],
      columns=unique_statements,
  )
  for i, row in unique_participants.iterrows():
    participant_id = row['rdud']
    predictions = results[i].strip().split('\n')
    for stmt, pred_line in zip(unique_statements, predictions):
      approval_matrix.loc[participant_id, stmt] = bool(
          'agree' in pred_line.lower() and 'disagree' not in pred_line.lower()
      )

  # Assume that a missing response implies disapproval.
  approval_matrix.fillna(False, inplace=True)

  if args.verbose:
    print('\n--- Participant Vote Counts ---')
    for participant_id in approval_matrix.index:
      agrees = approval_matrix.loc[participant_id].sum()
      disagrees = len(approval_matrix.columns) - agrees
      print(f'Participant {participant_id}: A={agrees}, D={disagrees})')

    print('\n--- Vote Counts per Opinion ---')
    for opinion in approval_matrix.columns:
      approvals = approval_matrix[opinion].sum()
      disapprovals = len(approval_matrix) - approvals
      print(f'Opinion: "{opinion}" (A={approvals}, D={disapprovals})')

  # Run the greedy selection algorithm for each topic

  # Run the greedy selection algorithm for each topic
  all_voter_ids = df['rdud'].unique().tolist()
  topics = df['topic'].unique()

  print('\n--- Running Representative Selection Algorithm ---\n')

  for topic in topics:
    if pd.isna(topic) or topic.strip() == '':
      continue

    print(f'Topic: {topic}')

    topic_df = (
        df[df['topic'] == topic].drop_duplicates(subset=['opinion']).copy()
    )

    if len(topic_df) < 3:
      print('  -> Not enough unique statements to select 3.\n')
      continue

    top_3_statements = run_greedy_selection(
        topic_df, all_voter_ids, approval_matrix, k=3
    )

    print('  Top 3 Representative & Bridging Statements:')
    for i, stmt in enumerate(top_3_statements, 1):
      print(f'  {i}. {stmt}')
    print('-' * 40)


if __name__ == '__main__':
  asyncio.run(main())
