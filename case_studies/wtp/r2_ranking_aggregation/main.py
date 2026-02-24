"""
Command line utility for aggregating R2 ranking data.
"""

import argparse
import asyncio
from case_studies.wtp.participation import load_and_merge_participant_data, get_r2_preferences_from_dataframe
from case_studies.wtp.social_choice.schulze import get_schulze_ranking


async def main():
  parser = argparse.ArgumentParser(description='Aggregate R2 ranking data.')
  parser.add_argument(
      '--r2_url', required=True, help='Google Sheet URL for R2 data'
  )
  args = parser.parse_args()

  # Load participant data
  df = load_and_merge_participant_data(r2_url=args.r2_url)

  # Get preferences from the dataframe
  preferences_by_topic = get_r2_preferences_from_dataframe(df)

  # Run Schulze ranking for each topic
  for topic, preferences in preferences_by_topic.items():
    print(f'\n--- Results for topic: {topic} ---')
    schulze_results = get_schulze_ranking(preferences)
    for i, proposition in enumerate(schulze_results['top_propositions'], 1):
      print(f'{i}. {proposition}')


if __name__ == '__main__':
  asyncio.run(main())
