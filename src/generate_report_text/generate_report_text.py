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
"""
Generates report text based on categorized quotes spreadsheet.

Input csv should contain topic, opinion, and quote columns.
Additional context file can be plain text or contain markdown.

Example usage:
python3 -m src.generate_report_text.generate_report_text \
  --input_csv <INPUT_CSV> \
  --additional_context_file <ADDITIONAL_CONTEXT_TEXT_FILE> \
  --output_dir <OUTPUT_DIR> \
  --model_name gemini-2.5-pro
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import pandas as pd

from src import runner_utils
from src.models import genai_model
from src.generate_report_text import generate_report_text_prompts


async def generate_text_in_parallel(
    model: genai_model.GenaiModel, prompt_obj_list: list[dict]
) -> pd.DataFrame:
  # prompt_obj_list members should have prompt, optional topic, opinion fields
  # lambda function extracts text field from response
  response_df, _, _, _ = await model.process_prompts_concurrently(
      prompt_obj_list, lambda x, _: x['text']
  )
  return response_df.sort_values('job_id', ascending=True)


async def generate_text(model: genai_model.GenaiModel, prompt: str) -> str:
  """Utility function to run a single prompt through the model."""
  # lambda function extracts text field from response
  response_df, _, _, _ = await model.process_prompts_concurrently(
      [{'prompt': prompt}], lambda x, _: x['text']
  )
  return response_df.iloc[0].result


def get_opinion_size(
    topic: str, opinion: str, categorized_quotes_df: pd.DataFrame
) -> int:
  # filter by both topic and opinion, in case the same opinion appears in multiple
  # topics, e.g. "Other"
  topic_df = categorized_quotes_df[categorized_quotes_df['topic'] == topic]
  return topic_df[topic_df['opinion'] == opinion].shape[0]


def get_opinions_per_topic(df: pd.DataFrame) -> dict[str, list[str]]:
  """Returns dict mapping from topic name to list of opinions for that topic."""
  return df.groupby('topic')['opinion'].unique().to_dict()


def get_summary_for_opinion(
    topic: str, opinion: str, opinion_summaries_df: pd.DataFrame
) -> str:
  # filter by both topic and opinion, in case the same opinion appears in multiple
  # topics, e.g. "Other"
  topic_df = opinion_summaries_df[opinion_summaries_df['topic'] == topic]
  opinion_df = topic_df[topic_df['opinion'] == opinion]
  if len(opinion_df) != 1:
    raise ValueError('Incorrect opinion count for %s, %s' % (topic, opinion))
  return opinion_df.iloc[0]['result']


async def generate_topic_summaries(
    model: genai_model.GenaiModel,
    categorized_quotes_df: pd.DataFrame,
    opinion_summaries_df: pd.DataFrame,
    additional_context: str,
) -> pd.DataFrame:
  """Generates topic summaries based on opinion summaries."""
  topic_summaries_request_prompts = []
  all_topics = opinion_summaries_df['topic'].unique().tolist()
  opinions_per_topic = get_opinions_per_topic(categorized_quotes_df)
  for topic, opinions in opinions_per_topic.items():
    # Get dict of opinion to summary, only for opinions in that topic
    opinion_summaries_for_topic = {
        op: get_summary_for_opinion(topic, op, opinion_summaries_df)
        for op in opinions
    }
    opinion_sizes_for_topic = {
        op: get_opinion_size(topic, op, categorized_quotes_df)
        for op in opinions
    }
    topic_summaries_request_prompts.append({
        'prompt': generate_report_text_prompts.get_topic_summary_prompt(
            topic,
            additional_context,
            opinion_summaries_for_topic,
            opinion_sizes_for_topic,
            all_topics,
        ),
        'topic': topic,
    })
  print('Summarizing %d topics' % len(topic_summaries_request_prompts))
  return await generate_text_in_parallel(model, topic_summaries_request_prompts)


async def generate_opinion_summaries(
    model: genai_model.GenaiModel,
    categorized_quotes_df: pd.DataFrame,
    additional_context: str,
) -> pd.DataFrame:
  """Generates opinion summaries based on categorized quotes."""
  opinions_per_topic = get_opinions_per_topic(categorized_quotes_df)
  opinion_summaries_request_prompts = []
  for topic, opinions in opinions_per_topic.items():
    topic_df = categorized_quotes_df[categorized_quotes_df['topic'] == topic]
    for opinion in opinions:
      quotes = topic_df[topic_df['opinion'] == opinion]['quote'].tolist()
      opinion_summaries_request_prompts.append({
          'prompt': generate_report_text_prompts.get_opinion_summary_prompt(
              topic, opinion, additional_context, quotes, opinions_per_topic
          ),
          'topic': topic,
          'opinion': opinion,
      })
  print('Summarizing %d opinions' % len(opinion_summaries_request_prompts))
  return await generate_text_in_parallel(
      model, opinion_summaries_request_prompts
  )


async def generate_overview_summary(
    model: genai_model.GenaiModel,
    topic_summaries_df: pd.DataFrame,
    additional_context: str,
) -> str:
  topic_summaries = topic_summaries_df.set_index('topic')['result'].to_dict()
  overview_prompt = generate_report_text_prompts.get_overview_prompt(
      additional_context, topic_summaries
  )
  print('Creating overview')
  return await generate_text(model, overview_prompt)


def write_json_file(data: dict, output_dir: str, filename: str):
  """Writes CSV file to disk, ensuring directory exists."""
  os.makedirs(output_dir, exist_ok=True)  # make sure dir exists
  full_filepath = os.path.join(output_dir, filename)
  with open(full_filepath, 'w') as file:
    file.write(json.dumps(data, indent=2))
  print(f'Wrote {full_filepath}')


def get_combined_report_data(
    overview_text: str,
    topic_summaries_df: pd.DataFrame,
    opinion_summaries_df: pd.DataFrame,
) -> tuple[dict, dict]:
  """Creates final data for report, returns tuple of data_with_opinions, data_without_opinions."""
  # Create data with overview, topic, and opinion summaries
  topic_map = {}
  for _, op_row in opinion_summaries_df.iterrows():
    topic = op_row['topic']
    opinion = op_row['opinion']
    opinion_summary = op_row['result']
    topic_summary = topic_summaries_df[
        topic_summaries_df['topic'] == topic
    ].iloc[0]['result']
    if topic not in topic_map:
      # initialize the topic object with it' summary
      topic_map[topic] = {
          'title': topic,
          'text': topic_summary,
          'sub_contents': [],
      }
    # add this opinion summary
    topic_map[topic]['sub_contents'].append(
        {'title': opinion, 'text': opinion_summary}
    )
  data_with_opinions = {
      'text': overview_text,
      'sub_contents': list(topic_map.values()),
  }
  # Create copy without opinions
  data_without_opinions = copy.deepcopy(data_with_opinions)
  for topic_obj in data_without_opinions['sub_contents']:
    del topic_obj['sub_contents']
  return data_with_opinions, data_without_opinions


async def main():
  parser = argparse.ArgumentParser(description='Generates the report text.')
  parser.add_argument(
      '--input_csv',
      required=True,
      help='Path to CSV output of categorization_runner.py',
  )
  runner_utils.add_additional_context_args(
      parser,
      help_str="Additional context for the categorization.",
  )
  parser.add_argument(
      '--output_dir', required=True, help='Path to output directory.'
  )
  parser.add_argument(
      '--model_name',
      type=str,
      default='gemini-2.5-pro',
      help='The name of the Vertex AI model to use. Default: gemini-2.5-pro.',
  )
  args = parser.parse_args()

  model = genai_model.GenaiModel(model_name=args.model_name)

  # Load data and additional context
  categorized_quotes_df = pd.read_csv(args.input_csv)
  additional_context = runner_utils.get_additional_context(args)

  # Drop Other from both topics and opinions
  categorized_quotes_df = categorized_quotes_df[
      categorized_quotes_df['topic'] != 'Other'
  ]
  categorized_quotes_df = categorized_quotes_df[
      categorized_quotes_df['opinion'] != 'Other'
  ]

  # First summaries all opinions
  opinion_summaries_df = await generate_opinion_summaries(
      model, categorized_quotes_df, additional_context
  )

  # Create topic summaries based on opinion summaries
  topic_summaries_df = await generate_topic_summaries(
      model, categorized_quotes_df, opinion_summaries_df, additional_context
  )

  # Create overview summary based on topic summaries
  overview_text = await generate_overview_summary(
      model, topic_summaries_df, additional_context
  )

  # Create final report data and write to disk
  data_with_opinions, data_without_opinions = get_combined_report_data(
      overview_text, topic_summaries_df, opinion_summaries_df
  )
  write_json_file(data_without_opinions, args.output_dir, 'report_data.json')
  write_json_file(
      data_with_opinions, args.output_dir, 'report_data_with_opinions.json'
  )


if __name__ == '__main__':
  asyncio.run(main())
