# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import pandas as pd


def main():
  """
  Reads a CSV file, renames columns, and saves the result to a new CSV file.
  """
  parser = argparse.ArgumentParser(description='Rename columns in a CSV file.')
  parser.add_argument(
      '--input_csv',
      type=str,
      required=True,
      help='The path to the input CSV file.',
  )
  parser.add_argument(
      '--output_csv',
      type=str,
      required=True,
      help='The path to save the output CSV file.',
  )

  args = parser.parse_args()

  # Read the input CSV into a pandas DataFrame
  df = pd.read_csv(args.input_csv)

  df.rename(
      columns={
          # Open ended questions
          'Q48': 'visual_question_1',
          'GOV1': 'question_1',
          'GOV2': 'question_2',
          'GOV3': 'question_3',
          'GOV4': 'question_4',
          'GOV5': 'question_5',
          'GOV6': 'question_6',
          'GOV7': 'question_7',
          'GOV8': 'question_8',
          'GOV9': 'question_9',
          'GOV10': 'question_10',
          'GOV11': 'question_11',
          'GOV12': 'question_12',
          'GOV13': 'question_13',
          'GOV14': 'question_14',
          'GOV15': 'question_15',
          'GOV16': 'question_16',
          'GOV17': 'question_17',
          'GOV18': 'question_18',
          'GOV19': 'question_19',
          'GOV20': 'question_20',
          'GOV21': 'question_21',
          'GOV22': 'question_22',
          'GOV23': 'question_23',
          'GOV24': 'question_24',
          'GOV25': 'question_25',
          'GOV26': 'question_26',
          'GOV27': 'question_27',
          # Sets of three quotes to rank and an open ended response.
          'Q_Ranking_T1_10': 'ranking_1_q_1',
          'Q_Ranking_T1_11': 'ranking_1_q_2',
          'Q_Ranking_T1_12': 'ranking_1_q_3',
          'Q_Ranking_T1_13': 'ranking_1_q_4',
          'Q_Ranking_T1_14': 'ranking_1_q_5',
          'Q_Ranking_T1_15': 'ranking_1_q_6',
          'Q_Ranking_T1_16': 'ranking_1_q_7',
          'Q_Ranking_T1_17': 'ranking_1_q_8',
          'Q_Ranking_T1_18': 'ranking_1_q_9',
          'Q_Ranking_T1_Comment': 'ranking_1_q_c',
          'Q_Ranking_T2_10': 'ranking_2_q_1',
          'Q_Ranking_T2_11': 'ranking_2_q_2',
          'Q_Ranking_T2_12': 'ranking_2_q_3',
          'Q_Ranking_T2_13': 'ranking_2_q_4',
          'Q_Ranking_T2_14': 'ranking_2_q_5',
          'Q_Ranking_T2_15': 'ranking_2_q_6',
          'Q_Ranking_T2_Comment': 'ranking_2_q_c',
          'Q_Ranking_T3_10': 'ranking_3_q_1',
          'Q_Ranking_T3_11': 'ranking_3_q_2',
          'Q_Ranking_T3_12': 'ranking_3_q_3',
          'Q_Ranking_T3_13': 'ranking_3_q_4',
          'Q_Ranking_T3_14': 'ranking_3_q_5',
          'Q_Ranking_T3_Comment': 'ranking_3_q_c',
          'Q_Ranking_T4_10': 'ranking_4_q_1',
          'Q_Ranking_T4_11': 'ranking_4_q_2',
          'Q_Ranking_T4_12': 'ranking_4_q_3',
          'Q_Ranking_T4_13': 'ranking_4_q_4',
          'Q_Ranking_T4_14': 'ranking_4_q_5',
          'Q_Ranking_T4_15': 'ranking_4_q_6',
          'Q_Ranking_T4_16': 'ranking_4_q_7',
          'Q_Ranking_T4_Comment': 'ranking_4_q_c',
          # Meta Questions
          'Measures_Q3': 'meta_question_1',
          'Measures_Q6': 'meta_question_2',
          'Measures_Q8_1': 'meta_question_3',
          'Measures_Q8_2': 'meta_question_4',
          'Measures_Q8_3': 'meta_question_5',
          'Measures_Q8_4': 'meta_question_6',
          'Measures_Q8_5': 'meta_question_7',
          'Measures_Q9': 'meta_question_8',
      },
      inplace=True,
  )

  # Write the modified DataFrame to the output CSV file
  df.to_csv(args.output_csv, index=False)
  print(
      f'Successfully processed {args.input_csv} and saved to {args.output_csv}'
  )


if __name__ == '__main__':
  main()
