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

"""
Conversion utility to create a "Simple Format" survey from a CSV file.

"""

import argparse
from collections.abc import Sequence
import pandas as pd
from typing import TextIO


def write_question(output: TextIO, id: str, text: str, separator: str):
  clean_text = text.replace(':', ';')
  output.write(separator)
  output.write(id + '. ' + clean_text + '\n')
  output.write('\n')
  output.write('Strongly disagree\nSomewhat disagree\n')
  output.write('Neither agree nor disagree\n')
  output.write('Somewhat agree\nStrongly agree\n')


def create_survey(filename: str, output: str):
  props = pd.read_csv(filename, header=0)
  topics = props['topic']
  nuanced_rows = topics == 'Nuanced'
  topic_map = {
      element: str(index) for index, element in enumerate(set(topics))
  }

  with open(output, 'w') as output_file:
    output_file.write('[[Block:Topics]]\n')
    separator = ''
    for index, row in props.loc[~nuanced_rows].iterrows():
      tag = 'T' + topic_map[row['topic']] + 'R' + str(index)
      write_question(output_file, tag, row['text'], separator)
      separator = '\n\n'

    output_file.write('[[Block:NuancedTopics]]\n')
    separator = ''
    for index, row in props.loc[nuanced_rows].iterrows():
      tag = 'T' + topic_map[row['topic']] + 'R' + str(index)
      write_question(output_file, tag, row['text'], separator)
      separator = '\n\n'


def main():
  """Reads a set of questions from a exported CSV to create a Qualtrics Survey"""
  parser = argparse.ArgumentParser(
      description=(
          "Exported propositions used to create a Qualtrics Simple Format survey")
  )
  parser.add_argument(
      "--input_csv", required=True, help="Path to the input CSV file."
  )
  parser.add_argument(
      "--output_txt", required=True, help="Output filename."
  )
  args = parser.parse_args()

  create_survey(args.input_csv, args.output_txt)


if __name__ == "__main__":
  main()
