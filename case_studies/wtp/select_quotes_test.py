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

import pandas as pd
import unittest

from case_studies.wtp import select_quotes


class RunnerUtilsTest(unittest.TestCase):

  def test_select_gov_quotes_top_scores(self):
    # Create data with all unique rids
    data = [
        ['TextA', 'OpinionA', 1, 0.9],
        ['TextB', 'OpinionA', 2, 0.8],
        ['TextC', 'OpinionB', 3, 0.7],
        ['TextD', 'OpinionB', 4, 0.6]]
    df = pd.DataFrame(data,
        columns=['text', 'opinion', 'rid', select_quotes.AVERAGE_BRIDGING_COLUMN])
    gov_df = select_quotes.select_gov_quotes(df)
    gov_texts = set(gov_df['text'])
    self.assertEqual(len(gov_df), 2)
    self.assertIn('TextA', gov_texts)
    self.assertIn('TextC', gov_texts)

  def test_select_gov_quotes_unique_rids(self):
    # Create data where same rid has highest score in 2 opinions
    data = [
        ['TextA', 'OpinionA', 1, 0.9],
        ['TextB', 'OpinionA', 2, 0.8],
        ['TextC', 'OpinionB', 1, 0.7],
        ['TextD', 'OpinionB', 3, 0.6]]
    df = pd.DataFrame(data,
        columns=['text', 'opinion', 'rid', select_quotes.AVERAGE_BRIDGING_COLUMN])
    gov_df = select_quotes.select_gov_quotes(df)
    # all rids should be unique
    self.assertEqual(len(gov_df), gov_df['rid'].nunique())
    # only 1 of TextA or TextC should be picked,
    # even though both have highest score in their opinions
    gov_texts = set(gov_df['text'])
    self.assertTrue(
      ('TextA' in gov_texts and 'TextC' not in gov_texts) or
      ('TextA' not in gov_texts and 'TextC' in gov_texts))


if __name__ == "__main__":
  unittest.main()
