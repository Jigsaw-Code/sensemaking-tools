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
import unittest
from case_studies.wtp.input_parsing_lib import parse_topics_string


class TestParseTopicsString(unittest.TestCase):

  def test_empty_string(self):
    self.assertEqual(parse_topics_string(""), [])

  def test_single_topic(self):
    self.assertEqual(parse_topics_string("Topic A"), [{"name": "Topic A"}])

  def test_multiple_topics(self):
    expected = [{"name": "Topic A"}, {"name": "Topic B"}, {"name": "Topic C"}]
    self.assertEqual(parse_topics_string("Topic A;Topic B;Topic C"), expected)

  def test_topic_with_subtopic(self):
    expected = [{"name": "Topic A", "subtopics": [{"name": "Subtopic A.1"}]}]
    self.assertEqual(parse_topics_string("Topic A:Subtopic A.1"), expected)

  def test_multiple_topics_with_subtopics(self):
    expected = [
        {"name": "Topic A", "subtopics": [{"name": "Subtopic A.1"}]},
        {"name": "Topic B", "subtopics": [{"name": "Subtopic B.1"}]},
        {"name": "Topic C"},
    ]
    self.assertEqual(
        parse_topics_string(
            "Topic A:Subtopic A.1;Topic B:Subtopic B.1;Topic C"
        ),
        expected,
    )

  def test_topic_with_multiple_subtopics(self):
    expected = [{
        "name": "Topic A",
        "subtopics": [{"name": "Subtopic A.1"}, {"name": "Subtopic A.2"}],
    }]
    self.assertEqual(
        parse_topics_string("Topic A:Subtopic A.1;Topic A:Subtopic A.2"),
        expected,
    )

  def test_topic_with_subtopic_and_theme(self):
    expected = [{
        "name": "Topic A",
        "subtopics": [
            {"name": "Subtopic A.1", "subtopics": [{"name": "Theme X"}]}
        ],
    }]
    self.assertEqual(
        parse_topics_string("Topic A:Subtopic A.1:Theme X"), expected
    )

  def test_topic_with_subtopic_and_multiple_themes(self):
    expected = [{
        "name": "Topic A",
        "subtopics": [{
            "name": "Subtopic A.1",
            "subtopics": [{"name": "Theme X"}, {"name": "Theme Y"}],
        }],
    }]
    self.assertEqual(
        parse_topics_string(
            "Topic A:Subtopic A.1:Theme X;Topic A:Subtopic A.1:Theme Y"
        ),
        expected,
    )

  def test_topics_with_empty_sub_or_theme_parts(self):
    self.assertEqual(
        parse_topics_string("Topic A:;Topic B::ThemeY;Topic C"),
        [{"name": "Topic A"}, {"name": "Topic B"}, {"name": "Topic C"}],
    )

  def test_complex_nested_structure(self):
    input_str = (
        "Politics:US Elections:Debates;Economy:Global"
        " Markets;Politics:International Relations:Treaties;Economy:Global"
        " Markets:Stock Analysis"
    )
    expected = [
        {
            "name": "Politics",
            "subtopics": [
                {"name": "US Elections", "subtopics": [{"name": "Debates"}]},
                {
                    "name": "International Relations",
                    "subtopics": [{"name": "Treaties"}],
                },
            ],
        },
        {
            "name": "Economy",
            "subtopics": [{
                "name": "Global Markets",
                "subtopics": [{"name": "Stock Analysis"}],
            }],
        },
    ]
    self.assertEqual(parse_topics_string(input_str), expected)
