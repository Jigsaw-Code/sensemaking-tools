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

from typing import Any, Dict, List, Optional

Topic = Dict[str, Any]


def parse_topics_string(topics_string: str) -> List[Topic]:
  """Parses a topics string into a (possibly) nested list of Topic objects.

  The input string format is like
  "Topic1:Subtopic1:ThemeA;Topic2:Subtopic2;Topic3".
  Each segment is separated by ';'. Within a segment, topic, subtopic, and
  sub-subtopic (theme) are separated by ':'.

  Args:
      topics_string: A string representing the topics and their hierarchy.

  Returns:
      A list of Topic dictionaries. Each dictionary has a 'name' key,
      and optionally a 'subtopics' key if it has children.
      Example:
      [
          {'name': 'Topic A', 'subtopics': [
              {'name': 'Subtopic A.1', 'subtopics': [{'name': 'Theme A'}]}
          ]},
          {'name': 'Topic B', 'subtopics': [{'name': 'Subtopic B.1'}]},
          {'name': 'Topic C'}
      ]
  """
  if not topics_string:
    return []

  topics_map: Dict[str, Topic] = {}
  topic_entries = topics_string.split(";")
  for entry in topic_entries:
    parts = [p.strip() for p in entry.split(":")]
    topic_name = parts[0]
    if not topic_name:
      continue
    if topic_name not in topics_map:
      topics_map[topic_name] = {"name": topic_name}
    current_topic_obj = topics_map[topic_name]

    subtopic_name: Optional[str] = None
    if len(parts) > 1 and parts[1]:
      subtopic_name = parts[1]
    if subtopic_name:
      if "subtopics" not in current_topic_obj:
        current_topic_obj["subtopics"] = []

      # Find or create subtopic
      current_subtopic_obj: Optional[Topic] = None
      for sub_obj in current_topic_obj["subtopics"]:
        if sub_obj["name"] == subtopic_name:
          current_subtopic_obj = sub_obj
          break

      if current_subtopic_obj is None:
        new_subtopic_obj: Topic = {"name": subtopic_name}
        current_topic_obj["subtopics"].append(new_subtopic_obj)
        current_subtopic_obj = new_subtopic_obj

      subsubtopic_name: Optional[str] = None
      if len(parts) > 2 and parts[2]:
        subsubtopic_name = parts[2]
      if subsubtopic_name:
        if "subtopics" not in current_subtopic_obj:
          current_subtopic_obj["subtopics"] = []

        # Sub-subtopics are leaf nodes in this context, just add their name
        current_subtopic_obj["subtopics"].append({"name": subsubtopic_name})

  return list(topics_map.values())
