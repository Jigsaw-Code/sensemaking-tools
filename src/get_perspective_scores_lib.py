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

from googleapiclient import discovery


def init_client(api_key: str):
  return discovery.build(
      'commentanalyzer',
      'v1alpha1',
      developerKey=api_key,
      discoveryServiceUrl='https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1',
      static_discovery=False,
  )


# Migrating to Gemini soon
def score_text(client, text: str, attributes: list[str]) -> dict[str, float]:
  requested_attributes = {attribute: {} for attribute in attributes}

  request = {
      'comment': {'text': text},
      'requestedAttributes': requested_attributes,
      'languages': ['en'],
  }
  response = client.comments().analyze(body=request).execute()
  return {
      attribute: extract_attribute_score(response, attribute)
      for attribute in attributes
  }


def extract_attribute_score(perspective_response, attribute):
  return perspective_response['attributeScores'][attribute]['summaryScore'][
      'value'
  ]
