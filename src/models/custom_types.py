# Copyright 2026 Google LLC
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

from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, Field


class FlatTopic(BaseModel):
  """Represents a simple topic with just a name."""

  name: str


class FlatTopicList(BaseModel):
  """Represents a list of flat topics."""

  topics: list[FlatTopic]


class NestedTopic(BaseModel):
  """Represents a topic that can contain subtopics.

  The subtopics themselves can be FlatTopic or NestedTopic, allowing for
  recursive structures.
  """

  name: str
  subtopics: List[Topic] = Field(
      default_factory=list, description="A list of subtopics under this topic."
  )


Topic = Union[NestedTopic, FlatTopic]


class StatementRecord(BaseModel):
  """Represents a statement that has been categorized with one or more topics.

  Typically used for the direct output of an LLM after a categorization step.
  """

  id: str = Field(description="The unique identifier of the statement.")
  topics: list[FlatTopic] = Field(
      description="A list of topics assigned to the statement.",
  )
  quote_id: Optional[str] = Field(
      default=None,
      description=(
          "The id of the specific quote within the statement, if applicable."
      ),
  )


class StatementRecordList(BaseModel):
  """Wrapper for a list of StatementRecords to use as a schema for the LLM."""

  items: list[StatementRecord]


class Quote(BaseModel):
  """Represents an extracted quote from a statement, associated with a specific topic or opinion."""

  id: str = Field(description="The unique identifier for the quote.")
  text: str = Field(description="The text content of the quote.")
  topic: Topic = Field(
      description=(
          "The topic or opinion specifically associated with this quote."
      )
  )


class Statement(BaseModel):
  """Represents a single statement (e.g., a user response) to be processed."""

  # TODO: Ensure the topics field is initialized as an empty list by default. This will eliminate the need for if statement.topics is None: checks.

  id: str = Field(description="The unique identifier for the statement.")
  text: str = Field(description="The text content of the statement.")
  topics: Optional[List[Topic]] = Field(
      default_factory=list,
      description="A list of topics assigned to the overall statement.",
  )
  quotes: Optional[List[Quote]] = Field(
      default_factory=list,
      description="A list of specific quotes extracted from the statement.",
  )


class EvaluationResult(BaseModel):
  score: int = Field(description="The evaluation score.")
  reasoning: str = Field(description="The reasoning for the score.")


class OpinionResponseSchema(BaseModel):
  """Non-recursive schema for opinion generation to avoid SDK recursion errors."""

  name: str
  subtopics: list[FlatTopic]


class ScoreResponse(BaseModel):
  """Schema for score generation for moderation and bridging."""

  score: float = Field(description="The estimated probability (0.0 to 1.0).")
