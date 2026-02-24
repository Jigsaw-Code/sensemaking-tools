// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { VertexModel } from "../../models/vertex_model";
import { GroupedSummaryStats } from "../../stats/group_informed";
import { Comment } from "../../types";
import { IntroSummary } from "./intro";

const TEST_COMMENTS: Comment[] = [
  {
    id: "1",
    text: "comment 1",
    topics: [{ name: "Topic A", subtopics: [{ name: "Subtopic A.1" }] }],
  },
  {
    id: "2",
    text: "comment 2",
    topics: [{ name: "Topic A", subtopics: [{ name: "Subtopic A.1" }] }],
  },
  {
    id: "3",
    text: "comment 3",
    topics: [{ name: "Topic A", subtopics: [{ name: "Subtopic A.2" }] }],
  },
  {
    id: "4",
    text: "comment 4",
    topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
  },
];

describe("IntroTest", () => {
  it("should create an intro section", async () => {
    expect(
      await new IntroSummary(
        new GroupedSummaryStats(TEST_COMMENTS),
        new VertexModel("project123", "global")
      ).getSummary()
    ).toEqual({
      title: "## Introduction",
      text: `This report summarizes the results of public input, encompassing:
 * __4 statements__
 * __0 votes__
 * 2 topics
 * 3 subtopics

All voters were anonymous.`,
    });
  });
});
