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
import { CommentWithVoteInfo, VoteTally } from "../../types";
import { AllTopicsSummary, TopicSummary } from "./topics";

// Mock the model response. This mock needs to be set up to return response specific for each test.
let mockThemesSummary: jest.SpyInstance;
let mockCommonGroundSummary: jest.SpyInstance;
let mockDifferencesSummary: jest.SpyInstance;
let mockGenerateText: jest.SpyInstance;

const TEST_COMMENTS: CommentWithVoteInfo[] = [
  {
    id: "1",
    text: "comment 1",
    voteInfo: {
      "0": new VoteTally(10, 5, 0),
      "1": new VoteTally(5, 10, 5),
    },
    topics: [{ name: "Topic A", subtopics: [{ name: "Subtopic A.1" }] }],
  },
  {
    id: "2",
    text: "comment 2",
    voteInfo: {
      "0": new VoteTally(10, 5, 0),
      "1": new VoteTally(5, 10, 5),
    },
    topics: [{ name: "Topic A", subtopics: [{ name: "Subtopic A.1" }] }],
  },
  {
    id: "3",
    text: "comment 3",
    voteInfo: {
      "0": new VoteTally(10, 5, 0),
      "1": new VoteTally(5, 10, 5),
    },
    topics: [{ name: "Topic A", subtopics: [{ name: "Subtopic A.2" }] }],
  },
  {
    id: "4",
    text: "comment 4",
    voteInfo: {
      "0": new VoteTally(10, 5, 0),
      "1": new VoteTally(5, 10, 5),
    },
    topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
  },
  {
    id: "5",
    text: "comment 5",
    voteInfo: {
      "0": new VoteTally(10, 5, 0),
      "1": new VoteTally(5, 10, 5),
    },
    topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
  },
];

describe("AllTopicsSummaryTest", () => {
  beforeEach(() => {
    mockThemesSummary = jest.spyOn(TopicSummary.prototype, "getThemesSummary");
    mockCommonGroundSummary = jest.spyOn(TopicSummary.prototype, "getCommonGroundSummary");
    mockDifferencesSummary = jest.spyOn(TopicSummary.prototype, "getDifferencesOfOpinionSummary");
    mockGenerateText = jest
      .spyOn(VertexModel.prototype, "generateText")
      .mockResolvedValue("A recursive summary...");
  });

  afterEach(() => {
    mockThemesSummary.mockRestore();
    mockCommonGroundSummary.mockRestore();
    mockDifferencesSummary.mockRestore();
    mockGenerateText.mockRestore();
  });
  it("should create a properly formatted topics summary", async () => {
    // Mock the LLM calls
    mockThemesSummary.mockReturnValue(
      Promise.resolve({
        text: "Themes were...",
      })
    );
    mockCommonGroundSummary.mockReturnValue(
      Promise.resolve({
        text: "Some points of common ground...",
        title: "Common ground between groups: ",
      })
    );
    mockDifferencesSummary.mockReturnValue(
      Promise.resolve({
        text: "Areas of disagreement between groups...",
        title: "Differences of opinion: ",
      })
    );

    expect(
      await new AllTopicsSummary(
        new GroupedSummaryStats(TEST_COMMENTS),
        new VertexModel("project123", "global"),
        undefined,
        "en"
      ).getSummary()
    ).toEqual({
      title: "## Topics",
      text: "From the statements submitted, 2 high level topics were identifiedas well as 3 subtopics. Based on voting patterns between the opinion groups described above, both points of common ground as well as differences of opinion between the groups have been identified and are described below.\n\n",
      subContents: [
        {
          title: "### Topic A (3 statements)",
          text: "This topic included 1 subtopic, comprising a total of 3 statements.",
          subContents: [
            {
              text: "A recursive summary...",
              type: "TopicSummary",
            },
            {
              text: "This subtopic had high alignment compared to the other subtopics.",
              title: "#### Subtopic A.1 (2 statements)",
              subContents: [
                {
                  text: "Themes were...",
                },
                {
                  text: "Some points of common ground...",
                  title: "Common ground between groups: ",
                },
                {
                  text: "Areas of disagreement between groups...",
                  title: "Differences of opinion: ",
                },
              ],
            },
          ],
        },
        {
          title: "### Topic B (2 statements)",
          text: "This topic included 1 subtopic, comprising a total of 2 statements.",
          subContents: [
            {
              text: "A recursive summary...",
              type: "TopicSummary",
            },
            {
              text: "This subtopic had high alignment compared to the other subtopics.",
              title: "#### Subtopic B.1 (2 statements)",
              subContents: [
                {
                  text: "Themes were...",
                },
                {
                  text: "Some points of common ground...",
                  title: "Common ground between groups: ",
                },
                {
                  text: "Areas of disagreement between groups...",
                  title: "Differences of opinion: ",
                },
              ],
            },
          ],
        },
      ],
    });
  });
});
