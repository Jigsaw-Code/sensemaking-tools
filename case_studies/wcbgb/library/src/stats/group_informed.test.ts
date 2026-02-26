// Copyright 2024 Google LLC
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

import { GroupedSummaryStats } from "./group_informed";
import { Comment, VoteTally } from "../types";

const TEST_COMMENTS = [
  {
    id: "1",
    text: "comment1",
    voteInfo: {
      "0": new VoteTally(20, 10, 0),
      "1": new VoteTally(5, 10, 5),
    },
  },
  {
    id: "2",
    text: "comment2",
    voteInfo: {
      "0": new VoteTally(2, 5, 3),
      "1": new VoteTally(5, 3, 2),
    },
  },
];

describe("GroupSummaryStats Test", () => {
  it("should get the total number of votes from multiple comments", () => {
    const summaryStats = new GroupedSummaryStats(TEST_COMMENTS);
    expect(summaryStats.voteCount).toEqual(70);
  });

  it("SummaryStats should get the total number of comments", () => {
    const summaryStats = new GroupedSummaryStats(TEST_COMMENTS);
    expect(summaryStats.commentCount).toEqual(2);
  });

  it("should count comments by topic", () => {
    const comments: Comment[] = [
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
    ];

    const statsByTopic = new GroupedSummaryStats(comments).getStatsByTopic();
    expect(statsByTopic[0].commentCount).toEqual(3);
    expect(statsByTopic[0]?.subtopicStats?.map((subtopic) => subtopic.commentCount)).toEqual([
      2, 1,
    ]);
  });

  it("should sort topics by comment count and put 'Other' topics and subtopics last", () => {
    const comments: Comment[] = [
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
        topics: [{ name: "Other", subtopics: [{ name: "Subtopic Other.1" }] }],
      },
      {
        id: "5",
        text: "comment 5",
        topics: [{ name: "Other", subtopics: [{ name: "Subtopic Other.1" }] }],
      },
      { id: "6", text: "comment 6", topics: [{ name: "Other", subtopics: [{ name: "Other" }] }] },
      { id: "7", text: "comment 7", topics: [{ name: "Other", subtopics: [{ name: "Other" }] }] },
      { id: "8", text: "comment 8", topics: [{ name: "Other", subtopics: [{ name: "Other" }] }] },

      {
        id: "9",
        text: "comment 9",
        topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
      },
      {
        id: "10",
        text: "comment 10",
        topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
      },
      {
        id: "11",
        text: "comment 11",
        topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
      },
      {
        id: "12",
        text: "comment 12",
        topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.1" }] }],
      },
      {
        id: "13",
        text: "comment 13",
        topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.2" }] }],
      },
      {
        id: "14",
        text: "comment 14",
        topics: [{ name: "Topic B", subtopics: [{ name: "Subtopic B.2" }] }],
      },
    ];

    const statsByTopic = new GroupedSummaryStats(comments).getStatsByTopic();
    expect(statsByTopic.map((topic) => topic.name)).toEqual(["Topic B", "Topic A", "Other"]);
  });

  it("should get the representative comments for a given group", () => {
    const representativeComments = new GroupedSummaryStats(
      TEST_COMMENTS
    ).getGroupRepresentativeComments("0");
    expect(representativeComments.length).toEqual(1);
    expect(representativeComments[0].id).toEqual("1");
  });

  it("should return empty array if there are no comments with vote tallies", () => {
    const commentsWithoutVotes: Comment[] = [
      { id: "1", text: "comment1" },
      { id: "2", text: "comment2" },
    ];
    expect(
      new GroupedSummaryStats(commentsWithoutVotes).getGroupRepresentativeComments("0")
    ).toEqual([]);
  });
});
