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

import { VoteTally } from "../types";
import { MajoritySummaryStats } from "./majority_vote";

const TEST_COMMENTS = [
  // Everyone Agrees
  {
    id: "1",
    text: "comment1",
    voteInfo: {
      "0": new VoteTally(20, 1, 2),
    },
  },
  // Everyone Disagrees
  {
    id: "2",
    text: "comment2",
    voteInfo: {
      "0": new VoteTally(2, 50, 3),
    },
  },
  // Low Alignment - Split Votes
  {
    id: "3",
    text: "comment3",
    voteInfo: {
      "0": new VoteTally(10, 11, 3),
    },
  },
  // Low Alignment if you ignore pass votes
  {
    id: "4",
    text: "comment4",
    voteInfo: { "0": new VoteTally(33, 33, 33) },
  },
  // High Uncertainty
  {
    id: "5",
    text: "comment5",
    voteInfo: { "0": new VoteTally(3, 4, 150) },
  },
  {
    id: "6",
    text: "comment6",
    voteInfo: { "0": new VoteTally(3, 4, 150) },
  },
  {
    id: "7",
    text: "comment7",
    voteInfo: { "0": new VoteTally(3, 4, 150) },
  },
];

describe("MajoritySummaryStats Test", () => {
  it("should get the comments with the most common ground", () => {
    const summaryStats = new MajoritySummaryStats(TEST_COMMENTS);

    // Of the 3 test comments only the two representing high agreement should be returned.
    const commonGroundComments = summaryStats.getCommonGroundComments(3);
    expect(commonGroundComments.length).toEqual(2);
    expect(commonGroundComments.map((comment) => comment.id).sort()).toEqual(["1", "2"]);
  });

  it("should get the comments with the most agreement", () => {
    const summaryStats = new MajoritySummaryStats(TEST_COMMENTS);

    const commonGroundComment = summaryStats.getCommonGroundAgreeComments(1);
    expect(commonGroundComment[0].id).toEqual("1");
  });

  it("should get the comments with the most disagreement", () => {
    const summaryStats = new MajoritySummaryStats(TEST_COMMENTS);

    const commonGroundComment = summaryStats.getCommonGroundDisagreeComments(1);
    expect(commonGroundComment[0].id).toEqual("2");
  });

  it("should get the comments with the most difference of opinion", () => {
    const summaryStats = new MajoritySummaryStats(TEST_COMMENTS);

    const differenceComments = summaryStats.getDifferenceOfOpinionComments(10);
    expect(differenceComments.length).toEqual(2);
    expect(differenceComments.map((comment) => comment.id).sort()).toEqual(["3", "4"]);
  });

  it("should get the comments with the most uncertainty", () => {
    const summaryStats = new MajoritySummaryStats(TEST_COMMENTS);

    const uncertainComments = summaryStats.getUncertainComments(10);
    expect(uncertainComments.length).toEqual(3);
    expect(uncertainComments.map((comment) => comment.id).sort()).toEqual(["5", "6", "7"]);
  });
});
