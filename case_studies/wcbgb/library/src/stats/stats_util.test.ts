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

import { VoteTally } from "../types";
import {
  getAgreeRate,
  getGroupInformedConsensus,
  getGroupAgreeProbDifference,
  getDisagreeRate,
  getGroupInformedDisagreeConsensus,
  getMinDisagreeProb,
  getMinAgreeProb,
  getCommentVoteCount,
  getTotalAgreeRate,
} from "./stats_util";

describe("stats utility functions", () => {
  it("should get the agree probability for a given vote tally", () => {
    expect(getAgreeRate(new VoteTally(10, 5, 5), true)).toBeCloseTo((10 + 1) / (20 + 2));
  });

  it("should handle vote tallies with zero counts", () => {
    expect(getAgreeRate(new VoteTally(0, 0, 0), true, true)).toBeCloseTo(0.5);
    expect(getAgreeRate(new VoteTally(0, 5), true, true)).toBeCloseTo(1 / 7);
    expect(getAgreeRate(new VoteTally(5, 0), true, true)).toBeCloseTo(6 / 7);
  });

  it("should get the comment vote count with groups", () => {
    expect(
      getCommentVoteCount(
        {
          id: "1",
          text: "hello",
          voteInfo: {
            "0": new VoteTally(10, 5, 0),
            "1": new VoteTally(5, 10, 5),
          },
        },
        true
      )
    ).toEqual(35);
  });

  it("should get the comment vote count without groups", () => {
    expect(
      getCommentVoteCount(
        {
          id: "1",
          text: "hello",
          voteInfo: new VoteTally(10, 5),
        },
        true
      )
    ).toEqual(15);
  });

  it("should get the total agree rate across groups for a given comment", () => {
    expect(
      getTotalAgreeRate(
        {
          "0": new VoteTally(10, 5, 0),
          "1": new VoteTally(5, 10, 5),
        },
        true,
        false
      )
    ).toEqual(15 / 35);
  });

  it("should get the total agree rate for a given comment", () => {
    expect(getTotalAgreeRate(new VoteTally(10, 5, 0), true, false)).toEqual(10 / 15);
  });

  it("should get the group informed consensus for a given comment", () => {
    expect(
      getGroupInformedConsensus({
        id: "1",
        text: "comment1",
        voteInfo: {
          "0": new VoteTally(10, 5, 0),
          "1": new VoteTally(5, 10, 5),
        },
      })
    ).toBeCloseTo(((11 / 17) * 6) / 22);
  });

  it("should get the minimum agree probability across groups for a given comment", () => {
    expect(
      getMinAgreeProb(
        {
          id: "1",
          text: "comment1",
          voteInfo: {
            "0": new VoteTally(10, 5, 0),
            "1": new VoteTally(5, 10, 5),
          },
        },
        true
      )
    ).toBeCloseTo(3 / 11);
  });

  it("should get the disagree probability for a given vote tally", () => {
    expect(getDisagreeRate(new VoteTally(10, 5, 5), true, true)).toBeCloseTo((5 + 1) / (20 + 2));
  });

  it("should handle vote tallies with zero counts", () => {
    expect(getDisagreeRate(new VoteTally(0, 0, 0), true, true)).toBeCloseTo(0.5);
    expect(getDisagreeRate(new VoteTally(0, 5), true, true)).toBeCloseTo(6 / 7);
    expect(getDisagreeRate(new VoteTally(5, 0), true, true)).toBeCloseTo(1 / 7);
  });

  it("should get the group informed consensus for a given comment", () => {
    expect(
      getGroupInformedDisagreeConsensus({
        id: "1",
        text: "comment1",
        voteInfo: {
          "0": new VoteTally(10, 5, 0),
          "1": new VoteTally(5, 10, 5),
        },
      })
    ).toBeCloseTo(((11 / 17) * 6) / 22);
  });

  it("should get the minimum disagree probability across groups for a given comment", () => {
    expect(
      getMinDisagreeProb({
        id: "1",
        text: "comment1",
        voteInfo: {
          "0": new VoteTally(5, 10, 0),
          "1": new VoteTally(10, 5, 5),
        },
      })
    ).toBeCloseTo(3 / 11);
  });

  it("should get the group agree difference for a given comment and group", () => {
    expect(
      getGroupAgreeProbDifference(
        {
          id: "1",
          text: "comment1",
          voteInfo: {
            "0": new VoteTally(1, 2, 0),
            "1": new VoteTally(3, 1, 0),
          },
        },
        "0"
      )
    ).toBeCloseTo(2 / 5 - 2 / 3);
  });
});
