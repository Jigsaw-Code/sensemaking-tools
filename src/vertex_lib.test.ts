// Copyright 2023 Google LLC
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

import { getPrompt, formatCommentsWithVotes } from "./vertex_lib";

describe("VertexLibTest", () => {
  it("should create a prompt", () => {
    expect(getPrompt("Summarize this:", ["comment1", "comment2"])).toEqual(
      "Summarize this: comment1,comment2",
    );
  });
  it("should format comments with vote tallies via formatCommentsWithVotes", () => {
    expect(
      formatCommentsWithVotes([
        {
          commentText: "comment1",
          groupVoteTallies: {
            "0": {
              agreeCount: 10,
              disagreeCount: 5,
              passCount: 0,
              totalCount: 15,
            },
            "1": {
              agreeCount: 5,
              disagreeCount: 10,
              passCount: 5,
              totalCount: 20,
            },
          },
        },
        {
          commentText: "comment2",
          groupVoteTallies: {
            "0": {
              agreeCount: 2,
              disagreeCount: 5,
              passCount: 3,
              totalCount: 10,
            },
            "1": {
              agreeCount: 5,
              disagreeCount: 3,
              passCount: 2,
              totalCount: 10,
            },
          },
        },
      ]),
    ).toEqual([
      `comment1
  vote info per group: {"0":{"agreeCount":10,"disagreeCount":5,"passCount":0,"totalCount":15},"1":{"agreeCount":5,"disagreeCount":10,"passCount":5,"totalCount":20}}`,
      `comment2
  vote info per group: {"0":{"agreeCount":2,"disagreeCount":5,"passCount":3,"totalCount":10},"1":{"agreeCount":5,"disagreeCount":3,"passCount":2,"totalCount":10}}`,
    ]);
  });
});
