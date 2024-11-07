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

import { getPrompt } from "./sensemaker_utils";

describe("SensemakerUtilsTest", () => {
  it("should create a prompt", () => {
    expect(getPrompt("Summarize this.", ["comment1", "comment2"])).toEqual(
      `Instructions:
Summarize this.

Comments:
comment1
comment2`
    );
  });

  it("should include additional instructions in the prompt", () => {
    expect(
      getPrompt(
        "Summarize this.",
        ["comment1", "comment2"],
        "This is for a town hall style deliberation"
      )
    ).toEqual(
      `Instructions:
Summarize this.

Additional context:
This is for a town hall style deliberation

Comments:
comment1
comment2`
    );
  });
});
