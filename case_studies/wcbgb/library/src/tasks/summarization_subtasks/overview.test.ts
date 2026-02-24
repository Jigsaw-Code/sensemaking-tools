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

import { isMdListValid, removeEmptyLines } from "./overview";

describe("IntroTest", () => {
  it("should remove empty lines", async () => {
    expect(
      removeEmptyLines(`
* Item 1

* Item 2


* Item 3
`)
    ).toEqual("* Item 1\n* Item 2\n* Item 3");
  });

  it("should remove empty lines with spaces", async () => {
    expect(
      removeEmptyLines(`
* Item 1
  
* Item 2
 

* Item 3
`)
    ).toEqual("* Item 1\n* Item 2\n* Item 3");
  });

  it("should remove empty lines with carriage returns", async () => {
    expect(
      removeEmptyLines(`
* Item 1\r

* Item 2\r\r
\r
* Item 3
`)
    ).toEqual("* Item 1\n* Item 2\n* Item 3");
  });

  it("should remove empty lines with spaces and carriage returns", async () => {
    expect(
      removeEmptyLines(`
* Item 1\r
  \r
* Item 2\r\r
 \r
* Item 3
`)
    ).toEqual("* Item 1\n* Item 2\n* Item 3");
  });

  it("should remove empty lines with spaces and carriage returns and tabs", async () => {
    expect(
      removeEmptyLines(`
* Item 1\r
  \r\t
* Item 2\r\r
 \r\t
* Item 3
`)
    ).toEqual("* Item 1\n* Item 2\n* Item 3");
  });

  it("should remove empty lines with spaces and carriage returns and tabs and newlines", async () => {
    expect(
      removeEmptyLines(`
* Item 1\r
  \r\t\n
* Item 2\r\r
 \r\t\n
* Item 3
`)
    ).toEqual("* Item 1\n* Item 2\n* Item 3");
  });
});

describe("isMdListValid", () => {
  it("should return false if a line does not match the expected format", async () => {
    expect(
      isMdListValid("* **Topic 1**: Summary\n* **Topic 2**: Summary\nTopic 3: Summary", [
        "Topic 1",
        "Topic 2",
        "Topic 3",
      ])
    ).toEqual(false);
  });

  it("should return false if some topic names don't match the expected order", async () => {
    expect(
      isMdListValid("* **Topic 1**: Summary\n* **Topic 3**: Summary\n* **Topic 2**: Summary", [
        "Topic 1",
        "Topic 2",
        "Topic 3",
      ])
    ).toEqual(false);
  });

  it("should return true if all lines match the expected format and topic order", async () => {
    // asterisks before colon: **:
    expect(
      isMdListValid("* **Topic 1**: Summary\n* **Topic 2**: Summary\n* **Topic 3**: Summary", [
        "Topic 1",
        "Topic 2",
        "Topic 3",
      ])
    ).toEqual(true);
    // asterisks after colon: :**
    expect(
      isMdListValid("* **Topic 1:** Summary\n* **Topic 2:** Summary\n* **Topic 3:** Summary", [
        "Topic 1",
        "Topic 2",
        "Topic 3",
      ])
    ).toEqual(true);
    // extra space between bullet point and topic name: *   **Topic 1:**
    expect(
      isMdListValid(
        "*   **Topic 1:** Summary\n*   **Topic 2:** Summary\n*   **Topic 3:** Summary",
        ["Topic 1", "Topic 2", "Topic 3"]
      )
    ).toEqual(true);
  });
});
