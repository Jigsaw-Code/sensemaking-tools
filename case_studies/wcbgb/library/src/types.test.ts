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

import {
  VoteTally,
  isTopicType,
  isVoteTallyType,
  isCommentType,
  isCommentRecordType,
  Summary,
  SummaryContent,
  CitationFormat,
} from "./types";

describe("Types Test", () => {
  it("The total votes should be the sum of all the different VoteTally values", () => {
    expect(new VoteTally(1, 2, 3).getTotalCount(true)).toEqual(6);
    expect(new VoteTally(1, 2).getTotalCount(true)).toEqual(3);
  });

  it("Valid VoteTallies should pass isVoteTallyType", () => {
    expect(isVoteTallyType({ agreeCount: 2, disagreeCount: 12 })).toBeTruthy();
    expect(isVoteTallyType({ agreeCount: 2, disagreeCount: 12, passCount: 0 })).toBeTruthy();
    expect(isVoteTallyType(new VoteTally(10, 5))).toBeTruthy();
  });

  it("Invalid VoteTallies should fail isVoteTallyType", () => {
    expect(isVoteTallyType({})).toBeFalsy();
    expect(isVoteTallyType({ agreeCount: "2" })).toBeFalsy();
  });

  it("Valid Comment should pass isCommentType", () => {
    expect(isCommentType({ id: "2", text: "hello" })).toBeTruthy();
    expect(
      isCommentType({
        id: "2",
        text: "hello",
        voteInfo: { "group 1": { agreeCount: 1, disagreeCount: 2 } },
      })
    ).toBeTruthy();
    expect(
      isCommentType({
        id: "2",
        text: "hello",
        voteInfo: { agreeCount: 1, disagreeCount: 2, passCount: 3 },
      })
    ).toBeTruthy();
  });

  it("Invalid Comment should fail isCommentType", () => {
    expect(isCommentType({})).toBeFalsy();
    // Vote Tally counts must be of type number
    expect(
      isCommentType({
        id: "2",
        text: "hello",
        voteInfo: { "group 1": { agreeCount: "1", disagreeCount: "2" } },
      })
    ).toBeFalsy();
  });

  it("Valid CommentRecord should pass isCommentRecordType", () => {
    expect(isCommentRecordType({ id: "123", topics: [] })).toBeTruthy();
  });

  it("Invalid CommentRecord should fail isCommentRecordType", () => {
    // ID is required.
    expect(isCommentRecordType({ topics: [{ name: "Healthcare" }] })).toBeFalsy();
    // ID must be of type string.
    expect(isCommentRecordType({ id: 1, topics: [{ name: "Healthcare" }] })).toBeFalsy();
    // Topics must be valid, the second one is missing a name.
    expect(
      isCommentRecordType({
        id: 1,
        topics: [{ name: "Healthcare" }, { subtopics: { name: "Public Parks" } }],
      })
    ).toBeFalsy();
  });

  it("Valid Topics should pass isTopicType", () => {
    expect(isTopicType({ name: "Test Topic" })).toBeTruthy();
    expect(
      isTopicType({ name: "Test Topic", subtopics: [{ name: "Test Subtopic" }] })
    ).toBeTruthy();
  });

  it("Invalid Topics should not pass isTopicType", () => {
    expect(isTopicType({})).toBeFalsy();
    expect(isTopicType({ name: 2 })).toBeFalsy();
    expect(isTopicType({ name: 2, subtopics: [{}] })).toBeFalsy();
    // The object has one valid subtopic and one invalid subtopic.
    expect(
      isTopicType({ name: "Test Topic", subtopics: [{ name: "Test Subtopic" }, {}] })
    ).toBeFalsy();
  });

  describe("Summary", () => {
    describe("getText", () => {
      const chunks: SummaryContent[] = [
        { text: "Claim 1 text", citations: ["id1", "id2"] },
        { text: "Claim 2 text.", citations: ["id3"] },
        { text: " Filler text" },
      ];
      const summary = new Summary(chunks, []);

      it("should return XML formatted summary", () => {
        const expectedXML = `Claim 1 text <citation comment_id=id1><citation comment_id=id2> 
Claim 2 text. <citation comment_id=id3> 
 Filler text`;
        expect(summary.getText("XML")).toBe(expectedXML);
      });

      it("should return MARKDOWN formatted summary", () => {
        const expectedMarkdown = `Claim 1 text [id1,id2] 
Claim 2 text. [id3] 
 Filler text`;
        expect(summary.getText("MARKDOWN")).toBe(expectedMarkdown);
      });

      it("should throw an error for unsupported format", () => {
        expect(() => summary.getText("UNSUPPORTED" as CitationFormat)).toThrow(
          "Unsupported citation type: UNSUPPORTED"
        );
      });
    });
  });
});
