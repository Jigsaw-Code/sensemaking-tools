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
  findMissingComments,
  validateCommentRecords,
  categorizeWithRetry,
  categorizeCommentsRecursive,
  getTopicDepthFromTopics,
} from "./categorization";
import { CommentRecord, Comment, Topic, NestedTopic } from "../types";
import { VertexModel } from "../models/vertex_model";
import { Model } from "../models/model";

// mock retry timeout
jest.mock("../models/model_util", () => {
  const originalModule = jest.requireActual("../models/model_util");
  return {
    __esModule: true,
    ...originalModule,
    RETRY_DELAY_MS: 0,
  };
});

// Mock the model response. This mock needs to be set up to return response specific for each test.
let mockGenerateData: jest.SpyInstance;

describe("CategorizationTest", () => {
  beforeEach(() => {
    mockGenerateData = jest.spyOn(VertexModel.prototype, "generateData");
  });

  afterEach(() => {
    mockGenerateData.mockRestore();
  });
  it("should retry categorization with all missing comments", async () => {
    const comments: Comment[] = [
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
      { id: "3", text: "Comment 3" },
    ];
    const instructions = "Categorize the comments based on these topics:  [{'name': 'Topic 1'}]";
    const commentsWithTextAndTopics = [
      {
        id: "1",
        text: "Comment 1",
        topics: [{ name: "Topic 1" }],
      },
      {
        id: "2",
        text: "Comment 2",
        topics: [{ name: "Topic 1" }],
      },
      {
        id: "3",
        text: "Comment 3",
        topics: [{ name: "Topic 1" }],
      },
    ];

    // The first response is incorrectly missing all comments, and then
    // on retry the text is present.
    mockGenerateData
      .mockReturnValueOnce(Promise.resolve([]))
      .mockReturnValueOnce(Promise.resolve(commentsWithTextAndTopics));

    const commentRecords = await categorizeWithRetry(
      new VertexModel("project", "location", "gemini-1000"),
      instructions,
      comments,
      [{ name: "Topic 1" }]
    );

    expect(mockGenerateData).toHaveBeenCalledTimes(2);
    expect(commentRecords).toEqual(commentsWithTextAndTopics);
  });

  it("should retry categorization with some missing comments", async () => {
    const comments: Comment[] = [
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
      { id: "3", text: "Comment 3" },
    ];
    const instructions = "Categorize the comments based on these topics:  [{'name': 'Topic 1'}]";
    const commentsWithTextAndTopics = [
      {
        id: "1",
        text: "Comment 1",
        topics: [{ name: "Topic 1" }],
      },
      {
        id: "2",
        text: "Comment 2",
        topics: [{ name: "Topic 1" }],
      },
      {
        id: "3",
        text: "Comment 3",
        topics: [{ name: "Topic 1" }],
      },
    ];

    // The first mock response includes only one comment, and for the next
    // response the two missing comments are returned.
    mockGenerateData
      .mockReturnValueOnce(Promise.resolve([commentsWithTextAndTopics[0]]))
      .mockReturnValueOnce(
        Promise.resolve([commentsWithTextAndTopics[1], commentsWithTextAndTopics[2]])
      );

    const commentRecords = await categorizeWithRetry(
      new VertexModel("project", "location", "gemini-1000"),
      instructions,
      comments,
      [{ name: "Topic 1" }]
    );

    expect(mockGenerateData).toHaveBeenCalledTimes(2);
    expect(commentRecords).toEqual(commentsWithTextAndTopics);
  });

  it('should assign "Other" topic to comments that failed categorization after max retries', async () => {
    const comments: Comment[] = [
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
      { id: "3", text: "Comment 3" },
    ];
    const topics = '[{"name": "Topic 1", "subtopics": []}]';
    const instructions = "Categorize the comments based on these topics: " + topics;
    const topicsJson = [{ name: "Topic 1", subtopics: [] }];

    // Mock the model to always return an empty response. This simulates a
    // categorization failure.
    mockGenerateData.mockReturnValue(Promise.resolve([]));

    const commentRecords = await categorizeWithRetry(
      new VertexModel("project", "location", "gemini-1000"),
      instructions,
      comments,
      topicsJson
    );

    expect(mockGenerateData).toHaveBeenCalledTimes(3);

    const expected = [
      {
        id: "1",
        text: "Comment 1",
        topics: [{ name: "Other" }],
      },
      {
        id: "2",
        text: "Comment 2",
        topics: [{ name: "Other" }],
      },
      {
        id: "3",
        text: "Comment 3",
        topics: [{ name: "Other" }],
      },
    ];
    expect(commentRecords).toEqual(expected);
  });

  it("should not categorize comments if they already have topics", async () => {
    const comments: Comment[] = [
      { id: "1", text: "Comment 1", topics: [{ name: "Topic A" }] },
      { id: "2", text: "Comment 2", topics: [{ name: "Topic B" }] },
    ];
    const topicDepth = 1;
    // Mock the Model class and its methods
    const mockModel = {
      categorizationBatchSize: 5,
      generateData: jest.fn(),
      generateText: jest.fn(),
    } as Model;

    const categorizedComments = await categorizeCommentsRecursive(comments, topicDepth, mockModel);

    // we expect to return the comments as they were passed, no extra calls to the model
    expect(categorizedComments).toEqual(comments);
    expect(mockGenerateData).not.toHaveBeenCalled();
  });
});

describe("validateCommentRecord", () => {
  const inputComments: Comment[] = [
    { id: "1", text: "Comment 1" },
    { id: "2", text: "Comment 2" },
  ];

  const topics: Topic[] = [
    { name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] },
    { name: "Topic 2", subtopics: [{ name: "Subtopic 2" }] },
  ];

  it("should return all comments as valid with correct input", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [{ name: "Subtopic 2" }] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(2);
    expect(commentsWithInvalidTopics.length).toBe(0);
  });

  it("should filter out extra comments", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [{ name: "Subtopic 2" }] }],
      },
      {
        id: "3",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(2);
    expect(commentsWithInvalidTopics.length).toBe(0);
  });

  it("should filter out comments with empty topics", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
      { id: "2", topics: [] },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(1);
    expect(commentsWithInvalidTopics.length).toBe(1);
  });

  it("should filter out comments with empty subtopics", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(1);
    expect(commentsWithInvalidTopics.length).toBe(1);
  });

  it("should filter out comments with invalid topic names", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Invalid Topic", subtopics: [{ name: "Subtopic 2" }] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(1);
    expect(commentsWithInvalidTopics.length).toBe(1);
  });

  it("should filter out a comment with one valid and one invalid topic name", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [
          { name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] },
          { name: "Invalid Topic", subtopics: [{ name: "Subtopic 2" }] },
        ],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [{ name: "Subtopic 2" }] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(1); // Only Comment 2 should pass
    expect(commentsWithInvalidTopics.length).toBe(1); // Comment 1 should fail
  });

  it("should filter out comments with invalid subtopic names", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Topic 1", subtopics: [{ name: "Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [{ name: "Invalid Subtopic" }] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(1);
    expect(commentsWithInvalidTopics.length).toBe(1);
  });

  it("should filter out a comment with one valid and one invalid subtopic name", () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [
          {
            name: "Topic 1",
            subtopics: [{ name: "Subtopic 1" }, { name: "Invalid Subtopic" }],
          },
        ],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [{ name: "Subtopic 2" }] }],
      },
    ];
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topics
    );
    expect(commentsPassedValidation.length).toBe(1); // Only Comment 2 should pass
    expect(commentsWithInvalidTopics.length).toBe(1); // Comment 1 should fail
  });

  it('should allow "Other" as a valid topic or subtopic name', () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Other", subtopics: [{ name: "Other Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Topic 2", subtopics: [{ name: "Other" }] }],
      },
    ];
    const topicsWithOther = topics.concat([
      { name: "Other", subtopics: [{ name: "Other Subtopic 1" }] },
    ]);
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topicsWithOther
    );
    expect(commentsPassedValidation.length).toBe(2);
    expect(commentsWithInvalidTopics.length).toBe(0);
  });

  it('should fiter our comments with an invalid subtopic name in the "Other" category', () => {
    const commentRecords: CommentRecord[] = [
      {
        id: "1",
        topics: [{ name: "Other", subtopics: [{ name: "Other Subtopic 1" }] }],
      },
      {
        id: "2",
        topics: [{ name: "Other", subtopics: [{ name: "Some invalid subtopic name" }] }],
      },
    ];
    const topicsWithOther = topics.concat([
      { name: "Other", subtopics: [{ name: "Other Subtopic 1" }] },
    ]);
    const { commentsPassedValidation, commentsWithInvalidTopics } = validateCommentRecords(
      commentRecords,
      inputComments,
      topicsWithOther
    );
    expect(commentsPassedValidation.length).toBe(1);
    expect(commentsWithInvalidTopics.length).toBe(1);
  });
});

describe("findMissingComments", () => {
  it("should return an empty array when all comments are present", () => {
    const commentRecords: CommentRecord[] = [
      { id: "1", topics: [] },
      { id: "2", topics: [] },
    ];
    const inputComments: Comment[] = [
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
    ];
    const missingComments = findMissingComments(commentRecords, inputComments);
    expect(missingComments).toEqual([]);
  });

  it("should return missing comments when some are not present", () => {
    const commentRecords: CommentRecord[] = [{ id: "1", topics: [] }];
    const inputComments: Comment[] = [
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
      { id: "3", text: "Comment 3" },
    ];
    const missingComments = findMissingComments(commentRecords, inputComments);
    expect(missingComments).toEqual([
      { id: "2", text: "Comment 2" },
      { id: "3", text: "Comment 3" },
    ]);
  });

  it("should return all comments when none are present", () => {
    const commentRecords: CommentRecord[] = [];
    const inputComments: Comment[] = [
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
    ];
    const missingComments = findMissingComments(commentRecords, inputComments);
    expect(missingComments).toEqual([
      { id: "1", text: "Comment 1" },
      { id: "2", text: "Comment 2" },
    ]);
  });
});

describe("getTopicDepthFromTopics", () => {
  it("should return 0 for empty topics array", () => {
    const topics: Topic[] = [];
    const depth = getTopicDepthFromTopics(topics);
    expect(depth).toBe(0);
  });

  it("should return 1 for topics with no subtopics", () => {
    const topics: Topic[] = [{ name: "Topic A" }, { name: "Topic B" }];
    const depth = getTopicDepthFromTopics(topics);
    expect(depth).toBe(1);
  });

  it("should return 2 for topics with one level of subtopics", () => {
    const topics: NestedTopic[] = [
      {
        name: "Topic A",
        subtopics: [{ name: "Subtopic A1" }, { name: "Subtopic A2" }],
      },
    ];
    const depth = getTopicDepthFromTopics(topics);
    expect(depth).toBe(2);
  });

  it("should return 3 for topics with two levels of subtopics", () => {
    const topics: NestedTopic[] = [
      {
        name: "Topic A",
        subtopics: [
          {
            name: "Subtopic A1",
            subtopics: [{ name: "SubSubtopic A1.1" }, { name: "SubSubtopic A1.2" }],
          } as NestedTopic,
        ],
      },
    ];
    const depth = getTopicDepthFromTopics(topics);
    expect(depth).toBe(3);
  });
});
