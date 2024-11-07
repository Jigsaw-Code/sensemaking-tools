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

import { generateCommentsWithModel, generateJSON, generateTopicsWithModel } from "./vertex_model";
import { GenerativeModel } from "@google-cloud/vertexai";
import { Comment, Topic } from "../types";

// Mock the VertexAI module - this mock will be used when the module is imported within a test run.
jest.mock("@google-cloud/vertexai", () => {
  // Mock the model response. This mock needs to be set up to return response specific for each test.
  const generateContentStreamMock = jest.fn();
  return {
    // Mock `generateContentStream` function within VertexAI module
    VertexAI: jest.fn(() => ({
      getGenerativeModel: jest.fn(() => ({
        generateContentStream: generateContentStreamMock,
      })),
    })),
    // Expose the mocked function, so we can get it within a test using `jest.requireMock`, and spy on its invocations.
    generateContentStreamMock: generateContentStreamMock,
    // Mock other imports from VertexAI module
    HarmBlockThreshold: {},
    HarmCategory: {},
    SchemaType: { ARRAY: 0, OBJECT: 1, STRING: 2 },
  };
});

function mockSingleModelResponse(generateContentStreamMock: jest.Mock, responseMock: string) {
  generateContentStreamMock.mockImplementationOnce(() =>
    Promise.resolve({
      response: {
        candidates: [{ content: { parts: [{ text: responseMock }] } }],
      },
    })
  );
}

describe("ModelTest", () => {
  // if we can pass the model mock directly to the function where it's being called (instead of relying on import),
  // then we don't need to use something like `jest.requireMock("@google-cloud/vertexai").generativeJsonModelMock`
  const generativeJsonModelMock = {
    generateContentStream: jest.fn(),
  };
  const generateContentStreamMock = generativeJsonModelMock.generateContentStream;

  beforeEach(() => {
    // Reset the mock before each test
    generateContentStreamMock.mockClear();
  });

  describe("generateContent", () => {
    it("should retry on rate limit error and return valid JSON", async () => {
      const expectedJSON = [{ result: "success" }];

      // Mock the first call to throw a rate limit error
      generateContentStreamMock.mockImplementationOnce(() => {
        throw new Error("429 Too Many Requests");
      });

      // Mock the second call to return the expected JSON
      mockSingleModelResponse(generateContentStreamMock, JSON.stringify(expectedJSON));

      const result = await generateJSON(
        "Some instructions",
        generativeJsonModelMock as unknown as GenerativeModel
      );

      // Assert that the mock was called twice (initial call + retry)
      expect(generateContentStreamMock).toHaveBeenCalledTimes(2);

      // Assert that the result is the expected JSON
      expect(result).toEqual(expectedJSON);
    });

    it("should generate valid topics", async () => {
      const expectedTopics: Topic[] = [{ name: "Topic 1" }, { name: "Topic 2" }];
      mockSingleModelResponse(generateContentStreamMock, JSON.stringify(expectedTopics));

      const result = await generateTopicsWithModel(
        "Some instructions",
        generativeJsonModelMock as unknown as GenerativeModel
      );

      expect(generateContentStreamMock).toHaveBeenCalledTimes(1);

      expect(result).toEqual(expectedTopics);
    });

    it("should generate valid comments", async () => {
      const expectedComments: Comment[] = [
        { id: "1", text: "hi", topics: [{ name: "Topic 1" }] },
        { id: "1", text: "hi", topics: [{ name: "Topic 1" }] },
      ];
      mockSingleModelResponse(generateContentStreamMock, JSON.stringify(expectedComments));

      const result = await generateCommentsWithModel(
        "Some instructions",
        generativeJsonModelMock as unknown as GenerativeModel
      );

      expect(generateContentStreamMock).toHaveBeenCalledTimes(1);

      expect(result).toEqual(expectedComments);
    });
  });
});
