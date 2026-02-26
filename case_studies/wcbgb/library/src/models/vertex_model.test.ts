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

import { Type } from "@sinclair/typebox";
import { VertexModel } from "./vertex_model";
import { MAX_LLM_RETRIES } from "./model_util";

// mock retry timeout
jest.mock("./model_util", () => {
  const originalModule = jest.requireActual("./model_util");
  return {
    __esModule: true,
    ...originalModule,
    RETRY_DELAY_MS: 0,
  };
});

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

describe("VertexAI test", () => {
  const model = new VertexModel("my-project", "us-central1", "models/gemini-pro");
  const { generateContentStreamMock } = jest.requireMock("@google-cloud/vertexai");

  beforeEach(() => {
    // Reset the mock before each test
    generateContentStreamMock.mockClear();
  });

  // These tests are specifically for the VertexModel class logic, rather than the implementation logic of the VertexAI calls
  describe("generateContent", () => {
    it("should retry on rate limit error and return valid JSON", async () => {
      const expectedJSON = [{ result: "success" }];

      // Mock the first call to throw a rate limit error
      generateContentStreamMock.mockImplementationOnce(() => {
        throw new Error("429 Too Many Requests");
      });

      // Mock the second call to return the expected JSON
      mockSingleModelResponse(generateContentStreamMock, JSON.stringify(expectedJSON));

      const result = JSON.parse(
        await model.callLLM("Some instructions", model.getGenerativeModel())
      );

      // Assert that the mock was called twice (initial call + retry)
      expect(generateContentStreamMock).toHaveBeenCalledTimes(2);

      // Assert that the result is the expected JSON
      expect(result).toEqual(expectedJSON);
    }, 15000); // no luck mocking retry timeout for this test - so letting it run longer for now

    it("should generate valid text", async () => {
      const expectedText = "This is some text.";
      mockSingleModelResponse(generateContentStreamMock, expectedText);

      const result = await model.generateText("Some instructions");

      expect(generateContentStreamMock).toHaveBeenCalledTimes(1);

      expect(result).toEqual(expectedText);
    });

    it("should generate valid structured data that matches the schema", async () => {
      const expectedStructuredData = { key1: "value1", key2: 2 };
      // the TypeBox spec:
      const schema = Type.Object({
        key1: Type.String(),
        key2: Type.Number(),
      });

      mockSingleModelResponse(generateContentStreamMock, JSON.stringify(expectedStructuredData));

      const result = await model.generateData("Some instructions", schema);

      expect(generateContentStreamMock).toHaveBeenCalledTimes(1);

      expect(result).toEqual(expectedStructuredData);
    });

    it("should throw an error when generated data does not match the schema", async () => {
      const expectedStructuredData = { key1: 1, key2: "value2" };
      // the TypeBox spec:
      const schema = Type.Object({
        key1: Type.String(),
        key2: Type.Number(),
      });

      mockSingleModelResponse(generateContentStreamMock, JSON.stringify(expectedStructuredData));
      await expect(async () => {
        await model.generateData("Some instructions", schema);
      }).rejects.toThrow(
        `Failed after ${MAX_LLM_RETRIES} attempts: Failed to get a valid model response.`
      );
    });
  });
});
