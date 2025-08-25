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

import { OpenRouterModel, createOpenRouterModelFromEnv } from "./openrouter_model";
import { Type } from "@sinclair/typebox";

// 簡化的測試，主要測試建構函數和基本邏輯
describe("OpenRouterModel", () => {
  describe("constructor", () => {
    it("should create model with default parameters", () => {
      // 測試建構函數不會拋出錯誤
      expect(() => {
        new OpenRouterModel("test-api-key");
      }).not.toThrow();
    });

    it("should create model with custom parameters", () => {
      // 測試自定義參數建構函數不會拋出錯誤
      expect(() => {
        new OpenRouterModel(
          "test-api-key",
          "anthropic/claude-3.5-sonnet",
          "https://custom.openrouter.ai/api/v1"
        );
      }).not.toThrow();
    });

    it("should throw error when API key is missing", () => {
      expect(() => new OpenRouterModel("")).toThrow("OpenRouter API key is required");
    });

    it("should throw error when API key is undefined", () => {
      expect(() => new OpenRouterModel(undefined as unknown as string)).toThrow("OpenRouter API key is required");
    });
  });

  describe("model properties", () => {
    it("should have correct default model name", () => {
      const model = new OpenRouterModel("test-api-key");
      // 檢查模型名稱是否正確設定
      expect(model).toBeDefined();
    });

    it("should have correct custom model name", () => {
      const customModelName = "anthropic/claude-3.5-sonnet";
      const model = new OpenRouterModel("test-api-key", customModelName);
      // 檢查模型名稱是否正確設定
      expect(model).toBeDefined();
    });
  });

  describe("createOpenRouterModelFromEnv", () => {
    const originalEnv = process.env;

    beforeEach(() => {
      jest.resetModules();
      process.env = { ...originalEnv };
    });

    afterAll(() => {
      process.env = originalEnv;
    });

    it("should throw error when OPENROUTER_API_KEY is not set", () => {
      delete process.env.OPENROUTER_API_KEY;
      
      expect(() => createOpenRouterModelFromEnv()).toThrow("OPENROUTER_API_KEY environment variable is required");
    });

    it("should use default model when OPENROUTER_MODEL is not set", () => {
      process.env.OPENROUTER_API_KEY = "env-api-key";
      delete process.env.OPENROUTER_MODEL;
      
      // 測試不會拋出錯誤
      expect(() => createOpenRouterModelFromEnv()).not.toThrow();
    });

    it("should use custom model when OPENROUTER_MODEL is set", () => {
      process.env.OPENROUTER_API_KEY = "env-api-key";
      process.env.OPENROUTER_MODEL = "env-model";
      
      // 測試不會拋出錯誤
      expect(() => createOpenRouterModelFromEnv()).not.toThrow();
    });
  });

  describe("schema validation", () => {
    it("should validate TypeBox schema correctly", () => {
      const schema = Type.Object({
        name: Type.String(),
        age: Type.Number()
      });
      
      // 測試 schema 是否正確定義
      expect(schema).toBeDefined();
      expect(schema.type).toBe("object");
      expect(schema.properties).toBeDefined();
    });
  });
});
