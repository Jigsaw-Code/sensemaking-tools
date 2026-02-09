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

import { describe, it, expect } from "@jest/globals";
import { getThemesPrompt, THEMES_PROMPT } from "./prompts";
import { SupportedLanguage } from "./languages";

describe("Multi-language Prompts", () => {
  describe("THEMES_PROMPT", () => {
    it("should have prompts for all supported languages", () => {
      const supportedLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
      
      supportedLanguages.forEach(lang => {
        expect(THEMES_PROMPT[lang]).toBeDefined();
        expect(THEMES_PROMPT[lang]).toBeTruthy();
        expect(typeof THEMES_PROMPT[lang]).toBe("string");
      });
    });

    it("should contain placeholder for topic name", () => {
      const supportedLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
      
      supportedLanguages.forEach(lang => {
        expect(THEMES_PROMPT[lang]).toContain("{topicName}");
      });
    });

    it("should contain criteria section", () => {
      const supportedLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
      
      supportedLanguages.forEach(lang => {
        expect(THEMES_PROMPT[lang]).toContain("<criteria");
        expect(THEMES_PROMPT[lang]).toContain("</criteria>");
      });
    });

    it("should contain output format section", () => {
      const supportedLanguages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];
      
      supportedLanguages.forEach(lang => {
        expect(THEMES_PROMPT[lang]).toContain("<output_format");
        expect(THEMES_PROMPT[lang]).toContain("</output_format>");
      });
    });
  });

  describe("getThemesPrompt", () => {
    it("should replace topic name placeholder", () => {
      const topicName = "Climate Change";
      const result = getThemesPrompt("en", topicName);
      
      expect(result).toContain(topicName);
      expect(result).not.toContain("{topicName}");
    });

    it("should work with Chinese topic names", () => {
      const topicName = "氣候變遷";
      const result = getThemesPrompt("zh-TW", topicName);
      
      expect(result).toContain(topicName);
      expect(result).not.toContain("{topicName}");
    });

    it("should work with Japanese topic names", () => {
      const topicName = "気候変動";
      const result = getThemesPrompt("ja", topicName);
      
      expect(result).toContain(topicName);
      expect(result).not.toContain("{topicName}");
    });

    it("should fallback to English for unsupported language", () => {
      const topicName = "Test Topic";
      const result = getThemesPrompt("invalid-lang" as SupportedLanguage, topicName);
      
      expect(result).toContain(topicName);
      expect(result).not.toContain("{topicName}");
      // Should fallback to English content
      expect(result).toContain("Please write a concise bulleted list");
    });

    it("should handle empty topic name", () => {
      const result = getThemesPrompt("en", "");
      
      expect(result).toBeDefined();
      expect(result).not.toContain("{topicName}");
    });

    it("should handle special characters in topic name", () => {
      const topicName = "AI & Machine Learning (2024)";
      const result = getThemesPrompt("en", topicName);
      
      expect(result).toContain(topicName);
      expect(result).not.toContain("{topicName}");
    });
  });

  describe("Language-specific content", () => {
    it("should have English content in English prompt", () => {
      const prompt = THEMES_PROMPT["en"];
      expect(prompt).toContain("Please write");
      expect(prompt).toContain("statements");
      expect(prompt).toContain("Impartiality");
    });

    it("should have Traditional Chinese content in zh-TW prompt", () => {
      const prompt = THEMES_PROMPT["zh-TW"];
      expect(prompt).toContain("請撰寫");
      expect(prompt).toContain("陳述");
      expect(prompt).toContain("公正性");
    });

    it("should have Simplified Chinese content in zh-CN prompt", () => {
      const prompt = THEMES_PROMPT["zh-CN"];
      expect(prompt).toContain("请撰写");
      expect(prompt).toContain("陈述");
      expect(prompt).toContain("公正性");
    });

    it("should have French content in fr prompt", () => {
      const prompt = THEMES_PROMPT["fr"];
      expect(prompt).toContain("Veuillez rédiger");
      expect(prompt).toContain("déclarations");
      expect(prompt).toContain("Impartialité");
    });

    it("should have Spanish content in es prompt", () => {
      const prompt = THEMES_PROMPT["es"];
      expect(prompt).toContain("Por favor, escriba");
      expect(prompt).toContain("declaraciones");
      expect(prompt).toContain("Imparcialidad");
    });

    it("should have Japanese content in ja prompt", () => {
      const prompt = THEMES_PROMPT["ja"];
      expect(prompt).toContain("作成してください");
      expect(prompt).toContain("声明文");
      expect(prompt).toContain("公平性");
    });
  });
});
