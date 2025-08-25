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

// Example usage of multilingual prompts

import { getThemesPrompt, THEMES_PROMPT } from "../templates/l10n/prompts";
import { SupportedLanguage } from "../templates/l10n/languages";

async function main() {
  console.log("=== 多語言提示詞使用範例 ===\n");

  // 範例主題名稱
  const sampleTopics = [
    "Climate Change",
    "氣候變遷",
    "気候変動",
    "Changement climatique",
    "Cambio climático"
  ];

  // 支援的語言
  const languages: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];

  console.log("1. 顯示所有語言的提示詞結構...\n");
  
  languages.forEach(lang => {
    console.log(`=== ${lang.toUpperCase()} ===`);
    const prompt = THEMES_PROMPT[lang];
    console.log(`長度: ${prompt.length} 字元`);
    console.log(`包含主題名稱佔位符: ${prompt.includes("{topicName}")}`);
    console.log(`包含標準章節: ${prompt.includes("<criteria") && prompt.includes("<output_format")}`);
    console.log("");
  });

  console.log("2. 動態生成不同語言的提示詞...\n");
  
  sampleTopics.forEach((topic, index) => {
    const lang = languages[index % languages.length] as SupportedLanguage;
    console.log(`主題: "${topic}" (語言: ${lang})`);
    
    const localizedPrompt = getThemesPrompt(lang, topic);
    console.log(`提示詞長度: ${localizedPrompt.length} 字元`);
    console.log(`主題名稱已替換: ${localizedPrompt.includes(topic)}`);
    console.log(`預覽: ${localizedPrompt.substring(0, 100)}...`);
    console.log("");
  });

  console.log("3. 語言特定內容檢查...\n");
  
  // 檢查每種語言的關鍵詞
  const languageKeywords = {
    "en": ["Please write", "statements", "Impartiality"],
    "zh-TW": ["請撰寫", "陳述", "公正性"],
    "zh-CN": ["请撰写", "陈述", "公正性"],
    "fr": ["Veuillez rédiger", "déclarations", "Impartialité"],
    "es": ["Por favor, escriba", "declaraciones", "Imparcialidad"],
    "ja": ["作成してください", "声明文", "公平性"]
  };

  languages.forEach(lang => {
    const prompt = THEMES_PROMPT[lang];
    const keywords = languageKeywords[lang];
    
    console.log(`${lang}:`);
    keywords.forEach(keyword => {
      const found = prompt.includes(keyword);
      console.log(`  ${keyword}: ${found ? "✓" : "✗"}`);
    });
    console.log("");
  });

  console.log("4. 錯誤處理測試...\n");
  
  // 測試不支援的語言
  const unsupportedLang = "invalid-lang" as SupportedLanguage;
  const fallbackPrompt = getThemesPrompt(unsupportedLang, "Test Topic");
  
  console.log(`不支援的語言 "${unsupportedLang}":`);
  console.log(`  回退到英文: ${fallbackPrompt.includes("Please write")}`);
  console.log(`  主題名稱已替換: ${fallbackPrompt.includes("Test Topic")}`);
  console.log(`  不包含佔位符: ${!fallbackPrompt.includes("{topicName}")}`);

  console.log("\n=== 範例執行完成 ===");
}

// 執行主函數
main().catch(console.error);
