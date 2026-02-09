import { SupportedLanguage } from "./languages";
import { Summary } from "../../src/types";

/**
 * 將報告中的 "Other" 標記轉換為對應語言
 * @param content 報告內容或 Summary 對象
 * @param output_lang 輸出語言
 * @returns 轉換後的內容
 */
export function translateSummary(content: Summary, output_lang: SupportedLanguage): Summary {
  if (output_lang === "en") {
    return content;
  }

  const otherTranslations: Record<SupportedLanguage, string> = {
    "en": "Other",
    "zh-TW": "其他",
    "zh-CN": "其他",
    "fr": "Autre",
    "es": "Otro",
    "ja": "その他"
  };

  // 處理 Summary 對象
  if (content instanceof Summary) {
    const translatedContents = content.contents.map(contentItem => {
      const translatedContent = { ...contentItem };
      
      // 轉換標題
      if (translatedContent.title) {
        translatedContent.title = translateString(translatedContent.title, otherTranslations[output_lang]);
      }
      
      // 轉換文本
      if (translatedContent.text) {
        translatedContent.text = translateString(translatedContent.text, otherTranslations[output_lang]);
      }
      
      // 遞歸處理子內容
      if (translatedContent.subContents) {
        translatedContent.subContents = translatedContent.subContents.map(subContent => {
          const translatedSubContent = { ...subContent };
          if (translatedSubContent.title) {
            translatedSubContent.title = translateString(translatedSubContent.title, otherTranslations[output_lang]);
          }
          if (translatedSubContent.text) {
            translatedSubContent.text = translateString(translatedSubContent.text, otherTranslations[output_lang]);
          }
          return translatedSubContent;
        });
      }
      
      return translatedContent;
    });

    return new Summary(translatedContents, content.comments);
  }

  return content;
}

/**
 * 輔助函式：轉換字符串中的 "Other" 標記
 */
function translateString(text: string, translation: string): string {
  const otherRegex = /\bOther\b/g;
  return text.replace(otherRegex, translation);
}
