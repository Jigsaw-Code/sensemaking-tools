import { SupportedLanguage } from "./languages";

// Special topic names that need localization
export const TOPIC_NAMES = {
  other: {
    "en": "Other",
    "zh-TW": "其他",
    "zh-CN": "其他",
    "fr": "Autre",
    "es": "Otros",
    "ja": "その他"
  },
  uncategorized: {
    "en": "Uncategorized",
    "zh-TW": "未分類",
    "zh-CN": "未分类",
    "fr": "Non catégorisé",
    "es": "Sin categorizar",
    "ja": "未分類"
  }
};

export function getTopicName(
  topicType: keyof typeof TOPIC_NAMES,
  lang: SupportedLanguage
): string {
  return TOPIC_NAMES[topicType][lang] || TOPIC_NAMES[topicType]["en"];
}

// Function to localize any topic name, with fallback to original if not found
export function localizeTopicName(
  topicName: string,
  lang: SupportedLanguage
): string {
  // Check if it's a special topic name that we have translations for
  const lowerTopicName = topicName.toLowerCase();
  if (lowerTopicName === "other") {
    return getTopicName("other", lang);
  }
  if (lowerTopicName === "uncategorized") {
    return getTopicName("uncategorized", lang);
  }
  
  // For other topic names, return as is (they should already be in the correct language)
  return topicName;
}
