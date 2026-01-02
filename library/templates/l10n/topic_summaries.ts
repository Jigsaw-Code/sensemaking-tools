import { SupportedLanguage } from "./languages";

// Topic summary related text
export const TOPIC_SUMMARIES = {
  topicSummary: {
    "en": "This topic included {subtopicCount} subtopic{subtopicPlural}, comprising a total of {statementCount} statement{statementPlural}.",
    "zh-TW": "此主題包含 {subtopicCount} 個子主題{subtopicPlural}，總共包含 {statementCount} 個意見{statementPlural}。",
    "zh-CN": "此主题包含 {subtopicCount} 个子主题{subtopicPlural}，总共包含 {statementCount} 个意见{statementPlural}。",
    "fr": "Ce sujet comprenait {subtopicCount} sous-sujet{subtopicPlural}, comprenant un total de {statementCount} déclaration{statementPlural}.",
    "es": "Este tema incluyó {subtopicCount} subtema{subtopicPlural}, que comprende un total de {statementCount} declaración{statementPlural}.",
    "ja": "このトピックには{subtopicCount}のサブトピック{subtopicPlural}が含まれており、合計{statementCount}の声明{statementPlural}で構成されています。"
  },
  relativeAgreement: {
    "en": "This subtopic had {level} compared to the other subtopics.",
    "zh-TW": "此子主題與其他子主題相比具有 {level}。",
    "zh-CN": "此子主题与其他子主题相比具有 {level}。",
    "fr": "Ce sous-sujet avait {level} par rapport aux autres sous-sujets.",
    "es": "Este subtema tenía {level} en comparación con los otros subtemas.",
    "ja": "このサブトピックは、他のサブトピックと比較して{level}でした。"
  }
};

export function getTopicSummaryText(
  section: keyof typeof TOPIC_SUMMARIES,
  lang: SupportedLanguage,
  replacements: Record<string, string | number>
): string {
  let text = TOPIC_SUMMARIES[section][lang] || TOPIC_SUMMARIES[section]["en"];
  
  Object.entries(replacements).forEach(([key, value]) => {
    text = text.replace(new RegExp(`{${key}}`, 'g'), value.toString());
  });
  
  return text;
}

// Helper function to get plural forms
export function getPluralForm(count: number, lang: SupportedLanguage): string {
  if (count === 1) return "";
  
  switch (lang) {
    case "zh-TW":
    case "zh-CN":
    case "ja":
      return "";
    case "fr":
    case "es":
      return "s";
    default: // en
      return "s";
  }
}
