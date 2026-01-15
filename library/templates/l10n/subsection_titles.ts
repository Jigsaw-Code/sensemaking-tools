import { SupportedLanguage } from "./languages";

// Subsection titles and labels
export const SUBSECTION_TITLES = {
  prominentThemes: {
    "en": "Prominent themes were:",
    "zh-TW": "主要主題包括：",
    "zh-CN": "主要主题包括：",
    "fr": "Les thèmes principaux étaient :",
    "es": "Los temas prominentes fueron:",
    "ja": "主要なテーマは以下の通りです："
  },
  commonGround: {
    "en": "Common ground:",
    "zh-TW": "共同點：",
    "zh-CN": "共同点：",
    "fr": "Terrain d'entente :",
    "es": "Puntos en común:",
    "ja": "共通点："
  },
  commonGroundBetweenGroups: {
    "en": "Common ground between groups:",
    "zh-TW": "群組間的共同點：",
    "zh-CN": "群组间的共同点：",
    "fr": "Terrain d'entente entre les groupes :",
    "es": "Puntos en común entre grupos:",
    "ja": "グループ間の共通点："
  },
  differencesOfOpinion: {
    "en": "Differences of opinion:",
    "zh-TW": "意見分歧：",
    "zh-CN": "意见分歧：",
    "fr": "Différences d'opinion :",
    "es": "Diferencias de opinión:",
    "ja": "意見の相違："
  },
  otherStatements: {
    "en": "**Other statements** ({count} statements",
    "zh-TW": "**其他意見** ({count} 個意見",
    "zh-CN": "**其他意见** ({count} 个意见",
    "fr": "**Autres déclarations** ({count} déclarations",
    "es": "**Otras declaraciones** ({count} declaraciones",
    "ja": "**その他の声明** ({count} 件の声明"
  }
};

export function getSubsectionTitle(
  section: keyof typeof SUBSECTION_TITLES,
  lang: SupportedLanguage,
  count?: number
): string {
  let title = SUBSECTION_TITLES[section][lang] || SUBSECTION_TITLES[section]["en"];
  
  if (count !== undefined) {
    title = title.replace("{count}", count.toString());
  }
  
  return title;
}
