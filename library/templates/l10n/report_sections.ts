import { SupportedLanguage } from "./languages";

// Report section titles and headers
export const REPORT_SECTIONS = {
  introduction: {
    "en": "## Introduction",
    "zh-TW": "## 簡介",
    "zh-CN": "## 简介",
    "fr": "## Introduction",
    "es": "## Introducción",
    "ja": "## はじめに"
  },
  overview: {
    "en": "## Overview",
    "zh-TW": "## 概述",
    "zh-CN": "## 概述",
    "fr": "## Aperçu",
    "es": "## Resumen",
    "ja": "## 概要"
  },
  topics: {
    "en": "## Topics",
    "zh-TW": "## 主題",
    "zh-CN": "## 主题",
    "fr": "## Sujets",
    "es": "## Temas",
    "ja": "## トピック"
  },
  topSubtopics: {
    "en": "## Top {count} Most Discussed Subtopics",
    "zh-TW": "## 前 {count} 個最常討論的子主題",
    "zh-CN": "## 前 {count} 个最常讨论的子主题",
    "fr": "## Top {count} des sous-sujets les plus discutés",
    "es": "## Top {count} de subtemas más discutidos",
    "ja": "## 最も議論された上位 {count} のサブトピック"
  },
  opinionGroups: {
    "en": "## Opinion Groups",
    "zh-TW": "## 意見群組",
    "zh-CN": "## 意见群组",
    "fr": "## Groupes d'opinion",
    "es": "## Grupos de opinión",
    "ja": "## 意見グループ"
  }
};

export function getReportSectionTitle(section: keyof typeof REPORT_SECTIONS, lang: SupportedLanguage, count?: number): string {
  let title = REPORT_SECTIONS[section][lang] || REPORT_SECTIONS[section]["en"];
  
  if (count !== undefined) {
    title = title.replace("{count}", count.toString());
  }
  
  return title;
}
