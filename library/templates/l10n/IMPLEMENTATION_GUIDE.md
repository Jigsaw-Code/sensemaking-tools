# 本地化系統實施指南

本指南說明如何將現有的 summary 類重構為使用新的本地化系統。

## 🎯 目標

1. **保持提示語為英文**：所有 LLM 提示語保持英文，確保模型理解任務要求
2. **通過語言前綴控制輸出**：在呼叫 LLM 時添加語言前綴來控制輸出語言
3. **報告內容本地化**：報告的標題、章節、描述等根據 `output_lang` 顯示相應語言
4. **支援多語言擴展**：輕鬆添加新語言支援（如法語、德語等）

## 🔧 實施步驟

### 步驟 1：導入本地化模組

```typescript
import { 
  getReportSectionTitle, 
  getReportContent, 
  getSubsectionTitle,
  getTopicSummaryText,
  getPluralForm,
  type SupportedLanguage 
} from '../templates/l10n';
```

### 步驟 2：更新類的建構函數

```typescript
export class YourSummaryClass extends RecursiveSummary<YourInputType> {
  constructor(input: YourInputType, model: Model, additionalContext?: string, output_lang: SupportedLanguage = "en") {
    super(input, model, additionalContext, output_lang);
  }
}
```

### 步驟 3：重構 getSummary 方法

#### 之前（硬編碼英文）：
```typescript
async getSummary(): Promise<SummaryContent> {
  return {
    title: "## Introduction",
    text: "This report summarizes the results of public input..."
  };
}
```

#### 之後（使用本地化）：
```typescript
async getSummary(): Promise<SummaryContent> {
  const lang = this.output_lang;
  
  const title = getReportSectionTitle("introduction", lang);
  const text = getReportContent("introduction", "text", lang);
  
  return { title, text };
}
```

### 步驟 4：處理動態內容

#### 使用替換參數：
```typescript
const overviewText = getReportContent("topics", "overview", lang, {
  topicCount: 5,
  subtopicsText: ", as well as 12 subtopics",
  groupsText: " between opinion groups,",
  groupsBetweenText: "between groups "
});
```

#### 處理複數形式：
```typescript
const text = getTopicSummaryText("topicSummary", lang, {
  subtopicCount: 3,
  subtopicPlural: getPluralForm(3, lang),
  statementCount: 15,
  statementPlural: getPluralForm(15, lang)
});
```

## 📝 具體重構範例

### IntroSummary 類重構

```typescript
// 之前
export class IntroSummary extends RecursiveSummary<SummaryStats> {
  getSummary(): Promise<SummaryContent> {
    let text: string;
    let title: string;
    
    if (this.output_lang === "zh-TW") {
      text = `本報告總結了公眾意見的結果，包含：\n`;
      text += ` * __${this.input.commentCount.toLocaleString()} 個意見__\n`;
      // ... 更多硬編碼的中文文字
      title = "## 簡介";
    } else {
      text = `This report summarizes the results of public input, encompassing:\n`;
      text += ` * __${this.input.commentCount.toLocaleString()} statements__\n`;
      // ... 更多硬編碼的英文文字
      title = "## Introduction";
    }
    
    return Promise.resolve({ title: title, text: text });
  }
}

// 之後
export class IntroSummary extends RecursiveSummary<SummaryStats> {
  getSummary(): Promise<SummaryContent> {
    const lang = this.output_lang;
    
    const title = getReportSectionTitle("introduction", lang);
    const text = getReportContent("introduction", "text", lang);
    const statementsLabel = getReportContent("introduction", "statements", lang);
    const votesLabel = getReportContent("introduction", "votes", lang);
    const topicsLabel = getReportContent("introduction", "topics", lang);
    const subtopicsLabel = getReportContent("introduction", "subtopics", lang);
    const anonymousText = getReportContent("introduction", "anonymous", lang);
    
    const content = `${text}\n` +
      ` * __${this.input.commentCount.toLocaleString()} ${statementsLabel}__\n` +
      ` * __${this.input.voteCount.toLocaleString()} ${votesLabel}__\n` +
      ` * ${this.input.getStatsByTopic().length} ${topicsLabel}\n` +
      ` * ${this.getSubtopicCount()} ${subtopicsLabel}\n\n` +
      `${anonymousText}`;
    
    return Promise.resolve({ title, text: content });
  }
  
  private getSubtopicCount(): number {
    const statsByTopic = this.input.getStatsByTopic();
    return statsByTopic.map(topic => topic.subtopicStats?.length || 0).reduce((a, b) => a + b, 0);
  }
}
```

### OverviewSummary 類重構

```typescript
// 之前
async getSummary(): Promise<SummaryContent> {
  let preamble: string;
  let title: string;
  
  if (this.output_lang === "zh-TW") {
    preamble = `以下是對話中討論主題的高層次概述...`;
    title = "## 概述";
  } else {
    preamble = `Below is a high level overview of the topics...`;
    title = "## Overview";
  }
  
  return { title: title, text: preamble + result };
}

// 之後
async getSummary(): Promise<SummaryContent> {
  const lang = this.output_lang;
  
  const title = getReportSectionTitle("overview", lang);
  const preamble = getReportContent("overview", "preamble", lang);
  
  return { title, text: preamble + result };
}
```

## 🌍 添加新語言支援

### 1. 更新語言配置

```typescript
// 在 languages.ts 中
export type SupportedLanguage = "en" | "zh-TW" | "fr" | "de";

export const SUPPORTED_LANGUAGES: SupportedLanguage[] = ["en", "zh-TW", "fr", "de"];

export const LANGUAGE_PREFIXES: Record<SupportedLanguage, string> = {
  "en": "",
  "zh-TW": "請用繁體中文回答",
  "fr": "Veuillez répondre en français",
  "de": "Bitte antworten Sie auf Deutsch"
};
```

### 2. 添加翻譯

```typescript
// 在 report_sections.ts 中
export const REPORT_SECTIONS = {
  introduction: {
    "en": "## Introduction",
    "zh-TW": "## 簡介",
    "fr": "## Introduction",
    "de": "## Einführung"  // 添加德語翻譯
  }
  // ... 其他章節
};
```

## ✅ 檢查清單

- [ ] 導入本地化模組
- [ ] 更新建構函數參數類型
- [ ] 重構硬編碼的標題和文字
- [ ] 使用 `getReportSectionTitle()` 獲取章節標題
- [ ] 使用 `getReportContent()` 獲取內容文字
- [ ] 使用 `getSubsectionTitle()` 獲取子章節標題
- [ ] 處理動態內容的替換參數
- [ ] 處理複數形式
- [ ] 測試所有支援的語言
- [ ] 確保 LLM 提示語保持英文

## 🚀 優勢

1. **維護性**：所有文字集中在一個地方管理
2. **擴展性**：輕鬆添加新語言支援
3. **一致性**：確保所有地方使用相同的翻譯
4. **類型安全**：TypeScript 類型檢查防止錯誤
5. **模組化**：清晰的職責分離

## 🔍 測試

使用提供的測試檔案驗證本地化系統：

```bash
npx ts-node library/templates/l10n/test_localization.ts
```

這將測試所有支援的語言和功能。
