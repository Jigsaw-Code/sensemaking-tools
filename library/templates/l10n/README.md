# Localization (l10n) System

This directory contains the localization system for the Sensemaker reports, supporting multiple languages including English, Traditional Chinese, and French.

## Structure

- `languages.ts` - Language configuration and validation
- `report_sections.ts` - Report section titles and headers
- `report_content.ts` - Report content and descriptions
- `subsection_titles.ts` - Subsection titles and labels
- `topic_summaries.ts` - Topic summary related text
- `index.ts` - Main export file

## Supported Languages

- `en` - English (default)
- `zh-TW` - Traditional Chinese
- `fr` - French

## Usage

### Basic Usage

```typescript
import { 
  getReportSectionTitle, 
  getReportContent, 
  getSubsectionTitle,
  getLanguagePrefix,
  type SupportedLanguage 
} from '../templates/l10n';

const lang: SupportedLanguage = "zh-TW";

// Get section title
const title = getReportSectionTitle("introduction", lang);
// Returns: "## 簡介"

// Get content with replacements
const content = getReportContent("topics", "overview", lang, {
  topicCount: 5,
  subtopicsText: ", as well as 12 subtopics",
  groupsText: " between the opinion groups described above,",
  groupsBetweenText: "between the groups "
});
// Returns localized text with replacements applied

// Get subsection title
const subtitle = getSubsectionTitle("prominentThemes", lang);
// Returns: "主要主題包括："

// Get language prefix for LLM calls
const prefix = getLanguagePrefix(lang);
// Returns: "請用繁體中文回答"
```

### Adding New Languages

1. Add the new language code to `SupportedLanguage` type in `languages.ts`
2. Add language name and prefix in `languages.ts`
3. Add translations for all text in each localization file
4. Update the `isValidLanguage` function if needed

### Adding New Text

1. Add the new text to the appropriate localization file
2. Provide translations for all supported languages
3. Export the getter function if needed
4. Update the main index file

## Design Principles

1. **Keep prompts in English**: All LLM prompts remain in English, only the output is localized
2. **Language prefixes**: Use language prefixes when calling LLMs to control output language
3. **Fallback to English**: If a translation is missing, fall back to English
4. **Type safety**: Use TypeScript types to ensure language codes are valid
5. **Easy extension**: Adding new languages should be straightforward

## Example: Adding French Support

```typescript
// In languages.ts
export type SupportedLanguage = "en" | "zh-TW" | "fr";

export const LANGUAGE_PREFIXES: Record<SupportedLanguage, string> = {
  "en": "",
  "zh-TW": "請用繁體中文回答",
  "fr": "Veuillez répondre en français"
};

// In report_sections.ts
export const REPORT_SECTIONS = {
  introduction: {
    "en": "## Introduction",
    "zh-TW": "## 簡介",
    "fr": "## Introduction"  // Add French translation
  }
  // ... other sections
};
```
