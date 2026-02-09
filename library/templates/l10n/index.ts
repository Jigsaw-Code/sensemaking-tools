// Main export file for the localization system
export * from './languages';
export * from './report_sections';
export * from './report_content';
export * from './subsection_titles';
export * from './topic_summaries';
export * from './statistics_messages';
export * from './topic_names';

// Re-export commonly used types and functions
export type { SupportedLanguage } from './languages';
export { getLanguagePrefix, isValidLanguage, getLanguageName } from "./languages";
export { getReportSectionTitle } from "./report_sections";
export { getReportContent } from "./report_content";
export { getSubsectionTitle } from "./subsection_titles";
export { getTopicSummaryText, getPluralForm } from "./topic_summaries";
export { getStatisticsMessage } from "./statistics_messages";
export { translateSummary } from "./translate_summary";