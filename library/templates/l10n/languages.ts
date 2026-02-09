// Supported languages configuration
export type SupportedLanguage = "en" | "zh-TW" | "zh-CN" | "fr" | "es" | "ja";

export const SUPPORTED_LANGUAGES: SupportedLanguage[] = ["en", "zh-TW", "zh-CN", "fr", "es", "ja"];

export const LANGUAGE_NAMES: Record<SupportedLanguage, string> = {
  "en": "English",
  "zh-TW": "繁體中文",
  "zh-CN": "简体中文",
  "fr": "Français",
  "es": "Español",
  "ja": "日本語"
};

export const LANGUAGE_PREFIXES: Record<SupportedLanguage, string> = {
  "en": "",
  "zh-TW": "你是一個只會寫繁體中文的AI，請一定要全文使用繁體中文回答",
  "zh-CN": "你是一个只会写简体中文的AI，请一定要全文使用简体中文回答",
  "fr": "Tu es une IA qui ne sait écrire qu'en français. Veuillez répondre en français.",
  "es": "Eres una IA que solo sabe escribir en español. Por favor responde en español.",
  "ja": "あなたは日本語しか書けないAIです。必ず日本語で回答してください。"
};

export function getLanguageName(lang: SupportedLanguage): string {
  return LANGUAGE_NAMES[lang] || "";
}

export function getLanguagePrefix(lang: SupportedLanguage): string {
  console.log(`[DEBUG] getLanguagePrefix() lang: ${lang}`);
  return LANGUAGE_PREFIXES[lang] || "";
}

export function isValidLanguage(lang: string): lang is SupportedLanguage {
  return SUPPORTED_LANGUAGES.includes(lang as SupportedLanguage);
}
