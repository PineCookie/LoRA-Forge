import { ref, shallowRef } from "vue";
import en from "./locales/en";
import ja from "./locales/ja";
import zh from "./locales/zh";

// English is the source language. UI text lives in ./locales (keyed, checked by the type
// checker); schema text (field descriptions, section titles) is English in the schema files and
// translated with catalogs the backend serves from lora_forge/schema/locales.

export const LOCALES = { en: "English", zh: "中文", ja: "日本語" } as const;
export type Locale = keyof typeof LOCALES;
export type MessageKey = keyof typeof en;
export type Messages = Record<MessageKey, string>;

const MESSAGES: Record<Locale, Messages> = { en, zh, ja };
const STORAGE_KEY = "lora-forge.locale";

function isLocale(value: unknown): value is Locale {
  return typeof value === "string" && Object.hasOwn(LOCALES, value);
}

/** The saved choice, else the first supported browser language, else English. */
function detectLocale(): Locale {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (isLocale(saved)) return saved;
  } catch {
    // storage blocked or not available (tests)
  }
  const languages = typeof navigator === "undefined" ? [] : (navigator.languages ?? [navigator.language]);
  for (const language of languages) {
    const base = language?.toLowerCase().split("-")[0];
    if (isLocale(base)) return base;
  }
  return "en";
}

export const locale = ref<Locale>(detectLocale());

export function setLocale(next: Locale) {
  locale.value = next;
  try {
    localStorage.setItem(STORAGE_KEY, next);
  } catch {
    // the choice then only lasts for this page load
  }
}

/** UI message `key` in the current language, with `{name}` fields filled from `params`. */
export function t(key: MessageKey, params: Record<string, string | number> = {}): string {
  return MESSAGES[locale.value][key].replace(/\{(\w+)\}/g, (field, name: string) => (name in params ? String(params[name]) : field));
}

const schemaCatalogs = shallowRef<Partial<Record<Locale, Record<string, string>>>>({});

export function setSchemaCatalogs(catalogs: Partial<Record<Locale, Record<string, string>>>) {
  schemaCatalogs.value = catalogs;
}

/** Translate schema text; text without a translation is shown in English. */
export function tr(source: string): string {
  return schemaCatalogs.value[locale.value]?.[source] ?? source;
}
