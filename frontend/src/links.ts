import type { Locale } from "./i18n";

export const REPOSITORY = "https://github.com/PineCookie/lora-forge";

/** A user guide in docs/ (English, with a `-zh` Chinese version), in the UI language where there is one. */
export function guideUrl(name: string, locale: Locale) {
  return `${REPOSITORY}/blob/main/docs/${name}${locale === "zh" ? "-zh" : ""}.md`;
}

/** A file's page on Hugging Face (or the mirror the launcher set, see /api/runtime). */
export function huggingFaceFileUrl(endpoint: string | undefined, repo: string, path: string) {
  return `${(endpoint || "https://huggingface.co").replace(/\/$/, "")}/${repo}/blob/main/${path}`;
}
