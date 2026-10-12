import { computed, ref, watchEffect } from "vue";

export type ThemeChoice = "system" | "light" | "dark";

export const THEME_CHOICES: ThemeChoice[] = ["system", "light", "dark"];
// index.html reads the same key before the app loads, so the page does not flash light first
const STORAGE_KEY = "lora-forge.theme";

function savedChoice(): ThemeChoice {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (THEME_CHOICES.includes(saved as ThemeChoice)) return saved as ThemeChoice;
  } catch {
    // storage blocked
  }
  return "system";
}

const media = typeof matchMedia === "undefined" ? null : matchMedia("(prefers-color-scheme: dark)");
const systemDark = ref(media?.matches ?? false);
media?.addEventListener("change", (event) => (systemDark.value = event.matches));

export const themeChoice = ref<ThemeChoice>(savedChoice());
export const theme = computed(() => (themeChoice.value === "system" ? (systemDark.value ? "dark" : "light") : themeChoice.value));

export function setTheme(choice: ThemeChoice) {
  themeChoice.value = choice;
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // the choice then only lasts for this page load
  }
}

/** Keep `<html data-theme>` in sync with the choice and the system setting. */
export function applyTheme() {
  watchEffect(() => {
    document.documentElement.dataset.theme = theme.value;
  });
}
