import { computed, ref, watchEffect } from "vue";

// The size of the whole UI. "auto" grows it on wide windows (4K screens show the page at its CSS size,
// so text that fits a 1080p screen looks tiny there); the others are fixed factors. Applied as CSS zoom
// on <html>; viewport-height lengths divide by --ui-scale (tokens.css), since zoom scales them too.

export type UiScaleChoice = "auto" | "1" | "1.25" | "1.5";

export const UI_SCALE_CHOICES: UiScaleChoice[] = ["auto", "1", "1.25", "1.5"];
const STORAGE_KEY = "lora-forge.ui-scale";

function savedChoice(): UiScaleChoice {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (UI_SCALE_CHOICES.includes(saved as UiScaleChoice)) return saved as UiScaleChoice;
  } catch {
    // storage blocked
  }
  return "auto";
}

const width = ref(typeof window === "undefined" ? 1920 : window.innerWidth);
if (typeof window !== "undefined") window.addEventListener("resize", () => (width.value = window.innerWidth));

/** auto: 100% up to a 1920 px wide window, then growing to 140% at 3840 px (a 4K screen at 100%). */
export function autoScale(windowWidth: number): number {
  if (windowWidth <= 1920) return 1;
  return Math.min(1.4, Math.round((1 + ((windowWidth - 1920) / 1920) * 0.4) * 20) / 20);
}

export const uiScaleChoice = ref<UiScaleChoice>(savedChoice());
export const uiScale = computed(() => (uiScaleChoice.value === "auto" ? autoScale(width.value) : Number(uiScaleChoice.value)));

export function setUiScale(choice: UiScaleChoice) {
  uiScaleChoice.value = choice;
  try {
    localStorage.setItem(STORAGE_KEY, choice);
  } catch {
    // the choice then only lasts for this page load
  }
}

/** Keep <html>'s zoom and --ui-scale in sync with the choice and the window width. */
export function applyUiScale() {
  watchEffect(() => {
    const root = document.documentElement;
    root.style.zoom = uiScale.value === 1 ? "" : String(uiScale.value);
    root.style.setProperty("--ui-scale", String(uiScale.value));
  });
}
