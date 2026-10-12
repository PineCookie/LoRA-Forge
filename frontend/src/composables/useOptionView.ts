import { ref } from "vue";

/** "essentials" shows the common options (config/essentials.ts), "all" every option. */
export type OptionView = "essentials" | "all";

const STORAGE_KEY = "lora-forge.option-view";

function savedView(): OptionView {
  try {
    if (localStorage.getItem(STORAGE_KEY) === "all") return "all";
  } catch {
    // storage blocked
  }
  return "essentials";
}

// shared by every trainer page and remembered per browser
export const optionView = ref<OptionView>(savedView());

export function setOptionView(view: OptionView) {
  optionView.value = view;
  try {
    localStorage.setItem(STORAGE_KEY, view);
  } catch {
    // the choice then only lasts for this page load
  }
}
