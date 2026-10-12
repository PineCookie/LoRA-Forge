import { t, type MessageKey } from "../i18n";

// sd-scripts' progress bar descriptions; bars without one are the dataset caching passes
const PHASES: Record<string, MessageKey> = {
  "": "runs.phase.preparing",
  "loading image sizes": "runs.phase.preparing",
  "get image size": "runs.phase.preparing",
  "read caption": "runs.phase.preparing",
  steps: "runs.phase.training",
  "validation steps": "runs.phase.validation",
  "epoch validation steps": "runs.phase.validation",
  Sampling: "runs.phase.sampling",
  "Denoising steps": "runs.phase.sampling",
};

/** A progress bar's phase in the UI language (unknown descriptions are shown as they are). */
export function phaseLabel(desc: string): string {
  const key = PHASES[desc];
  return key ? t(key) : desc;
}

const MODELS: Record<string, string> = {
  "anima-lora": "Anima",
  "flux-lora": "FLUX",
  "sd3-lora": "SD3.5",
  "sdxl-lora": "SDXL",
  "sd-lora": "SD1.5",
  "sd-dreambooth": "Dreambooth",
  "sdxl-finetune": "SDXL fine-tune",
};

export const modelLabel = (type: string) => MODELS[type] ?? type;
