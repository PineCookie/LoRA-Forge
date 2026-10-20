import type { MessageKey } from "../i18n";
import type { FormValues } from "./form";
import { resolutionOf, stepCount } from "./hints";
import type { ModelFileState } from "./modelFiles";
import { RESOLUTIONS } from "./resolutions";
import { isFieldVisible } from "./visibility";

// Problems the page can see before Start: the ones the backend or the trainer would refuse (errors,
// Start waits for them), required files not picked yet (todo, Start waits too) and settings that run
// but are probably not meant (warnings). Shown next to the option and in the run summary.

export type CheckLevel = "error" | "todo" | "warn";

export interface Check {
  field: string;
  level: CheckLevel;
  key: MessageKey;
  params?: Record<string, string | number>;
}

export interface CheckContext {
  schemaName: string;
  values: FormValues;
  /** the dataset folder could not be read (null while loading or unset) */
  datasetError: string | null;
  /** its `<repeats>_<name>` subfolders, once read (null while loading or unset) */
  datasetSubfolders?: number | null;
  fileStates?: Record<string, ModelFileState | "checking">;
}

// the files each page cannot start without (config.py and the trainers refuse an empty one)
const REQUIRED: Record<string, string[]> = {
  "anima-lora": ["qwen3", "vae"],
  "krea2-lora": ["text_encoder", "vae"],
  "moss-image-lora": ["text_encoder", "vae"],
};

// config.py check_full_fp16_optimizer: fp16 weights turn these into NaN or stop them learning
const FULL_FP16_BROKEN = new Set([
  "adamw", "radamschedulefree", "pytorch_optimizer.came", "prodigy", "prodigyplus.prodigyplusschedulefree",
  "dadaptation", "dadaptadam", "dadaptadagrad", "dadaptadanip", "dadaptlion", "dadaptsgd",
]);
const FULL_FP16_DEGRADED = new Set(["lion", "sgdnesterov", "sgdnesterov8bit"]);

const FILE_PROBLEMS = new Set<ModelFileState>(["missing", "wrong", "quantized", "fp8", "turbo"]);

const text = (value: unknown) => String(value ?? "").trim();
const number = (value: unknown) => (text(value) === "" ? NaN : Number(value));

export function formChecks({ schemaName, values, datasetError, datasetSubfolders = null, fileStates = {} }: CheckContext): Check[] {
  const checks: Check[] = [];
  const has = (name: string) => name in values;

  const required = ["pretrained_model_name_or_path", ...(schemaName === "sdxl-finetune" ? [] : ["train_data_dir"]), ...(REQUIRED[schemaName] ?? [])];
  for (const field of required) if (has(field) && !text(values[field])) checks.push({ field, level: "todo", key: "check.required" });
  if (datasetError && text(values.train_data_dir)) checks.push({ field: "train_data_dir", level: "error", key: "check.dataset", params: { error: datasetError } });
  // the backend refuses a folder without them (the images would not be found); sdxl-finetune reads metadata instead
  else if (datasetSubfolders === 0 && schemaName !== "sdxl-finetune") checks.push({ field: "train_data_dir", level: "error", key: "check.noSubfolders" });

  for (const [field, state] of Object.entries(fileStates)) {
    if (FILE_PROBLEMS.has(state as ModelFileState) && isFieldVisible(field, values)) {
      checks.push({ field, level: "warn", key: `modelFiles.state.${state}` as MessageKey });
    }
  }

  const size = resolutionOf(values.resolution);
  if (has("resolution") && !size) checks.push({ field: "resolution", level: "error", key: "check.resolution" });
  // a fixed size (buckets off) reaches the model as it is: U-Nets need multiples of 64, DiTs of 16
  const step = RESOLUTIONS[schemaName]?.step;
  if (size && step && values.enable_bucket === false && (size[0] % step || size[1] % step)) {
    checks.push({ field: "resolution", level: "error", key: "check.resolutionStep", params: { step } });
  }

  for (const field of ["lr_warmup_steps", "lr_decay_steps"]) {
    if (has(field) && isFieldVisible(field, values) && stepCount(values[field], 1) === null) checks.push({ field, level: "error", key: "check.stepsOrFraction" });
  }

  const fullFp16 = values.training_precision === "full-fp16" || values.full_precision === "full_fp16";
  const optimizer = text(values.optimizer_type).toLowerCase();
  const precisionField = has("training_precision") ? "training_precision" : "full_precision";
  if (fullFp16 && FULL_FP16_BROKEN.has(optimizer)) {
    checks.push({ field: precisionField, level: "error", key: "check.fullFp16Optimizer", params: { optimizer: text(values.optimizer_type) } });
  } else if (fullFp16 && FULL_FP16_DEGRADED.has(optimizer)) {
    checks.push({ field: precisionField, level: "warn", key: "check.fullFp16Degraded", params: { optimizer: text(values.optimizer_type) } });
  }

  // sd-scripts refuses cached text encoder outputs for captions that change per step (an import can
  // bring both; in the form, rules.ts switches one off)
  if (values.cache_text_encoder_outputs === true) {
    const varying = [
      values.shuffle_caption === true && "shuffle_caption",
      number(values.caption_tag_dropout_rate) > 0 && "caption_tag_dropout_rate",
      number(values.caption_dropout_rate) > 0 && schemaName !== "anima-lora" && "caption_dropout_rate",
    ];
    for (const field of varying) if (field) checks.push({ field, level: "error", key: "check.textEncoderCache" });
  }

  if (number(values.network_alpha) > number(values.network_dim)) checks.push({ field: "network_alpha", level: "warn", key: "check.alphaAboveDim" });
  if (number(values.save_every_n_epochs) > number(values.max_train_epochs)) checks.push({ field: "save_every_n_epochs", level: "warn", key: "check.saveEvery" });

  return checks;
}

/** Start waits until these are gone. */
export const blocksStart = (check: Check) => check.level !== "warn";
