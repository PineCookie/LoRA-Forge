import type { MessageKey } from "../i18n";
import type { FormValues } from "./form";
import type { ModelFileState } from "./modelFiles";
import { parseResolution } from "./normalize";
import { isFieldVisible } from "./visibility";

// Live hints next to an option's value ("the first 161 of 3,220 steps") and the step math behind them.
// Pure: the page passes what it knows (form values, the dataset size, the model file checks).

export interface HintContext {
  values: FormValues;
  /** images x repeats of the dataset folder; null while unknown */
  imagesWithRepeats: number | null;
  /** model file checks by field (ModelFiles) */
  fileStates?: Record<string, ModelFileState | "checking">;
}

export interface Hint {
  key: MessageKey;
  params?: Record<string, string | number>;
  tone?: "ok" | "warn";
}

const number = (value: unknown) => (String(value ?? "").trim() === "" ? NaN : Number(value));
const positive = (value: unknown, fallback = 1) => {
  const n = number(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
};
const fmt = (n: number) => n.toLocaleString();

/** Steps per epoch: images with repeats over the effective batch (a lower bound; buckets add a few). */
export function stepsPerEpoch(values: FormValues, imagesWithRepeats: number | null): number | null {
  if (!imagesWithRepeats) return null;
  return Math.ceil(imagesWithRepeats / (positive(values.train_batch_size) * positive(values.gradient_accumulation_steps)));
}

export function totalSteps(values: FormValues, imagesWithRepeats: number | null): number | null {
  const perEpoch = stepsPerEpoch(values, imagesWithRepeats);
  const epochs = number(values.max_train_epochs);
  return perEpoch === null || !(epochs > 0) ? null : perEpoch * epochs;
}

/** sd-scripts' and musubi-tuner's reading of lr_warmup_steps / lr_decay_steps: a whole number counts
 * steps, a value below 1 is a fraction of all steps. null for a value they would refuse. */
export function stepCount(value: unknown, total: number): number | null {
  const n = number(value);
  if (String(value ?? "").trim() === "") return 0;
  if (!Number.isFinite(n) || n < 0 || (n >= 1 && !Number.isInteger(n))) return null;
  return n < 1 ? Math.floor(n * total) : n;
}

/** Width x height of the resolution field, or null when it does not parse. */
export function resolutionOf(value: unknown): [number, number] | null {
  const size = parseResolution(value);
  return size && size[0] > 0 && size[1] > 0 ? size : null;
}

const FILE_HINTS: Partial<Record<ModelFileState | "checking", Hint>> = {
  ok: { key: "hint.fileOk", tone: "ok" },
  checking: { key: "modelFiles.state.checking" },
  repo: { key: "modelFiles.state.repo", tone: "ok" },
  missing: { key: "modelFiles.state.missing", tone: "warn" },
  wrong: { key: "modelFiles.state.wrong", tone: "warn" },
  quantized: { key: "modelFiles.state.quantized", tone: "warn" },
  fp8: { key: "modelFiles.state.fp8", tone: "warn" },
  turbo: { key: "modelFiles.state.turbo", tone: "warn" },
};

/** The hint for option `name`, or null. */
export function fieldHint(name: string, context: HintContext): Hint | null {
  const { values, imagesWithRepeats } = context;
  const total = totalSteps(values, imagesWithRepeats);
  const fileState = context.fileStates?.[name];
  if (fileState && fileState !== "empty") return FILE_HINTS[fileState] ?? null;

  switch (name) {
    case "train_data_dir":
      return imagesWithRepeats === null ? null : { key: "hint.dataset", params: { images: fmt(imagesWithRepeats) } };
    case "resolution": {
      // buckets take only the pixel count from it; without them it is every image's exact size
      const size = resolutionOf(values.resolution);
      if (!size) return null;
      if (values.enable_bucket === false) return { key: "hint.resolutionFixed", params: { width: size[0], height: size[1] } };
      return { key: "hint.resolutionArea", params: { mp: ((size[0] * size[1]) / 1e6).toFixed(2) } };
    }
    case "max_train_epochs":
      return total === null ? null : { key: "hint.totalSteps", params: { steps: fmt(total) } };
    case "train_batch_size":
    case "gradient_accumulation_steps": {
      const perEpoch = stepsPerEpoch(values, imagesWithRepeats);
      const effective = positive(values.train_batch_size) * positive(values.gradient_accumulation_steps);
      if (name === "gradient_accumulation_steps") return effective > positive(values.train_batch_size) ? { key: "hint.effectiveBatch", params: { batch: effective } } : null;
      return perEpoch === null ? null : { key: "hint.stepsPerEpoch", params: { steps: fmt(perEpoch) } };
    }
    case "save_every_n_epochs": {
      const every = number(values.save_every_n_epochs);
      const epochs = number(values.max_train_epochs);
      if (!(every > 0) || !(epochs > 0)) return null;
      return { key: "hint.checkpoints", params: { count: Math.max(1, Math.floor(epochs / every)) } };
    }
    case "sample_every_n_epochs": {
      const every = number(values.sample_every_n_epochs);
      const epochs = number(values.max_train_epochs);
      return every > 0 && epochs > 0 ? { key: "hint.previewRounds", params: { count: Math.floor(epochs / every) } } : null;
    }
    case "lr_warmup_steps": {
      if (total === null) return null;
      const steps = stepCount(values.lr_warmup_steps, total);
      if (steps === null) return null;
      return steps ? { key: "hint.warmup", params: { steps: fmt(steps), total: fmt(total) } } : { key: "hint.off" };
    }
    case "lr_decay_steps": {
      if (total === null) return null;
      const steps = stepCount(values.lr_decay_steps, total);
      return steps ? { key: "hint.decay", params: { steps: fmt(steps) } } : null;
    }
    case "network_alpha": {
      const dim = number(values.network_dim);
      const alpha = number(values.network_alpha);
      return dim > 0 && alpha > 0 ? { key: "hint.loraScale", params: { scale: +(alpha / dim).toFixed(3) } } : null;
    }
    case "caption_dropout_rate":
    case "caption_tag_dropout_rate": {
      const rate = number(values[name]);
      return rate > 0 ? { key: "hint.dropout", params: { percent: Math.round(rate * 100) } } : null;
    }
    case "lr_scheduler":
      return isFieldVisible("prodigyplus_d_coef", values) ? { key: "hint.scheduleFree" } : null;
    default:
      return null;
  }
}
