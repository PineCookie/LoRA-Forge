import { beforeAll, describe, expect, it } from "vitest";
import { autoScale } from "../src/composables/useUiScale";
import { locale } from "../src/i18n";
import { blocksStart, formChecks } from "../src/config/checks";
import { initialFormValues } from "../src/config/form";
import { fieldHint, stepCount, stepsPerEpoch, totalSteps } from "../src/config/hints";
import { sectionSummary } from "../src/config/summaries";
import { conditionOf } from "../src/config/visibility";
import { loadGroups } from "./helpers";

// what the trainer page tells about the values: hints, checks, section summaries

const anima = () => initialFormValues(loadGroups("anima-lora"));

describe("step math", () => {
  it("counts steps per epoch and in all from the dataset size", () => {
    const values = { ...anima(), train_batch_size: "2", gradient_accumulation_steps: "2", max_train_epochs: "10" };
    expect(stepsPerEpoch(values, 322)).toBe(81);
    expect(totalSteps(values, 322)).toBe(810);
    expect(totalSteps(values, null)).toBeNull();
  });

  it("reads warmup and decay as sd-scripts does: whole steps, or a fraction below 1", () => {
    expect(stepCount("0.05", 3220)).toBe(161);
    expect(stepCount("100", 3220)).toBe(100);
    expect(stepCount("", 3220)).toBe(0);
    expect(stepCount("1.5", 3220)).toBeNull();
    expect(stepCount("-1", 3220)).toBeNull();
  });
});

describe("hints", () => {
  const values = { ...anima(), lr_scheduler: "cosine", lr_warmup_steps: "0.05", max_train_epochs: "10", train_batch_size: "1" };

  it("turns a warmup fraction into steps of this run", () => {
    expect(fieldHint("lr_warmup_steps", { values, imagesWithRepeats: 322 })).toEqual({ key: "hint.warmup", params: { steps: "161", total: "3,220" } });
    expect(fieldHint("lr_warmup_steps", { values, imagesWithRepeats: null })).toBeNull();
  });

  it("shows the model file check on the path field", () => {
    expect(fieldHint("vae", { values, imagesWithRepeats: null, fileStates: { vae: "ok" } })?.tone).toBe("ok");
    expect(fieldHint("vae", { values, imagesWithRepeats: null, fileStates: { vae: "wrong" } })?.tone).toBe("warn");
  });
});

describe("checks", () => {
  it("asks for the required files and blocks Start until they are set", () => {
    const values = { ...anima(), pretrained_model_name_or_path: "", qwen3: "", vae: "" };
    const checks = formChecks({ schemaName: "anima-lora", values, datasetError: null });
    expect(checks.filter((check) => check.level === "todo").map((check) => check.field)).toEqual(["pretrained_model_name_or_path", "qwen3", "vae"]);
    expect(checks.every(blocksStart)).toBe(true);
  });

  it("refuses a warmup sd-scripts would read as more than the run, unless the field is hidden", () => {
    const base = { ...anima(), pretrained_model_name_or_path: "m", qwen3: "q", vae: "v", lr_warmup_steps: "1.5" };
    expect(formChecks({ schemaName: "anima-lora", values: { ...base, lr_scheduler: "cosine" }, datasetError: null })).toEqual([
      { field: "lr_warmup_steps", level: "error", key: "check.stepsOrFraction" },
    ]);
    expect(formChecks({ schemaName: "anima-lora", values: { ...base, lr_scheduler: "constant" }, datasetError: null })).toEqual([]);
  });

  it("refuses the optimizers that do not train with fp16 weights, on both precision fields", () => {
    const animaChecks = formChecks({ schemaName: "anima-lora", values: { ...anima(), training_precision: "full-fp16", optimizer_type: "Prodigy" }, datasetError: null });
    expect(animaChecks).toContainEqual({ field: "training_precision", level: "error", key: "check.fullFp16Optimizer", params: { optimizer: "Prodigy" } });
    const flux = { ...initialFormValues(loadGroups("flux-lora")), full_precision: "full_fp16", optimizer_type: "AdamW" };
    expect(formChecks({ schemaName: "flux-lora", values: flux, datasetError: null }).map((check) => check.field)).toContain("full_precision");
  });

  it("warns about settings that run but are probably not meant", () => {
    const values = { ...anima(), pretrained_model_name_or_path: "m", qwen3: "q", vae: "v", network_dim: "16", network_alpha: "32", save_every_n_epochs: "20" };
    const checks = formChecks({ schemaName: "anima-lora", values, datasetError: null });
    expect(checks.map((check) => [check.field, check.level])).toEqual([["network_alpha", "warn"], ["save_every_n_epochs", "warn"]]);
    expect(checks.some(blocksStart)).toBe(false);
  });
});

describe("section summaries", () => {
  beforeAll(() => {
    locale.value = "en";
  });

  it("say what a collapsed section holds", () => {
    const values = { ...anima(), max_train_epochs: "10", train_batch_size: "2", optimizer_type: "AdamW8bit", learning_rate: "1e-4", lr_scheduler: "cosine" };
    expect(sectionSummary("training", values)).toBe("10 epochs · batch 2 · AdamW8bit · 1e-4 · cosine");
    expect(sectionSummary("training", { ...values, optimizer_type: "prodigyplus.ProdigyPlusScheduleFree" })).toContain("ProdigyPlus · 1e-4 · schedule-free");
    expect(sectionSummary("model", { ...values, pretrained_model_name_or_path: "./sd-models/anima-base.safetensors" })).toBe("anima-base.safetensors");
  });
});

describe("UI size", () => {
  it("grows on wide windows only", () => {
    expect([1280, 1920, 2560, 3840, 5120].map(autoScale)).toEqual([1, 1, 1.15, 1.4, 1.4]);
  });
});

describe("visibility conditions", () => {
  it("describe when a hidden option shows", () => {
    expect(conditionOf("lr_warmup_steps")).toEqual({ controller: "lr_scheduler", values: ["constant"], not: true });
    expect(conditionOf("lr_decay_steps")).toEqual({ controller: "lr_scheduler", values: ["warmup_stable_decay"], not: false });
    expect(conditionOf("network_dim")).toBeUndefined();
  });
});
