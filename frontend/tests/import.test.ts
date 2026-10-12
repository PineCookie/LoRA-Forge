import { describe, expect, it } from "vitest";
import { expandImportedArgs } from "../src/config/import";
import { normalizeTrainingConfig, type TrainingConfig } from "../src/config/normalize";
import { fromToml, toToml } from "../src/config/toml";

// form values -> export (normalizeTrainingConfig) -> TOML -> import (expandImportedArgs) -> form ->
// submit must reproduce the export, and a later form edit must win.

const UNSET = "<form default>";

function roundTrip(form: TrainingConfig) {
  const exported = structuredClone(form);
  normalizeTrainingConfig(exported);
  const imported = expandImportedArgs(fromToml(toToml(exported)));
  // the form only has its own fields; a field the import does not restore keeps its default
  const reloaded = Object.fromEntries(Object.keys(form).map((key) => [key, key in imported ? imported[key] : UNSET]));
  return { exported, reloaded };
}

function submit(form: TrainingConfig) {
  const config = structuredClone(form);
  for (const [key, value] of Object.entries(config)) if (value === UNSET) delete config[key];
  normalizeTrainingConfig(config);
  return config;
}

// form fields hold strings; compare numbers and their string forms as equal
const loose = (config: TrainingConfig) => JSON.parse(JSON.stringify(config, (_key, value) => (typeof value === "number" ? String(value) : value)));

describe("config import round-trip", () => {
  const lycoris = {
    network_module: "lycoris.kohya", conv_dim: 4, conv_alpha: 1, dropout: 0, lycoris_algo: "lokr", lokr_factor: 8, train_norm: true,
    optimizer_type: "Prodigy", prodigy_d_coef: "2.0", prodigy_d0: "1e-6", lr_warmup_steps: 100,
    full_precision: "full_fp16", mixed_precision: "bf16", learning_rate: 1,
    network_args_custom: ["train_llm_adapter=true"], optimizer_args_custom: ["decouple=False"],
    enable_block_weights: true, down_lr_weight: "1,1,1,1,1,1,1,1,1,1,1,0", mid_lr_weight: "1",
    up_lr_weight: "0,1,1,1,1,1,1,1,1,1,1,1", block_lr_zero_threshold: 0.1,
  };

  it("re-submits a lycoris + prodigy + block weights + full_fp16 config unchanged", () => {
    const { exported, reloaded } = roundTrip(lycoris);
    expect(loose(submit(reloaded))).toEqual(loose(exported));
    expect(reloaded.full_precision).toBe("full_fp16");
    expect(reloaded.network_args_custom).toEqual(["train_llm_adapter=true"]);
    expect(reloaded.optimizer_args_custom).toEqual(["decouple=False"]);
  });

  it("lets edits made after the import win", () => {
    const { reloaded } = roundTrip(lycoris);
    const edited = submit({ ...reloaded, conv_dim: 8, prodigy_d0: "1e-5", lr_warmup_steps: 0 });
    expect(edited.network_args).toContain("conv_dim=8");
    expect(edited.network_args).not.toContain("conv_dim=4");
    expect(edited.optimizer_args).toContain("d0=1e-5");
    expect(edited.optimizer_args).not.toContain("d0=1e-6");
    expect(edited.optimizer_args).not.toContain("safeguard_warmup=True");
    expect(edited.full_fp16).toBe(true);
    expect(edited.mixed_precision).toBe("fp16");
  });

  it("keeps d_coef custom for DAdaptation and does not duplicate the rule's args", () => {
    const { exported, reloaded } = roundTrip({ optimizer_type: "DAdaptAdam", learning_rate: 1, optimizer_args_custom: ["d_coef=2"], network_module: "networks.dylora", dylora_unit: 8 });
    expect(reloaded.optimizer_args_custom).toEqual(["d_coef=2"]);
    expect(reloaded.dylora_unit).toBe("8");
    expect(loose(submit(reloaded))).toEqual(loose(exported));
  });

  it("round-trips a plain AdamW config and clears stale custom args", () => {
    const { exported, reloaded } = roundTrip({ optimizer_type: "AdamW8bit", learning_rate: 0.0001, full_precision: "none", network_args_custom: [], optimizer_args_custom: [] });
    expect(reloaded.full_precision).toBe("none");
    expect(loose(submit(reloaded))).toEqual(loose(exported));
  });
});

describe("toml", () => {
  it("keeps strings with quotes, # and = intact", () => {
    const config = { output_name: 'a "b" #c = d', network_args: ["x=1", "y=a,b"], seed: 42, cache_latents: true };
    expect(fromToml(toToml(config))).toEqual(config);
  });

  it("rejects tables", () => {
    expect(() => fromToml("[general]\nbatch_size = 1")).toThrow();
  });
});
