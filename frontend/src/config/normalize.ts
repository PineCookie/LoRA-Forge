import { schemaOptions, type Group } from "../schema/groups";

// A training config as sent to /api/run and exported as JSON/TOML. Values are whatever the
// schema field produces (strings, numbers, booleans, string arrays), so this stays loose.
export type TrainingConfig = Record<string, any>;

export function isResolutionField(name: string) {
  return name === "resolution";
}

function parseResolutionValue(value: unknown): [string, string] | null {
  const match = String(value).trim().match(/^(\d+)\s*(?:[xX×,])\s*(\d+)$/);
  return match ? [match[1], match[2]] : null;
}

export function formatResolutionForInput(value: unknown): string {
  const parsed = parseResolutionValue(value);
  return parsed ? `${parsed[0]}x${parsed[1]}` : String(value ?? "");
}

export function normalizeResolutionForConfig(value: unknown): string {
  const parsed = parseResolutionValue(value);
  return parsed ? `${parsed[0]},${parsed[1]}` : String(value).trim();
}

/** Typed form values -> config: drops empty fields, converts numbers/arrays, then normalizes. */
export function cleanConfig(values: TrainingConfig, groups: Group[]): TrainingConfig {
  const config: TrainingConfig = {};
  const fieldMap = new Map(groups.flatMap((group) => group.fields.map((field) => [field.name, field.schema] as const)));
  for (const [name, value] of Object.entries(values)) {
    const schema = fieldMap.get(name);
    if (!schema) continue;
    if (schema.type !== "boolean" && schema.type !== "const" && value === "") continue;
    // form-only numbers become *_args text (d_coef=2.0), so they keep the text as typed
    if (schema.type === "number") config[name] = FORM_ONLY_KEYS.includes(name) ? String(value).trim() : Number(value);
    else if (schema.type === "enum" && schemaOptions(schema)?.every((option) => typeof option === "number")) config[name] = Number(value);
    else if (schema.type === "array") config[name] = Array.isArray(value) ? value : String(value).split(/\r?\n/).filter(Boolean);
    else config[name] = value;
  }
  normalizeTrainingConfig(config);
  return config;
}

interface OptimizerRule {
  match: (optimizer: string) => boolean;
  values: Record<string, string | number>;
  args: (config: TrainingConfig, optimizer: string) => (string | null)[];
}

// Values and optimizer_args an optimizer needs. `values` are also forced into the form while the
// optimizer is selected; `args` are appended to optimizer_args on submit.
export const optimizerRules: OptimizerRule[] = [
  {
    match: (optimizer) => optimizer.toLowerCase().startsWith("dadapt"),
    values: { learning_rate: 1, unet_lr: 1, text_encoder_lr: 1 },
    args: (_config, optimizer) => (["DAdaptation", "DAdaptAdam"].includes(optimizer) ? ["decouple=True", "weight_decay=0.01"] : []),
  },
  {
    match: (optimizer) => optimizer.toLowerCase() === "prodigy",
    values: { max_grad_norm: 0 },
    args: (config) => [
      "decouple=True",
      "weight_decay=0.01",
      "use_bias_correction=True",
      config.prodigy_d_coef !== undefined ? `d_coef=${config.prodigy_d_coef}` : null,
      config.lr_warmup_steps ? "safeguard_warmup=True" : null,
      config.prodigy_d0 ? `d0=${config.prodigy_d0}` : null,
    ],
  },
  {
    match: (optimizer) => ["prodigyplus.ProdigyPlusScheduleFree", "ProdigyPlusScheduleFree"].includes(optimizer),
    values: { learning_rate: 1, unet_lr: 1, text_encoder_lr: 1, lr_scheduler: "constant", max_grad_norm: 0 },
    args: (config) => [
      config.prodigyplus_d_coef !== undefined ? `d_coef=${config.prodigyplus_d_coef}` : null,
      config.prodigyplus_betas ? `betas=${config.prodigyplus_betas}` : null,
      config.prodigyplus_schedulefree_c !== undefined ? `schedulefree_c=${config.prodigyplus_schedulefree_c}` : null,
      config.prodigyplus_prodigy_steps !== undefined ? `prodigy_steps=${config.prodigyplus_prodigy_steps}` : null,
    ],
  },
];

export function optimizerRuleFor(optimizer: unknown) {
  const value = String(optimizer ?? "");
  return optimizerRules.find((rule) => rule.match(value));
}

function applyOptimizerRuleToConfig(config: TrainingConfig) {
  const optimizer = String(config.optimizer_type ?? "");
  const rule = optimizerRuleFor(optimizer);
  if (!rule) return;
  Object.assign(config, rule.values);
  config.optimizer_args.push(...rule.args(config, optimizer).filter(Boolean));
}

// compile_shapes (a LoRA-Forge option) -> sd-scripts' compile_dynamic; `bounded` also sets
// compile_dynamic_seq, which the backend turns into run_entry.py's patch (it is not an sd-scripts key)
const COMPILE_SHAPES: Record<string, { compile_dynamic: string; compile_dynamic_seq?: boolean }> = {
  bounded: { compile_dynamic: "false", compile_dynamic_seq: true },
  static: { compile_dynamic: "false" },
  auto: { compile_dynamic: "auto" },
  dynamic: { compile_dynamic: "true" },
};

/** compile_shapes for imported compile_dynamic / compile_dynamic_seq values (the inverse of COMPILE_SHAPES). */
export function compileShapesOf(config: TrainingConfig): string | undefined {
  if ("compile_dynamic_seq" in config) {
    if (config.compile_dynamic_seq) return "bounded";
  } else if (!("compile_dynamic" in config)) {
    return undefined;
  }
  const dynamic = String(config.compile_dynamic ?? "false").toLowerCase();
  if (dynamic === "true") return "dynamic";
  if (dynamic === "auto") return "auto";
  // a config from before compile_dynamic_seq existed had false only because nothing else worked;
  // bounded trains the same and compiles once
  return "compile_dynamic_seq" in config ? "static" : "bounded";
}

function appendNetworkArg(config: TrainingConfig, name: string) {
  const value = typeof config[name] === "string" ? config[name].trim() : config[name];
  if (value !== undefined && value !== "") config.network_args.push(`${name}=${value}`);
}

const FORM_ONLY_KEYS = [
  "network_args_custom",
  "network_reg_dims",
  "network_reg_lrs",
  "optimizer_args_custom",
  "enable_base_weight",
  "enable_block_weights",
  "enable_preview",
  "conv_dim",
  "conv_alpha",
  "dropout",
  "lycoris_algo",
  "lokr_factor",
  "dylora_unit",
  "down_lr_weight",
  "mid_lr_weight",
  "up_lr_weight",
  "block_lr_zero_threshold",
  "train_norm",
  "prodigy_d0",
  "prodigy_d_coef",
  "prodigyplus_d_coef",
  "prodigyplus_betas",
  "prodigyplus_schedulefree_c",
  "prodigyplus_prodigy_steps",
];

const PREVIEW_KEYS = [
  "randomly_choice_prompt",
  "prompt_file",
  "positive_prompts",
  "negative_prompts",
  "sample_width",
  "sample_height",
  "sample_cfg",
  "sample_seed",
  "sample_steps",
  "sample_sampler",
  "sample_every_n_epochs",
  "sample_prompts",
];

/** Turn form-only fields into the sd-scripts options they stand for (network_args, optimizer_args, ...). */
export function normalizeTrainingConfig(config: TrainingConfig) {
  config.network_args ??= [];
  config.optimizer_args ??= [];

  appendNetworkArg(config, "network_reg_dims");
  appendNetworkArg(config, "network_reg_lrs");

  if (config.network_module === "lycoris.kohya") {
    if (config.conv_dim !== undefined) config.network_args.push(`conv_dim=${config.conv_dim}`);
    if (config.conv_alpha !== undefined) config.network_args.push(`conv_alpha=${config.conv_alpha}`);
    if (config.dropout !== undefined) config.network_args.push(`dropout=${config.dropout}`);
    if (config.lycoris_algo) config.network_args.push(`algo=${config.lycoris_algo}`);
    if (config.lokr_factor) config.network_args.push(`factor=${config.lokr_factor}`);
    if (config.train_norm) config.network_args.push("train_norm=True");
  } else if (config.network_module === "networks.dylora") {
    if (config.dylora_unit !== undefined) config.network_args.push(`unit=${config.dylora_unit}`);
  }

  applyOptimizerRuleToConfig(config);

  if (!["huber", "smooth_l1"].includes(config.loss_type)) {
    delete config.huber_schedule;
    delete config.huber_c;
  }

  if (!["sd-lora", "sdxl-lora", "sd-dreambooth", "sdxl-finetune"].includes(config.model_train_type)) {
    delete config.v2;
    delete config.v_parameterization;
  }
  if (!["sd-lora", "sd-dreambooth"].includes(config.model_train_type)) delete config.v2;
  // sdxl_train.py only uses the text encoder learning rates with --train_text_encoder
  if (!config.train_text_encoder) {
    delete config.learning_rate_te1;
    delete config.learning_rate_te2;
  }
  if (!config.v_parameterization) delete config.scale_v_pred_loss_like_noise_pred;

  if (config.enable_block_weights) {
    for (const name of ["down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold"]) {
      if (config[name] !== undefined) config.network_args.push(`${name}=${config[name]}`);
    }
  }

  if (config.enable_base_weight) {
    if (typeof config.base_weights === "string") config.base_weights = config.base_weights.split(/\r?\n/).filter(Boolean);
    if (typeof config.base_weights_multiplier === "string") {
      config.base_weights_multiplier = config.base_weights_multiplier.split(/\r?\n/).filter(Boolean).map(Number);
    }
  } else {
    delete config.base_weights;
    delete config.base_weights_multiplier;
  }

  if (Array.isArray(config.network_args_custom)) config.network_args.push(...config.network_args_custom.filter(Boolean));
  if (Array.isArray(config.optimizer_args_custom)) config.optimizer_args.push(...config.optimizer_args_custom.filter(Boolean));

  if (config.full_precision === "full_fp16") {
    config.full_fp16 = true;
    config.full_bf16 = false;
  } else if (config.full_precision === "full_bf16") {
    config.full_fp16 = false;
    config.full_bf16 = true;
  } else if (config.full_precision === "none") {
    config.full_fp16 = false;
    config.full_bf16 = false;
  }
  delete config.full_precision;

  if (!config.enable_preview) for (const key of PREVIEW_KEYS) delete config[key];

  if (config.compile_shapes !== undefined) {
    if (config.compile) Object.assign(config, COMPILE_SHAPES[config.compile_shapes] ?? {});
    delete config.compile_shapes;
  }
  // sd-scripts takes 150 or 225; 75 (CLIP's own length) is its default, without the option
  if (Number(config.max_token_length) === 75) delete config.max_token_length;

  for (const key of Object.keys(config)) {
    if (typeof config[key] === "string" && /(_dir|_path|weights|vae|qwen3|t5xxl|clip_l|clip_g|ae|resume)$/.test(key)) {
      config[key] = config[key].replaceAll("\\", "/");
    }
    if (config[key] === "" || (Array.isArray(config[key]) && config[key].length === 0)) delete config[key];
  }

  for (const key of FORM_ONLY_KEYS) delete config[key];

  if (!config.network_args?.length) delete config.network_args;
  if (!config.optimizer_args?.length) delete config.optimizer_args;

  normalizeConstraintCombinations(config);
}

// Last line of defence before submit/export: sd-scripts asserts on these combinations, and an
// imported config never passes through the form's live coupling.
function normalizeConstraintCombinations(config: TrainingConfig) {
  if (config.compile_fullgraph) config.split_attn = false;

  if (config.blocks_to_swap) {
    config.cpu_offload_checkpointing = false;
    config.unsloth_offload_checkpointing = false;
  }
  if (config.cpu_offload_checkpointing || config.unsloth_offload_checkpointing) {
    delete config.blocks_to_swap;
  }

  if (["anima-lora", "sdxl-lora"].includes(config.model_train_type) && config.cache_text_encoder_outputs) {
    config.network_train_unet_only = true;
    config.network_train_text_encoder_only = false;
  }

  // full_precision became full_fp16/full_bf16 above; sd-scripts requires matching mixed_precision
  if (config.full_fp16) config.mixed_precision = "fp16";
  else if (config.full_bf16) config.mixed_precision = "bf16";
}

const SECRET_KEYS = ["wandb_api_key", "huggingface_token", "hf_token"];

export function redactConfigForLog(config: TrainingConfig): TrainingConfig {
  const copy = { ...config };
  for (const key of SECRET_KEYS) {
    if (copy[key]) copy[key] = "<redacted>";
  }
  return copy;
}
