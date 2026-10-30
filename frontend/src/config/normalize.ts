import { schemaOptions, type Group } from "../schema/groups";

// A training config as sent to /api/run and exported as JSON/TOML. Values are whatever the
// schema field produces (strings, numbers, booleans, string arrays), so this stays loose.
export type TrainingConfig = Record<string, any>;

export function isResolutionField(name: string) {
  return name === "resolution";
}

// "1024" (a square), "1024x768", "1024×768", "1024*768", "1024,768" or "1024 768"; also the full-width
// digits and separators Chinese / Japanese input methods type
const RESOLUTION = /^(\d+)(?:\s*[xX×*,，ｘＸ＊]\s*|\s+)(\d+)$/;

/** Width and height of a resolution as people type it; null when it does not parse. */
export function parseResolution(value: unknown): [number, number] | null {
  const text = String(value ?? "")
    .trim()
    .replace(/[０-９]/g, (digit) => String.fromCharCode(digit.charCodeAt(0) - 0xfee0));
  if (/^\d+$/.test(text)) return [Number(text), Number(text)];
  const match = RESOLUTION.exec(text);
  return match ? [Number(match[1]), Number(match[2])] : null;
}

export function formatResolutionForInput(value: unknown): string {
  const parsed = parseResolution(value);
  return parsed ? `${parsed[0]}x${parsed[1]}` : String(value ?? "");
}

/** sd-scripts', musubi-tuner's and the MOSS-Image trainer's form: "width,height". */
export function normalizeResolutionForConfig(value: unknown): string {
  const parsed = parseResolution(value);
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
      config.lr_warmup_steps && config.lr_scheduler !== "constant" ? "safeguard_warmup=True" : null,
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

// compile_shapes (a LoRA-Forge option) -> sd-scripts' compile_dynamic
const COMPILE_SHAPES: Record<string, { compile_dynamic: string }> = {
  auto: { compile_dynamic: "auto" },
  static: { compile_dynamic: "false" },
  dynamic: { compile_dynamic: "true" },
};

/** compile_shapes for imported compile_dynamic / compile_dynamic_seq values (the inverse of COMPILE_SHAPES). */
export function compileShapesOf(config: TrainingConfig): string | undefined {
  if (!("compile_dynamic_seq" in config) && !("compile_dynamic" in config)) return undefined;
  const dynamic = String(config.compile_dynamic ?? "false").toLowerCase();
  if (dynamic === "true") return "dynamic";
  // compile_dynamic_seq was LoRA Forge's `bounded` mode, which `auto` replaces since PyTorch 2.15; a config from
  // before compile_dynamic_seq existed had false only because auto failed then
  if (dynamic === "auto" || config.compile_dynamic_seq || !("compile_dynamic_seq" in config)) return "auto";
  return "static";
}

// training_precision (Anima, SD1.5, SDXL: one choice) as sd-scripts' keys. Its fp16 choices also take
// LoRA-Forge's fp16_accumulation (a field of its own, applied by run_entry.py). Most optimizers fail
// with full-fp16's fp16 LoRA weights (config.py refuses those).
const TRAINING_PRECISION: Record<string, TrainingConfig> = {
  bf16: { mixed_precision: "bf16", full_bf16: false, full_fp16: false },
  "full-bf16": { mixed_precision: "bf16", full_bf16: true, full_fp16: false },
  fp16: { mixed_precision: "fp16", full_bf16: false, full_fp16: false },
  "full-fp16": { mixed_precision: "fp16", full_bf16: false, full_fp16: true },
  fp32: { mixed_precision: "no", full_bf16: false, full_fp16: false },
};

/** training_precision for imported precision keys (the inverse of TRAINING_PRECISION). */
export function trainingPrecisionOf(config: TrainingConfig): string | undefined {
  const mixed = config.mixed_precision ?? (config.full_fp16 ? "fp16" : config.full_bf16 ? "bf16" : undefined);
  if (mixed === undefined) return undefined;
  if (mixed === "no") return "fp32";
  if (mixed === "fp16") return config.full_fp16 ? "full-fp16" : "fp16";
  return config.full_bf16 ? "full-bf16" : "bf16";
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
  if (config.training_precision !== undefined) {
    Object.assign(config, TRAINING_PRECISION[config.training_precision] ?? {});
    delete config.training_precision;
  }

  if (!config.enable_preview) for (const key of PREVIEW_KEYS) delete config[key];

  if (config.compile_shapes !== undefined) {
    if (config.compile) Object.assign(config, COMPILE_SHAPES[config.compile_shapes] ?? {});
    delete config.compile_shapes;
  }
  // only compiled blocks have the recompute planner it sets
  if (!config.compile || config.activation_memory_budget === "off") delete config.activation_memory_budget;
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

  // H2D-only block swap needs gradient checkpointing (sd-scripts and musubi refuse it otherwise); the
  // checkpoint offloads are only read with it, and a leftover one would still cancel blocks_to_swap
  if (!config.gradient_checkpointing) {
    delete config.block_swap_h2d_only;
    delete config.cpu_offload_checkpointing;
    delete config.gradient_checkpointing_cpu_offload;
  }

  if (config.blocks_to_swap) {
    config.cpu_offload_checkpointing = false;
    config.unsloth_offload_checkpointing = false;
  }
  if (config.cpu_offload_checkpointing || config.unsloth_offload_checkpointing) {
    delete config.blocks_to_swap;
  }
  // the trainers read the swap mode only when blocks are swapped
  if (!config.blocks_to_swap) {
    delete config.block_swap_h2d_only;
    delete config.use_pinned_memory_for_block_swap;
  }
  // Krea 2: an imported config with both keeps int8 (config.py does the same)
  if (config.convrot_int8) delete config.fp8_base;

  // scheduler settings only for the scheduler that reads them (sd-scripts' warmup_stable_decay would
  // also take a restarts count left over from cosine_with_restarts as its decay shape)
  if (config.lr_scheduler === "constant") delete config.lr_warmup_steps;
  if (config.lr_scheduler !== "cosine_with_restarts") delete config.lr_scheduler_num_cycles;
  if (config.lr_scheduler !== "warmup_stable_decay") delete config.lr_decay_steps;

  // the VRAM-saving options win over activation_memory_budget (config.py ignores it with them)
  if (config.blocks_to_swap || config.cpu_offload_checkpointing || config.unsloth_offload_checkpointing) {
    delete config.activation_memory_budget;
  }

  if (["anima-lora", "sdxl-lora"].includes(config.model_train_type) && config.cache_text_encoder_outputs) {
    config.network_train_unet_only = true;
    config.network_train_text_encoder_only = false;
  }

  // full_precision became full_fp16/full_bf16 above; sd-scripts requires matching mixed_precision
  if (config.full_fp16) config.mixed_precision = "fp16";
  else if (config.full_bf16) config.mixed_precision = "bf16";
  // fp16 accumulation only changes fp16 matrix multiplies (kept as false too, so an import restores it)
  if (config.mixed_precision !== "fp16") delete config.fp16_accumulation;
  // the SDXL VAE gives NaN in fp16 (it encodes the latents in the mixed precision otherwise)
  if (["sdxl-lora", "sdxl-finetune"].includes(config.model_train_type) && config.mixed_precision === "fp16") config.no_half_vae = true;
}

const SECRET_KEYS = ["wandb_api_key", "huggingface_token", "hf_token"];

export function redactConfigForLog(config: TrainingConfig): TrainingConfig {
  const copy = { ...config };
  for (const key of SECRET_KEYS) {
    if (copy[key]) copy[key] = "<redacted>";
  }
  return copy;
}
