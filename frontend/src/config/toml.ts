import { t } from "../i18n";
import type { TrainingConfig } from "./normalize";

// Minimal TOML for the flat training configs this UI exports and imports (no tables). Section
// comments are English in every UI language, so an exported file does not depend on it.

const TOML_KEY_ORDER: { section: string; keys: string[] }[] = [
  { section: "Model", keys: ["pretrained_model_name_or_path", "qwen3", "vae", "llm_adapter_path", "t5_tokenizer_path", "resume"] },
  {
    section: "Dataset",
    keys: [
      "train_data_dir", "reg_data_dir", "prior_loss_weight", "resolution", "enable_bucket", "min_bucket_reso", "max_bucket_reso",
      "bucket_reso_steps", "bucket_no_upscale", "cache_latents", "cache_latents_to_disk", "caption_extension", "shuffle_caption",
      "keep_tokens", "keep_tokens_separator", "max_token_length", "caption_dropout_rate", "caption_dropout_every_n_epochs",
      "caption_tag_dropout_rate", "weighted_captions", "persistent_data_loader_workers",
    ],
  },
  {
    section: "Network",
    keys: [
      "network_module", "network_weights", "network_dim", "network_alpha", "network_dropout", "scale_weight_norms", "enable_base_weight",
      "base_weights", "base_weights_multiplier", "network_train_unet_only", "network_train_text_encoder_only",
    ],
  },
  {
    section: "Training",
    keys: [
      "train_batch_size", "max_train_epochs", "gradient_checkpointing", "gradient_accumulation_steps", "learning_rate", "unet_lr",
      "text_encoder_lr", "optimizer_type", "lr_scheduler", "lr_warmup_steps", "lr_restart_cycles", "lr_scheduler_num_cycles", "loss_type",
      "optimizer_args",
    ],
  },
  {
    section: "Anima",
    keys: [
      "timestep_sampling", "weighting_scheme", "sigmoid_scale", "discrete_flow_shift", "logit_mean", "logit_std", "mode_scale",
      "qwen3_max_token_length", "t5_max_token_length", "llm_adapter_lr", "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr",
    ],
  },
  {
    section: "Precision and speed",
    keys: [
      "mixed_precision", "full_precision", "xformers", "sdpa", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
      "attn_mode", "split_attn", "blocks_to_swap", "cpu_offload_checkpointing", "unsloth_offload_checkpointing", "vae_chunk_size",
      "vae_disable_cache", "qwen_image_vae_2d", "text_encoder_batch_size", "vae_batch_size",
    ],
  },
  {
    section: "torch.compile",
    keys: ["compile", "compile_backend", "compile_mode", "compile_dynamic", "compile_dynamic_seq", "compile_fullgraph", "compile_cache_size_limit", "cuda_allow_tf32", "cuda_cudnn_benchmark"],
  },
  {
    section: "Saving",
    keys: [
      "output_name", "output_dir", "save_model_as", "save_precision", "save_every_n_epochs", "save_state", "save_last_n_epochs",
      "save_n_epoch_ratio", "log_with", "logging_dir",
    ],
  },
  {
    section: "Preview images",
    keys: [
      "enable_preview", "positive_prompts", "negative_prompts", "sample_width", "sample_height", "sample_cfg", "sample_seed", "sample_steps",
      "sample_sampler", "sample_every_n_epochs", "randomly_choice_prompt", "prompt_file",
    ],
  },
  {
    section: "Data augmentation",
    keys: ["color_aug", "flip_aug", "random_crop", "noise_offset", "min_snr_gamma", "ip_noise_gamma", "multires_noise_iterations", "multires_noise_discount"],
  },
  { section: "Other", keys: ["seed", "network_args"] },
];

const KEY_ORDER = new Map(
  TOML_KEY_ORDER.flatMap(({ section, keys }) => keys.map((key) => [key, section] as const)).map(([key, section], order) => [key, { section, order }]),
);

function tomlValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(tomlValue).join(", ")}]`;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return JSON.stringify(String(value));
}

/** Serialize a config, grouped into commented sections in a stable key order. */
export function toToml(config: TrainingConfig): string {
  const entries = Object.entries(config);
  const known = entries.filter(([key]) => KEY_ORDER.has(key)).sort(([a], [b]) => KEY_ORDER.get(a)!.order - KEY_ORDER.get(b)!.order);
  const unknown = entries.filter(([key]) => !KEY_ORDER.has(key));

  const lines: string[] = [];
  let currentSection: string | null = null;
  for (const [key, value] of known) {
    const { section } = KEY_ORDER.get(key)!;
    if (section !== currentSection) {
      currentSection = section;
      lines.push(`# --- ${section} ---`);
    }
    lines.push(`${key} = ${tomlValue(value)}`);
  }
  if (unknown.length) {
    lines.push("# --- Other ---");
    for (const [key, value] of unknown) lines.push(`${key} = ${tomlValue(value)}`);
  }
  return `${lines.join("\n")}\n`;
}

function indexOutsideQuotes(line: string, target: string) {
  let quote: string | null = null;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if ((char === '"' || char === "'") && line[index - 1] !== "\\") {
      quote = quote === char ? null : quote || char;
    }
    if (char === target && !quote) return index;
  }
  return -1;
}

function parseTomlString(value: string): string {
  if (value.startsWith("'")) return value.slice(1, -1);
  try {
    return JSON.parse(value);
  } catch {
    return value.slice(1, -1);
  }
}

function parseTomlArray(source: string): unknown[] {
  const values: unknown[] = [];
  let quote: string | null = null;
  let current = "";
  let depth = 0;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if ((char === '"' || char === "'") && source[index - 1] !== "\\") quote = quote === char ? null : quote || char;
    if (!quote && char === "[") depth += 1;
    if (!quote && char === "]") depth -= 1;
    if (char === "," && !quote && depth === 0) {
      if (current.trim()) values.push(parseTomlValue(current.trim()));
      current = "";
    } else {
      current += char;
    }
  }
  if (current.trim()) values.push(parseTomlValue(current.trim()));
  return values;
}

function parseTomlValue(value: string): unknown {
  if (value === "true") return true;
  if (value === "false") return false;
  if (/^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) return Number(value);
  if (value.startsWith("[") && value.endsWith("]")) return parseTomlArray(value.slice(1, -1));
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) return parseTomlString(value);
  return value;
}

/** Parse a flat training-config TOML; tables are rejected. */
export function fromToml(text: string): TrainingConfig {
  const result: TrainingConfig = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const commentAt = indexOutsideQuotes(rawLine, "#");
    const line = (commentAt >= 0 ? rawLine.slice(0, commentAt) : rawLine).trim();
    if (!line || line.startsWith("[[")) continue;
    if (line.startsWith("[")) throw new Error(t("trainer.tomlTables"));
    const separator = indexOutsideQuotes(line, "=");
    if (separator < 1) continue;
    result[line.slice(0, separator).trim()] = parseTomlValue(line.slice(separator + 1).trim());
  }
  return result;
}
