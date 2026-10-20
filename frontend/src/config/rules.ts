import type { FormValues } from "./form";
import { optimizerRuleFor } from "./normalize";

// Live coupling between form fields. These mutate the form values in place, like a user edit.

function setField(values: FormValues, name: string, value: string | boolean) {
  if (!(name in values)) return;
  values[name] = typeof values[name] === "boolean" ? Boolean(value) : String(value);
}

function checked(values: FormValues, name: string) {
  return values[name] === true;
}

const SD15_TYPES = ["sd-lora", "sd-dreambooth"];
const SDXL_TYPES = ["sdxl-lora", "sdxl-finetune"];
const FP16_PRECISIONS = ["fp16", "full-fp16"];

function budgetOn(values: FormValues) {
  return !["", "off"].includes(String(values.activation_memory_budget ?? ""));
}

/** Force the values the selected optimizer requires (e.g. learning_rate=1 for DAdaptation). */
export function applyDependentValues(values: FormValues) {
  const rule = optimizerRuleFor(values.optimizer_type);
  if (!rule) return;
  for (const [name, value] of Object.entries(rule.values)) {
    if (name in values) values[name] = String(value);
  }
}

/**
 * Keep combinations that sd-scripts asserts on out of the form. `edited` is the field the user just
 * changed (null for a full pass, e.g. after an import); the full pass afterwards also fixes
 * imported configs.
 */
export function applyExclusiveRules(values: FormValues, edited: string | null) {
  const trainType = String(values.model_train_type ?? "");
  const textEncoderCachingLocksUnet = ["anima-lora", "sdxl-lora"].includes(trainType);

  // sd-scripts' SD1.5 trainers assert a bucket step that is a multiple of 64 (an imported config may
  // carry SDXL's 32)
  if (SD15_TYPES.includes(trainType) && Number(values.bucket_reso_steps) % 64) setField(values, "bucket_reso_steps", "64");

  if (edited === "compile_fullgraph" && checked(values, edited)) setField(values, "split_attn", false);
  if (edited === "split_attn" && checked(values, edited)) setField(values, "compile_fullgraph", false);

  if (edited === "blocks_to_swap" && Number(values.blocks_to_swap) > 0) {
    setField(values, "cpu_offload_checkpointing", false);
    setField(values, "unsloth_offload_checkpointing", false);
  }
  if (edited === "cpu_offload_checkpointing" && checked(values, edited)) {
    setField(values, "unsloth_offload_checkpointing", false);
    setField(values, "blocks_to_swap", "");
  }
  if (edited === "unsloth_offload_checkpointing" && checked(values, edited)) {
    setField(values, "cpu_offload_checkpointing", false);
    setField(values, "blocks_to_swap", "");
  }
  // Krea 2: int8 and fp8 base weights are alternatives
  if (edited === "convrot_int8" && checked(values, edited)) setField(values, "fp8_base", false);
  if (edited === "fp8_base" && checked(values, edited)) setField(values, "convrot_int8", false);
  // Krea 2: one Turbo source for the previews
  if (edited === "turbo_dit" && String(values.turbo_dit ?? "").trim()) setField(values, "turbo_lora", "");
  if (edited === "turbo_lora" && String(values.turbo_lora ?? "").trim()) setField(values, "turbo_dit", "");
  // Krea 2: musubi refuses H2D-only block swap without gradient checkpointing; the checkpoint offloads
  // are only read with it (a leftover one would still cancel blocks_to_swap)
  if (edited === "gradient_checkpointing" && !checked(values, edited)) {
    setField(values, "block_swap_h2d_only", false);
    setField(values, "cpu_offload_checkpointing", false);
    setField(values, "gradient_checkpointing_cpu_offload", false);
  }
  if (edited === "block_swap_h2d_only" && checked(values, edited)) setField(values, "gradient_checkpointing", true);

  // activation_memory_budget spends VRAM for speed, the others save VRAM: the last edited one wins
  if (edited === "activation_memory_budget" && budgetOn(values)) {
    setField(values, "blocks_to_swap", "");
    setField(values, "cpu_offload_checkpointing", false);
    setField(values, "unsloth_offload_checkpointing", false);
  }
  if (
    (edited === "blocks_to_swap" && Number(values.blocks_to_swap) > 0) ||
    (["cpu_offload_checkpointing", "unsloth_offload_checkpointing"].includes(edited ?? "") && checked(values, edited ?? ""))
  ) {
    setField(values, "activation_memory_budget", "off");
  }

  // sd-scripts cannot cache text encoder outputs for captions that change per step: shuffled tags,
  // tag dropout, and (except Anima, whose cache supports it) whole-caption dropout. The last edit wins.
  const captionsVary = (name: string) =>
    (name === "shuffle_caption" && checked(values, name)) ||
    (name === "caption_tag_dropout_rate" && Number(values[name]) > 0) ||
    (name === "caption_dropout_rate" && Number(values[name]) > 0 && trainType !== "anima-lora");
  if (edited && "cache_text_encoder_outputs" in values && captionsVary(edited)) {
    setField(values, "cache_text_encoder_outputs", false);
    setField(values, "cache_text_encoder_outputs_to_disk", false);
  }
  if (edited === "cache_text_encoder_outputs" && checked(values, edited)) {
    setField(values, "shuffle_caption", false);
    setField(values, "caption_tag_dropout_rate", "");
    if (trainType !== "anima-lora") setField(values, "caption_dropout_rate", "");
  }

  // FLUX: Chroma is not guidance-distilled; it trains with guidance 0 and sigmoid sampling and
  // samples with real CFG (sd-scripts' FLUX guide)
  if (edited === "model_type") {
    const chroma = values.model_type === "chroma";
    setField(values, "guidance_scale", chroma ? "0" : "1");
    setField(values, "timestep_sampling", chroma ? "sigmoid" : "flux_shift");
    setField(values, "sample_cfg", chroma ? "4" : "1");
  }

  // only Anima/SDXL assert "unet_only or no TE caching"; FLUX/SD3 cache one encoder while
  // training the other
  if (textEncoderCachingLocksUnet) {
    if (edited === "cache_text_encoder_outputs" && checked(values, edited)) {
      setField(values, "network_train_text_encoder_only", false);
      setField(values, "network_train_unet_only", true);
    }
    if (edited === "network_train_text_encoder_only" && checked(values, edited)) {
      setField(values, "cache_text_encoder_outputs", false);
      setField(values, "cache_text_encoder_outputs_to_disk", false);
      setField(values, "network_train_unet_only", false);
    }
  }

  // full pass, same priority as normalizeConstraintCombinations(): an explicit block count wins
  // over the offload flags
  if (checked(values, "compile_fullgraph")) setField(values, "split_attn", false);
  if ((Number(values.blocks_to_swap) || 0) > 0) {
    setField(values, "cpu_offload_checkpointing", false);
    setField(values, "unsloth_offload_checkpointing", false);
  } else if (checked(values, "cpu_offload_checkpointing") || checked(values, "unsloth_offload_checkpointing")) {
    setField(values, "blocks_to_swap", "");
  }
  // an imported config with both: the VRAM-saving options win, as in config.py
  if ((Number(values.blocks_to_swap) || 0) > 0 || checked(values, "cpu_offload_checkpointing") || checked(values, "unsloth_offload_checkpointing")) {
    setField(values, "activation_memory_budget", "off");
  }
  if (textEncoderCachingLocksUnet && checked(values, "cache_text_encoder_outputs")) {
    setField(values, "network_train_text_encoder_only", false);
    setField(values, "network_train_unet_only", true);
  }

  // an older config's 255 (sd-scripts only takes 150 / 225, and 255 trained as 225)
  if ("max_token_length" in values && !["75", "150", "225"].includes(String(values.max_token_length))) setField(values, "max_token_length", "225");

  // sd-scripts requires mixed_precision to match full_fp16 / full_bf16
  if (values.full_precision === "full_fp16") setField(values, "mixed_precision", "fp16");
  else if (values.full_precision === "full_bf16") setField(values, "mixed_precision", "bf16");
  // the SDXL VAE gives NaN in fp16: with the fp16 training precisions it stays in fp32
  if (SDXL_TYPES.includes(trainType) && FP16_PRECISIONS.includes(String(values.training_precision))) setField(values, "no_half_vae", true);
}
