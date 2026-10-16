import { controllerOf } from "./visibility";

// The options shown in the "Essentials" view, for every model; names a page does not have are
// ignored. The trainer page also keeps every option the user edited, and a search looks through
// all options, so nothing in effect is hidden.

const ESSENTIAL_FIELDS = new Set([
  // model files
  "pretrained_model_name_or_path",
  "model_type",
  "qwen3",
  "vae",
  "text_encoder",
  "ae",
  "clip_l",
  "clip_g",
  "t5xxl",
  "v_parameterization",
  // data and output
  "train_data_dir",
  "resolution",
  "output_name",
  "output_dir",
  "save_every_n_epochs",
  // training
  "max_train_epochs",
  "train_batch_size",
  "seed",
  "optimizer_type",
  "learning_rate",
  "unet_lr",
  "text_encoder_lr",
  "lr_scheduler",
  "lr_warmup_steps",
  "network_module",
  "network_dim",
  "network_alpha",
  "timestep_sampling",
  // preview images (the prompts and sampler settings appear with it, see below)
  "enable_preview",
  // captions
  "shuffle_caption",
  "keep_tokens",
  // speed and memory
  "training_precision",
  "mixed_precision",
  "blocks_to_swap",
  "fp8_base",
  "convrot_int8",
  "attn_mode",
  "qwen_image_vae_2d",
  "compile",
]);

// Sub-options of an essential option that stay under "All options": compile is on by default,
// and its settings are for tuning, not for getting started.
const ADVANCED_FIELDS = new Set([
  "compile_backend",
  "compile_mode",
  "compile_shapes",
  "compile_fullgraph",
  "compile_cache_size_limit",
  "activation_memory_budget",
]);

/**
 * Whether an option belongs to Essentials: listed above, or a sub-option of one that is (shown
 * only once that option turns it on, e.g. ProdigyPlus' settings, the preview prompts).
 */
export function isEssential(name: string): boolean {
  if (ESSENTIAL_FIELDS.has(name)) return true;
  if (ADVANCED_FIELDS.has(name)) return false;
  const controller = controllerOf(name);
  return controller !== undefined && isEssential(controller);
}
