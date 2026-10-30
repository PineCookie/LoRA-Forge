import { controllerOf } from "./visibility";

// The options a trainer page section shows from the start, for every model; names a page does not
// have are ignored. The others are behind the section's "+ more"; the page also keeps every option
// the user edited, and a search looks through all options, so nothing in effect is hidden.

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
  // training (unet_lr / text_encoder_lr stay behind "+ more": empty, they follow learning_rate)
  "max_train_epochs",
  "train_batch_size",
  "optimizer_type",
  "learning_rate",
  "lr_scheduler",
  "lr_warmup_steps",
  "network_module",
  "network_dim",
  "network_alpha",
  "timestep_sampling",
  // preview images (the prompts and sampler settings appear with it, see below)
  "enable_preview",
  // speed and memory
  "training_precision",
  "mixed_precision",
  "blocks_to_swap",
  "fp8_base",
  "convrot_int8",
  "qwen_image_vae_2d",
  "compile",
  // the page only has it with two or more GPUs, and then which one trains matters
  "gpu_ids",
]);

// Sub-options of an essential option that stay behind "+ more": compile is on by default, and its
// settings are for tuning, not for getting started.
const ADVANCED_FIELDS = new Set([
  "compile_backend",
  "compile_mode",
  "compile_shapes",
  "compile_fullgraph",
  "compile_cache_size_limit",
  "activation_memory_budget",
  // timestep_sampling's fine-tuning
  "sigmoid_scale",
  "discrete_flow_shift",
]);

/**
 * Whether an option is shown without "+ more": listed above, or a sub-option of one that is (shown
 * only once that option turns it on, e.g. ProdigyPlus' settings, the preview prompts).
 */
export function isEssential(name: string): boolean {
  if (ESSENTIAL_FIELDS.has(name)) return true;
  if (ADVANCED_FIELDS.has(name)) return false;
  const controller = controllerOf(name);
  return controller !== undefined && isEssential(controller);
}
