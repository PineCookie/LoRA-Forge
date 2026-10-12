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
  "mixed_precision",
  "blocks_to_swap",
  "fp8_base",
  "attn_mode",
  "qwen_image_vae_2d",
  "compile",
]);

/**
 * Whether an option belongs to Essentials: listed above, or a sub-option of one that is (shown
 * only once that option turns it on, e.g. ProdigyPlus' settings, the preview prompts, compile's
 * settings).
 */
export function isEssential(name: string): boolean {
  if (ESSENTIAL_FIELDS.has(name)) return true;
  const controller = controllerOf(name);
  return controller !== undefined && isEssential(controller);
}
