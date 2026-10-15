import type { FormValues } from "./form";

// field -> [controlling field, value(s) for which the field is shown]
const VISIBILITY_RULES: Record<string, [string, string | boolean | string[]]> = {
  prodigy_d0: ["optimizer_type", "Prodigy"],
  prodigy_d_coef: ["optimizer_type", "Prodigy"],
  prodigyplus_d_coef: ["optimizer_type", "prodigyplus.ProdigyPlusScheduleFree"],
  prodigyplus_betas: ["optimizer_type", "prodigyplus.ProdigyPlusScheduleFree"],
  prodigyplus_schedulefree_c: ["optimizer_type", "prodigyplus.ProdigyPlusScheduleFree"],
  prodigyplus_prodigy_steps: ["optimizer_type", "prodigyplus.ProdigyPlusScheduleFree"],
  lycoris_algo: ["network_module", "lycoris.kohya"],
  conv_dim: ["network_module", "lycoris.kohya"],
  conv_alpha: ["network_module", "lycoris.kohya"],
  dropout: ["network_module", "lycoris.kohya"],
  train_norm: ["network_module", "lycoris.kohya"],
  lokr_factor: ["lycoris_algo", "lokr"],
  dylora_unit: ["network_module", "networks.dylora"],
  base_weights: ["enable_base_weight", true],
  base_weights_multiplier: ["enable_base_weight", true],
  down_lr_weight: ["enable_block_weights", true],
  mid_lr_weight: ["enable_block_weights", true],
  up_lr_weight: ["enable_block_weights", true],
  block_lr_zero_threshold: ["enable_block_weights", true],
  randomly_choice_prompt: ["enable_preview", true],
  prompt_file: ["enable_preview", true],
  positive_prompts: ["enable_preview", true],
  negative_prompts: ["enable_preview", true],
  sample_width: ["enable_preview", true],
  sample_height: ["enable_preview", true],
  sample_cfg: ["enable_preview", true],
  sample_seed: ["enable_preview", true],
  sample_steps: ["enable_preview", true],
  sample_sampler: ["enable_preview", true],
  sample_every_n_epochs: ["enable_preview", true],
  sample_prompts: ["enable_preview", true],
  // sd-scripts ignores both with bucket_no_upscale (it sizes each bucket from its image)
  min_bucket_reso: ["bucket_no_upscale", false],
  max_bucket_reso: ["bucket_no_upscale", false],
  huber_schedule: ["loss_type", ["huber", "smooth_l1"]],
  huber_c: ["loss_type", ["huber", "smooth_l1"]],
  v2: ["model_train_type", ["sd-lora", "sd-dreambooth"]],
  v_parameterization: ["model_train_type", ["sd-lora", "sdxl-lora", "sd-dreambooth", "sdxl-finetune"]],
  scale_v_pred_loss_like_noise_pred: ["v_parameterization", true],
  // Anima per-block compile: these only matter while compile is enabled
  compile_backend: ["compile", true],
  compile_mode: ["compile", true],
  compile_shapes: ["compile", true],
  compile_fullgraph: ["compile", true],
  compile_cache_size_limit: ["compile", true],
  activation_memory_budget: ["compile", true],
  // Krea 2 (musubi-tuner)
  convrot_int8_bwd: ["convrot_int8", true],
  turbo_dit: ["enable_preview", true],
  wandb_api_key: ["log_with", "wandb"],
  // SDXL fine-tune: the text encoders are only trained with train_text_encoder
  learning_rate_te1: ["train_text_encoder", true],
  learning_rate_te2: ["train_text_encoder", true],
};

/** The field whose value shows or hides `name`, if any. */
export function controllerOf(name: string): string | undefined {
  return VISIBILITY_RULES[name]?.[0];
}

/** Whether a field is shown for the current form values. Hidden fields are still submitted. */
export function isFieldVisible(name: string, values: FormValues): boolean {
  const rule = VISIBILITY_RULES[name];
  if (!rule) return true;
  const [controller, shownFor] = rule;
  if (!(controller in values)) return false;
  const current = String(values[controller]);
  return (Array.isArray(shownFor) ? shownFor : [shownFor]).map(String).includes(current);
}
