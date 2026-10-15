// The sections of every trainer page, in page order, and the options in each, in order. The schema
// files only define the options; this list decides where they appear, so every model page uses
// the same sections. A page shows the sections and options it has. Section titles are English
// and translated like schema text (lora_forge/schema/locales). tests/ui.test.ts fails while an
// option of any page is missing here.

export type Tone = "general" | "model" | "data" | "save" | "optimize" | "network" | "observe" | "performance";

/** The broader sections the Essentials view merges these into. */
export type MergedSectionId = "model" | "data" | "training" | "network" | "preview" | "speed";

export interface SectionLayout {
  title: string;
  tone: Tone;
  merged: MergedSectionId;
  fields: string[];
}

export const SECTION_LAYOUT: SectionLayout[] = [
  {
    title: "Model",
    tone: "model",
    merged: "model",
    fields: [
      "model_train_type", "model_type", "pretrained_model_name_or_path", "qwen3", "vae", "text_encoder", "turbo_dit", "turbo_lora", "ae", "clip_l",
      "clip_g", "t5xxl", "llm_adapter_path", "t5_tokenizer_path", "v2", "v_parameterization", "scale_v_pred_loss_like_noise_pred",
      "clip_skip", "apply_t5_attn_mask",
    ],
  },
  {
    title: "Dataset",
    tone: "data",
    merged: "data",
    fields: [
      "train_data_dir", "reg_data_dir", "prior_loss_weight", "resolution", "enable_bucket", "bucket_reso_steps", "bucket_no_upscale",
      "min_bucket_reso", "max_bucket_reso",
    ],
  },
  {
    title: "Captions",
    tone: "data",
    merged: "data",
    fields: [
      "caption_extension", "shuffle_caption", "keep_tokens", "keep_tokens_separator", "weighted_captions", "caption_dropout_rate",
      "caption_dropout_every_n_epochs", "caption_tag_dropout_rate", "max_token_length", "qwen3_max_token_length",
      "t5_max_token_length", "t5xxl_max_token_length", "no_token_padding",
    ],
  },
  { title: "Data augmentation", tone: "data", merged: "data", fields: ["color_aug", "flip_aug", "random_crop"] },
  {
    title: "Network",
    tone: "network",
    merged: "network",
    fields: [
      "network_module", "network_weights", "network_dim", "network_alpha", "network_dropout", "scale_weight_norms",
      // LyCORIS / DyLoRA settings, shown for their module
      "lycoris_algo", "conv_dim", "conv_alpha", "dropout", "train_norm", "lokr_factor", "dylora_unit",
      "network_train_unet_only", "network_train_text_encoder_only", "train_t5xxl", "network_reg_dims", "network_reg_lrs",
      "network_args_custom",
      "enable_block_weights", "down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold",
      "enable_base_weight", "base_weights", "base_weights_multiplier",
    ],
  },
  {
    title: "Training",
    tone: "optimize",
    merged: "training",
    fields: ["max_train_epochs", "train_batch_size", "gradient_accumulation_steps", "seed", "stop_text_encoder_training"],
  },
  {
    title: "Learning rate and optimizer",
    tone: "optimize",
    merged: "training",
    fields: [
      "optimizer_type", "learning_rate", "unet_lr", "text_encoder_lr", "learning_rate_te", "train_text_encoder", "learning_rate_te1",
      "learning_rate_te2", "lr_scheduler",
      "lr_scheduler_num_cycles", "lr_warmup_steps", "max_grad_norm",
      // the optimizer's own settings, shown for it
      "prodigy_d0", "prodigy_d_coef", "prodigyplus_d_coef", "prodigyplus_betas", "prodigyplus_schedulefree_c",
      "prodigyplus_prodigy_steps", "optimizer_args_custom",
    ],
  },
  {
    title: "Timesteps and loss",
    tone: "optimize",
    merged: "training",
    fields: [
      "timestep_sampling", "sigmoid_scale", "discrete_flow_shift", "model_prediction_type", "guidance_scale", "weighting_scheme",
      "min_timestep", "max_timestep", "loss_type", "huber_schedule", "huber_c", "min_snr_gamma", "debiased_estimation_loss",
      "show_timesteps_resolution", "show_timesteps_offset",
    ],
  },
  { title: "Noise", tone: "optimize", merged: "training", fields: ["noise_offset", "multires_noise_iterations", "multires_noise_discount"] },
  {
    title: "Saving",
    tone: "save",
    merged: "data",
    fields: ["output_name", "output_dir", "save_model_as", "save_precision", "save_every_n_epochs", "save_state", "save_last_n_epochs_state", "resume"],
  },
  {
    title: "Preview images",
    tone: "observe",
    merged: "preview",
    fields: [
      "enable_preview", "randomly_choice_prompt", "prompt_file", "sample_prompts", "positive_prompts", "negative_prompts", "sample_width",
      "sample_height", "sample_cfg", "sample_steps", "sample_seed", "sample_sampler", "sample_every_n_epochs",
    ],
  },
  { title: "Logging", tone: "observe", merged: "data", fields: ["log_with", "wandb_api_key", "logging_dir", "log_prefix", "log_tracker_name"] },
  {
    title: "Precision",
    tone: "performance",
    merged: "speed",
    fields: ["mixed_precision", "full_precision", "convrot_int8", "convrot_int8_bwd", "fp8_base", "fp8_base_unet", "no_half_vae"],
  },
  {
    title: "Caching and VAE",
    tone: "performance",
    merged: "speed",
    fields: [
      "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
      "text_encoder_batch_size", "vae_batch_size", "qwen_image_vae_2d", "vae_chunk_size", "vae_disable_cache",
    ],
  },
  {
    title: "Memory",
    tone: "performance",
    merged: "speed",
    fields: [
      "gradient_checkpointing", "activation_memory_budget", "gradient_checkpointing_cpu_offload", "blocks_to_swap",
      "use_pinned_memory_for_block_swap", "block_swap_h2d_only", "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
      "split_attn", "lowram",
    ],
  },
  {
    title: "Speed",
    tone: "performance",
    merged: "speed",
    fields: ["attn_mode", "xformers", "sdpa", "max_data_loader_n_workers", "persistent_data_loader_workers", "cuda_allow_tf32", "cuda_cudnn_benchmark"],
  },
  {
    title: "torch.compile",
    tone: "performance",
    merged: "speed",
    fields: ["compile", "compile_shapes", "compile_mode", "compile_backend", "compile_fullgraph", "compile_cache_size_limit"],
  },
  { title: "Multiple GPUs", tone: "performance", merged: "speed", fields: ["gpu_ids", "ddp_timeout", "ddp_gradient_as_bucket_view"] },
];

/** Where an option without a place in the layout goes (a test keeps this empty). */
export const FALLBACK_SECTION: SectionLayout = { title: "Other", tone: "general", merged: "training", fields: [] };

const BY_TITLE = new Map(SECTION_LAYOUT.map((section) => [section.title, section]));

export const sectionByTitle = (title: string) => BY_TITLE.get(title) ?? FALLBACK_SECTION;
