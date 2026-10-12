Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("anima-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("Anima DiT model path"),
        qwen3: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen3-0.6B model path: a safetensors file or a model folder"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen-Image VAE model path"),
        llm_adapter_path: Schema.string().role('filepicker', { type: "model-file" }).description("Optional: a separate LLM adapter weights file. When empty, the adapter is read from the DiT"),
        t5_tokenizer_path: Schema.string().role('filepicker', { type: "folder" }).description("Optional: T5 tokenizer folder. When empty, the configs/t5_old folder bundled with sd-scripts is used"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
    }).description("Model"),

    Schema.object({
        train_data_dir: SHARED_SCHEMAS.RAW.DATASET_SETTINGS.train_data_dir,
        output_name: SHARED_SCHEMAS.RAW.SAVE_FIELDS.output_name,
        output_dir: SHARED_SCHEMAS.RAW.SAVE_FIELDS.output_dir,
        logging_dir: SHARED_SCHEMAS.RAW.LOG_FIELDS.logging_dir,
    }).description("⚡ Quick setup"),

    Schema.object({
        timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid").description("Timestep sampling"),
        show_timesteps: Schema.union(["console", "image"]).description("Output format of the timestep distribution preview. Run it with the “Preview timesteps” button; ignored when training. ⚠ Needs sd-scripts v0.11.1+"),
        show_timesteps_resolution: Schema.string().default("1024").description("Image resolution the preview assumes (H or H,W); used by flux_shift"),
        show_timesteps_offset: Schema.number().step(0.01).default(0.0).description("Timestep offset for the preview only; no effect with sigma/uniform. ⚠ Needs sd-scripts main (> v0.11.1)"),
        weighting_scheme: Schema.union(["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"]).default("uniform").description("Timestep loss weighting scheme"),
        sigmoid_scale: Schema.number().step(0.001).default(1.0).description("Sigmoid scale"),
        discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("Discrete flow shift"),
        qwen3_max_token_length: Schema.number().step(1).default(512).description("Qwen3 max token length"),
        t5_max_token_length: Schema.number().step(1).default(512).description("T5 tokenizer max token length"),
        llm_adapter_lr: Schema.number().step(0.000001).default(0).description("LLM adapter learning rate. Freezing it is the official advice for LoRA training; 0 freezes it"),
        self_attn_lr: Schema.number().step(0.000001).description("Self-attention learning rate. Empty uses the main learning rate; 0 freezes it"),
        cross_attn_lr: Schema.number().step(0.000001).description("Cross-attention learning rate. Empty uses the main learning rate; 0 freezes it"),
        mlp_lr: Schema.number().step(0.000001).description("MLP learning rate. Empty uses the main learning rate; 0 freezes it"),
        mod_lr: Schema.number().step(0.000001).description("AdaLN modulation learning rate. Empty uses the main learning rate; 0 freezes it"),
    }).description("Anima options"),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("1024,1024").description("Training resolution, width x height. Anima supports 512x512 to 1536x1536; both must be multiples of 16."),
            enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
            min_bucket_reso: Schema.number().default(512).description("Minimum ARB bucket resolution"),
            max_bucket_reso: Schema.number().default(2048).description("Maximum ARB bucket side length. For square images keep Anima at 1536 or below; for 2:1 and other wide images raise it to 2048 or more, but VRAM use grows with the total pixel count"),
            bucket_reso_steps: Schema.number().default(16).description("ARB bucket resolution step. Anima: 16 recommended"),
        }, ["train_data_dir"])
    ).description("Dataset"),

    // Saving (output_name / output_dir moved to ⚡ Quick setup)
    Schema.intersect([
        Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.SAVE_FIELDS, {}, ["output_name", "output_dir"])).description("Saving"),
        Schema.union([
            Schema.object({
                save_state: Schema.const(true).required(),
                save_last_n_epochs_state: Schema.number().min(1).description("Keep only the training states of the last n epochs"),
            }),
            Schema.object({})
        ])
    ]),

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).default(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(true).description("Train the DiT/U-Net only. Required when caching text encoder outputs"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("Train the text encoder only"),
    }).description("Training"),

    // Learning rate & optimizer
    SHARED_SCHEMAS.LR_OPTIMIZER,

    // Loss
    SHARED_SCHEMAS.LOSS_SETTINGS_NO_WEIGHTING,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora_anima"]).default("networks.lora_anima").description("Network module to train"),
            network_weights: Schema.string().role('filepicker').description("Continue training from an existing Anima LoRA: enter its path"),
            network_dim: Schema.number().min(1).default(32).description("Network dimension (rank). The official advice is to start LoRA training at rank 32"),
            network_alpha: Schema.number().min(1).default(32).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_dropout: Schema.number().step(0.01).default(0).description("Dropout probability"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("Max-norm regularization. If used, 1 is recommended"),
            network_args_custom: Schema.array(String).role('table').description("Custom network_args, one per line. Example: train_llm_adapter=true"),
            enable_base_weight: Schema.boolean().default(false).description("Enable base weights (difference training)"),
        }).description("Network"),

        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // Preview images
    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("Generate preview images during training"),
        }).description("Preview images"),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                randomly_choice_prompt: Schema.boolean().default(false).description("Use a random caption from the dataset as the preview prompt"),
                prompt_file: Schema.string().role('textarea').description("Preview prompt file. When set, the prompts in the file are used and the options below are ignored."),
                positive_prompts: Schema.string().role('textarea').default("masterpiece, best quality, score_7, safe, 1girl, solo").description("Prompt"),
                negative_prompts: Schema.string().role('textarea').default("worst quality, low quality, score_1, score_2, score_3, artist name").description("Negative Prompt"),
                sample_width: Schema.number().default(1024).description("Preview width"),
                sample_height: Schema.number().default(1024).description("Preview height"),
                sample_cfg: Schema.number().min(1).max(30).default(4.5).description("CFG scale; Anima: 4-5 recommended"),
                sample_seed: Schema.number().default(2333).description("Seed"),
                sample_steps: Schema.number().min(1).max(300).default(40).description("Sampling steps; Anima: 30-50 recommended"),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                sample_every_n_epochs: Schema.number().default(2).description("Generate preview images every N epochs"),
            }),
            Schema.object({}),
        ]),
    ]),

    // Logging (logging_dir moved to ⚡ Quick setup)
    Schema.intersect([
        Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.LOG_FIELDS, {}, ["logging_dir"])).description("Logging"),
        Schema.union([
            Schema.object({
                log_with: Schema.const("wandb").required(),
                wandb_api_key: Schema.string().required().description("wandb API key"),
            }),
            Schema.object({}),
        ]),
    ]),

    // Captions
    // weighted_captions has no effect on Anima (strategy_anima has no weighted tokenization), so the field is hidden
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length", "weighted_captions"])).description("Captions"),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options. Anima does not use clip_skip/v2/v_parameterization, so SHARED_SCHEMAS.OTHER is not reused here.
    Schema.object({
        seed: Schema.number().default(1337).description("Random seed"),
    }).description("Other"),

    // Speed and memory
    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
            mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("Mixed precision; `bf16` works on RTX 30 series and newer. Full precision mode adjusts this automatically"),
            full_precision: Schema.union(["none", "full_fp16", "full_bf16"]).default("full_bf16").description("Full precision mode; sets mixed precision to the matching fp16/bf16 (sd-scripts requires both to agree)"),
            cache_text_encoder_outputs: Schema.boolean().default(true).description("Cache text encoder outputs to save VRAM. Needs shuffle_caption off; turns on network_train_unet_only (the text encoder cannot be trained at the same time)"),
            cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("Cache text encoder outputs to disk"),
            cache_latents: Schema.boolean().default(true).description("Cache image latents (VAE outputs) to save VRAM"),
            cache_latents_to_disk: Schema.boolean().default(true).description("Cache image latents to disk"),
            // --xformers / --sdpa are overridden by --attn_mode, so only attn_mode is exposed
            attn_mode: Schema.union(["torch", "xformers", "flash"]).default("torch").description("Attention implementation. torch = PyTorch SDPA; xformers / flash need their library installed (flash-attn)"),
            split_attn: Schema.boolean().default(false).description("Split attention per sample to save VRAM. Anima has no attention mask, so xformers is not required; cannot be combined with compile_fullgraph"),
            blocks_to_swap: Schema.number().min(0).step(1).description("Number of blocks swapped to the CPU (more saves VRAM but is slower). Cannot be combined with the cpu/unsloth offload options; checking either one clears this field"),
            cpu_offload_checkpointing: Schema.boolean().default(false).description("Offload gradient checkpoints to the CPU: less VRAM but slower. Cannot be combined with blocks_to_swap or unsloth offload"),
            unsloth_offload_checkpointing: Schema.boolean().default(false).description("Offload activations to CPU RAM asynchronously (faster than cpu offload). Cannot be combined with the two options above"),
            vae_chunk_size: Schema.number().min(2).step(2).description("Spatial chunk size for VAE encoding/decoding (an even number). Empty disables chunking; with qwen_image_vae_2d it barely affects peak VRAM"),
            vae_disable_cache: Schema.boolean().default(false).description("Disable the VAE's internal cache to save VRAM. No effect with qwen_image_vae_2d, which has no temporal cache"),
            qwen_image_vae_2d: Schema.boolean().default(false).description("Image-only 2D Qwen-Image VAE (Conv3d→Conv2d): about 2× faster with about 1/3 of the peak VRAM; best combined with cache_latents. ⚠ Needs sd-scripts v0.11.1+"),
            text_encoder_batch_size: Schema.number().min(1).description("Batch size for caching text encoder outputs; empty uses the dataset batch size"),
        }, ["fp8_base", "fp8_base_unet", "no_half_vae", "lowram", "full_fp16", "full_bf16", "xformers", "sdpa"])
    ).description("Speed and memory"),

    // torch.compile (Anima compiles block by block)
    Schema.object({
        compile: Schema.boolean().default(false).description("Speed up Anima with per-block torch.compile (needs Triton); cannot be combined with --torch_compile. The options below only apply when this is on"),
        compile_backend: Schema.union(["inductor", "eager", "aot_eager"]).default("inductor").description("torch.compile backend"),
        compile_mode: Schema.union(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]).default("default").description("Compile mode. max-autotune is the fastest but the first compile is slow"),
        compile_dynamic: Schema.union(["auto", "true", "false"]).default("false").description("Dynamic shape mode. Defaults to false: on torch 2.14, dynamic=true/auto triggers a Dynamo assertion on the first step (PyTorch #193194, fixed in PR #193254, expected in 2.15), so it is off to keep training working. When off, each bucket compiles once; keep compile_cache_size_limit at 32. On Windows, true needs the VS 2022 C++ compiler"),
        compile_fullgraph: Schema.boolean().default(false).description("Full-graph compile mode. Checking it turns off split_attn"),
        compile_cache_size_limit: Schema.number().min(1).default(32).description("torch._dynamo cache size limit; 32 recommended"),
        cuda_allow_tf32: Schema.boolean().default(false).description("Allow TF32 (Ampere+), which can improve performance; works without compile"),
        cuda_cudnn_benchmark: Schema.boolean().default(false).description("Enable cuDNN benchmark, which may improve performance; works without compile"),
    }).description("torch.compile / CUDA performance (⚠ needs sd-scripts v0.11.1+; dynamic shapes default to false, see PyTorch #193194)"),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
