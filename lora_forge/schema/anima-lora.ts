Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("anima-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("Anima DiT model path"),
        qwen3: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen3-0.6B model path: a safetensors file or a model folder"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen-Image VAE model path"),
        llm_adapter_path: Schema.string().role('filepicker', { type: "model-file" }).description("Optional: a separate LLM adapter weights file. When empty, the adapter is read from the DiT"),
        t5_tokenizer_path: Schema.string().role('filepicker', { type: "folder" }).description("Optional: T5 tokenizer folder. When empty, the configs/t5_old folder bundled with sd-scripts is used"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
    }),

    Schema.object({
        timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid").description("Timestep sampling"),
        show_timesteps_resolution: Schema.string().default("1024").description("Image size the timestep preview assumes, as `H` or `H,W`; only flux_shift uses it"),
        show_timesteps_offset: Schema.number().step(0.01).default(0.0).description("Timestep offset for the preview only; no effect with sigma/uniform. ⚠ Needs sd-scripts main (> v0.11.1)"),
        weighting_scheme: Schema.union(["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"]).default("uniform").description("Timestep loss weighting scheme"),
        sigmoid_scale: Schema.number().step(0.001).default(1.0).description("Sigmoid scale"),
        discrete_flow_shift: Schema.number().min(0).step(0.001).default(1.0).description("Discrete flow shift"),
        qwen3_max_token_length: Schema.number().min(1).step(1).default(512).description("Qwen3 max token length"),
        t5_max_token_length: Schema.number().min(1).step(1).default(512).description("T5 tokenizer max token length"),
        // sd-scripts' self_attn_lr / cross_attn_lr / mlp_lr / mod_lr / llm_adapter_lr only apply to full
        // fine-tuning (anima_train.py); for a LoRA, network_reg_lrs sets learning rates per module
    }),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("1024,1024").description("Training resolution, width x height. Anima supports 512x512 to 1536x1536; both must be multiples of 16."),
            enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
            min_bucket_reso: Schema.number().min(16).step(16).default(512).description("Shortest bucket side. Only used with bucket_no_upscale off, when images are fitted to a fixed set of buckets of about the training resolution's pixel count; images more stretched than min_bucket_reso and max_bucket_reso allow are cropped"),
            max_bucket_reso: Schema.number().min(16).step(16).default(2048).description("Longest bucket side. Only used with bucket_no_upscale off; buckets keep about the training resolution's pixel count, so this only limits very wide or tall images"),
            bucket_reso_steps: Schema.number().min(16).step(16).default(16).description("ARB bucket resolution step: images are fitted to sizes in steps of this many pixels, so smaller steps crop less. Anima accepts multiples of 16; with compile_shapes = bounded more bucket sizes cost no extra compiles"),
        })
    ),

    // Saving
    Schema.intersect([
        Schema.object(SHARED_SCHEMAS.RAW.SAVE_FIELDS),
        Schema.union([
            Schema.object({
                save_state: Schema.const(true).required(),
                save_last_n_epochs_state: Schema.number().min(1).step(1).description("Keep only the training states of the last n epochs"),
            }),
            Schema.object({})
        ])
    ]),

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).step(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing: saves VRAM by running each block's forward pass again during the backward pass, which is slower. With compile on, activation_memory_budget (below) can replace it to use spare VRAM for speed"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).default(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(true).description("Train the DiT/U-Net only. Required when caching text encoder outputs"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("Train the text encoder only"),
    }),

    // Learning rate & optimizer
    SHARED_SCHEMAS.LR_OPTIMIZER,

    // Loss
    SHARED_SCHEMAS.LOSS_SETTINGS_NO_WEIGHTING,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora_anima"]).default("networks.lora_anima").description("Network module to train"),
            network_weights: Schema.string().role('filepicker').description("Continue training from an existing Anima LoRA: enter its path"),
            network_dim: Schema.number().min(1).step(1).default(32).description("Network dimension (rank). The official advice is to start LoRA training at rank 32"),
            network_alpha: Schema.number().min(1).default(32).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description("Dropout probability"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("Max-norm regularization. If used, 1 is recommended"),
            network_args_custom: Schema.array(String).role('table').description("Custom network_args, one per line. Example: train_llm_adapter=true"),
            enable_base_weight: Schema.boolean().default(false).description("Enable base weights (difference training)"),
        }),

        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // Preview images
    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("Generate preview images during training"),
        }),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                randomly_choice_prompt: Schema.boolean().default(false).description("Use a random caption from the dataset as the preview prompt"),
                prompt_file: Schema.string().role('textarea').description("Preview prompt file. When set, the prompts in the file are used and the options below are ignored."),
                positive_prompts: Schema.string().role('textarea').default("masterpiece, best quality, score_7, safe, 1girl, solo").description("Prompt"),
                negative_prompts: Schema.string().role('textarea').default("worst quality, low quality, score_1, score_2, score_3, artist name").description("Negative Prompt"),
                sample_width: Schema.number().min(64).step(16).default(1024).description("Preview width"),
                sample_height: Schema.number().min(64).step(16).default(1024).description("Preview height"),
                sample_cfg: Schema.number().min(1).max(30).default(4.5).description("CFG scale; Anima: 4-5 recommended"),
                sample_seed: Schema.number().min(0).step(1).default(2333).description("Seed"),
                sample_steps: Schema.number().min(1).max(300).default(40).description("Sampling steps; Anima: 30-50 recommended"),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                sample_every_n_epochs: Schema.number().min(1).step(1).default(2).description("Generate preview images every N epochs"),
            }),
            Schema.object({}),
        ]),
    ]),

    // Logging
    Schema.intersect([
        Schema.object(SHARED_SCHEMAS.RAW.LOG_FIELDS),
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
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length", "weighted_captions"])),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options. Anima does not use clip_skip/v2/v_parameterization, so SHARED_SCHEMAS.OTHER is not reused here.
    Schema.object({
        seed: Schema.number().min(0).step(1).default(1337).description("Random seed"),
    }),

    // Speed and memory
    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
            // one choice in place of mixed_precision + full_fp16/full_bf16; normalize.ts expands it
            training_precision: Schema.union(["bf16", "full-bf16", "fp16", "full-fp16", "fp32"]).default("full-bf16").description("Precision of the training math and of the LoRA weights while they train (the saved file uses save_precision).<br>`bf16`: math in bf16, LoRA weights in fp32.<br>`full-bf16`: math and LoRA weights in bf16 (sd-scripts' full_bf16). The standard, for RTX 30 series and newer.<br>`fp16`: math in fp16, LoRA weights in fp32. For GPUs without bf16, or with fp16_accumulation for speed; a little more VRAM than the bf16 choices.<br>`full-fp16`: math and LoRA weights in fp16 (sd-scripts' full_fp16), a little less VRAM than `fp16`. Only with AdamW8bit, PagedAdamW8bit, Lion8bit, PagedLion8bit or AdaFactor: other optimizers stop learning or turn the weights into NaN.<br>`fp32`: everything in fp32. Slowest, for debugging"),
            // LoRA-Forge's own option (run_entry.py), shown for the fp16 choices
            fp16_accumulation: Schema.boolean().default(true).description("Let fp16 matrix multiplies add up in fp16 too: much faster on GeForce RTX cards (no gain on data-center or RTX PRO cards), still about as precise as bf16"),
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
            qwen_image_vae_2d: Schema.boolean().default(true).description("Image-only 2D Qwen-Image VAE (Conv3d→Conv2d): about 2× faster with about 1/3 of the peak VRAM; best combined with cache_latents. ⚠ Needs sd-scripts v0.11.1+"),
            text_encoder_batch_size: Schema.number().min(1).step(1).description("Batch size for caching text encoder outputs; empty uses the dataset batch size"),
        }, ["mixed_precision", "fp8_base", "fp8_base_unet", "no_half_vae", "lowram", "full_fp16", "full_bf16", "xformers", "sdpa"])
    ),

    // torch.compile (Anima compiles block by block)
    Schema.object({
        compile: Schema.boolean().default(true).description("Speed up Anima with per-block torch.compile (needs Triton, which LoRA Forge installs). The first steps of a run are slower while it compiles; turn it off if compiling fails. Cannot be combined with --torch_compile. The options below only apply when this is on"),
        compile_backend: Schema.union(["inductor", "eager", "aot_eager"]).default("inductor").description("torch.compile backend"),
        compile_mode: Schema.union(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]).default("default").description("Compile mode. max-autotune is the fastest but the first compile is slow"),
        // LoRA-Forge's own option: normalize.ts turns it into sd-scripts' compile_dynamic (+ compile_dynamic_seq for bounded)
        compile_shapes: Schema.union(["bounded", "static", "auto", "dynamic"]).default("bounded").description("How different bucket and batch sizes are compiled.<br>`bounded` (recommended): one graph for every bucket up to 2048x2048 px and every batch size from 2 to train_batch_size; anything outside gets its own graph and the Runs page says so. With blocks_to_swap it works like `static`.<br>`static`: one graph per bucket size and batch size, up to compile_cache_size_limit.<br>`auto` / `dynamic`: torch.compile's own dynamic shapes (sd-scripts' compile_dynamic). On torch 2.14 they fail at the first step (PyTorch #193194, fixed in 2.15); on Windows `dynamic` needs the VS 2022 C++ compiler"),
        compile_fullgraph: Schema.boolean().default(false).description("Full-graph compile mode. Checking it turns off split_attn"),
        compile_cache_size_limit: Schema.number().min(1).step(1).default(64).description("Most graphs torch.compile builds per block before it stops compiling new shapes (they then run uncompiled, slower). With compile_shapes = static: one per bucket, one more per bucket whose last batch is smaller, and more for sample images. The Runs page shows the count and warns when the limit is reached"),
        // LoRA-Forge's own option: run_entry.py sets the budget (estimates it for auto) and turns gradient checkpointing off
        activation_memory_budget: Schema.union(["off", "auto", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4", "0.5", "0.6", "0.8", "1"]).default("off").description("A faster replacement for gradient checkpointing that uses spare VRAM (needs compile): each compiled block keeps this fraction of its activations for the backward pass and recomputes only the rest.<br>`off`: gradient_checkpointing works as set.<br>`auto`: when training starts, picks the largest fraction that fits in the free VRAM and turns gradient checkpointing off; with too little free VRAM, gradient checkpointing stays on.<br>A fraction: used as is, with gradient checkpointing off. Higher is faster but needs more VRAM; if a run fails at its first step for lack of VRAM, pick a lower one.<br>The saved config keeps your gradient_checkpointing setting. Not used with blocks_to_swap or the offload options"),
        cuda_allow_tf32: Schema.boolean().default(false).description("Allow TF32 (Ampere+), which can improve performance; works without compile"),
        cuda_cudnn_benchmark: Schema.boolean().default(false).description("Enable cuDNN benchmark, which may improve performance; works without compile"),
    }),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
