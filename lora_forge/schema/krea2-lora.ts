// Krea 2 LoRA with kohya-ss/musubi-tuner (krea2_train_network.py). Option names are musubi-tuner's;
// lora_forge/training/krea2.py turns the dataset, attention and quantization fields into its arguments.
Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("krea2-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("Krea 2 RAW DiT file. Train on RAW, not Turbo. The ComfyUI `krea2_raw_int8_convrot` file trains as it is with convrot_int8; `krea2_raw_bf16` is needed for fp8_base or bf16"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen-Image VAE model path"),
        text_encoder: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen3-VL-4B-Instruct as one safetensors file (ComfyUI's `qwen3vl_4b_bf16`), used to cache the captions before training and for preview images"),
        turbo_dit: Schema.string().role('filepicker', { type: "model-file" }).description("Optional: the Krea 2 Turbo DiT file, used only for preview images (few steps, no CFG). Not with convrot_int8 yet; without it previews come from RAW"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
    }),

    Schema.object({
        timestep_sampling: Schema.union(["krea2_shift", "shift", "sigma", "uniform", "sigmoid", "flux_shift"]).default("krea2_shift").description("Timestep sampling. `krea2_shift` shifts each sample like Krea 2 does at inference for its resolution (256-1280 px); `shift` uses one discrete_flow_shift for all"),
        discrete_flow_shift: Schema.number().min(0).step(0.001).default(2.5).description("Discrete flow shift for `shift`: 2.5 matches Krea 2 at 1024x1024 (about 1.6 at 256 px, 3.2 at 1280 px)"),
        sigmoid_scale: Schema.number().step(0.001).default(1.0).description("Sigmoid scale"),
        weighting_scheme: Schema.union(["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"]).default("none").description("Timestep loss weighting scheme"),
        min_timestep: Schema.number().min(0).max(999).step(1).description("Lowest timestep to train on (0-999); empty = 0"),
        max_timestep: Schema.number().min(1).max(1000).step(1).description("Highest timestep to train on (1-1000); empty = 1000"),
    }),

    Schema.object({
        train_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).default("./train/").description("Training dataset folder"),
        resolution: Schema.string().default("1024,1024").description("Training resolution, width x height. Krea 2 trains at about 1024x1024; multiples of 16"),
        enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
        bucket_no_upscale: Schema.boolean().default(true).description("Give each image a bucket of its own aspect ratio: larger images are scaled down to the training resolution's pixel count, smaller ones keep their size"),
        caption_extension: Schema.string().default(".txt").description("Caption file extension"),
    }),

    // Saving
    Schema.intersect([
        Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.SAVE_FIELDS, {
            save_precision: Schema.union(["bf16", "fp16", "float"]).default("bf16").description("Save precision"),
        }, ["save_model_as"])),
        Schema.union([
            Schema.object({
                save_state: Schema.const(true).required(),
                save_last_n_epochs_state: Schema.number().min(1).step(1).description("Keep only the training states of the last n epochs"),
            }),
            Schema.object({})
        ])
    ]),

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(16).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).step(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).default(1).description("Gradient accumulation steps"),
    }),

    // Learning rate & optimizer: musubi-tuner knows AdamW, AdamW8bit and AdaFactor by name and any other
    // optimizer by its import path; the GUI's ProdigyPlus settings become optimizer_args as on the other pages
    Schema.intersect([
        Schema.object({
            optimizer_type: Schema.union(["AdamW8bit", "AdamW", "AdaFactor", "prodigyplus.ProdigyPlusScheduleFree"]).default("AdamW8bit").description("Optimizer"),
            learning_rate: Schema.number().min(0).default("1e-4").description("Learning rate; 1e-4 is the Krea 2 authors' starting point for rank 32"),
            lr_scheduler: Schema.union(["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"]).default("constant").description("Learning rate scheduler"),
            lr_warmup_steps: Schema.number().min(0).step(1).default(0).description('Learning rate warmup steps'),
            max_grad_norm: Schema.number().min(0).step(0.1).default(1.0).description("Max gradient norm for clipping, 0 = no clipping. sd-scripts default 1.0; 0 is recommended for adaptive optimizers such as Prodigy / ProdigyPlus (set automatically when you switch optimizers)"),
        }),
        Schema.union([
            Schema.object({
                lr_scheduler: Schema.const('cosine_with_restarts'),
                lr_scheduler_num_cycles: Schema.number().min(1).step(1).default(1).description('Number of restarts'),
            }),
            Schema.object({}),
        ]),
        Schema.union([
            Schema.object({
                optimizer_type: Schema.const('prodigyplus.ProdigyPlusScheduleFree').required(),
                prodigyplus_d_coef: Schema.number().min(0).step(0.1).default("1.0").description("Prodigy Plus d_coef. Official default 1.0; try 2 or higher to help the LR grow"),
                prodigyplus_betas: Schema.string().default("(0.95, 0.99)").description("Prodigy Plus betas as a Python tuple, e.g. (0.95, 0.99)"),
                prodigyplus_schedulefree_c: Schema.number().min(0).default("20").description("Prodigy Plus schedulefree_c. This UI defaults to 20 (official default 0)"),
                prodigyplus_prodigy_steps: Schema.number().min(0).step(1).default("0").description("Prodigy Plus prodigy_steps. Official default 0; on older versions, 5-25% of the total steps"),
            }),
            Schema.object({}),
        ]),
        Schema.object({
            optimizer_args_custom: Schema.array(String).role('table').description('Custom optimizer_args, one per line'),
        })
    ]),

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora_krea2"]).default("networks.lora_krea2").description("Network module to train"),
            network_weights: Schema.string().role('filepicker', { type: "model-file" }).description("Continue training from an existing Krea 2 LoRA: enter its path"),
            network_dim: Schema.number().min(1).step(1).default(32).description("Network dimension (rank). The official advice is to start LoRA training at rank 32"),
            network_alpha: Schema.number().min(1).default(32).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description("Dropout probability"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("Max-norm regularization. If used, 1 is recommended"),
            network_args_custom: Schema.array(String).role('table').description("Custom network_args, one per line. By default the LoRA covers all 264 Linear layers; see musubi-tuner's docs/krea2.md for attention-only targets"),
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
                prompt_file: Schema.string().role('textarea').description("Preview prompt file. When set, the prompts in the file are used and the options below are ignored."),
                positive_prompts: Schema.string().role('textarea').default("a photo of a cat sitting on a windowsill, soft morning light").description("Prompt"),
                negative_prompts: Schema.string().role('textarea').default("").description("Negative Prompt"),
                sample_width: Schema.number().min(64).step(16).default(1024).description("Preview width"),
                sample_height: Schema.number().min(64).step(16).default(1024).description("Preview height"),
                sample_cfg: Schema.number().min(1).max(30).default(5.5).description("CFG scale; Krea 2 RAW: 5.5 recommended (Turbo previews ignore it)"),
                sample_seed: Schema.number().min(0).step(1).default(2333).description("Seed"),
                sample_steps: Schema.number().min(1).max(300).default(28).description("Sampling steps; Krea 2 RAW: 28 recommended, Turbo: 8"),
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

    Schema.object({
        seed: Schema.number().min(0).step(1).default(1337).description("Random seed"),
    }),

    // Speed and memory
    Schema.object({
        mixed_precision: Schema.union(["bf16", "fp16", "no"]).default("bf16").description("Mixed precision; Krea 2 trains in bf16"),
        convrot_int8: Schema.boolean().default(true).description("Keep the frozen base weights in ConvRot int8: half the VRAM of bf16 and a real int8 matmul (needs Triton, which LoRA Forge installs). Loads ComfyUI's `int8_convrot` file directly, or quantizes a bf16 file when it loads. Instead of fp8_base"),
        convrot_int8_bwd: Schema.union(["bf16", "int8"]).default("bf16").description("Backward pass with convrot_int8: `bf16` is exact, `int8` is faster without gradient checkpointing but quantizes the gradients a little"),
        fp8_base: Schema.boolean().default(false).description("Keep the frozen base weights in scaled fp8 (needs a bf16 DiT file; sets fp8_scaled too). Same VRAM as convrot_int8 but slower on Krea 2, which dequantizes per forward. Turning it on turns convrot_int8 off"),
        attn_mode: Schema.union(["sdpa", "flash_attn", "sage_attn", "xformers"]).default("sdpa").description("Attention implementation. sdpa = PyTorch; flash_attn needs flash-attn, sage_attn needs SageAttention, xformers needs xformers"),
        split_attn: Schema.boolean().default(false).description("Split attention per sample to save a little VRAM; advised with any attention other than sdpa"),
        blocks_to_swap: Schema.number().min(0).max(26).step(1).default(10).description("Blocks kept in system RAM and moved to the GPU when used (up to 26 of 28); the step time barely changes. With int8 weights, 10 fits a 16 GB card at 1024x1024. 0 keeps the whole model on the GPU"),
        use_pinned_memory_for_block_swap: Schema.boolean().default(false).description("Pinned RAM for the swapped blocks: faster transfers, but on Windows the pinned memory counts against shared GPU memory"),
        block_swap_h2d_only: Schema.boolean().default(false).description("Experimental: swapped blocks are only copied RAM to GPU, never back (the base is frozen). Saves the return copy"),
        gradient_checkpointing_cpu_offload: Schema.boolean().default(false).description("Keep the checkpointed activations in system RAM instead of VRAM: less VRAM, slower"),
        max_data_loader_n_workers: Schema.number().min(0).step(1).default(0).description("Processes that prepare the batches; the latents and captions are cached, so 0 is fine and starts fastest"),
        persistent_data_loader_workers: Schema.boolean().default(false).description("Keep the dataloader workers alive to shorten the pause between epochs."),
    }),

    // torch.compile (musubi-tuner compiles the 28 main blocks; quantized Linears and swapped blocks stay out)
    Schema.object({
        compile: Schema.boolean().default(true).description("Speed up Krea 2 with torch.compile of the 28 main blocks (needs Triton, which LoRA Forge installs). The first steps of a run are slower while it compiles; turn it off if compiling fails. Works with convrot_int8, fp8_base and blocks_to_swap"),
        compile_mode: Schema.union(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]).default("default").description("Compile mode. max-autotune is the fastest but the first compile is slow"),
        compile_fullgraph: Schema.boolean().default(false).description("Full-graph compile mode. Checking it turns off split_attn"),
        compile_cache_size_limit: Schema.number().min(1).step(1).default(64).description("Most graphs torch.compile builds per block before it stops compiling new shapes (they then run uncompiled, slower): about one per bucket size and batch size"),
        cuda_allow_tf32: Schema.boolean().default(false).description("Allow TF32 (Ampere+), which can improve performance; works without compile"),
        cuda_cudnn_benchmark: Schema.boolean().default(false).description("Enable cuDNN benchmark, which may improve performance; works without compile"),
    }),
]);
