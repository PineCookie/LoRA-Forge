// MOSS-Image LoRA with LoRA Forge's own trainer (lora_forge/trainers/moss_image); option names follow
// sd-scripts'. lora_forge/training/moss_image.py checks the model files and writes the preview prompts.
Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("moss-image-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("MOSS-Image DiT file: ComfyUI's `moss_image_base_bf16` (or OpenMOSS's `transformer` folder). Train on Base: the LoRA also works on SFT. The int8 files cannot train"),
        text_encoder: Schema.string().role('filepicker', { type: "model-file" }).description("Required. Qwen3.5-9B as one bf16 safetensors file (ComfyUI's `qwen3.5_9b_bf16`), used to cache the captions before training and for the preview prompts. Layers that do not fit in VRAM run from RAM"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).description("Required. OpenMOSS's VAE (`vae/diffusion_pytorch_model.safetensors` of MOSS-Image-Base), which gives the latents the model was trained on; ComfyUI's `flux2-vae` also works (latents within about 1%)"),
    }),

    Schema.object({
        logit_mean: Schema.number().step(0.01).default(-1.09).description("Timestep sampling: the mean of MOSS-Image's logit-normal distribution (-1.09 as in its report). Higher values train more on low-noise steps (fine detail), lower values on high-noise steps (composition)"),
        logit_std: Schema.number().min(0.01).step(0.01).default(0.95).description("Timestep sampling: the spread of the logit-normal distribution (0.95 as in the MOSS-Image report)"),
    }),

    Schema.object({
        train_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).default("./train/").description("Training dataset folder"),
        resolution: Schema.string().default("1024,1024").description("Training resolution, width x height: each image is scaled to this many pixels at its own aspect ratio (MOSS-Image trained at about 1024x1024). Sides are rounded down to multiples of 16"),
        enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
        bucket_no_upscale: Schema.boolean().default(true).description("Give each image a bucket of its own aspect ratio: larger images are scaled down to the training resolution's pixel count, smaller ones keep their size"),
        caption_extension: Schema.string().default(".txt").description("Caption file extension. Each line of a caption file is one caption of its image, and each step trains on one of them at random: e.g. a full description on one line and a short one with the name on the next"),
        caption_dropout_rate: Schema.number().min(0).max(0.5).step(0.01).default(0).description("Share of steps that train on an empty caption, so the LoRA also learns the empty prompt that CFG compares against. Helps a style apply everywhere (0.1 is common); for a character it can make it appear without its name. 0 = off"),
    }),

    // Saving
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.SAVE_FIELDS, {
        save_precision: Schema.union(["fp16", "bf16", "float"]).default("fp16").description("Save precision of the LoRA file. `fp16` keeps a little more detail than `bf16` (LoRA weights stay far inside fp16's range); `float` is fp32, twice the size"),
    }, ["save_model_as", "save_state"])),

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(10).description("Maximum training epochs"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).default(1).description("Gradient accumulation steps. MOSS-Image trains one image per step, so this is the way to larger batches"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing: saves VRAM by running each block's forward pass again during the backward pass. Keep it on with 16 GB of VRAM: the DiT alone takes about 10 GB"),
    }),

    // Learning rate & optimizer: the GUI's ProdigyPlus settings become optimizer_args as on the other pages
    Schema.intersect([
        Schema.object({
            optimizer_type: Schema.union(["AdamW", "AdamW8bit", "AdaFactor", "prodigyplus.ProdigyPlusScheduleFree"]).default("AdamW").description("Optimizer"),
            learning_rate: Schema.number().min(0).default("1e-4").description("Learning rate; 1e-4 for AdamW / AdamW8bit at rank 32"),
            lr_scheduler: Schema.union(["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "warmup_stable_decay"]).default("cosine").description("Learning rate scheduler. `cosine` lowers the learning rate toward 0 over the run, so late steps change the LoRA less; MOSS-Image LoRAs overtrained quickly at a constant rate in our tests. Schedule-free optimizers (ProdigyPlus) always run constant"),
            lr_warmup_steps: Schema.number().min(0).default(0).description("Learning rate warmup: a whole number of steps (100 = the first 100 steps), or below 1 a fraction of all steps (0.05 = the first 5%, which scales with the run's length). 0 = off"),
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
                lr_scheduler: Schema.const('warmup_stable_decay'),
                lr_decay_steps: Schema.number().min(0).default(0.2).description("Length of the decay at the end of `warmup_stable_decay`: a whole number of steps, or below 1 a fraction of all steps (0.2 = the last 20%). Between the warmup and the decay the learning rate stays at its full value"),
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

    Schema.object({
        network_weights: Schema.string().role('filepicker', { type: "model-file" }).description("Continue training from an existing MOSS-Image LoRA of the same rank: enter its path"),
        network_dim: Schema.number().min(1).step(1).default(32).description("Network dimension (rank). The LoRA covers the attention and feed-forward layers of all 42 blocks; rank 32 is about 74M parameters"),
        network_alpha: Schema.number().min(1).default(32).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
        network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description("Dropout probability"),
        lora_targets: Schema.union(["all", "attention"]).default("all").description("Layers the LoRA trains. `all`: attention and feed-forward. `attention`: attention only (33M instead of 74M parameters at rank 32); the feed-forward layers, which hold much of how things are drawn, stay the base model's, so its rendering style is kept better. Characters and outfits are usually still learned; very unusual small details may be less exact"),
    }),

    // Preview images
    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("Generate preview images during training"),
        }),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                prompt_file: Schema.string().role('textarea').description("Preview prompt file, one prompt per line, e.g. `a cat --n blurry --w 1024 --h 1024 --s 30 --l 3 --d 42`. When set, the options below are ignored"),
                positive_prompts: Schema.string().role('textarea').default("A photo of a cat sitting on a windowsill, soft morning light.").description("Prompt"),
                negative_prompts: Schema.string().role('textarea').default("").description("Negative Prompt"),
                sample_width: Schema.number().min(64).step(16).default(1024).description("Preview width"),
                sample_height: Schema.number().min(64).step(16).default(1024).description("Preview height"),
                sample_cfg: Schema.number().min(1).max(30).default(3).description("CFG scale, 3 recommended. As in OpenMOSS's pipeline, the negative pass skips block 22"),
                sample_seed: Schema.number().min(0).step(1).default(2333).description("Seed"),
                sample_steps: Schema.number().min(1).max(300).default(30).description("Sampling steps: 30 is enough for previews (OpenMOSS's default is 50)"),
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

    // Precision and speed. training_precision becomes mixed_precision (normalize.ts), as on the other pages
    Schema.object({
        training_precision: Schema.union(["bf16", "fp16"]).default("bf16").description("Precision of the DiT's matrix multiplies (the LoRA weights always train in fp32).<br>`bf16`: the standard.<br>`fp16`: the block matrix multiplies run in fp16 with loss scaling. With fp16_accumulation about a quarter faster per step on GeForce RTX cards, and as close to fp32 as bf16 in our tests"),
        fp16_accumulation: Schema.boolean().default(true).description("Let fp16 matrix multiplies add up in fp16 too: much faster on GeForce RTX cards (no gain on data-center or RTX PRO cards), still about as precise as bf16"),
        attn_mode: Schema.union(["auto", "flash_attn", "sdpa"]).default("auto").description("Attention implementation. auto = flash_attn when flash-attn is installed (faster), otherwise sdpa (PyTorch)"),
        compile: Schema.boolean().default(true).description("Speed up MOSS-Image with per-block torch.compile (about 30% faster per step; needs Triton, which LoRA Forge installs). The first steps of a run are slower while it compiles; turn it off if compiling fails"),
        compile_cache_size_limit: Schema.number().min(1).step(1).default(64).description("Most graphs torch.compile builds per block before it stops compiling new shapes (they then run uncompiled, slower). MOSS-Image needs only a few for all bucket sizes"),
    }),
]);
