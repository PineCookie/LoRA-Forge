Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("sdxl-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("Base model file"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).description("(Optional) VAE model file; replaces the VAE inside the base model"),
        v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
        scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
    }),

    // Dataset: SDXL trains at about 1 megapixel
    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("1024,1024").description("Training resolution, width x height. Non-square works, but both must be multiples of 64."),
            min_bucket_reso: Schema.number().min(32).step(32).default(256).description("Shortest bucket side. Only used with bucket_no_upscale off, when images are fitted to a fixed set of buckets of about the training resolution's pixel count; images more stretched than min_bucket_reso and max_bucket_reso allow are cropped"),
            max_bucket_reso: Schema.number().min(32).step(32).default(2048).description("Longest bucket side. Only used with bucket_no_upscale off; buckets keep about the training resolution's pixel count, so this only limits very wide or tall images"),
            bucket_reso_steps: Schema.number().min(32).step(32).default(32).description("ARB bucket resolution step: images are fitted to sizes in steps of this many pixels, so smaller steps crop less. SDXL accepts multiples of 32"),
        })
    ),

    // Saving
    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).step(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(true).description("Train the U-Net only; recommended for SDXL LoRAs"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("Train the text encoder only"),
    }),

    // Learning rate & optimizer
    SHARED_SCHEMAS.LR_OPTIMIZER,

    // Loss
    SHARED_SCHEMAS.LOSS_SETTINGS,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora", "networks.dylora", "networks.oft", "lycoris.kohya"]).default("networks.lora").description("Network module to train"),
            network_weights: Schema.string().role('filepicker').description("Continue training from an existing LoRA: enter its path"),
            network_dim: Schema.number().min(1).step(1).default(32).description("Network dimension, commonly 4-128. Bigger is not always better; a lower dim uses less VRAM"),
            network_alpha: Schema.number().min(1).default(32).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description('Dropout probability (not compatible with LyCORIS; use LyCORIS\' own dropout)'),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("Max-norm regularization. If used, 1 is recommended"),
            network_args_custom: Schema.array(String).role('table').description('Custom network_args, one per line'),
            enable_block_weights: Schema.boolean().default(false).description('Enable block-wise learning rates (network module networks.lora only)'),
            enable_base_weight: Schema.boolean().default(false).description('Enable base weights (difference training)'),
        }),

        SHARED_SCHEMAS.LYCORIS_MAIN,
        SHARED_SCHEMAS.LYCORIS_LOKR,
        SHARED_SCHEMAS.NETWORK_OPTION_DYLORA,

        // SDXL's U-Net has 9 down / 3 mid / 9 up blocks (networks/lora.py SDXL_NUM_OF_BLOCKS)
        Schema.union([
            Schema.object({
                enable_block_weights: Schema.const(true).required(),
                down_lr_weight: Schema.string().role('folder').default("1,1,1,1,1,1,1,1,1").description("Learning rate weights of the U-Net encoder (down) blocks, 9 values"),
                mid_lr_weight: Schema.string().role('folder').default("1,1,1").description("Learning rate weights of the U-Net mid blocks, 3 values"),
                up_lr_weight: Schema.string().role('folder').default("1,1,1,1,1,1,1,1,1").description("Learning rate weights of the U-Net decoder (up) blocks, 9 values"),
                block_lr_zero_threshold: Schema.number().step(0.01).default(0).description("Block learning rates below this threshold are set to 0"),
            }),
            Schema.object({}),
        ]),

        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // Preview images, at SDXL's size
    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("Generate preview images during training"),
        }),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                randomly_choice_prompt: Schema.boolean().default(false).description("Use a random caption from the dataset as the preview prompt"),
                prompt_file: Schema.string().role('textarea').description("Preview prompt file. When set, the prompts in the file are used and the options below are ignored."),
                positive_prompts: Schema.string().role('textarea').default("masterpiece, best quality, 1girl, solo").description("Prompt"),
                negative_prompts: Schema.string().role('textarea').default("lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry").description("Negative Prompt"),
                sample_width: Schema.number().min(64).step(8).default(1024).description("Preview width"),
                sample_height: Schema.number().min(64).step(8).default(1024).description("Preview height"),
                sample_cfg: Schema.number().min(1).max(30).default(7).description("CFG Scale"),
                sample_seed: Schema.number().min(0).step(1).default(2333).description("Seed"),
                sample_steps: Schema.number().min(1).max(300).default(28).description("Sampling steps"),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                sample_every_n_epochs: Schema.number().min(1).step(1).default(2).description("Generate preview images every N epochs"),
            }),
            Schema.object({}),
        ]),
    ]),

    // Logging
    SHARED_SCHEMAS.LOG_SETTINGS,

    // Captions
    Schema.object(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options. SDXL does not use clip_skip, so SHARED_SCHEMAS.OTHER is not reused here.
    Schema.object({
        seed: Schema.number().min(0).step(1).default(1337).description("Random seed"),
    }),

    // Speed and memory
    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
            ...SHARED_SCHEMAS.RAW.TRAINING_PRECISION,
            no_half_vae: Schema.boolean().description("Keep the VAE in full precision (fp32). Always on with the `fp16` and `full-fp16` training precisions: the SDXL VAE gives NaN in fp16"),
            cache_text_encoder_outputs: Schema.boolean().default(true).description("Cache text encoder outputs to save VRAM. Needs shuffle_caption off; turns on network_train_unet_only (the text encoder cannot be trained at the same time)"),
        }, ["mixed_precision", "full_fp16", "full_bf16"])
    ),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
