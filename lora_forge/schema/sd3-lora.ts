Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("sd3-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("SD3 model path"),
        clip_l: Schema.string().role('filepicker', { type: "model-file" }).description("clip_l model file"),
        clip_g: Schema.string().role('filepicker', { type: "model-file" }).description("clip_g model file"),
        t5xxl: Schema.string().role('filepicker', { type: "model-file" }).description("t5xxl model file"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
    }),

    Schema.object({
        t5xxl_max_token_length: Schema.number().min(1).step(1).description("T5XXL max token length (empty = automatic)"),
        train_t5xxl: Schema.boolean().default(false).description("Train T5XXL (not recommended)"),
    }),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("768,768").description("Training resolution: width x height, or one number for a square (1024 = 1024x1024). With buckets on (the default), each image is scaled to about this many pixels at its own aspect ratio; with them off, every image is cropped to exactly this size"),
            enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
            min_bucket_reso: Schema.number().min(32).step(32).default(256).description("Shortest bucket side. Only used with bucket_no_upscale off, when images are fitted to a fixed set of buckets of about the training resolution's pixel count; images more stretched than min_bucket_reso and max_bucket_reso allow are cropped"),
            max_bucket_reso: Schema.number().min(32).step(32).default(2048).description("Longest bucket side. Only used with bucket_no_upscale off; buckets keep about the training resolution's pixel count, so this only limits very wide or tall images"),
            bucket_reso_steps: Schema.number().min(32).step(32).default(32).description("ARB bucket resolution step: images are fitted to sizes in steps of this many pixels, so smaller steps crop less. SD3.5 accepts multiples of 32"),
        })
    ),

    // Saving
    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(20).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).step(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).default(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(true).description("Train the U-Net only"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("Train the text encoder only"),
    }),

    // Learning rate & optimizer
    SHARED_SCHEMAS.LR_OPTIMIZER,

    // Loss
    SHARED_SCHEMAS.LOSS_SETTINGS,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora_sd3", "lycoris.kohya"]).default("networks.lora_sd3").description("Network module to train"),
            network_weights: Schema.string().role('filepicker').description("Continue training from an existing LoRA: enter its path"),
            network_dim: Schema.number().min(1).step(1).default(4).description("Network dimension, commonly 4-128. Bigger is not always better; a lower dim uses less VRAM"),
            network_alpha: Schema.number().min(1).default(1).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_args_custom: Schema.array(String).role('table').description('Custom network_args, one per line'),
            enable_base_weight: Schema.boolean().default(false).description('Enable base weights (difference training)'),
        }),

        // LyCORIS options
        SHARED_SCHEMAS.LYCORIS_MAIN,
        SHARED_SCHEMAS.LYCORIS_LOKR,

        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // Preview images
    SHARED_SCHEMAS.PREVIEW_IMAGE,

    // Logging
    SHARED_SCHEMAS.LOG_SETTINGS,

    // Captions
    // FLUX has no max_token_length
    // weighted_captions has no effect on SD3 (strategy_sd3 has no weighted tokenization), so the field is hidden
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length", "weighted_captions"])),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options. SD3.5 does not use clip_skip, so SHARED_SCHEMAS.OTHER is not reused here.
    Schema.object({
        seed: Schema.number().min(0).step(1).default(1337).description("Random seed"),
    }),

    // Full precision
    SHARED_SCHEMAS.FULL_PRECISION_MODE,

    // Speed and memory
    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
            fp8_base: Schema.boolean().default(true).description("Use FP8 precision for the base model"),
            fp8_base_unet: Schema.boolean().description("Use FP8 precision for the U-Net only (not CLIP-L)"),
            sdpa: Schema.boolean().default(true).description("Enable SDPA"),
            cache_text_encoder_outputs: Schema.boolean().default(true).description("Cache text encoder outputs to save VRAM. Needs shuffle_caption off"),
            cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("Cache text encoder outputs to disk"),
        }, ["xformers", "full_fp16", "full_bf16"])
    ),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
