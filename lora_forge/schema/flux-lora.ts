Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("flux-lora").disabled().description("Training type"),
        model_type: Schema.union(["flux", "chroma"]).default("flux").description("FLUX model type (Chroma is supported)"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("FLUX model path"),
        ae: Schema.string().role('filepicker', { type: "model-file" }).description("AE model file"),
        clip_l: Schema.string().role('filepicker', { type: "model-file" }).description("clip_l model file"),
        t5xxl: Schema.string().role('filepicker', { type: "model-file" }).description("t5xxl model file"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
    }).description("Model"),

    Schema.object({
        timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift"]).default("sigmoid").description("Timestep sampling"),
        show_timesteps: Schema.union(["console", "image"]).description("Output format of the timestep distribution preview. Run it with the “Preview timesteps” button; ignored when training. ⚠ Needs sd-scripts v0.11.1+"),
        show_timesteps_resolution: Schema.string().default("1024").description("Image resolution the preview assumes (H or H,W); used by flux_shift"),
        show_timesteps_offset: Schema.number().step(0.01).default(0.0).description("Timestep offset for the preview only; no effect with sigma/uniform. ⚠ Needs sd-scripts main (> v0.11.1)"),
        sigmoid_scale: Schema.number().step(0.001).default(1.0).description("Sigmoid scale"),
        model_prediction_type: Schema.union(["raw", "additive", "sigma_scaled"]).default("raw").description("Model prediction type"),
        discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("Discrete flow shift of the Euler scheduler"),
        guidance_scale: Schema.number().step(0.01).default(1.0).description("CFG guidance scale"),
        t5xxl_max_token_length: Schema.number().step(1).description("T5XXL max token length (empty = automatic)"),
        train_t5xxl: Schema.boolean().default(false).description("Train T5XXL (not recommended)"),
        // apply attention mask to T5-XXL encode and FLUX double blocks
        apply_t5_attn_mask: Schema.boolean().default(true).description("Apply the attention mask to the T5-XXL encoder and the FLUX double blocks"),
    }).description("FLUX options"),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("768,768").description("Training resolution, width x height. Non-square works, but both must be multiples of 64."),
            enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
            min_bucket_reso: Schema.number().default(256).description("Minimum ARB bucket resolution"),
            max_bucket_reso: Schema.number().default(2048).description("Maximum ARB bucket resolution"),
            bucket_reso_steps: Schema.number().default(64).description("ARB bucket resolution step; FLUX needs a value above 64"),
        })
    ).description("Dataset"),

    // Saving
    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(20).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(true).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).default(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(true).description("Train the U-Net only"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("Train the text encoder only"),
    }).description("Training"),

    // Learning rate & optimizer
    SHARED_SCHEMAS.LR_OPTIMIZER,

    // Loss
    SHARED_SCHEMAS.LOSS_SETTINGS_NO_WEIGHTING,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora_flux", "networks.oft_flux", "lycoris.kohya"]).default("networks.lora_flux").description("Network module to train"),
            network_weights: Schema.string().role('filepicker').description("Continue training from an existing LoRA: enter its path"),
            network_dim: Schema.number().min(1).default(2).description("Network dimension, commonly 4-128. Bigger is not always better; a lower dim uses less VRAM"),
            network_alpha: Schema.number().min(1).default(16).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_dropout: Schema.number().step(0.01).default(0).description('Dropout probability (not compatible with LyCORIS; use LyCORIS\' own dropout)'),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("Max-norm regularization. If used, 1 is recommended"),
            network_args_custom: Schema.array(String).role('table').description('Custom network_args, one per line'),
            enable_base_weight: Schema.boolean().default(false).description('Enable base weights (difference training)'),
        }).description("Network"),

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
    // weighted_captions has no effect on FLUX (strategy_flux has no weighted tokenization), so the field is hidden
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length", "weighted_captions"])).description("Captions"),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options
    SHARED_SCHEMAS.OTHER,

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
    ).description("Speed and memory"),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
