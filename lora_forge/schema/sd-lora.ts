Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("sd-lora").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("Base model file"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
        vae: Schema.string().role('filepicker', { type: "model-file" }).description("(Optional) VAE model file; replaces the VAE inside the base model"),
        v2: Schema.boolean().default(false).description("Enable for base models built on SD 2.0 or later"),
        v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
        scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
    }),

    // Dataset
    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            bucket_reso_steps: Schema.number().min(64).step(64).default(64).description("ARB bucket resolution step: images are fitted to sizes in steps of this many pixels, so smaller steps crop less. SD1.5 needs multiples of 64"),
        })
    ),

    // Saving
    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).step(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(false).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(false).description("Train the U-Net only"),
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
        // SD1.5's U-Net: 12 down / 1 mid / 12 up blocks
        SHARED_SCHEMAS.NETWORK_OPTION_BLOCK_WEIGHTS,
        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // Preview images
    SHARED_SCHEMAS.PREVIEW_IMAGE,

    // Logging
    SHARED_SCHEMAS.LOG_SETTINGS,

    // Captions
    Schema.object(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options (seed, clip_skip)
    SHARED_SCHEMAS.OTHER,

    // Speed and memory; text encoder output caching is SDXL-only in sd-scripts' LoRA trainers
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, SHARED_SCHEMAS.RAW.TRAINING_PRECISION,
        ["mixed_precision", "full_fp16", "full_bf16", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"])),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
