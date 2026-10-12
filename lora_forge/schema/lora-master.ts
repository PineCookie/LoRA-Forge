Schema.intersect([
    Schema.intersect([
        Schema.object({
            model_train_type: Schema.union(["sd-lora", "sdxl-lora"]).default("sd-lora").description("Training type"),
            pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).description("Base model file"),
            resume: Schema.string().role('filepicker', { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
            vae: Schema.string().role('filepicker', { type: "model-file" }).description("(Optional) VAE model file; replaces the VAE inside the base model"),
        }).description("Model"),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-lora"),
                v2: Schema.boolean().default(false).description("Enable for base models built on SD 2.0 or later"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-lora").required(),
                v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
                scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
            }),
            Schema.object({
                model_train_type: Schema.const("sdxl-lora").required(),
                v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
                scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
            }),
            Schema.object({}),
        ]),
    ]),

    // Dataset
    Schema.object(SHARED_SCHEMAS.RAW.DATASET_SETTINGS).description("Dataset"),

    // Saving
    SHARED_SCHEMAS.SAVE_SETTINGS,

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).default(1).description("Batch size; higher uses more VRAM"),
        gradient_checkpointing: Schema.boolean().default(false).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).description("Gradient accumulation steps"),
        network_train_unet_only: Schema.boolean().default(false).description("Train the U-Net only; recommended for SDXL LoRAs"),
        network_train_text_encoder_only: Schema.boolean().default(false).description("Train the text encoder only"),
    }).description("Training"),

    // Learning rate & optimizer
    SHARED_SCHEMAS.LR_OPTIMIZER,

    // Loss
    SHARED_SCHEMAS.LOSS_SETTINGS,

    Schema.intersect([
        Schema.object({
            network_module: Schema.union(["networks.lora", "networks.dylora", "networks.oft", "lycoris.kohya"]).default("networks.lora").description("Network module to train"),
            network_weights: Schema.string().role('filepicker').description("Continue training from an existing LoRA: enter its path"),
            network_dim: Schema.number().min(1).default(32).description("Network dimension, commonly 4-128. Bigger is not always better; a lower dim uses less VRAM"),
            network_alpha: Schema.number().min(1).default(32).description("Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"),
            network_dropout: Schema.number().step(0.01).default(0).description('Dropout probability (not compatible with LyCORIS; use LyCORIS\' own dropout)'),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("Max-norm regularization. If used, 1 is recommended"),
            network_args_custom: Schema.array(String).role('table').description('Custom network_args, one per line'),
            enable_block_weights: Schema.boolean().default(false).description('Enable block-wise learning rates (network module networks.lora only)'),
            enable_base_weight: Schema.boolean().default(false).description('Enable base weights (difference training)'),
        }).description("Network"),

        // LyCORIS options
        SHARED_SCHEMAS.LYCORIS_MAIN,
        SHARED_SCHEMAS.LYCORIS_LOKR,

        // DyLoRA options
        SHARED_SCHEMAS.NETWORK_OPTION_DYLORA,

        // Block-wise learning rate options
        SHARED_SCHEMAS.NETWORK_OPTION_BLOCK_WEIGHTS,

        SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
    ]),

    // Preview images
    SHARED_SCHEMAS.PREVIEW_IMAGE,

    // Logging
    SHARED_SCHEMAS.LOG_SETTINGS,

    // Captions
    Schema.object(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS).description("Captions"),

    // Noise
    SHARED_SCHEMAS.NOISE_SETTINGS,

    // Data augmentation
    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    // Other options
    SHARED_SCHEMAS.OTHER,

    // Full precision
    SHARED_SCHEMAS.FULL_PRECISION_MODE,

    // Speed and memory
    Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {}, ["full_fp16", "full_bf16"])).description("Speed and memory"),

    // Distributed training
    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);
