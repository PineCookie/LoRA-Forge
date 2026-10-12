Schema.intersect([
    Schema.object({
        pretrained_model_name_or_path: Schema.string().role('filepicker', {type: "model-file"}).description("Base model file"),
    }).description("Model"),

    Schema.object({
        train_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).default("./train/aki").description("Training dataset folder"),
        reg_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).description("Regularization dataset folder. Leave empty to train without regularization images"),
        resolution: Schema.string().default("512,512").description("Training resolution, width x height. Non-square works, but both must be multiples of 64."),
    }).description("Dataset"),

    Schema.object({
        output_name: Schema.string().default("model_name").description("Output name"),
        output_dir: Schema.string().default("./output").role('filepicker', { type: "folder" }).description("Output folder"),
        save_every_n_epochs: Schema.number().default(2).description("Save the model every N epochs"),
    }).description("Saving"),

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).default(1).description("Batch size"),
    }).description("Training"),

    Schema.intersect([
        Schema.object({
            unet_lr: Schema.string().default("1e-4").description("U-Net learning rate"),
            text_encoder_lr: Schema.string().default("1e-5").description("Text encoder learning rate"),
            lr_scheduler: Schema.union([
                "cosine",
                "cosine_with_restarts",
                "constant",
                "constant_with_warmup",
            ]).default("cosine_with_restarts").description("Learning rate scheduler"),
            lr_warmup_steps: Schema.number().default(0).description('Learning rate warmup steps'),
        }).description("Learning rate and optimizer"),
        Schema.object({
            min_snr_gamma: Schema.number().step(0.1).description("Min-SNR gamma; 5 recommended. Similar in effect to Debiased Estimation loss, so avoid enabling both"),
            debiased_estimation_loss: Schema.boolean().default(false).description("Use Debiased Estimation loss. Similar in effect to Min-SNR, so avoid enabling both"),
            loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("Loss function"),
        }).description("Loss"),
        Schema.union([
            Schema.object({
                loss_type: Schema.const('huber').required(),
                huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber loss schedule; only used when loss_type is huber or smooth_l1"),
                huber_c: Schema.number().step(0.001).default(0.1).description("Huber loss decay parameter; only used when loss_type is huber or smooth_l1"),
            }),
            Schema.object({
                loss_type: Schema.const('smooth_l1').required(),
                huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber loss schedule; only used when loss_type is huber or smooth_l1"),
                huber_c: Schema.number().step(0.001).default(0.1).description("Huber loss decay parameter; only used when loss_type is huber or smooth_l1"),
            }),
            Schema.object({}),
        ]),
        Schema.union([
            Schema.object({
                lr_scheduler: Schema.const('cosine_with_restarts'),
                lr_scheduler_num_cycles: Schema.number().default(1).description('Number of restarts'),
            }),
            Schema.object({}),
        ]),
        Schema.object({
            optimizer_type: Schema.union([
                "AdamW8bit",
                "Lion",
            ]).default("AdamW8bit").description("Optimizer"),
        })
    ]),

    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description('Generate preview images during training'),
        }).description('Preview images'),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                sample_prompts: Schema.string().role('textarea').default(window.__LORA_FORGE__.SAMPLE_PROMPTS_DEFAULT).description(window.__LORA_FORGE__.SAMPLE_PROMPTS_DESCRIPTION),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                sample_every_n_epochs: Schema.number().default(2).description("Generate preview images every N epochs"),
            }),
            Schema.object({}),
        ]),
    ]),

    Schema.intersect([
        Schema.object({
            network_weights: Schema.string().role('filepicker', { type: "model-file", internal: "model-saved-file" }).description("Continue training from an existing LoRA: enter its path"),
            network_dim: Schema.number().min(8).max(256).step(8).default(32).description("Network dimension, commonly 4-128. Bigger is not always better; a lower dim uses less VRAM"),
            network_alpha: Schema.number().min(1).default(32).description(
                "Common values: equal to network_dim, network_dim/2, or 1. A smaller alpha needs a higher learning rate"
            ),
        }).description("Network"),
    ]),

    Schema.object({
        shuffle_caption: Schema.boolean().default(true).description("Shuffle caption tokens during training"),
        keep_tokens: Schema.number().min(0).max(255).step(1).default(0).description("Keep the first N tokens in place when shuffling"),
    }).description("Captions"),

    Schema.object({
        mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("fp16").description("Mixed precision; `bf16` also works on RTX 30 series and newer"),
        no_half_vae: Schema.boolean().description("Keep the VAE in full precision; use it when you get NaN detected in latents errors"),
        xformers: Schema.boolean().default(true).description("Enable xformers"),
        cache_latents: Schema.boolean().default(true).description("Cache image latents (VAE outputs) to save VRAM")
    }).description("Speed and memory"),
]);
