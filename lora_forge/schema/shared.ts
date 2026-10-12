(function () {
    let data = {
        RAW: {
            DATASET_SETTINGS: {
                train_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).default("./train/").description("Training dataset folder"),
                reg_data_dir: Schema.string().role('filepicker', { type: "folder", internal: "train-dir" }).description("Regularization dataset folder. Leave empty to train without regularization images"),
                prior_loss_weight: Schema.number().step(0.1).default(1.0).description("Regularization: prior loss weight"),
                resolution: Schema.string().default("512,512").description("Training resolution, width x height. Non-square works, but both must be multiples of 64."),
                enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
                min_bucket_reso: Schema.number().default(256).description("Minimum ARB bucket resolution"),
                max_bucket_reso: Schema.number().default(1024).description("Maximum ARB bucket resolution"),
                bucket_reso_steps: Schema.number().default(64).description("ARB bucket resolution step: images are fitted to sizes in steps of this many pixels, so smaller steps crop less. SD1.5 needs multiples of 64"),
                bucket_no_upscale: Schema.boolean().default(true).description("Do not upscale images to fit ARB buckets"),
            },
            CAPTION_SETTINGS: {
                caption_extension: Schema.string().default(".txt").description("Caption file extension"),
                shuffle_caption: Schema.boolean().default(false).description("Shuffle caption tokens during training"),
                weighted_captions: Schema.boolean().description("Use weighted tokens; not recommended together with shuffle_caption"),
                keep_tokens: Schema.number().min(0).max(255).step(1).default(0).description("Keep the first N tokens in place when shuffling"),
                keep_tokens_separator: Schema.string().description("Separator that ends the kept tokens"),
                max_token_length: Schema.number().default(255).description("Max token length"),
                caption_dropout_rate: Schema.number().min(0).step(0.01).description("Probability of dropping the whole caption (the image is trained without caption or class token)"),
                caption_dropout_every_n_epochs: Schema.number().min(0).max(100).step(1).description("Drop all captions every N epochs"),
                caption_tag_dropout_rate: Schema.number().min(0).step(0.01).description("Probability of dropping each comma-separated tag"),
            },
            PRECISION_CACHE_BATCH: {
                mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("Mixed precision; use `fp16` on GPUs older than the RTX 30 series"),
                full_fp16: Schema.boolean().description("Full FP16 precision"),
                full_bf16: Schema.boolean().description("Full BF16 precision"),
                no_half_vae: Schema.boolean().description("Keep the VAE in full precision"),
                xformers: Schema.boolean().default(false).description("Enable xformers"),
                sdpa: Schema.boolean().description("Enable SDPA"),
                lowram: Schema.boolean().default(false).description("Low-RAM mode: loads the U-Net, text encoder and VAE directly into VRAM"),
                cache_latents: Schema.boolean().default(true).description("Cache image latents (VAE outputs) to save VRAM"),
                cache_latents_to_disk: Schema.boolean().default(true).description("Cache image latents to disk"),
                cache_text_encoder_outputs: Schema.boolean().description("Cache text encoder outputs to save VRAM. Needs shuffle_caption off"),
                cache_text_encoder_outputs_to_disk: Schema.boolean().description("Cache text encoder outputs to disk"),
                persistent_data_loader_workers: Schema.boolean().default(true).description("Keep the dataloader workers alive to shorten the pause between epochs."),
                vae_batch_size: Schema.number().min(1).description("VAE encoding batch size"),
            },
            SAVE_FIELDS: {
                output_name: Schema.string().default("model_name").description("Output name"),
                output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./output").description("Output folder"),
                save_model_as: Schema.union(["safetensors", "pt", "ckpt"]).default("safetensors").description("Save format"),
                save_precision: Schema.union(["fp16", "float", "bf16"]).default("bf16").description("Save precision"),
                save_every_n_epochs: Schema.number().default(2).description("Save the model every N epochs"),
                save_state: Schema.boolean().default(false).description("Save the training state; `resume` can continue training from it"),
            },
            LOG_FIELDS: {
                log_with: Schema.union(["tensorboard", "wandb"]).default("tensorboard").description("Logger"),
                log_prefix: Schema.string().description("Log prefix"),
                log_tracker_name: Schema.string().description("Log tracker name"),
                logging_dir: Schema.string().role('filepicker', { type: "folder" }).default("./logs").description("Log folder"),
            },
        },

        LYCORIS_MAIN: Schema.union([
            Schema.object({
                network_module: Schema.const('lycoris.kohya').required(),
                lycoris_algo: Schema.union(["locon", "loha", "lokr", "ia3", "dylora", "glora", "diag-oft", "boft"]).default("locon").description('LyCORIS algorithm'),
                conv_dim: Schema.number().default(4),
                conv_alpha: Schema.number().default(1),
                dropout: Schema.number().step(0.01).default(0).description('Dropout probability; 0-0.5 recommended. Not supported by LoHa/LoKr/(IA)^3 yet'),
                train_norm: Schema.boolean().default(false).description('Train the norm layers; not supported by (IA)^3'),
            }),
            Schema.object({}),
        ]),

        LYCORIS_LOKR: Schema.union([
            Schema.object({
                lycoris_algo: Schema.const('lokr').required(),
                lokr_factor: Schema.number().min(-1).default(-1).description('Commonly 4 to infinity (-1 means infinity)'),
            }),
            Schema.object({}),
        ]),

        NETWORK_OPTION_DYLORA: Schema.union([
            Schema.object({
                network_module: Schema.const('networks.dylora').required(),
                dylora_unit: Schema.number().min(1).default(4).description('DyLoRA unit. 1 is the smallest and the slowest; 4, 8, 12 or 16 are typical'),
            }),
            Schema.object({}),
        ]),

        NETWORK_OPTION_BASEWEIGHT: Schema.union([
            Schema.object({
                enable_base_weight: Schema.const(true).required(),
                base_weights: Schema.string().role('textarea').description("LoRA files merged into the base model, one path per line"),
                base_weights_multiplier: Schema.string().role('textarea').description("Weights of the merged LoRAs, one number per line"),
            }),
            Schema.object({}),
        ]),

        NETWORK_OPTION_BLOCK_WEIGHTS: Schema.union([
            Schema.object({
                enable_block_weights: Schema.const(true).required(),
                down_lr_weight: Schema.string().role('folder').default("1,1,1,1,1,1,1,1,1,1,1,1").description("Learning rate weights of the U-Net encoder (down) blocks, 12 values"),
                mid_lr_weight: Schema.string().role('folder').default("1").description("Learning rate weight of the U-Net mid block, 1 value"),
                up_lr_weight: Schema.string().role('folder').default("1,1,1,1,1,1,1,1,1,1,1,1").description("Learning rate weights of the U-Net decoder (up) blocks, 12 values"),
                block_lr_zero_threshold: Schema.number().step(0.01).default(0).description("Block learning rates below this threshold are set to 0"),
            }),
            Schema.object({}),
        ]),

        SAVE_SETTINGS: Schema.intersect([
            Schema.object({
                output_name: Schema.string().default("model_name").description("Output name"),
                output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./output").description("Output folder"),
                save_model_as: Schema.union(["safetensors", "pt", "ckpt"]).default("safetensors").description("Save format"),
                save_precision: Schema.union(["fp16", "float", "bf16"]).default("bf16").description("Save precision"),
                save_every_n_epochs: Schema.number().default(2).description("Save the model every N epochs"),
                save_state: Schema.boolean().default(false).description("Save the training state; `resume` can continue training from it"),
            }).description("Saving"),
            Schema.union([
                Schema.object({
                    save_state: Schema.const(true).required(),
                    save_last_n_epochs_state: Schema.number().min(1).description("Keep only the training states of the last n epochs"),
                }),
                Schema.object({})
            ])
        ]),

        LR_OPTIMIZER: Schema.intersect([
            Schema.object({
                optimizer_type: Schema.union([
                    "AdamW",
                    "AdamW8bit",
                    "PagedAdamW8bit",
                    "RAdamScheduleFree",
                    "Lion",
                    "Lion8bit",
                    "PagedLion8bit",
                    "SGDNesterov",
                    "SGDNesterov8bit",
                    "DAdaptation",
                    "DAdaptAdam",
                    "DAdaptAdaGrad",
                    "DAdaptAdanIP",
                    "DAdaptLion",
                    "DAdaptSGD",
                    "AdaFactor",
                    "Prodigy",
                    "prodigyplus.ProdigyPlusScheduleFree",
                    "pytorch_optimizer.CAME"
                ]).default("AdamW8bit").description("Optimizer"),
            }),

            Schema.object({
                learning_rate: Schema.string().default("1e-4").description("Overall learning rate; ignored once separate U-Net and text encoder learning rates are set."),
                unet_lr: Schema.string().default("1e-4").description("U-Net learning rate"),
                text_encoder_lr: Schema.string().default("1e-5").description("Text encoder learning rate"),
                lr_scheduler: Schema.union([
                    "linear",
                    "cosine",
                    "cosine_with_restarts",
                    "polynomial",
                    "constant",
                    "constant_with_warmup",
                ]).default("cosine_with_restarts").description("Learning rate scheduler"),
                lr_warmup_steps: Schema.number().default(0).description('Learning rate warmup steps'),
                max_grad_norm: Schema.number().min(0).step(0.1).default(1.0).description("Max gradient norm for clipping, 0 = no clipping. sd-scripts default 1.0; 0 is recommended for adaptive optimizers such as Prodigy / ProdigyPlus (set automatically when you switch optimizers)"),
            }).description("Learning rate and optimizer"),

            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const('cosine_with_restarts'),
                    lr_scheduler_num_cycles: Schema.number().default(1).description('Number of restarts'),
                }),
                Schema.object({}),
            ]),

            Schema.union([
                Schema.object({
                    optimizer_type: Schema.const('Prodigy').required(),
                    prodigy_d0: Schema.string().description("Prodigy d0: the initial LR estimate; try 1e-7, 1e-6 or 1e-5"),
                    prodigy_d_coef: Schema.string().default("2.0").description("Prodigy d_coef. Default 2.0"),
                }),
                Schema.object({
                    optimizer_type: Schema.const('prodigyplus.ProdigyPlusScheduleFree').required(),
                    prodigyplus_d_coef: Schema.string().default("1.0").description("Prodigy Plus d_coef. Official default 1.0; try 2 or higher to help the LR grow"),
                    prodigyplus_betas: Schema.string().default("(0.95, 0.99)").description("Prodigy Plus betas as a Python tuple, e.g. (0.95, 0.99)"),
                    prodigyplus_schedulefree_c: Schema.string().default("20").description("Prodigy Plus schedulefree_c. This UI defaults to 20 (official default 0)"),
                    prodigyplus_prodigy_steps: Schema.string().default("0").description("Prodigy Plus prodigy_steps. Official default 0; on older versions, 5-25% of the total steps"),
                }),
                Schema.object({}),
            ]),

            Schema.object({
                optimizer_args_custom: Schema.array(String).role('table').description('Custom optimizer_args, one per line'),
            })
        ]),

        LOSS_SETTINGS: Schema.intersect([
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
        ]),

        LOSS_SETTINGS_NO_WEIGHTING: Schema.intersect([
            Schema.object({
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
        ]),

        FULL_PRECISION_MODE: Schema.object({
            full_precision: Schema.union(["none", "full_fp16", "full_bf16"]).default("none").description("Full precision mode. full_fp16 and full_bf16 cannot both be enabled"),
        }).description("Full precision"),

        PREVIEW_IMAGE: Schema.intersect([
            Schema.object({
                enable_preview: Schema.boolean().default(false).description('Generate preview images during training'),
            }).description('Preview images'),

            Schema.union([
                Schema.object({
                    enable_preview: Schema.const(true).required(),
                    randomly_choice_prompt: Schema.boolean().default(false).description('Use a random caption from the dataset as the preview prompt'),
                    prompt_file: Schema.string().role('textarea').description('Preview prompt file. When set, the prompts in the file are used and the options below are ignored.'),
                    positive_prompts: Schema.string().role('textarea').default('masterpiece, best quality, 1girl, solo').description("Prompt"),
                    negative_prompts: Schema.string().role('textarea').default('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry').description("Negative Prompt"),
                    sample_width: Schema.number().default(512).description('Preview width'),
                    sample_height: Schema.number().default(512).description('Preview height'),
                    sample_cfg: Schema.number().min(1).max(30).default(7).description('CFG Scale'),
                    sample_seed: Schema.number().default(2333).description('Seed'),
                    sample_steps: Schema.number().min(1).max(300).default(24).description('Sampling steps'),
                    sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                    sample_every_n_epochs: Schema.number().default(2).description("Generate preview images every N epochs"),
                }),
                Schema.object({}),
            ]),
        ]),

        LOG_SETTINGS: Schema.intersect([
            Schema.object({
                log_with: Schema.union(["tensorboard", "wandb"]).default("tensorboard").description("Logger"),
                log_prefix: Schema.string().description("Log prefix"),
                log_tracker_name: Schema.string().description("Log tracker name"),
                logging_dir: Schema.string().role('filepicker', { type: "folder" }).default("./logs").description("Log folder"),
            }).description('Logging'),

            Schema.union([
                Schema.object({
                    log_with: Schema.const("wandb").required(),
                    wandb_api_key: Schema.string().required().description("wandb API key"),
                }),
                Schema.object({}),
            ]),
        ]),

        NOISE_SETTINGS: Schema.object({
            noise_offset: Schema.number().step(0.01).description("Add a noise offset during training to improve very dark or very bright images; 0.1 recommended if used"),
            multires_noise_iterations: Schema.number().step(1).description("Multi-resolution (pyramid) noise iterations; 6-10 recommended. Cannot be combined with noise_offset"),
            multires_noise_discount: Schema.number().step(0.01).description("Multi-resolution (pyramid) noise discount; 0.3-0.8 recommended. Needs multires_noise_iterations above"),
        }).description("Noise"),

        DATA_ENCHANCEMENT: Schema.object({
            color_aug: Schema.boolean().description("Color augmentation"),
            flip_aug: Schema.boolean().description("Flip augmentation"),
            random_crop: Schema.boolean().description("Random crop"),
        }).description("Data augmentation"),

        OTHER: Schema.object({
            seed: Schema.number().default(1337).description("Random seed"),
            clip_skip: Schema.number().role("slider").min(0).max(12).step(1).default(2).description("CLIP skip (number of final CLIP layers to skip)"),
        }).description("Other"),

        DISTRIBUTED_TRAINING: Schema.object({
            ddp_timeout: Schema.number().min(0).description("Distributed training timeout"),
            ddp_gradient_as_bucket_view: Schema.boolean(),
        }).description("Distributed training"),

    }

    return data
})()
