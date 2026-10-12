Schema.intersect([
    Schema.intersect([
        Schema.object({
            model_train_type: Schema.union(["sd-dreambooth", "sdxl-finetune"]).default("sd-dreambooth").description("Training type"),
            pretrained_model_name_or_path: Schema.string().role("filepicker", { type: "model-file" }).description("Base model file"),
            resume: Schema.string().role("filepicker", { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
            vae: Schema.string().role("filepicker", { type: "model-file" }).description("(Optional) VAE model file; replaces the VAE inside the base model"),
        }).description("Model"),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-dreambooth"),
                v2: Schema.boolean().default(false).description("Enable for base models built on SD 2.0 or later"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-dreambooth").required(),
                v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
                scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
            }),
            Schema.object({
                model_train_type: Schema.const("sdxl-finetune").required(),
                v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
                scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
            }),
            Schema.object({}),
        ]),
    ]),

    Schema.object({
        train_data_dir: Schema.string().role("filepicker", { type: "folder" }).default("./train/aki").description("Training dataset folder"),
        reg_data_dir: Schema.string().role("filepicker", { type: "folder" }).description("Regularization dataset folder. Leave empty to train without regularization images"),
        prior_loss_weight: Schema.number().step(0.1).description("Regularization: prior loss weight"),
        resolution: Schema.string().default("512,512").description("Training resolution, width x height. Non-square works, but both must be multiples of 64."),
        enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
        min_bucket_reso: Schema.number().default(256).description("Minimum ARB bucket resolution"),
        max_bucket_reso: Schema.number().default(1024).description("Maximum ARB bucket resolution"),
        bucket_reso_steps: Schema.number().default(64).description("ARB bucket resolution step; SDXL can use 32"),
    }).description("Dataset"),

    Schema.object({
        output_name: Schema.string().default("model_name").description("Output name"),
        output_dir: Schema.string().role("filepicker", { type: "folder" }).default("./output").description("Output folder"),
        save_model_as: Schema.union(["safetensors", "pt", "ckpt"]).default("safetensors").description("Save format"),
        save_precision: Schema.union(["fp16", "float", "bf16"]).default("fp16").description("Save precision"),
        save_every_n_epochs: Schema.number().default(2).description("Save the model every N epochs"),
        save_state: Schema.boolean().description("Save the training state; `resume` can continue training from it"),
    }).description("Saving"),

    Schema.object({
        max_train_epochs: Schema.number().min(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).default(1).description("Batch size"),
        stop_text_encoder_training: Schema.number().min(-1).description("sd-dreambooth only. Stop training the text encoder at step N; -1 does not train the text encoder"),
        gradient_checkpointing: Schema.boolean().default(false).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).description("Gradient accumulation steps"),
    }).description("Training"),


    Schema.intersect([
        Schema.object({
            optimizer_type: Schema.union([
                "AdamW",
                "AdamW8bit",
                "PagedAdamW8bit",
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
                "prodigyplus.ProdigyPlusScheduleFree"
            ]).default("AdamW8bit").description("Optimizer"),
        }),

        Schema.object({
            learning_rate: Schema.string().default("1e-6").description("Learning rate"),
        }).description("Learning rate and optimizer"),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sd-dreambooth"),
                learning_rate_te: Schema.string().default("5e-7").description("Text encoder learning rate"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                model_train_type: Schema.const("sdxl-finetune").required(),
                learning_rate_te1: Schema.string().default("5e-7").description("SDXL text encoder 1 (ViT-L) learning rate"),
                learning_rate_te2: Schema.string().default("5e-7").description("SDXL text encoder 2 (BiG-G) learning rate"),
            }),
            Schema.object({}),
        ]),

        Schema.object({
            lr_scheduler: Schema.union([
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]).default("cosine_with_restarts").description("Learning rate scheduler"),
            lr_warmup_steps: Schema.number().default(0).description("Learning rate warmup steps"),
        }),

        Schema.object({
            min_snr_gamma: Schema.number().step(0.1).description("Min-SNR gamma; 5 recommended. Similar in effect to Debiased Estimation loss, so avoid enabling both"),
            debiased_estimation_loss: Schema.boolean().default(false).description("Use Debiased Estimation loss. Similar in effect to Min-SNR, so avoid enabling both"),
            loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("Loss function"),
        }).description("Loss"),

        Schema.union([
            Schema.object({
                loss_type: Schema.const("huber").required(),
                huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber loss schedule; only used when loss_type is huber or smooth_l1"),
                huber_c: Schema.number().step(0.001).default(0.1).description("Huber loss decay parameter; only used when loss_type is huber or smooth_l1"),
            }),
            Schema.object({
                loss_type: Schema.const("smooth_l1").required(),
                huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber loss schedule; only used when loss_type is huber or smooth_l1"),
                huber_c: Schema.number().step(0.001).default(0.1).description("Huber loss decay parameter; only used when loss_type is huber or smooth_l1"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                lr_scheduler: Schema.const("cosine_with_restarts"),
                lr_scheduler_num_cycles: Schema.number().default(1).description("Number of restarts"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                optimizer_type: Schema.const("Prodigy").required(),
                prodigy_d0: Schema.string().description("Prodigy d0: the initial LR estimate; try 1e-7, 1e-6 or 1e-5"),
                prodigy_d_coef: Schema.string().default("2.0").description("Prodigy d_coef. Default 2.0"),
            }),
            Schema.object({
                optimizer_type: Schema.const("prodigyplus.ProdigyPlusScheduleFree").required(),
                prodigyplus_d_coef: Schema.string().default("1.0").description("Prodigy Plus d_coef. Official default 1.0; try 2 or higher to help the LR grow"),
                prodigyplus_betas: Schema.string().default("(0.9, 0.99)").description("Prodigy Plus betas as a Python tuple, e.g. (0.95, 0.99)"),
                prodigyplus_schedulefree_c: Schema.string().default("0").description("Prodigy Plus schedulefree_c. Official default 0"),
                prodigyplus_prodigy_steps: Schema.string().default("0").description("Prodigy Plus prodigy_steps. Official default 0; on older versions, 5-25% of the total steps"),
            }),
            Schema.object({}),
        ]),

        Schema.object({
            optimizer_args_custom: Schema.array(String).role("table").description("Custom optimizer_args, one per line"),
        })
    ]),

    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("Generate preview images during training"),
        }).description("Preview images"),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                sample_prompts: Schema.string().role("textarea").default("(masterpiece, best quality:1.2), 1girl, solo, --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry,  --w 512  --h 768  --l 7  --s 24  --d 1337").description("Preview prompt and options. `--n` starts the negative prompt<br>`--w` width, `--h` height<br>`--l`: CFG scale<br>`--s`: steps<br>`--d`: seed"),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                sample_every_n_epochs: Schema.number().default(2).description("Generate preview images every N epochs"),
            }),
            Schema.object({}),
        ]),
    ]),

    Schema.intersect([
        Schema.object({
            log_with: Schema.union(["tensorboard", "wandb"]).default("tensorboard").description("Logger"),
            log_prefix: Schema.string().description("Log prefix"),
            log_tracker_name: Schema.string().description("Log tracker name"),
            logging_dir: Schema.string().role("filepicker", { type: "folder" }).default("./logs").description("Log folder"),
        }).description("Logging"),

        Schema.union([
            Schema.object({
                log_with: Schema.const("wandb").required(),
                wandb_api_key: Schema.string().required().description("wandb API key"),
            }),
            Schema.object({}),
        ]),
    ]),

    Schema.object({
        caption_extension: Schema.string().default(".txt").description("Caption file extension"),
        shuffle_caption: Schema.boolean().default(true).description("Shuffle caption tokens during training"),
        weighted_captions: Schema.boolean().default(false).description("Use weighted tokens; not recommended together with shuffle_caption"),
        keep_tokens: Schema.number().min(0).max(255).step(1).default(0).description("Keep the first N tokens in place when shuffling"),
        keep_tokens_separator: Schema.string().description("Separator that ends the kept tokens"),
        max_token_length: Schema.number().default(255).description("Max token length"),
        caption_dropout_rate: Schema.number().min(0).max(1).step(0.1).description("Probability of dropping the whole caption (the image is trained without caption or class token)"),
        caption_dropout_every_n_epochs: Schema.number().min(0).max(100).step(1).description("Drop all captions every N epochs"),
        caption_tag_dropout_rate: Schema.number().min(0).max(1).step(0.1).description("Probability of dropping each comma-separated tag"),
    }).description("Captions"),

    Schema.object({
        noise_offset: Schema.number().step(0.0001).description("Add a noise offset during training to improve very dark or very bright images; 0.1 recommended if used"),
        multires_noise_iterations: Schema.number().step(1).description("Multi-resolution (pyramid) noise iterations; 6-10 recommended. Cannot be combined with noise_offset"),
        multires_noise_discount: Schema.number().step(0.1).description("Multi-resolution (pyramid) noise discount; 0.3-0.8 recommended. Needs multires_noise_iterations above"),
    }).description("Noise"),

    Schema.object({
        seed: Schema.number().default(1337).description("Random seed"),
        clip_skip: Schema.number().role("slider").min(0).max(12).step(1).default(2).description("CLIP skip (number of final CLIP layers to skip)"),
        no_token_padding: Schema.boolean().default(false).description("Disable token padding (same as the old Diffusers Dreambooth script)"),
    }).description("Advanced"),

    Schema.object({
        full_precision: Schema.union(["none", "full_fp16", "full_bf16"]).default("none").description("Full precision mode. full_fp16 and full_bf16 cannot both be enabled"),
    }).description("Full precision"),

    Schema.object({
        mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("fp16").description("Mixed precision"),
        xformers: Schema.boolean().default(false).description("Enable xformers"),
        sdpa: Schema.boolean().description("Enable SDPA"),
        lowram: Schema.boolean().default(false).description("Low-RAM mode: loads the U-Net, text encoder and VAE directly into VRAM"),
        cache_latents: Schema.boolean().default(true).description("Cache image latents"),
        cache_latents_to_disk: Schema.boolean().default(true).description("Cache image latents to disk"),
        persistent_data_loader_workers: Schema.boolean().default(true).description("Keep the dataloader workers alive to shorten the pause between epochs."),
        vae_batch_size: Schema.number().min(1).description("VAE encoding batch size"),
    }).description("Speed and memory"),

    Schema.object({
        ddp_timeout: Schema.number().min(0).description("Distributed training timeout"),
        ddp_gradient_as_bucket_view: Schema.boolean(),
    }).description("Distributed training"),
]);
