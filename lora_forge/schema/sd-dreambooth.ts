Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("sd-dreambooth").disabled().description("Training type"),
        pretrained_model_name_or_path: Schema.string().role("filepicker", { type: "model-file" }).description("Base model file"),
        resume: Schema.string().role("filepicker", { type: "folder" }).description("Resume training from a state saved with `save_state`: enter its folder"),
        vae: Schema.string().role("filepicker", { type: "model-file" }).description("(Optional) VAE model file; replaces the VAE inside the base model"),
        v2: Schema.boolean().default(false).description("Enable for base models built on SD 2.0 or later"),
        v_parameterization: Schema.boolean().default(false).description("V-pred models need v-parameterization, e.g. SD2 768-v, NoobXL V-Pred, or models fine-tuned from a V-pred model"),
        scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("V-pred models only: scale the v-prediction loss like the noise prediction loss"),
    }),

    Schema.object({
        train_data_dir: Schema.string().role("filepicker", { type: "folder" }).default("./train/aki").description("Training dataset folder"),
        reg_data_dir: Schema.string().role("filepicker", { type: "folder" }).description("Regularization dataset folder. Leave empty to train without regularization images"),
        prior_loss_weight: Schema.number().min(0).step(0.1).description("Regularization: prior loss weight"),
        resolution: Schema.string().default("512,512").description("Training resolution, width x height. Non-square works, but both must be multiples of 64."),
        enable_bucket: Schema.boolean().default(true).description("Enable ARB buckets so images can have different aspect ratios"),
        min_bucket_reso: Schema.number().min(64).step(64).default(256).description("Shortest bucket side. Only used with bucket_no_upscale off, when images are fitted to a fixed set of buckets of about the training resolution's pixel count; images more stretched than min_bucket_reso and max_bucket_reso allow are cropped"),
        max_bucket_reso: Schema.number().min(64).step(64).default(1024).description("Longest bucket side. Only used with bucket_no_upscale off; buckets keep about the training resolution's pixel count, so this only limits very wide or tall images"),
        bucket_reso_steps: Schema.number().min(64).step(64).default(64).description("ARB bucket resolution step: images are fitted to sizes in steps of this many pixels, so smaller steps crop less. SD1.5 needs multiples of 64"),
        bucket_no_upscale: Schema.boolean().default(false).description("Give each image a bucket of its own aspect ratio: larger images are scaled down to the training resolution's pixel count, smaller ones keep their size. min_bucket_reso and max_bucket_reso are then ignored"),
    }),

    Schema.object({
        output_name: Schema.string().default("model_name").description("Output name"),
        output_dir: Schema.string().role("filepicker", { type: "folder" }).default("./output").description("Output folder"),
        save_model_as: Schema.union(["safetensors", "pt", "ckpt"]).default("safetensors").description("Save format"),
        save_precision: Schema.union(["fp16", "float", "bf16"]).default("fp16").description("Save precision"),
        save_every_n_epochs: Schema.number().min(1).step(1).default(2).description("Save the model every N epochs"),
        save_state: Schema.boolean().description("Save the training state; `resume` can continue training from it"),
    }),

    Schema.object({
        max_train_epochs: Schema.number().min(1).step(1).default(10).description("Maximum training epochs"),
        train_batch_size: Schema.number().min(1).step(1).default(1).description("Batch size"),
        stop_text_encoder_training: Schema.number().min(-1).step(1).description("Stop training the text encoder at step N; -1 does not train the text encoder"),
        gradient_checkpointing: Schema.boolean().default(false).description("Gradient checkpointing"),
        gradient_accumulation_steps: Schema.number().min(1).step(1).description("Gradient accumulation steps"),
    }),

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
            learning_rate: Schema.number().min(0).default("1e-6").description("Learning rate"),
            learning_rate_te: Schema.number().min(0).default("5e-7").description("Text encoder learning rate"),
        }),

        Schema.object({
            lr_scheduler: Schema.union([
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
                "warmup_stable_decay",
            ]).default("cosine_with_restarts").description("Learning rate scheduler"),
            lr_warmup_steps: Schema.number().min(0).default(0).description("Learning rate warmup: a whole number of steps (100 = the first 100 steps), or below 1 a fraction of all steps (0.05 = the first 5%, which scales with the run's length). 0 = off. `constant` has no warmup; use `constant_with_warmup`"),
        }),

        Schema.object({
            min_snr_gamma: Schema.number().min(0).step(0.1).description("Min-SNR gamma; 5 recommended. Similar in effect to Debiased Estimation loss, so avoid enabling both"),
            debiased_estimation_loss: Schema.boolean().default(false).description("Use Debiased Estimation loss. Similar in effect to Min-SNR, so avoid enabling both"),
            loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("Loss function"),
        }),

        Schema.union([
            Schema.object({
                loss_type: Schema.const("huber").required(),
                huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber loss schedule; only used when loss_type is huber or smooth_l1"),
                huber_c: Schema.number().min(0).step(0.001).default(0.1).description("Huber loss decay parameter; only used when loss_type is huber or smooth_l1"),
            }),
            Schema.object({
                loss_type: Schema.const("smooth_l1").required(),
                huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber loss schedule; only used when loss_type is huber or smooth_l1"),
                huber_c: Schema.number().min(0).step(0.001).default(0.1).description("Huber loss decay parameter; only used when loss_type is huber or smooth_l1"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                lr_scheduler: Schema.const("cosine_with_restarts"),
                lr_scheduler_num_cycles: Schema.number().min(1).step(1).default(1).description("Number of restarts"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                lr_scheduler: Schema.const("warmup_stable_decay"),
                lr_decay_steps: Schema.number().min(0).default(0.2).description("Length of the decay at the end of `warmup_stable_decay`: a whole number of steps, or below 1 a fraction of all steps (0.2 = the last 20%). Between the warmup and the decay the learning rate stays at its full value"),
            }),
            Schema.object({}),
        ]),

        Schema.union([
            Schema.object({
                optimizer_type: Schema.const("Prodigy").required(),
                prodigy_d0: Schema.number().min(0).description("Prodigy d0: the initial LR estimate; try 1e-7, 1e-6 or 1e-5"),
                prodigy_d_coef: Schema.number().min(0).step(0.1).default("2.0").description("Prodigy d_coef. Default 2.0"),
            }),
            Schema.object({
                optimizer_type: Schema.const("prodigyplus.ProdigyPlusScheduleFree").required(),
                prodigyplus_d_coef: Schema.number().min(0).step(0.1).default("1.0").description("Prodigy Plus d_coef. Official default 1.0; try 2 or higher to help the LR grow"),
                prodigyplus_betas: Schema.string().default("(0.9, 0.99)").description("Prodigy Plus betas as a Python tuple, e.g. (0.95, 0.99)"),
                prodigyplus_schedulefree_c: Schema.number().min(0).default("0").description("Prodigy Plus schedulefree_c. Official default 0"),
                prodigyplus_prodigy_steps: Schema.number().min(0).step(1).default("0").description("Prodigy Plus prodigy_steps. Official default 0; on older versions, 5-25% of the total steps"),
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
        }),

        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                sample_prompts: Schema.string().role("textarea").default("(masterpiece, best quality:1.2), 1girl, solo, --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry,  --w 512  --h 768  --l 7  --s 24  --d 1337").description("Preview prompt and options. `--n` starts the negative prompt<br>`--w` width, `--h` height<br>`--l`: CFG scale<br>`--s`: steps<br>`--d`: seed"),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("Sampler for preview images"),
                sample_every_n_epochs: Schema.number().min(1).step(1).default(2).description("Generate preview images every N epochs"),
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
        }),

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
        max_token_length: Schema.union([75, 150, 225]).default(225).description("Longest caption in CLIP tokens; longer captions are cut off"),
        caption_dropout_rate: Schema.number().min(0).max(1).step(0.1).description("Probability of dropping the whole caption (the image is trained without caption or class token)"),
        caption_dropout_every_n_epochs: Schema.number().min(0).max(100).step(1).description("Drop all captions every N epochs"),
        caption_tag_dropout_rate: Schema.number().min(0).max(1).step(0.1).description("Probability of dropping each comma-separated tag"),
    }),

    Schema.object({
        noise_offset: Schema.number().min(0).step(0.0001).description("Add a noise offset during training to improve very dark or very bright images; 0.1 recommended if used"),
        multires_noise_iterations: Schema.number().min(0).step(1).description("Multi-resolution (pyramid) noise iterations; 6-10 recommended. Cannot be combined with noise_offset"),
        multires_noise_discount: Schema.number().min(0).max(1).step(0.1).description("Multi-resolution (pyramid) noise discount; 0.3-0.8 recommended. Needs multires_noise_iterations above"),
    }),

    Schema.object({
        seed: Schema.number().min(0).step(1).default(1337).description("Random seed"),
        clip_skip: Schema.number().role("slider").min(0).max(12).step(1).default(2).description("CLIP skip (number of final CLIP layers to skip)"),
        no_token_padding: Schema.boolean().default(false).description("Disable token padding (same as the old Diffusers Dreambooth script)"),
    }),

    Schema.object({
        // fp16 as before (this page's old mixed_precision default); fp16 accumulation is untested on SD1.5
        // full fine-tunes, so it stays off unless chosen
        ...UpdateSchema(SHARED_SCHEMAS.RAW.TRAINING_PRECISION, {
            training_precision: SHARED_SCHEMAS.RAW.TRAINING_PRECISION.training_precision.default("fp16"),
            fp16_accumulation: SHARED_SCHEMAS.RAW.TRAINING_PRECISION.fp16_accumulation.default(false),
        }),
        xformers: Schema.boolean().default(false).description("Enable xformers"),
        sdpa: Schema.boolean().default(true).description("PyTorch's SDPA attention: much faster and less VRAM than sd-scripts' plain attention, which runs when neither SDPA nor xformers is on"),
        lowram: Schema.boolean().default(false).description("Low-RAM mode: loads the U-Net, text encoder and VAE directly into VRAM"),
        cache_latents: Schema.boolean().default(true).description("Cache image latents"),
        cache_latents_to_disk: Schema.boolean().default(true).description("Cache image latents to disk"),
        persistent_data_loader_workers: Schema.boolean().default(true).description("Keep the dataloader workers alive to shorten the pause between epochs."),
        max_data_loader_n_workers: Schema.number().min(0).step(1).description("Processes that prepare the batches. Empty: chosen when training starts, none when the latents are cached (a batch is then a few small file reads) and 2 otherwise. Each one takes seconds to start; sd-scripts' own default is 8"),
        vae_batch_size: Schema.number().min(1).description("VAE encoding batch size"),
    }),

    Schema.object({
        ddp_timeout: Schema.number().min(0).step(1).description("Distributed training timeout in minutes"),
        ddp_gradient_as_bucket_view: Schema.boolean().description("Pass gradient_as_bucket_view to DDP: saves some VRAM when training on several GPUs"),
    }),
]);
