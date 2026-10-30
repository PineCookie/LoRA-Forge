# Anima LoRA training

[中文](anima-zh.md)

What to know when training Anima LoRAs with LoRA-Forge: which options take effect, what each one
needs, and the usual traps. For what the options mean, see sd-scripts' own
[`anima_train_network.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md)
and [`anima_torch_compile.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_torch_compile.md).

## sd-scripts version

sd-scripts ignores config keys it does not know, so on an old checkout an option can look set and do
nothing. The GUI checks this for you: the **System info** panel on the home page shows the compatibility of
the installed checkout, and a run that needs a newer one is refused with a message.

| Feature | Needs |
|---|---|
| per-block `torch.compile`, `cuda_allow_tf32`, `cuda_cudnn_benchmark`, `qwen_image_vae_2d`, timestep preview | v0.11.1 |
| per-subfolder timestep offsets, `show_timesteps_offset` | main (after v0.11.1) |
| `block_swap_h2d_only`, `use_pinned_memory_for_block_swap` | v0.12.2 |

Tested with v0.12.3. Update with `.\run_gui.ps1 update-sd-scripts` (`main` by default, `--branch`
for another branch or tag) and run `uv sync` afterwards; starting the GUI never updates sd-scripts.

## Model files

All three are in [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) under
`split_files/`. The page's **Model files** panel links each one and checks the path you enter:
whether the file exists and is the right one (for example not the Qwen-Image 2.1 VAE).

| Option | File | Notes |
|---|---|---|
| `pretrained_model_name_or_path` | `diffusion_models/anima-base-v1.0.safetensors` (4.2 GB) | train on Anima-Base; ComfyUI's file (`net.` prefix) works too |
| `qwen3` | `text_encoders/qwen_3_06b_base.safetensors` (1.2 GB), Qwen3-0.6B | a file, a folder or a Hugging Face repo id; **required** |
| `vae` | `vae/qwen_image_vae.safetensors` (254 MB), the Qwen-Image VAE | **required**; not the Qwen-Image 2.1 VAE |
| `llm_adapter_path` | separate LLM adapter | read from the DiT file when empty |

## Options with limits

| Option | |
|---|---|
| attention | PyTorch's SDPA, which runs on flash kernels since PyTorch 2.15, so the page has no attention option. An older config's `attn_mode = flash` or `xformers` is used only when that package is installed, otherwise `torch` |
| `split_attn` | saves VRAM only; cannot be combined with `compile_fullgraph` |
| `timestep_sampling` | `sigma`, `uniform`, `sigmoid` (default), `shift`, `flux_shift` |
| `discrete_flow_shift` | only used with `timestep_sampling = sigma` or `shift` |
| `weighting_scheme` | Anima only implements `sigma_sqrt` and `cosmap`; anything else is uniform |
| `vae_disable_cache`, `vae_chunk_size` | no effect (cache) or little effect (chunks) with `qwen_image_vae_2d` |
| `network_reg_lrs` | learning rate per module, by regex (`.*self_attn.*=1e-4,.*mlp.*=5e-5`). sd-scripts' `self_attn_lr`, `cross_attn_lr`, `mlp_lr`, `mod_lr` and `llm_adapter_lr` only apply to full fine-tuning, so the LoRA page does not offer them |

## Speed and memory

- **Caching**: `cache_latents` and `cache_text_encoder_outputs` (optionally to disk) are on by
  default. Caching the text encoder outputs turns on `network_train_unet_only`. After changing
  caption settings such as `caption_dropout_rate`, delete the caches so they are rebuilt.
- **2D VAE** (`qwen_image_vae_2d`): converts the VAE's 3D convolutions to equivalent 2D ones; same
  latents, about twice as fast, a third of the peak VRAM. The cache is the same, so switching needs
  no re-caching.
- **`compile`** (on by default): per-block `torch.compile`. The first steps of a run are slow while
  it compiles, later ones clearly faster; the compiled code is cached, so later runs start faster.
  It needs no C++ compiler (only `compile_shapes = dynamic` does). If compiling fails, turn
  `compile` off: training then runs uncompiled.
  - `compile_shapes` sets how different bucket and batch sizes are compiled (sd-scripts'
    `compile_dynamic`):
    - **`auto`** (the default): the first shape gets a graph of its own; from the second one on, the
      sizes that changed become dynamic and share a graph. A few more can follow (a dimension that
      only changes later, a smaller last batch, batch size 1). On a test with 6 bucket sizes: 2 graphs
      instead of 6, the same step time and loss, and 22% less time for the whole run with an empty
      compile cache.
    - **`static`**: every bucket size (and every smaller last batch) compiles its own graph, up to
      `compile_cache_size_limit` (64); past that, new sizes run uncompiled. The Runs page shows the
      graph count and warns when the limit is reached.
    - **`dynamic`**: one graph for all sizes from the start (1 graph in the same test, the same step
      time). On Windows it needs the VS 2022 C++ compiler, which `run_gui.ps1` loads when it finds
      it. If compiling fails with `C1083` (cannot open a compiler-generated file), the compile cache's
      path is too long: set `TORCHINDUCTOR_CACHE_DIR` to a short folder such as `C:\torch_cache`.
  - Because extra bucket sizes cost at most a few compiles with `auto` or `dynamic`, `bucket_reso_steps = 16` (the
    default) is fine: on a 36-image dataset it cut the average crop from 1.32% (step 64) to 0.34%
    for 2 more batches per epoch.
  - `activation_memory_budget` (default `off`) trades spare VRAM for speed in place of gradient
    checkpointing, which runs each block's forward twice: every compiled block keeps up to this
    fraction of its activations and recomputes only the rest; the losses stay the same. Higher is
    faster but needs more VRAM. `auto` picks the fraction when training starts, from the free VRAM
    and the largest batch (its choice is in the console output); with too little free VRAM it
    keeps gradient checkpointing. Whenever a budget is used, gradient checkpointing is off for that
    run; the saved config keeps your `gradient_checkpointing` setting. A fraction compiles once the
    first time it is used. Such runs
    also use PyTorch's expandable memory segments, which stop the memory from fragmenting (not when
    `PYTORCH_CUDA_ALLOC_CONF` is set, with several GPUs or with a `compile_mode` that uses CUDA
    graphs). A run that does not fit then fails at its first step instead of slowing down in system
    RAM, and the Runs page says so: pick a lower fraction. Not used with `blocks_to_swap` or the
    offload options.
  - Not with the older `torch_compile` option.
- **`cuda_allow_tf32`, `cuda_cudnn_benchmark`**: independent of `compile` (Ampere or newer).
- **ProdigyPlusScheduleFree**: LoRA-Forge runs its optimizer step batched, updating all LoRA tensors
  of the same shape together instead of one by one. Same math and same saved state; about 7% less
  time per step, which makes it about as fast as AdamW8bit. Its options `use_bias_correction`,
  `use_adopt`, `use_cautious`, `use_grams`, `use_orthograd`, `use_focus` and `use_speed` keep the
  original step. `LORA_FORGE_PATCHES=0` turns this off with LoRA-Forge's other patches.
- **Less VRAM**: `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`,
  `cpu_offload_checkpointing`. `blocks_to_swap` and the two offload options exclude each other;
  the GUI clears the conflicting one when you turn one on.
- **Precision** (`training_precision`): sets two things at once, the precision of the training
  math (sd-scripts' `mixed_precision`) and of the LoRA weights while they train (`full_bf16` /
  `full_fp16`, else fp32). The saved file uses `save_precision`, independent of this.

  | Choice | Math | LoRA weights | Use |
  |---|---|---|---|
  | `bf16` | bf16 | fp32 | |
  | `full-bf16` (default) | bf16 | bf16 (sd-scripts' `full_bf16`) | The standard (RTX 30 series and newer) |
  | `fp16` | fp16 | fp32 | GPUs without bf16, or for speed with `fp16_accumulation`; a little more VRAM than bf16 |
  | `full-fp16` | fp16 | fp16 (sd-scripts' `full_fp16`) | A little less VRAM than `fp16`; only with AdamW8bit, PagedAdamW8bit, Lion8bit, PagedLion8bit or AdaFactor |
  | `fp32` | fp32 | fp32 | Slowest; debugging |

  With `fp16` and `full-fp16`, **`fp16_accumulation`** (on by default) lets the matrix multiplies add
  up in fp16 too: much faster on GeForce RTX cards (no gain on data-center or RTX PRO cards), still
  about as precise as bf16. It is LoRA-Forge's own key in the exported TOML; sd-scripts never sees it.
  With fp16 weights (`full_fp16`, on every page) the other optimizers fail: AdamW, RAdamScheduleFree
  and CAME turn the weights into NaN, Prodigy, ProdigyPlus and DAdaptation stop learning, so the GUI
  refuses to start those pairs. Lion and SGDNesterov train but lose small updates (a console warning).

## Per-subfolder timestep offsets

Lets subsets train on different noise ranges: **negative values favour low noise (detail),
positive values high noise (structure)**.

- Works with `timestep_sampling = sigmoid`, `shift` or `flux_shift`; not with `sigma` or `uniform`.
- Training only; validation stays unbiased so losses stay comparable.
- Keep to -0.5 .. 0.5. `shift` / `flux_shift` already lean to high noise, so start positive offsets
  at +0.25 or less.
- Set them per `repeats_name` subfolder in the dataset panel; check the distribution with
  **Preview timesteps** first. Offsets need no re-caching.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| A speed option changes nothing | sd-scripts is too old and ignores it; check **System info** on the home page |
| Loss becomes NaN | update PyTorch; Anima needs a recent 2.x |
| A run fails at its first step with an Inductor or Triton error | turn off `compile` and report the error |
| `CUDA driver error: device not ready` on the first step | out of VRAM with `activation_memory_budget`: pick a lower fraction or `off` |
| Timestep preview shows nothing | its output is the run's console output on the Runs page |

### Before PyTorch 2.15

On torch 2.14 and earlier, `compile_dynamic = true` / `auto` failed on the first step with
`AssertionError: lift_tracked_freevar_to_input should not be called on root SubgraphTracer`, a Dynamo
bug with dynamic shapes and gradient checkpointing ([issue](https://github.com/pytorch/pytorch/issues/193194),
fixed by [PR #193254](https://github.com/pytorch/pytorch/pull/193254) in torch 2.15). LoRA Forge had its
own mode for it, `compile_shapes = bounded`, which torch 2.15 rejects; configs that use it now get `auto`.

## Runtime dependencies

- torch 2.15 for CUDA 13.2 (capped below 2.16 in `pyproject.toml`) with triton 3.9. Its own SDPA
  attention runs on flash kernels, so flash-attn and xformers are not installed.
- Updating torch or Triton invalidates the compile caches (`TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`); the next run compiles again once.

## Using the LoRA

- The file records the training settings (`ss_timestep_sampling`, `ss_sigmoid_scale`,
  `ss_discrete_flow_shift`, `ss_weighting_scheme`, ...).
- A DiT-only LoRA (the default, `network_train_unet_only`) loads in ComfyUI as it is. If you also
  trained the Qwen3 text encoder, convert it with sd-scripts' tool: in `scripts/sd-scripts`, run
  `uv run python -m networks.convert_anima_lora_to_comfy <source> <target>`.
- For preview images, CFG 4-5 and 30-50 steps work well.
