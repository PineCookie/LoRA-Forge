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

Tested with v0.12.0. Update with `.\run_gui.ps1 update-sd-scripts` (`main` by default, `--branch`
for another branch or tag) and run `uv sync` afterwards; starting the GUI never updates sd-scripts.

## Model files

| Option | File | Notes |
|---|---|---|
| `pretrained_model_name_or_path` | Anima DiT (`.safetensors`) | ComfyUI's file (`net.` prefix) works too |
| `qwen3` | Qwen3-0.6B text encoder | a file, a folder or a Hugging Face repo id; **required** |
| `vae` | Qwen-Image VAE | **required** |
| `llm_adapter_path` | separate LLM adapter | read from the DiT file when empty |

## Options with limits

| Option | |
|---|---|
| `attn_mode` | `torch` (PyTorch SDPA), `xformers` or `flash` (needs flash-attn, see [installing it](../README.md#flash-attn-optional)). Replaces the `xformers` / `sdpa` switches of other models |
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
  - `compile_shapes` sets how different bucket and batch sizes are compiled:
    - **`bounded`** (the default, a LoRA-Forge mode) compiles one graph for all bucket and batch
      sizes; only the token count and batch are dynamic. On a 20-bucket dataset that is 2 graphs
      instead of 28, with the same loss: an epoch was 11% faster with a warm compile cache and 55%
      faster with a cold one (batch 4). It covers buckets up to 2048x2048 px; anything outside is
      compiled separately and the Runs page says so. With `blocks_to_swap` it works like `static`.
    - **`static`**: every bucket size (and every smaller last batch) compiles its own graph, up to
      `compile_cache_size_limit` (64); past that, new sizes run uncompiled. The Runs page shows the
      graph count and warns when the limit is reached.
    - **`auto`** / **`dynamic`**: torch.compile's own dynamic shapes (sd-scripts'
      `compile_dynamic`). They fail on torch 2.14, see [PyTorch #193194](#pytorch-193194) below.
      On Windows, `dynamic` needs the VS 2022 C++ compiler.
  - Because extra bucket sizes cost no compiles with `bounded`, `bucket_reso_steps = 16` (the
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
- **Less VRAM**: `blocks_to_swap`, `gradient_checkpointing`, `unsloth_offload_checkpointing`,
  `cpu_offload_checkpointing`. `blocks_to_swap` and the two offload options exclude each other;
  the GUI clears the conflicting one when you turn one on.
- **Precision**: `bf16` with `full_bf16` is the recommended pair.

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
| `AssertionError: lift_tracked_freevar_to_input should not be called on root SubgraphTracer` on the first step | `compile_shapes` is `dynamic` or `auto`; see below |
| `CUDA driver error: device not ready` on the first step | out of VRAM with `activation_memory_budget`: pick a lower fraction or `off` |
| Lots of `remark: <file>:<line>:0: N instructions in function` | an older triton-windows (3.8.0.post28, [#54](https://github.com/triton-lang/triton-windows/issues/54)); the lock file installs post29, so run `uv sync` |
| Timestep preview shows nothing | its output is the run's console output on the Runs page |

### PyTorch #193194

With `compile_shapes = dynamic` or `auto` (sd-scripts' `compile_dynamic = true` / `auto`), torch
2.14 (and earlier) fails on the first step with the `lift_tracked_freevar_to_input` assertion
above: a Dynamo bug with dynamic shapes and gradient checkpointing
([issue](https://github.com/pytorch/pytorch/issues/193194), fixed by
[PR #193254](https://github.com/pytorch/pytorch/pull/193254), expected in torch 2.15). Use
`bounded` (the default), which gives the benefit of dynamic shapes without triggering it.

## Runtime dependencies

- torch is capped below 2.15 in `pyproject.toml`; triton-windows is 3.8.x (3.8 supports torch
  2.14 and 2.15).
- xformers and flash-attn need wheels built for the installed torch version.
- Updating torch or Triton invalidates the compile caches (`TORCHINDUCTOR_CACHE_DIR`,
  `TRITON_CACHE_DIR`); the next run compiles again once.

## Using the LoRA

- The file records the training settings (`ss_timestep_sampling`, `ss_sigmoid_scale`,
  `ss_discrete_flow_shift`, `ss_weighting_scheme`, ...).
- A DiT-only LoRA (the default, `network_train_unet_only`) loads in ComfyUI as it is. If you also
  trained the Qwen3 text encoder, convert it with sd-scripts' tool: in `scripts/sd-scripts`, run
  `uv run python -m networks.convert_anima_lora_to_comfy <source> <target>`.
- For preview images, CFG 4-5 and 30-50 steps work well.
