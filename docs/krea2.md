# Krea 2 LoRA training

[中文](krea2-zh.md)

Krea 2 LoRAs are trained with [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)
(its `krea2_train_network.py`), which LoRA Forge clones into `scripts/musubi-tuner` on first
start, next to sd-scripts. The page's options are musubi-tuner's; for what they mean, see its
[`docs/krea2.md`](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md). Krea 2
support is marked experimental there.

## Model files

Krea 2 is a 12.9B text-to-image model with the Qwen3-VL-4B text encoder and the Qwen-Image VAE
(the same VAE Anima uses). Download from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)
and [Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL):

| Option | File | Notes |
|---|---|---|
| `pretrained_model_name_or_path` | `krea2_raw_int8_convrot.safetensors` (13.5 GB) | the RAW model, pre-quantized; trains as it is with `convrot_int8` (the default). `krea2_raw_bf16` (26 GB) is needed for `fp8_base` or bf16 training |
| `vae` | `qwen_image_vae.safetensors` | **required** |
| `text_encoder` | `text_encoders/qwen3vl_4b_bf16.safetensors` | **required**: the captions are encoded with it before training; also used for preview images |
| `turbo_lora` | `loras/krea2_turbo_lora_rank_64_bf16.safetensors` | optional, Turbo preview images; works with every memory setting |
| `turbo_dit` | `krea2_turbo_*.safetensors` | optional, Turbo preview images instead of `turbo_lora`; only without `blocks_to_swap` and `convrot_int8`, so not on 16 GB cards |

**Train on RAW, use the LoRA on Turbo.** Turbo is a distilled few-step model without CFG, so
training on it would work against the distillation; musubi-tuner only trains on RAW. The LoRA
loads on either model in ComfyUI.

## How a run works

1. **Caching.** The latents (VAE) and the text encoder outputs are cached by musubi-tuner's two
   caching scripts before training starts; the Runs page shows them as "Caching latents" and
   "Caching text encoder outputs". Cache files are written next to the images, like sd-scripts'
   `.npz` files, and a second run on the same dataset only encodes new images.
2. **Training.** `krea2_train_network.py` with the page's options, through the same monitor as the
   other pages: progress, loss charts, preview images and the saved LoRA files.

The dataset folder has the same layout as for the other models (`<repeats>_<name>` subfolders
with images and `.txt` captions); LoRA Forge writes musubi-tuner's dataset TOML from it. Only
single-GPU training is offered (untested upstream with several GPUs).

## Memory and speed

- **`convrot_int8`** (default on): the frozen base weights stay in int8 and run a real int8
  matmul through Triton, which LoRA Forge installs. Half the VRAM of bf16, and faster than fp8 on
  Krea 2, whose fp8 path dequantizes on every forward. The LoRA itself trains in full precision.
- **`blocks_to_swap`** (default 18): blocks kept in system RAM and moved to the GPU when used.
  With `use_pinned_memory_for_block_swap` and `block_swap_h2d_only` (both on by default) the
  copies run alongside the computation, so swapping costs almost no time even with all 26
  swappable blocks. The default leaves a 16 GB card plenty of VRAM at 1024x1024; raise it for
  larger images or batches. 0 keeps the whole model on the GPU.
- **Pinned RAM** (`use_pinned_memory_for_block_swap`) cannot be paged out: about 0.5 GB per
  swapped block, so 9 GB at the default. Windows' Task Manager shows it as shared GPU memory,
  which is not a sign that the model spills. Before a run starts, LoRA Forge checks that the
  pinned blocks fit in the free RAM and otherwise asks you to swap fewer blocks, turn pinned RAM
  off or close other programs.
- **`gradient_checkpointing`** (default on) is needed at this model size, and
  `block_swap_h2d_only` requires it. Without it Krea 2 needs about 48 GB of VRAM; on a smaller
  card the run spills into system RAM instead of failing.
- **`compile`** (default on): torch.compile of the 28 main blocks; the quantized layers and the
  swapped blocks stay out automatically. The first steps compile.
- **`attn_mode`** (default `auto`): FlashAttention when flash-attn is installed, otherwise
  PyTorch's SDPA. FlashAttention makes each step about a fifth faster on Krea 2 with the same
  results, so install it if you can (see [flash-attn](../README.md#flash-attn-optional)).
  SageAttention is not offered: it has no backward pass, so it cannot train.
- A run that does not fit spills into system RAM on Windows and becomes very slow (minutes per
  step) rather than failing: raise `blocks_to_swap`.

## Training settings

The defaults follow musubi-tuner's Krea 2 example: rank and alpha 32, AdamW8bit at 1e-4,
16 epochs, `timestep_sampling = krea2_shift` (Krea 2's own resolution-dependent timestep shift;
`shift` with `discrete_flow_shift = 2.5` is the fixed equivalent at 1024x1024). The LoRA covers
all 264 Linear layers of the DiT by default; `network_args` can narrow it (see musubi-tuner's
docs). Optimal settings are not established yet upstream.

## Preview images

Previews need `text_encoder`. By default they come from the RAW model (28 steps, CFG 5.5, an
empty negative prompt unless you set one). With `turbo_lora` or `turbo_dit` set they come from
Turbo instead: 8 steps, no CFG, several times faster and closer to how the LoRA is mostly used.
`turbo_lora` is a LoRA that turns RAW into an approximation of Turbo, so it also works with block
swap; `turbo_dit` replaces the base weights, so it needs the whole model on the GPU, and not as int8.

## Updating musubi-tuner

Like sd-scripts, musubi-tuner is cloned at the tested version and never updated automatically:

```powershell
.\run_gui.ps1 update-musubi-tuner           # Windows
```

```sh
bash run_gui.sh update-musubi-tuner         # Linux
```

`--branch <name or tag>` picks another branch or tag.
