# MOSS-Image LoRA training

[中文](moss-image-zh.md)

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) (OpenMOSS, Apache-2.0) is a 5B text-to-image
model with strong text rendering and layout control. No upstream trainer supports it yet, so LoRA
Forge trains it with its own trainer (`lora_forge/trainers/moss_image`), which follows the recipe
in the [MOSS-Image technical report](https://github.com/OpenMOSS/MOSS-Image/tree/main/paper): the
same loss, the same timestep sampling, and the text encoder frozen.

## Model files

| Option | File | Notes |
|---|---|---|
| `pretrained_model_name_or_path` | `diffusion_models/moss_image_base_bf16.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) (9.5 GB) | the Base model; OpenMOSS's `transformer` folder from [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) also works. The int8 files cannot train |
| `text_encoder` | `text_encoders/qwen3.5_9b_bf16.safetensors` from [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5) (19.3 GB) | **required**, the bf16 file: the captions are encoded with it before training, and the preview prompts too |
| `vae` | `vae/diffusion_pytorch_model.safetensors` from [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main/vae) (250 MB) | **required**. This is the VAE the model was trained with. ComfyUI's `flux2-vae.safetensors` also works; its latents differ by less than 1% |

The tokenizer (a few MB) is downloaded from Hugging Face (`Qwen/Qwen3.5-9B`) on the first run.

**Train on Base, generate with either.** MOSS-Image-SFT is a light fine-tune of Base (its
weights differ by about 1%), and a LoRA trained on Base works on both. OpenMOSS's post-training
advice is written for Base. In our tests SFT gave nicer images with the same LoRA.

## Captions

MOSS-Image was trained on natural-language descriptions, so write sentences, not tag lists.
Visible text is rendered as written in the caption.

**Each line of a caption file is one caption of its image.** Each step trains on one of them at
random, as OpenMOSS trained on short, medium and long captions of the same image. For a character
this matters: if every caption describes the full look, the model learns to draw that description
and the name alone brings back little of it. Add a short line that only names the character and
what changes in that image:

```
An anime-style illustration of my_character from the chest up in her school uniform: a navy blazer over a white shirt with a striped tie, ... Plain white background.
my_character in her school uniform, smiling at the viewer.
```

A one-line caption file trains as before. OpenMOSS's
[user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) has the captioning
prompt they recommend for vision models (long, medium and short captions).

## How a run works

1. **Caching.** The latents (VAE) and the text-encoder outputs are cached before training; the
   Runs page shows them as "Caching latents" and "Caching text encoder outputs". Cache files are
   written next to the images (`*_moss_latent.safetensors`, `*_moss_te.safetensors`) and record
   what they were made from: a changed image, caption or model file is encoded again. The text
   encoder needs about 20 GB of free RAM; the part that does not fit in VRAM runs from RAM.
2. **Training** at batch 1: each image keeps its own caption length, as in OpenMOSS's pipeline.
   `gradient_accumulation_steps` gives larger batches. Each image gets a bucket of its own aspect
   ratio at the training resolution's pixel count, with sides in multiples of 16.
3. **Preview images** use OpenMOSS's sampler: Euler with shift 5 and CFG, block 22 skipped in the
   negative pass.

The dataset folder has the same layout as for the other models (`<repeats>_<name>` subfolders
with images and `.txt` captions). Only single-GPU training is offered.

## Memory and speed

On an RTX 5070 Ti (16 GB) at about 1 megapixel, rank 32: **2.1 s per step** with the defaults,
**1.5 s** with `training_precision = fp16`, 2.9 s without flash-attn. The run peaks at about
12-13 GB of VRAM.

- **`training_precision`** (default `bf16`): `fp16` runs the block matrix multiplies in fp16 with
  loss scaling, and with `fp16_accumulation` (on by default for `fp16`) they also add up in fp16,
  which GeForce RTX cards run about twice as fast: a quarter less time per step. Its gradients were
  as close to an fp32 reference as bf16's, and no step was skipped for overflow in our runs. The LoRA
  weights train in fp32 either way. Opt-in until more full trainings have compared it with bf16.
- **`gradient_checkpointing`** (default on): keep it on with 16 GB; the DiT alone takes about 10 GB.
- **`compile`** (default on): per-block torch.compile, about 30% faster. It builds a few graphs
  for all bucket sizes; the first steps are slower while it compiles.
- **`attn_mode`** (default `auto`): FlashAttention when [flash-attn](../README.md#flash-attn-optional)
  is installed, otherwise PyTorch's SDPA.

## Training settings

The defaults: rank and alpha 32 on the attention and feed-forward layers of all 42 blocks (about
74M parameters), AdamW at 1e-4, 10 epochs, and the report's timestep sampling (`logit_mean`
-1.09, `logit_std` 0.95). A character test LoRA (28 images, 545 steps) learned the outfit details by
epoch 2. Optimal settings are not established yet.

## Using the LoRA in ComfyUI

ComfyUI supports MOSS-Image natively (update it first); load the LoRA with *LoraLoaderModelOnly*.
OpenMOSS's sampling settings: 50 steps, CFG 3, euler / sgm_uniform, shift 5 (ComfyUI's default
for MOSS-Image), an empty negative prompt, and block 22 skipped in the negative pass: add
*SkipLayerGuidanceDiTSimple* with `single_layers` = `22` and `double_layers` empty. ComfyUI's own
example workflow uses 25 steps, CFG 2.5 and er_sde / simple, with the skip for the first half
of the steps only (`end_percent` 0.5).
