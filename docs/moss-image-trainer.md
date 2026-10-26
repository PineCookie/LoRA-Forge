# The MOSS-Image trainer

For contributors. The user guide is [moss-image.md](moss-image.md).

MOSS-Image has no upstream trainer, so LoRA Forge trains it with its own code in
`lora_forge/trainers/moss_image/`. It follows the
[MOSS-Image technical report](https://github.com/OpenMOSS/MOSS-Image/tree/main/paper) and OpenMOSS's
inference pipeline (`pipeline_moss_image.py` in
[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), Apache-2.0), from which the
model code is ported. Any change that touches the numbers below needs a check against those sources
and a training run (see [Checking a change](#checking-a-change)).

## Files

| File | Contents |
|---|---|
| `moss_train_network.py` | The entry script: reads the TOML, caches, builds the model, runs the loop, saves and samples |
| `moss_model.py` | The DiT (port of `MOSSImageTransformer`), its loader, `Fp16Linear` |
| `moss_lora.py` | LoRA on the block Linears, saving and loading |
| `moss_data.py` | Dataset scan, buckets, the VAE and text-encoder loaders, both caches |
| `moss_sampling.py` | Training timestep sampling, the preview sampler, prompt-line parsing |
| `moss_swap.py` | Block swap (`blocks_to_swap`): some blocks' frozen weights in RAM, copied to the GPU when used |
| `configs/` | OpenMOSS's text-encoder and VAE configs (only the shapes; weights come from the user's files) |

Around it: `lora_forge/schema/moss-image-lora.ts` (the page), `lora_forge/training/moss_image.py` (what
`prepare_training` does for this page: file checks, the preview prompt file, dropping other pages'
keys; also `check_model_file` for the page's Model files panel) and `frontend/src/config/modelFiles.ts`
(the download links).

## How a run starts

The GUI writes a flat TOML and launches it like the sd-scripts trainers:
`accelerate launch run_entry.py moss_train_network.py --config_file <toml>`. `run_entry.py` puts the
trainer's folder first on `sys.path` (the modules import each other by name, never `lora_forge`) and
installs its observers, so the Runs page works without anything MOSS-specific:

- progress: tqdm bars named `caching latents`, `caching text encoder outputs`, `steps` and `Sampling`;
- epochs: the `epoch N/M` line printed through `Accelerator.print`;
- metrics: `Accelerator.log` with sd-scripts' names (`loss/current`, `loss/average`,
  `loss/epoch_average`, `lr/unet`, and `lr/d*lr/unet`, `lr/d*eff_lr/unet` for Prodigy-type optimizers).
  The trainer uses accelerate only for logging (TensorBoard / wandb through `log_with`); it does not
  wrap the model.

`run_entry.py`'s batched ProdigyPlusScheduleFree step applies here too.

## The recipe

| | Value | Source |
|---|---|---|
| Prediction target | velocity `noise - x0`, from `x_t = (1 - σ) x0 + σ noise` | report Eq. 3 |
| Loss | MSE over the image tokens; text tokens are not in the loss | report Eq. 3 |
| Timestep input | `1000 σ` | report, pipeline |
| σ sampling | `S = 1 - sigmoid(Z)`, `Z ~ N(logit_mean, logit_std²)` (-1.09, 0.95), truncated at 0.95 and rescaled onto the 1000-step grid, mixed with 1% uniform | report Eq. 4, final pretraining interval |
| Text | Qwen3.5-9B's last hidden state for the chat template with an open think block (below), each caption at its own length (no padding) | pipeline; ComfyUI uses the same template |
| Latents | FLUX.2 VAE latent mean, 2×2 patchified to 128 channels, normalized with the VAE's batch-norm statistics | pipeline (inverse of its decode) |
| Images | one bucket per image: its aspect ratio at the resolution's pixel count, sides multiples of 16, resized to cover and center-cropped | |
| Batch | 1; `gradient_accumulation_steps` for larger batches | the pipeline runs one sample at a time |
| Previews | OpenMOSS's scheduler (shift 5 snapped to the training grid, Euler), CFG, block 22 skipped in the negative pass | pipeline defaults |

The chat template, with the caption stripped of surrounding whitespace:

```
<|im_start|>user
{caption}<|im_end|>
<|im_start|>assistant
<think>
```

The text encoder and the VAE are frozen, as OpenMOSS recommends. The LoRA covers `q`, `k`, `v`, `o`,
`ffn_up_gate` and `ffn_down` of all 42 blocks; the text-refiner blocks, the modulation and the heads
are not trained.

## Model files

- **DiT**: ComfyUI's single file stores q / k / v fused as `layers.N.proj.qkv`; `load_dit` splits it
  into OpenMOSS's separate projections. OpenMOSS's sharded `transformer` folder loads as it is.
  Quantized files (int8, fp8) are refused before the run starts.
- **Text encoder**: only the language model is built (`Qwen3_5TextModel`; for text-only input its
  positions equal the full model's). All weights are loaded **in bf16**, including the linear-attention
  `A_log` and gated norm weights the file stores in fp32: ComfyUI and OpenMOSS's pipeline round them
  too, so the LoRA trains on the conditioning it gets at inference. Keeping the norm weights in fp32
  moved the outputs by about 1.6% away from ComfyUI's (`A_log` makes no difference). The layers that do not fit in VRAM are
  dispatched to RAM by accelerate. The tokenizer comes from Hugging Face (`Qwen/Qwen3.5-9B`, identical
  to OpenMOSS's copy).
- **VAE**: OpenMOSS's file (diffusers names) or ComfyUI's `flux2-vae` (FLUX.2 names, converted at
  load). Their encoders differ slightly: ComfyUI's gives latents within about 1% of OpenMOSS's. Only
  the decoder width differs otherwise, and it is read from the file.

## Saved LoRA

Keys `lora_unet_layers_{i}_proj_{name}.lora_down.weight` / `.lora_up.weight` / `.alpha`, plus `ss_*`
metadata like sd-scripts'. ComfyUI maps `q` / `k` / `v` onto slices of its fused `qkv`
(`comfy/lora.py`), so the files load in ComfyUI as they are. `network_weights` continues from such a
file, adjusting for a different alpha.

## Caches

Written next to each image: `<image>_moss_latent.safetensors` and `<image>_moss_te.safetensors`. Their
metadata records what they were made from, and a cache whose metadata does not match is encoded again:

- latents: `format`, the bucket size, the image file (name, size, modification time), the VAE file;
- text: `format`, the caption variants (JSON list), the text-encoder file. The variants are stored as
  `hidden_0`, `hidden_1`, ...

Raise `CACHE_FORMAT` in `moss_data.py` whenever a change makes existing caches wrong.

Captions: every non-empty line of the caption file is one variant, and each step trains on one at
random (`read_captions`).

## Precision and speed

The DiT is bf16; the LoRA weights are fp32. With `mixed_precision = fp16` (the page's
`training_precision = fp16`) the six block Linears become `Fp16Linear`: the frozen bf16 weights are
converted to fp16 (exactly) and the matmuls run in fp16, with fp16 accumulation when
`fp16_accumulation` is on. Their inputs are normalized or capped (SiTU-GLU), so they stay far inside
fp16's range. The gradients need loss scaling (a `GradScaler`): without it they underflow, and the
attention LoRA got about 6× too little gradient.

Measurements (a 16 GB card, 1 MP, rank 32, batch 1):

| | s/step |
|---|---|
| SDPA, no compile | 2.93 |
| flash-attn + per-block compile (default) | 2.10 |
| + fp16 with fp16 accumulation | 1.54 |

Matmuls take about 68% of the GPU time and flash attention 23%. Against an fp32 reference computed on
the CPU, the LoRA gradients of bf16, fp16 with fp32 accumulation and fp16 with fp16 accumulation were
equally close (the bf16 residual stream they share limits all three), and the loss differed by
0.2-0.4% in each.

Per-block `torch.compile` compiles `Block.forward` once (`model.block_fn`), called inside non-reentrant
gradient checkpointing; it builds 4 graphs for all bucket sizes.

## Block swap

`moss_swap.BlockSwap` follows sd-scripts' / Musubi Tuner's H2D-only block swap (`LoRAStreamOffloader`,
Apache-2.0). The streaming blocks are spread evenly over the 42; the frozen Linear weights of each
(`nn.Linear.weight`, or the `Fp16Linear` buffer) are packed into one flat CPU buffer, pinned by default,
and copied into a ring of two GPU buffers on a side stream, never back. The LoRA weights are not part of
it and stay on the GPU with everything else. The model's block loop drives it: `before` waits (on the
GPU) for a block's copy, `after_forward` frees its slot and prefetches two streaming blocks ahead, and an
identity autograd function on each block's input (`hook_input`) runs `after_backward` once the block's
backward pass is done, which prefetches backwards. sd-scripts' module backward hooks would not fire here,
because the compiled `Block.forward` is called as a function. A slot is overwritten as soon as its block is
done, so gradient checkpointing is required: the recompute reads the weights again. With swapping, the
DiT loads into RAM, the LoRA is attached there, and `place()` then moves everything else to the GPU.

Checked: with PyTorch's deterministic math attention, 20 / 30 / 40 swapped blocks, pinned or not, give
bit-identical LoRA gradients and outputs (flash attention's backward is not deterministic, about 0.5%
between any two runs). Compile keeps working (2 graphs). A 16 GB card, 1 MP, through the GUI path:

| blocks_to_swap | s/step | VRAM (nvidia-smi) |
|---|---|---|
| 0 | 2.09 | 13.3 GB |
| 10 | 2.09 | 11.4 GB |
| 20 | 2.10 | 9.3 GB |
| 30 | 2.22 | 7.1 GB |
| 40 | 2.36 | 5.0 GB |
| 40, not pinned | 3.19 | 5.0 GB |

## Not supported yet

- Batch sizes above 1: the attention would need variable-length (varlen) batching.
- Several GPUs.
- Resuming from a saved state (`save_state` / `resume`).
- Training the text-refiner blocks or the modulation.

## Checking a change

1. `npm run typecheck`, `npm test` and `npm run build` in `frontend/` for page changes.
2. A short real run through the GUI (or `prepare_training` + the runner's accelerate command): the
   caching bars, steps, the epoch and the metrics on the Runs page, and the step time against the
   table above.
3. For changes to the model, the text encoder, the VAE or the loss: compare the outputs or caches
   with the previous code (they should match bit for bit unless the change is meant to alter them),
   render the preview prompts, and load a trained LoRA in ComfyUI.
