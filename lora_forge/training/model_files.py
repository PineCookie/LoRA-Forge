"""Checks of the model files a trainer page asks for, from the safetensors headers only (milliseconds,
even for a 19 GB file). Used by the pages' Model files panel (/api/model_file) and by the MOSS-Image
run preparation."""

import json
import os
import re
import struct

# dtypes of quantized checkpoints (ComfyUI's int8_convrot / fp8 files), which the trainers cannot use
QUANTIZED_DTYPES = {"I8", "U8", "F8_E4M3", "F8_E5M2"}
# a Hugging Face repository id, which sd-scripts downloads by itself (Anima's qwen3 accepts one)
REPO_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def headers(path: str) -> dict:
    """The tensor entries (name -> dtype / shape) of a safetensors file or of all files in a folder."""
    files = [os.path.join(path, name) for name in os.listdir(path) if name.endswith(".safetensors")] if os.path.isdir(path) else [path]
    entries = {}
    for file in files:
        try:
            with open(file, "rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
        except (OSError, ValueError, struct.error, MemoryError):
            continue
        entries.update({key: value for key, value in header.items() if isinstance(value, dict)})
    return entries


def is_quantized(path: str) -> bool:
    return any(entry.get("dtype") in QUANTIZED_DTYPES for entry in headers(path).values())


def _shape(entries: dict, *names: str):
    """The shape of the first of `names` the file has, with or without a key prefix (`net.`,
    `model.diffusion_model.`, `diffusion_model.`)."""
    for name in names:
        if name in entries:
            return entries[name].get("shape")
    for key, value in entries.items():
        if any(key.endswith("." + name) for name in names):
            return value.get("shape")
    return None


# What each file must contain, judged from tensor names and shapes
EXPECTED = {
    "moss-image-lora": {
        "pretrained_model_name_or_path": lambda e: _shape(e, "shared_block_modulation.weight") is not None
        and _shape(e, "layers.0.proj.ffn_up_gate.weight") is not None,
        # Qwen3.5-9B: hidden size 4096 (the 2B / 4B files are a different size)
        "text_encoder": lambda e: (_shape(e, "model.language_model.embed_tokens.weight", "language_model.embed_tokens.weight") or [0])[-1] == 4096,
        # FLUX.2's VAE: 128 batch-norm channels (the patchified latent)
        "vae": lambda e: _shape(e, "bn.running_mean") == [128] and _shape(e, "encoder.conv_in.weight") is not None,
    },
    "anima-lora": {
        # the Anima DiT with its LLM adapter (2048-wide blocks)
        "pretrained_model_name_or_path": lambda e: _shape(e, "x_embedder.proj.1.weight") == [2048, 68]
        and _shape(e, "llm_adapter.blocks.0.cross_attn.k_proj.weight") is not None,
        # Qwen3-0.6B: 151936 tokens x 1024
        "qwen3": lambda e: _shape(e, "model.embed_tokens.weight", "embed_tokens.weight") == [151936, 1024],
        # the Qwen-Image VAE with 16 latent channels (ComfyUI's or diffusers' names); the Qwen-Image 2.1 VAE has 64
        "vae": lambda e: (_shape(e, "conv2.weight") or [0])[0] == 16 or (_shape(e, "post_quant_conv.weight") or [0])[0] == 16,
    },
}


def check(model: str, field: str, path: str) -> str:
    """`empty`, `missing`, `repo` (a Hugging Face id the trainer downloads), `quantized`, `wrong`, `ok`,
    or `unknown` for a model or field without a check."""
    expected = EXPECTED.get(model, {}).get(field)
    if expected is None:
        return "unknown"
    path = str(path or "").strip()
    if not path:
        return "empty"
    if not os.path.exists(path):
        return "repo" if model == "anima-lora" and field == "qwen3" and REPO_ID.match(path) else "missing"
    entries = headers(path)
    if any(entry.get("dtype") in QUANTIZED_DTYPES for entry in entries.values()):
        return "quantized"
    return "ok" if entries and expected(entries) else "wrong"
