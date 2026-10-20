"""MOSS-Image dataset: `N_name` folders, one bucket per image, and the latent / text-encoder caches.

Both caches are written next to the images (like sd-scripts' .npz files) and record what they were
made from (bucket size, image file, model file, caption text), so a changed image, caption or model
is encoded again and nothing stale is reused.
"""

import json
import math
import os
import re
from dataclasses import dataclass, field

import torch

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SUBSET_FOLDER = re.compile(r"^(\d+)_(.+)$")
LATENT_SUFFIX = "_moss_latent.safetensors"
TEXT_SUFFIX = "_moss_te.safetensors"
# raise when a change makes existing caches wrong: they are then encoded again
CACHE_FORMAT = "1"
# Qwen3.5-9B's tokenizer and chat template (the weights come from the text_encoder file)
DEFAULT_TOKENIZER = "Qwen/Qwen3.5-9B"
MAX_TEXT_TOKENS = 4096
# VRAM kept free while the text encoder runs; the layers that do not fit stay in RAM
TEXT_ENCODER_HEADROOM_GIB = 2.5


@dataclass
class Item:
    image: str
    caption_file: str
    repeats: int
    captions: list = field(default_factory=list)
    bucket: tuple = (0, 0)

    @property
    def latent_cache(self):
        return os.path.splitext(self.image)[0] + LATENT_SUFFIX

    @property
    def text_cache(self):
        return os.path.splitext(self.image)[0] + TEXT_SUFFIX


def read_captions(path: str) -> list:
    """The caption variants of an image: each non-empty line of its caption file is one, and each
    step trains on one of them at random (OpenMOSS trained on short, medium and long captions of
    the same image). A one-line file is a single caption."""
    if not os.path.exists(path):
        return [""]
    with open(path, encoding="utf-8-sig") as f:  # Notepad may add a BOM
        lines = [line.strip() for line in f]
    return [line for line in lines if line] or [""]


def scan_dataset(train_data_dir: str, caption_extension: str) -> list:
    items = []
    for name in sorted(os.listdir(train_data_dir)):
        folder = os.path.join(train_data_dir, name)
        match = SUBSET_FOLDER.match(name)
        if not match or not os.path.isdir(folder):
            continue
        repeats = int(match.group(1))
        for file in sorted(os.listdir(folder)):
            if os.path.splitext(file)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            image = os.path.join(folder, file)
            caption_file = os.path.splitext(image)[0] + caption_extension
            items.append(Item(image, caption_file, repeats, read_captions(caption_file)))
    return items


def bucket_size(width: int, height: int, resolution: tuple, enable_bucket: bool, no_upscale: bool, step: int = 16) -> tuple:
    """Each image gets a bucket of its own aspect ratio with the pixel count of `resolution`; with
    no_upscale a smaller image keeps its size. Sides are multiples of 16 (VAE x8, patch x2)."""
    if not enable_bucket:
        return resolution
    area = resolution[0] * resolution[1]
    scale = 1.0 if no_upscale and width * height <= area else math.sqrt(area / (width * height))
    return max(step, int(width * scale) // step * step), max(step, int(height * scale) // step * step)


def _signature(path: str) -> str:
    stat = os.stat(path)
    return f"{os.path.basename(path)}:{stat.st_size}:{stat.st_mtime_ns}"


def _model_signature(path: str) -> str:
    if os.path.isdir(path):
        return os.path.basename(os.path.normpath(path)) + ":" + ",".join(
            f"{name}:{os.path.getsize(os.path.join(path, name))}" for name in sorted(os.listdir(path)) if name.endswith(".safetensors"))
    return f"{os.path.basename(path)}:{os.path.getsize(path)}"


def _cache_metadata(path: str):
    from safetensors import safe_open

    if not os.path.exists(path):
        return None
    try:
        with safe_open(path, framework="pt") as f:
            return f.metadata() or {}
    except Exception:
        return None


# ---------------------------------------------------------------- VAE and latents

def _weights_file(path: str, name: str) -> str:
    if os.path.isdir(path):
        file = os.path.join(path, name)
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} not found")
        return file
    return path


def load_vae(path: str, device):
    """OpenMOSS's VAE (diffusers names; the latents the DiT was trained on) or ComfyUI's
    `flux2-vae` (original FLUX.2 names, converted here): its encoder and batch-norm statistics give
    latents within about 1% of OpenMOSS's."""
    from diffusers import AutoencoderKLFlux2
    from safetensors.torch import load_file

    with open(os.path.join(CONFIG_DIR, "vae.json"), encoding="utf-8") as f:
        config = json.load(f)
    state = load_file(_weights_file(path, "diffusion_pytorch_model.safetensors"))
    if "encoder.down.0.block.0.conv1.weight" in state:
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        bn = {key: state.pop(key) for key in list(state) if key.startswith("bn.")}
        for old, new in (("encoder.quant_conv", "quant_conv"), ("decoder.post_quant_conv", "post_quant_conv")):
            for suffix in (".weight", ".bias"):
                state[new + suffix] = state.pop(old + suffix)
        state = convert_ldm_vae_checkpoint(state, config)
        state.update(bn)
    # the decoder width differs between the files (OpenMOSS ships a slimmer one)
    top = state["decoder.conv_in.weight"].shape[0]
    widths = config["decoder_block_out_channels"]
    config["decoder_block_out_channels"] = [width * top // widths[-1] for width in widths]
    vae = AutoencoderKLFlux2.from_config(config)
    vae.load_state_dict(state, strict=True)
    return vae.to(device).eval().requires_grad_(False)


def _bn(vae, reference):
    eps = vae.config.batch_norm_eps
    mean = vae.bn.running_mean.to(reference).view(1, -1, 1, 1)
    std = (vae.bn.running_var + eps).sqrt().to(reference).view(1, -1, 1, 1)
    return mean, std


@torch.no_grad()
def encode_image(vae, pixels):
    """[1, 3, H, W] in [-1, 1] -> the DiT's latent [1, 128, H/16, W/16]: 2x2 patches, batch-norm normalized."""
    z = vae.encode(pixels).latent_dist.mode()
    b, c, h, w = z.shape
    z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean, std = _bn(vae, z)
    return (z - mean) / std


@torch.no_grad()
def decode_latent(vae, latent):
    z = latent.to(device=vae.device, dtype=vae.dtype)
    mean, std = _bn(vae, z)
    z = z * std + mean
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0].clamp(-1, 1)


def load_image(path: str, bucket: tuple):
    """The image resized to cover the bucket and center-cropped, as [1, 3, H, W] in [-1, 1]."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = bucket
        scale = max(width / image.width, height / image.height)
        resized = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
        image = image.resize(resized, Image.LANCZOS)
        left, top = (resized[0] - width) // 2, (resized[1] - height) // 2
        image = image.crop((left, top, left + width, top + height))
        array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)[None] / 127.5 - 1


def assign_buckets(items: list, resolution: tuple, enable_bucket: bool, no_upscale: bool):
    from PIL import Image

    for item in items:
        with Image.open(item.image) as image:
            item.bucket = bucket_size(image.width, image.height, resolution, enable_bucket, no_upscale)


def cache_latents(items: list, vae_path: str, device) -> int:
    """Encodes the images whose cache is missing or stale; returns how many it encoded."""
    from safetensors.torch import save_file
    from tqdm import tqdm

    model = _model_signature(vae_path)
    todo = []
    for item in items:
        expected = {"format": CACHE_FORMAT, "bucket": f"{item.bucket[0]}x{item.bucket[1]}", "image": _signature(item.image), "vae": model}
        if _cache_metadata(item.latent_cache) != expected:
            todo.append((item, expected))
    if not todo:
        return 0
    vae = load_vae(vae_path, device)
    for item, metadata in tqdm(todo, desc="caching latents", smoothing=0):
        latent = encode_image(vae, load_image(item.image, item.bucket).to(device))[0]
        save_file({"latent": latent.float().cpu().contiguous()}, item.latent_cache, metadata=metadata)
    del vae
    return len(todo)


# ---------------------------------------------------------------- text encoder

class TextEncoder:
    """Qwen3.5-9B's language model (the vision tower is not needed) from one safetensors file
    (ComfyUI's `qwen3.5_9b_bf16`) or a folder of shards. Layers that do not fit in VRAM run from RAM."""

    def __init__(self, path: str, device, tokenizer: str = ""):
        import accelerate
        from safetensors import safe_open
        from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel

        with open(os.path.join(CONFIG_DIR, "text_encoder.json"), encoding="utf-8") as f:
            config = Qwen3_5TextConfig(**json.load(f)["text_config"])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer or DEFAULT_TOKENIZER)
        with torch.device("meta"):
            model = Qwen3_5TextModel._from_config(config, dtype=torch.bfloat16)
        state = {}
        files = _weights_file_list(path)
        for file in files:
            with safe_open(file, framework="pt") as f:
                for key in f.keys():
                    for prefix in ("model.language_model.", "language_model."):
                        if key.startswith(prefix):
                            # as stored: the linear-attention A_log and norm weights are fp32 on purpose
                            state[key[len(prefix):]] = f.get_tensor(key)
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if missing:
            raise ValueError(f"{path} is not a Qwen3.5-9B text encoder: missing {missing[:3]}")
        del state
        # the rotary inverse frequencies are buffers the file does not hold
        model.rotary_emb = type(model.rotary_emb)(config=config)
        model.eval().requires_grad_(False)
        index = torch.device(device).index or 0
        free = torch.cuda.mem_get_info(index)[0] / 1024 ** 3
        budget = max(1.0, free - TEXT_ENCODER_HEADROOM_GIB)
        device_map = accelerate.infer_auto_device_map(model, max_memory={index: f"{budget:.1f}GiB", "cpu": "1000GiB"},
                                                      no_split_module_classes=model._no_split_modules, dtype=torch.bfloat16)
        on_gpu = sum(1 for place in device_map.values() if place == index)
        print(f"LoRA-Forge: text encoder: {on_gpu} of {len(device_map)} parts on the GPU, the rest runs from RAM", flush=True)
        self.model = accelerate.dispatch_model(model, device_map=device_map, main_device=index)
        self.device = torch.device("cuda", index)

    def format(self, text: str) -> str:
        """The chat template as MOSS-Image was trained with it (user turn, open think block)."""
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(self.format(text), return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS).to(self.device)
        hidden = self.model(input_ids=inputs["input_ids"], use_cache=False).last_hidden_state[0]
        return hidden.to(torch.bfloat16).cpu().contiguous()


def _weights_file_list(path: str) -> list:
    if os.path.isdir(path):
        files = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {path}")
        return files
    return [path]


def cache_text(items: list, prompts: list, text_encoder_path: str, device, tokenizer: str = "", empty_cache: str = "") -> dict:
    """Encodes the captions whose cache is missing or stale, and the preview prompts (kept in memory,
    returned as {text: hidden}). With `empty_cache` (a file path), the empty prompt that caption
    dropout trains with is cached there and returned as {"": hidden}, so a run with every caption
    cached and no previews does not load the text encoder for it."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tqdm import tqdm

    model = _model_signature(text_encoder_path)
    todo = []
    for item in items:
        expected = {"format": CACHE_FORMAT, "captions": json.dumps(item.captions, ensure_ascii=False), "text_encoder": model}
        if _cache_metadata(item.text_cache) != expected:
            todo.append((item, expected))
    prompts = sorted(set(prompts))
    empty_expected = {"format": CACHE_FORMAT, "captions": "[\"\"]", "text_encoder": model}
    encode_empty = bool(empty_cache) and _cache_metadata(empty_cache) != empty_expected
    encoded = {}
    if todo or prompts or encode_empty:
        encoder = TextEncoder(text_encoder_path, device, tokenizer)
        for item, metadata in tqdm(todo, desc="caching text encoder outputs", smoothing=0):
            tensors = {f"hidden_{i}": encoder.encode(caption) for i, caption in enumerate(item.captions)}
            save_file(tensors, item.text_cache, metadata=metadata)
        if encode_empty:
            save_file({"hidden_0": encoder.encode("")}, empty_cache, metadata=empty_expected)
        encoded = {prompt: encoder.encode(prompt) for prompt in prompts}
        del encoder
    if empty_cache and "" not in encoded:
        with safe_open(empty_cache, framework="pt") as f:
            encoded[""] = f.get_tensor("hidden_0")
    return encoded


def load_cached(item: Item, variant: int):
    from safetensors import safe_open

    with safe_open(item.latent_cache, framework="pt") as f:
        latent = f.get_tensor("latent")
    with safe_open(item.text_cache, framework="pt") as f:
        hidden = f.get_tensor(f"hidden_{variant}")
    return latent, hidden
