"""MOSS-Image runs: what prepare_training does for LoRA Forge's own trainer (lora_forge/trainers/moss_image).

The trainer reads a flat TOML like sd-scripts' and caches the latents and text-encoder outputs itself,
so this only checks the model files, writes the preview prompt file and drops the form's keys the
trainer has no use for.
"""

import json
import os
import struct
from typing import Optional

from lora_forge.i18n import t

MODEL_TRAIN_TYPE = "moss-image-lora"
TRAINER = "./lora_forge/trainers/moss_image/moss_train_network.py"
SAMPLE_STEPS = 30
SAMPLE_CFG = 3.0
# form keys the trainer does not read (other pages' options an imported config may carry)
DROPPED_KEYS = (
    "network_module", "network_args", "train_batch_size", "full_precision", "full_bf16", "full_fp16",
    "save_model_as", "save_state", "resume", "max_data_loader_n_workers", "persistent_data_loader_workers",
    "randomly_choice_prompt", "sample_sampler", "enable_preview",
)
# dtypes of quantized checkpoints (ComfyUI's int8_convrot / fp8 files), which the trainer cannot load
QUANTIZED_DTYPES = {"I8", "U8", "F8_E4M3", "F8_E5M2"}


class MossConfigError(ValueError):
    """A problem the user has to fix; the message is shown in the UI as is."""


def _headers(path: str) -> dict:
    """The tensor entries (name -> dtype / shape) of a safetensors file or of all files in a folder,
    read from the headers only, so even the 19 GB text encoder takes milliseconds."""
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


def _quantized(path: str) -> bool:
    return any(entry.get("dtype") in QUANTIZED_DTYPES for entry in _headers(path).values())


def _shape(entries: dict, *names: str):
    for name in names:
        if name in entries:
            return entries[name].get("shape")
    return None


# What each model file of the page must contain, judged from its tensor names and shapes
EXPECTED_FILES = {
    "pretrained_model_name_or_path": lambda e: _shape(e, "shared_block_modulation.weight", "diffusion_model.shared_block_modulation.weight") is not None
    and _shape(e, "layers.0.proj.ffn_up_gate.weight", "diffusion_model.layers.0.proj.ffn_up_gate.weight") is not None,
    # Qwen3.5-9B: hidden size 4096 (the 2B / 4B files are a different size)
    "text_encoder": lambda e: (_shape(e, "model.language_model.embed_tokens.weight", "language_model.embed_tokens.weight") or [0, 0])[-1] == 4096,
    # FLUX.2's VAE: 128 batch-norm channels (the patchified latent)
    "vae": lambda e: _shape(e, "bn.running_mean") == [128] and _shape(e, "encoder.conv_in.weight") is not None,
}


def check_model_file(field: str, path: str) -> str:
    """The state of a model file the page asks for: `empty`, `missing`, `quantized`, `wrong` or `ok`."""
    path = str(path or "").strip()
    if not path:
        return "empty"
    if not os.path.exists(path):
        return "missing"
    entries = _headers(path)
    if any(entry.get("dtype") in QUANTIZED_DTYPES for entry in entries.values()):
        return "quantized"
    expected = EXPECTED_FILES.get(field)
    return "ok" if expected is None or (entries and expected(entries)) else "wrong"


def check_model_files(config: dict) -> None:
    for label, key in ((t("model.mossDit"), "pretrained_model_name_or_path"), (t("model.mossTextEncoder"), "text_encoder"),
                       (t("model.mossVae"), "vae")):
        value = str(config.get(key) or "").strip()
        if not value:
            raise MossConfigError(t("model.animaPathMissing", label=label))
        if not os.path.exists(value):
            raise MossConfigError(t("model.animaPathNotFound", label=label, path=value))
        if key != "vae" and _quantized(value):
            raise MossConfigError(t("run.mossQuantized", label=label, path=value))
        config[key] = value


def sample_prompts(config: dict) -> Optional[str]:
    """The preview prompt line (sd-scripts' format, which the trainer reads) from the form's preview
    fields, which it pops; None without a prompt."""
    positive = " ".join(str(config.pop("positive_prompts", "") or "").split())
    negative = " ".join(str(config.pop("negative_prompts", "") or "").split())
    width = int(config.pop("sample_width", 1024) or 1024)
    height = int(config.pop("sample_height", 1024) or 1024)
    cfg = float(config.pop("sample_cfg", SAMPLE_CFG) or SAMPLE_CFG)
    seed = config.pop("sample_seed", None)
    steps = int(config.pop("sample_steps", SAMPLE_STEPS) or SAMPLE_STEPS)
    if not positive:
        return None
    line = positive + (f" --n {negative}" if negative else "")
    line += f" --w {width} --h {height} --s {steps} --l {cfg:g}"
    if seed not in (None, ""):
        line += f" --d {int(seed)}"
    return line + "\n"


def prepare_config(config: dict, prompt_path: str) -> None:
    """Rewrite a submitted MOSS-Image config in place into the trainer's options."""
    check_model_files(config)
    prompt_file = str(config.pop("prompt_file", "") or "").strip()
    preview = bool(config.get("enable_preview"))
    prompts = sample_prompts(config)
    if not preview:
        config.pop("sample_every_n_epochs", None)
        config.pop("sample_prompts", None)
    elif prompt_file:
        if not os.path.exists(prompt_file):
            raise MossConfigError(t("run.promptFileMissing", path=prompt_file))
        config["sample_prompts"] = prompt_file
    elif prompts:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompts)
        config["sample_prompts"] = prompt_path
    for key in DROPPED_KEYS:
        config.pop(key, None)
