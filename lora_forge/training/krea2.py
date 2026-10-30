"""Krea 2 runs: what prepare_training does differently for kohya-ss/musubi-tuner.

musubi-tuner takes the same kind of flat TOML as sd-scripts (`--config_file`), but its dataset
is its own TOML (`[general]` + `[[datasets]]`, one per `repeats_name` subfolder) and the latents
and text-encoder outputs are cached by two separate scripts before training. run_entry.py runs
those two scripts (`--skip_existing`) in the training process from the environment set here.
"""

import importlib.util
import json
import os
from typing import Optional

import toml

from lora_forge.i18n import t
from lora_forge.training.dataset import subset_folders

MODEL_TRAIN_TYPE = "krea2-lora"
# JSON for run_entry.py: the caching scripts' arguments
CACHE_ENV = "LORA_FORGE_KREA2_CACHE"
# for run_entry.py: the Turbo LoRA, converted from ComfyUI's key names as musubi-tuner loads it
TURBO_LORA_ENV = "LORA_FORGE_KREA2_TURBO_LORA"

# GUI keys that become the dataset TOML rather than training arguments
DATASET_KEYS = ("resolution", "caption_extension", "train_batch_size", "enable_bucket", "bucket_no_upscale")
# GUI keys renamed for musubi-tuner
RENAMED_KEYS = {"pretrained_model_name_or_path": "dit"}
# sd-scripts-only keys a config may carry over (imported Anima configs); musubi's parser rejects unknown ones
DROPPED_KEYS = (
    "train_data_dir", "reg_data_dir", "prior_loss_weight", "min_bucket_reso", "max_bucket_reso", "bucket_reso_steps",
    "shuffle_caption", "keep_tokens", "caption_dropout_rate", "caption_dropout_every_n_epochs", "caption_tag_dropout_rate",
    "color_aug", "flip_aug", "random_crop", "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs",
    "cache_text_encoder_outputs_to_disk", "network_train_unet_only", "network_train_text_encoder_only", "unet_lr",
    "text_encoder_lr", "save_model_as", "loss_type", "noise_offset", "multires_noise_iterations",
    "multires_noise_discount", "clip_skip", "lowram", "cuda_allow_tf32", "cuda_cudnn_benchmark", "compile_dynamic",
    "compile_dynamic_seq", "activation_memory_budget", "qwen3", "llm_adapter_path", "fp8_base_unet",
    "vae_chunk_size", "vae_disable_cache", "qwen_image_vae_2d", "text_encoder_batch_size", "max_token_length",
    "weighted_captions", "full_precision", "persistent_data_loader_workers", "save_model_as",
    "sage_attn",  # SageAttention has no backward pass, so it cannot train
)
# Pinned block swap: the swapped blocks sit in page-locked RAM, which Windows cannot page out, so a
# run that pins more than is free can freeze the PC. The trainer needs about 2 GB on top of them.
MAIN_BLOCKS = 28
PINNED_RAM_HEADROOM_GB = 3.5
RAW_SAMPLE_STEPS = 28
TURBO_SAMPLE_STEPS = 8
# GUI attn_mode -> musubi's attention flag (one of them is required)
ATTENTION_FLAGS = {"sdpa": "sdpa", "torch": "sdpa", "flash": "flash_attn", "flash_attn": "flash_attn", "flash3": "flash3",
                   "xformers": "xformers"}


class Krea2ConfigError(ValueError):
    """A problem the user has to fix; the message is shown in the UI as is."""


def _resolution(value) -> list:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value][:2]
    parts = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    return [parts[0], parts[0]] if len(parts) == 1 else parts[:2]


def write_dataset_config(config: dict, train_data_dir: str, path: str) -> str:
    """musubi-tuner's dataset TOML for a `repeats_name` dataset folder; pops the dataset keys from
    `config`. Caches are written next to the images (musubi's default), like sd-scripts' .npz files."""
    datasets = [
        {"image_directory": os.path.join(train_data_dir, name).replace("\\", "/"), "num_repeats": num_repeats}
        for name, num_repeats, _class_tokens in subset_folders(train_data_dir)
    ]
    if not datasets:
        raise Krea2ConfigError(t("run.offsetsNoSubsets"))
    general = {
        "resolution": _resolution(config.pop("resolution", "1024,1024")),
        "caption_extension": config.pop("caption_extension", ".txt") or ".txt",
        "batch_size": int(config.pop("train_batch_size", 1) or 1),
        "enable_bucket": bool(config.pop("enable_bucket", True)),
        "bucket_no_upscale": bool(config.pop("bucket_no_upscale", False)),
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(toml.dumps({"general": general, "datasets": datasets}))
    return path.replace("\\", "/")


def prepare_config(config: dict, dataset_config_path: str, environment: dict) -> None:
    """Rewrite a submitted Krea 2 config in place into musubi-tuner's arguments."""
    train_data_dir = config.get("train_data_dir", "")
    for label, key in ((t("model.dit"), "pretrained_model_name_or_path"), (t("model.vae"), "vae")):
        value = str(config.get(key, "")).strip()
        if not value:
            raise Krea2ConfigError(t("model.animaPathMissing", label=label))
        if not os.path.isfile(value):
            raise Krea2ConfigError(t("model.animaPathNotFound", label=label, path=value))
    text_encoder = str(config.get("text_encoder", "")).strip()
    if text_encoder and not os.path.isfile(text_encoder):
        raise Krea2ConfigError(t("model.animaPathNotFound", label=t("model.textEncoder"), path=text_encoder))
    if not text_encoder:
        # caching needs it; samples during training too
        raise Krea2ConfigError(t("model.animaPathMissing", label=t("model.textEncoder")))
    _check_turbo(config)
    _check_pinned_ram(config)

    config["dataset_config"] = write_dataset_config(config, train_data_dir, dataset_config_path)
    for old, new in RENAMED_KEYS.items():
        if old in config:
            config[new] = config.pop(old)
    for key in DROPPED_KEYS:
        config.pop(key, None)
    config.setdefault("network_module", "networks.lora_krea2")
    config.setdefault("mixed_precision", "bf16")
    attn_mode = str(config.pop("attn_mode", "") or "").strip().lower()
    attention = ATTENTION_FLAGS.get(attn_mode)
    if attention is None and (attn_mode == "auto" or not any(config.get(flag) for flag in set(ATTENTION_FLAGS.values()))):
        # since PyTorch 2.15, SDPA runs on flash kernels (about as fast as flash-attn on Krea 2); a flash-attn
        # installed by hand is still used. The GUI and the trainer share one venv
        attention = "flash_attn" if importlib.util.find_spec("flash_attn") else "sdpa"
    # an older config may ask for a package LoRA Forge no longer installs; musubi would fail without it
    for flag, package in (("flash_attn", "flash_attn"), ("xformers", "xformers")):
        if (attention == flag or (attention is None and config.get(flag))) and importlib.util.find_spec(package) is None:
            config.pop(flag, None)
            attention = "sdpa"
    if attention:
        for flag in set(ATTENTION_FLAGS.values()):
            config.pop(flag, None)
        config[attention] = True
    if not config.get("gradient_checkpointing"):
        config.pop("block_swap_h2d_only", None)  # musubi refuses it without checkpointing
    if config.get("xformers"):
        config["split_attn"] = True  # musubi's xformers path needs it for Krea 2's grouped heads
    # ConvRot int8 and fp8 are alternatives; both need their pair of flags
    if config.get("convrot_int8"):
        config.pop("fp8_base", None)
        config.pop("fp8_scaled", None)
    elif config.get("fp8_base") or config.get("fp8_scaled"):
        config["fp8_base"] = config["fp8_scaled"] = True

    environment[CACHE_ENV] = json.dumps({
        "dataset_config": config["dataset_config"],
        "vae": config["vae"],
        "text_encoder": text_encoder,
    })
    if config.get("turbo_lora"):
        environment[TURBO_LORA_ENV] = config["turbo_lora"]


def pinned_block_gb(dit_file: str, quantized: bool) -> float:
    """RAM one swapped block takes: the main blocks hold almost all of the DiT file. The RAW bf16
    file (26 GB) is quantized to half when it loads with convrot_int8 or fp8; the pre-quantized
    files are about 13 GB already."""
    size_gb = os.path.getsize(dit_file) / 1024 ** 3
    return size_gb / MAIN_BLOCKS * (0.5 if quantized and size_gb > 20 else 1)


def _check_pinned_ram(config: dict) -> None:
    blocks = int(config.get("blocks_to_swap") or 0)
    if not blocks or not config.get("use_pinned_memory_for_block_swap"):
        return
    import psutil

    quantized = bool(config.get("convrot_int8") or config.get("fp8_base"))
    pinned = blocks * pinned_block_gb(config["pretrained_model_name_or_path"], quantized)
    free = psutil.virtual_memory().available / 1024 ** 3
    if pinned + PINNED_RAM_HEADROOM_GB > free:
        raise Krea2ConfigError(t("run.krea2PinnedRam", blocks=blocks, need=f"{pinned + PINNED_RAM_HEADROOM_GB:.0f}", free=f"{free:.0f}"))


def _check_turbo(config: dict) -> None:
    """The Turbo files only serve the previews: dropped without them; otherwise what musubi-tuner
    would refuse at startup is reported here."""
    for key in ("turbo_dit", "turbo_lora"):
        value = str(config.get(key) or "").strip()
        if not value or not config.get("sample_prompts"):
            config.pop(key, None)
        elif not os.path.isfile(value):
            raise Krea2ConfigError(t("model.animaPathNotFound", label=key, path=value))
    if config.get("turbo_dit") and config.get("turbo_lora"):
        raise Krea2ConfigError(t("run.krea2TurboBoth"))
    if config.get("turbo_dit") and (int(config.get("blocks_to_swap") or 0) > 0 or config.get("convrot_int8")):
        raise Krea2ConfigError(t("run.krea2TurboDit"))


def _turbo_source(config: dict) -> bool:
    return any(str(config.get(key) or "").strip() for key in ("turbo_dit", "turbo_lora"))


def sample_prompts(config: dict) -> Optional[str]:
    """musubi-tuner's prompt file line from the GUI's preview fields (pops them); None without a
    prompt. RAW previews use CFG (5.5, 28 steps); Turbo previews (turbo_dit / turbo_lora) run
    without CFG in a few steps."""
    positive = config.pop("positive_prompts", None)
    negative = str(config.pop("negative_prompts", "") or "").strip()
    width = config.pop("sample_width", 1024)
    height = config.pop("sample_height", 1024)
    cfg = float(config.pop("sample_cfg", 5.5) or 1)
    seed = config.pop("sample_seed", 2333)
    steps = int(config.pop("sample_steps", RAW_SAMPLE_STEPS) or RAW_SAMPLE_STEPS)
    config.pop("randomly_choice_prompt", None)
    config.pop("sample_sampler", None)
    if not positive or not str(positive).strip():
        return None
    line = str(positive).strip()
    if _turbo_source(config):
        # the distilled Turbo model runs without CFG; the RAW step count is far more than it needs
        steps = TURBO_SAMPLE_STEPS if steps == RAW_SAMPLE_STEPS else steps
        cfg = 1.0
    elif cfg > 1:
        # musubi's CFG is `--l` and only runs with a negative prompt. Krea 2's own default is an
        # empty one, which a prompt line cannot hold: a single space stands in for it (`--n` comes
        # first, as the line's trailing whitespace is stripped when musubi reads it)
        line += f" --n {negative or ' '}"
    return f"{line} --w {width} --h {height} --d {seed} --s {steps} --l {cfg:g}\n"


def musubi_pythonpath(src_dir: str, current: Optional[str]) -> str:
    """PYTHONPATH with musubi-tuner's `src` first, keeping what the GUI's process already has."""
    return src_dir if not current else os.pathsep.join([src_dir, current])
