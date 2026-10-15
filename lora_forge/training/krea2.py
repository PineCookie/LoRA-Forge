"""Krea 2 runs: what prepare_training does differently for kohya-ss/musubi-tuner.

musubi-tuner takes the same kind of flat TOML as sd-scripts (`--config_file`), but its dataset
is its own TOML (`[general]` + `[[datasets]]`, one per `repeats_name` subfolder) and the latents
and text-encoder outputs are cached by two separate scripts before training. run_entry.py runs
those two scripts (`--skip_existing`) in the training process from the environment set here.
"""

import json
import os
from typing import Optional

import toml

from lora_forge.i18n import t
from lora_forge.training.dataset import subset_folders

MODEL_TRAIN_TYPE = "krea2-lora"
# JSON for run_entry.py: the caching scripts' arguments
CACHE_ENV = "LORA_FORGE_KREA2_CACHE"

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
    "compile_dynamic_seq", "activation_memory_budget", "qwen3", "llm_adapter_path", "attn_mode", "fp8_base_unet",
    "vae_chunk_size", "vae_disable_cache", "qwen_image_vae_2d", "text_encoder_batch_size", "max_token_length",
    "weighted_captions", "full_precision", "persistent_data_loader_workers", "save_model_as",
)
# GUI attn_mode -> musubi's attention flag (one of them is required)
ATTENTION_FLAGS = {"sdpa": "sdpa", "torch": "sdpa", "flash": "flash_attn", "flash_attn": "flash_attn", "flash3": "flash3",
                   "sage_attn": "sage_attn", "xformers": "xformers"}


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

    config["dataset_config"] = write_dataset_config(config, train_data_dir, dataset_config_path)
    for old, new in RENAMED_KEYS.items():
        if old in config:
            config[new] = config.pop(old)
    for key in DROPPED_KEYS:
        config.pop(key, None)
    config.setdefault("network_module", "networks.lora_krea2")
    config.setdefault("mixed_precision", "bf16")
    attention = ATTENTION_FLAGS.get(str(config.pop("attn_mode", "") or "").strip().lower())
    if attention is None and not any(config.get(flag) for flag in set(ATTENTION_FLAGS.values())):
        attention = "sdpa"
    if attention:
        for flag in set(ATTENTION_FLAGS.values()):
            config.pop(flag, None)
        config[attention] = True
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


def sample_prompts(config: dict) -> Optional[str]:
    """musubi-tuner's prompt file line from the GUI's preview fields (pops them); None without a
    prompt. Krea 2 RAW samples with CFG (`--g`), 28 steps and guidance 5.5 are its defaults."""
    positive = config.pop("positive_prompts", None)
    negative = config.pop("negative_prompts", "")
    width = config.pop("sample_width", 1024)
    height = config.pop("sample_height", 1024)
    cfg = config.pop("sample_cfg", 5.5)
    seed = config.pop("sample_seed", 2333)
    steps = config.pop("sample_steps", 28)
    config.pop("randomly_choice_prompt", None)
    config.pop("sample_sampler", None)
    if not positive or not str(positive).strip():
        return None
    line = f"{str(positive).strip()} --w {width} --h {height} --d {seed} --s {steps} --g {cfg}"
    if negative and str(negative).strip():
        line += f" --n {str(negative).strip()}"
    return line + "\n"


def musubi_pythonpath(src_dir: str, current: Optional[str]) -> str:
    """PYTHONPATH with musubi-tuner's `src` first, keeping what the GUI's process already has."""
    return src_dir if not current else os.pathsep.join([src_dir, current])
