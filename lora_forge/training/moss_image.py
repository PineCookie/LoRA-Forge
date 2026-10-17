"""MOSS-Image runs: what prepare_training does for LoRA Forge's own trainer (lora_forge/trainers/moss_image).

The trainer reads a flat TOML like sd-scripts' and caches the latents and text-encoder outputs itself,
so this only checks the model files, writes the preview prompt file and drops the form's keys the
trainer has no use for.
"""

import os
from typing import Optional

from lora_forge.i18n import t
from lora_forge.training import model_files

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


class MossConfigError(ValueError):
    """A problem the user has to fix; the message is shown in the UI as is."""


def check_model_files(config: dict) -> None:
    for label, key in ((t("model.mossDit"), "pretrained_model_name_or_path"), (t("model.mossTextEncoder"), "text_encoder"),
                       (t("model.mossVae"), "vae")):
        value = str(config.get(key) or "").strip()
        if not value:
            raise MossConfigError(t("model.animaPathMissing", label=label))
        if not os.path.exists(value):
            raise MossConfigError(t("model.animaPathNotFound", label=label, path=value))
        if key != "vae" and model_files.is_quantized(value):
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
    # the web UI sends the preview fields only with previews on and never sends enable_preview itself
    # (a form-only key); a config built by hand may still say enable_preview = false
    preview = config.get("enable_preview", True)
    prompts = sample_prompts(config)
    if preview and prompt_file:
        if not os.path.exists(prompt_file):
            raise MossConfigError(t("run.promptFileMissing", path=prompt_file))
        config["sample_prompts"] = prompt_file
    elif preview and prompts:
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompts)
        config["sample_prompts"] = prompt_path
    else:
        config.pop("sample_every_n_epochs", None)
        config.pop("sample_prompts", None)
    for key in DROPPED_KEYS:
        config.pop(key, None)
