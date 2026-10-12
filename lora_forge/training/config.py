"""Turn a config submitted by the web UI into the files sd-scripts runs with."""

import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from typing import Optional, Tuple

import toml

from lora_forge import sd_scripts
from lora_forge.i18n import join_list, t
from lora_forge.log import log
from lora_forge.training import validation
from lora_forge.training.dataset import count_images, subset_folders

TRAINERS = {
    "sd-lora": "./scripts/sd-scripts/train_network.py",
    "sdxl-lora": "./scripts/sd-scripts/sdxl_train_network.py",

    "sd-dreambooth": "./scripts/sd-scripts/train_db.py",
    "sdxl-finetune": "./scripts/sd-scripts/sdxl_train.py",

    "sd3-lora": "./scripts/sd-scripts/sd3_train_network.py",
    "flux-lora": "./scripts/sd-scripts/flux_train_network.py",

    "anima-lora": "./scripts/sd-scripts/anima_train_network.py",
}

# Keys moved into the generated dataset_config; only used when a subset has a non-zero
# timestep offset. Every key must also exist in sd-scripts' dataset [general] schema, or
# config_util.sanitize_user_config rejects the whole file with "extra keys not allowed" --
# notably weighted_captions is a top-level CLI argument only, so it stays out of this set.
DATASET_CONFIG_KEYS = {
    "train_batch_size", "resolution", "enable_bucket", "min_bucket_reso",
    "max_bucket_reso", "bucket_reso_steps", "bucket_no_upscale",
    "caption_extension", "shuffle_caption", "keep_tokens",
    "keep_tokens_separator", "caption_dropout_rate",
    "caption_dropout_every_n_epochs", "caption_tag_dropout_rate", "color_aug",
    "flip_aug", "random_crop",
}

# Values never written to the log in clear text (the received config is logged for debugging).
SENSITIVE_CONFIG_KEYS = ("wandb_api_key", "huggingface_token", "hf_token")


class TrainingConfigError(ValueError):
    """A problem the user has to fix; the message is shown in the UI as is."""


@dataclass
class PreparedRun:
    toml_file: str
    trainer_file: str
    gpu_ids: Optional[list]
    cpu_threads: int
    # for the monitor's run record
    run_id: str
    model_train_type: str
    config: dict
    preview: bool
    # environment for run_entry.py: LoRA-Forge options that are not sd-scripts arguments
    environment: dict = field(default_factory=dict)


def redact_config(config: dict) -> dict:
    """Config copy for logging, with secrets masked."""
    return {key: ("<redacted>" if key in SENSITIVE_CONFIG_KEYS and value else value) for key, value in config.items()}


def _parse_resolution(value):
    if not isinstance(value, str):
        return value
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    return values[0] if len(values) == 1 else values


def _scan_subsets(root: str, is_reg: bool = False):
    """Subset configs for the `repeats_name` subfolders of `root`, mirroring sd-scripts'
    generate_dreambooth_subsets_config_by_subdirs() (same directory names, same class_tokens) so
    offset runs build the same subsets as the flat train_data_dir route."""
    subsets = []
    for name, num_repeats, class_tokens in subset_folders(root):
        subset = {
            "image_dir": os.path.join(root, name).replace("\\", "/"),
            "num_repeats": num_repeats,
            "class_tokens": class_tokens,
        }
        if is_reg:
            subset["is_reg"] = True
        subsets.append((name, subset))
    return subsets


def create_offset_dataset_config(config: dict, offsets: dict, dataset_path: str) -> bool:
    """Write a dataset_config with per-subset timestep offsets; False when there are none.

    Offsets are keyed by `repeats_name` folder name and only non-zero values are written into
    custom_attributes.
    """
    parsed_offsets = {}
    for name, value in offsets.items():
        try:
            offset = float(value)
        except (TypeError, ValueError):
            continue
        if offset != 0.0:
            parsed_offsets[name] = offset
    if not parsed_offsets:
        return False

    subsets = _scan_subsets(config.get("train_data_dir", ""))
    if not subsets:
        raise TrainingConfigError(t("run.offsetsNoSubsets"))

    unknown_names = set(parsed_offsets) - {name for name, _subset in subsets}
    if unknown_names:
        raise TrainingConfigError(t("run.offsetsUnknown", names=join_list(sorted(unknown_names))))

    general = {}
    for key in DATASET_CONFIG_KEYS:
        if key not in config:
            continue
        value = config.pop(key)
        if key == "train_batch_size":
            general["batch_size"] = value
            # general.batch_size drives the dataset; keep the key so args.train_batch_size
            # (LoRA metadata, DeepSpeed micro batch) does not fall back to 1
            config[key] = value
        elif key == "resolution":
            general[key] = _parse_resolution(value)
        else:
            general[key] = value

    dataset_subsets = []
    for name, subset in subsets:
        if name in parsed_offsets:
            subset["custom_attributes"] = {"timestep_sampling": {"offset": parsed_offsets[name]}}
        dataset_subsets.append(subset)

    reg_root = config.pop("reg_data_dir", "")
    if reg_root and os.path.isdir(reg_root):
        dataset_subsets.extend(subset for _name, subset in _scan_subsets(reg_root, is_reg=True))

    config.pop("train_data_dir", None)
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(toml.dumps({"general": general, "datasets": [{"subsets": dataset_subsets}]}))
    config["dataset_config"] = dataset_path.replace("\\", "/")
    return True


def get_sample_prompts(config: dict) -> Tuple[Optional[str], str]:
    # backward compatibility
    if "sample_prompts" in config and "positive_prompts" not in config:
        return None, config["sample_prompts"]

    train_data_dir = config.get("train_data_dir", "")
    sub_dir = [dir for dir in glob(os.path.join(train_data_dir, '*')) if os.path.isdir(dir)]

    positive_prompts = config.pop('positive_prompts', None)
    negative_prompts = config.pop('negative_prompts', '')
    sample_width = config.pop('sample_width', 512)
    sample_height = config.pop('sample_height', 512)
    sample_cfg = config.pop('sample_cfg', 7)
    sample_seed = config.pop('sample_seed', 2333)
    sample_steps = config.pop('sample_steps', 24)
    randomly_choice_prompt = config.pop('randomly_choice_prompt', False)

    if randomly_choice_prompt:
        if len(sub_dir) != 1:
            raise TrainingConfigError(t("run.randomPromptSubsets"))

        txt_files = glob(os.path.join(sub_dir[0], '*.txt'))
        if not txt_files:
            raise TrainingConfigError(t("run.randomPromptNoTxt"))
        sample_prompt_file = random.choice(txt_files)
        try:
            with open(sample_prompt_file, 'r', encoding='utf-8') as f:
                positive_prompts = f.read()
        except IOError:
            log.error(f"Failed to read {sample_prompt_file}")

    return positive_prompts, f'{positive_prompts} --n {negative_prompts}  --w {sample_width} --h {sample_height} --l {sample_cfg}  --s {sample_steps}  --d {sample_seed}'


def prepare_training(config: dict) -> PreparedRun:
    """Validate a submitted config and write the TOML (and prompt / dataset files) for sd-scripts.

    Raises TrainingConfigError with a message for the user when something has to be fixed.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # git-ignored and not shipped, so a fresh clone does not have it yet
    autosave_dir = os.path.join(os.getcwd(), "config", "autosave")
    os.makedirs(autosave_dir, exist_ok=True)
    toml_file = os.path.join(autosave_dir, f"{timestamp}.toml")

    validation.fix_config_types(config)
    subset_timestep_sampling_offsets = config.pop("subset_timestep_sampling_offsets", {})
    gpu_ids = config.pop("gpu_ids", None)

    model_train_type = config.pop("model_train_type", "sd-lora")
    trainer_file = TRAINERS.get(model_train_type)
    if trainer_file is None:
        raise TrainingConfigError(t("run.unknownType", type=model_train_type))

    # LoRA-Forge's own options, applied by run_entry.py; sd-scripts never sees them
    dynamic_seq = bool(config.pop("compile_dynamic_seq", False)) and model_train_type == "anima-lora" and bool(config.get("compile"))
    environment = {
        "LORA_FORGE_COMPILE_DYNAMIC_SEQ": "1" if dynamic_seq else "0",
        # the upper bound of the dynamic batch dimension
        "LORA_FORGE_BATCH_SIZE": str(config.get("train_batch_size") or 1),
    }

    train_data_dir = config.get("train_data_dir", "")
    if model_train_type != "sdxl-finetune":
        validated, message = validation.validate_data_dir(train_data_dir)
        if not validated:
            raise TrainingConfigError(message)

    # the image count only picks a dataloader thread suggestion
    cpu_threads = 8 if train_data_dir and os.path.isdir(train_data_dir) and count_images(train_data_dir) > 200 else 2

    if not str(config.get("pretrained_model_name_or_path", "")).strip():
        raise TrainingConfigError(t("run.noBaseModel"))
    validated, message = validation.validate_model(config["pretrained_model_name_or_path"], model_train_type)
    if not validated:
        raise TrainingConfigError(message)

    # qwen3 / vae are mandatory for Anima: report them here, not inside the trainer
    if model_train_type == "anima-lora":
        validated, message = validation.validate_anima_model_paths(config.get("qwen3"), config.get("vae"))
        if not validated:
            raise TrainingConfigError(message)

    # Anima / FLUX / SD3 have no weighted tokenization (their strategy does not implement it)
    if model_train_type in {"anima-lora", "flux-lora", "sd3-lora"} and config.get("weighted_captions"):
        raise TrainingConfigError(t("run.weightedCaptions"))

    # sd-scripts ignores unknown config keys, so an outdated checkout would accept these
    # options and do nothing; refuse them instead of running a silent no-op.
    if model_train_type in {"anima-lora", "flux-lora"}:
        requested = sd_scripts.requested_features(config, has_offsets=bool(subset_timestep_sampling_offsets))
        missing = sd_scripts.missing_features(requested)
        if missing:
            raise TrainingConfigError(sd_scripts.missing_features_message(missing))

    prompt_file = str(config.get("prompt_file", "")).strip()
    if prompt_file:
        if not os.path.exists(prompt_file):
            raise TrainingConfigError(t("run.promptFileMissing", path=prompt_file))
        config["sample_prompts"] = prompt_file
    else:
        positive_prompt, sample_prompts_arg = get_sample_prompts(config=config)
        if positive_prompt is not None and validation.is_prompt_like(sample_prompts_arg):
            sample_prompts_file = os.path.join(autosave_dir, f"{timestamp}-prompt.txt")
            with open(sample_prompts_file, "w", encoding="utf-8") as f:
                f.write(sample_prompts_arg)
            config["sample_prompts"] = sample_prompts_file
            log.info(f"Wrote prompts to file {sample_prompts_file}")

    if model_train_type in {"flux-lora", "anima-lora"}:
        dataset_toml_file = os.path.join(autosave_dir, f"{timestamp}-dataset.toml")
        if create_offset_dataset_config(config, subset_timestep_sampling_offsets, dataset_toml_file):
            log.info(f"Wrote per-subset timestep offset dataset config to {dataset_toml_file}")

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))

    return PreparedRun(
        toml_file=toml_file,
        trainer_file=trainer_file,
        gpu_ids=gpu_ids,
        cpu_threads=cpu_threads,
        run_id=timestamp,
        model_train_type=model_train_type,
        config=config,
        # sd-scripts only prints the timestep distribution and exits
        preview=bool(config.get("show_timesteps")),
        environment=environment,
    )
