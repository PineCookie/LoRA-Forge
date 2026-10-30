"""Turn a config submitted by the web UI into the files sd-scripts runs with."""

import importlib.util
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from typing import Optional, Tuple

import toml

from lora_forge import musubi_tuner, sd_scripts
from lora_forge.i18n import join_list, t
from lora_forge.log import log
from lora_forge.training import krea2, moss_image, validation
from lora_forge.training.dataset import count_images, subset_folders

TRAINERS = {
    "sd-lora": "./scripts/sd-scripts/train_network.py",
    "sdxl-lora": "./scripts/sd-scripts/sdxl_train_network.py",

    "sd-dreambooth": "./scripts/sd-scripts/train_db.py",
    "sdxl-finetune": "./scripts/sd-scripts/sdxl_train.py",

    "sd3-lora": "./scripts/sd-scripts/sd3_train_network.py",
    "flux-lora": "./scripts/sd-scripts/flux_train_network.py",

    "anima-lora": "./scripts/sd-scripts/anima_train_network.py",

    # kohya-ss/musubi-tuner (see lora_forge/training/krea2.py)
    "krea2-lora": "./scripts/musubi-tuner/src/musubi_tuner/krea2_train_network.py",

    # LoRA Forge's own trainer (see lora_forge/training/moss_image.py)
    "moss-image-lora": moss_image.TRAINER,
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


def resolve_dataloader_workers(config: dict) -> None:
    """max_data_loader_n_workers left empty: none when the latents are cached, else 2.

    Each worker is a separate process that is slow to start (sd-scripts' default of 8 delays the
    first step noticeably), and with cached latents there is little work to hand to them.
    """
    workers = config.get("max_data_loader_n_workers")
    if workers in (None, ""):
        workers = 0 if config.get("cache_latents") else 2
    config["max_data_loader_n_workers"] = int(workers)
    if config["max_data_loader_n_workers"] == 0:
        # PyTorch refuses persistent workers without workers
        config["persistent_data_loader_workers"] = False


MEMORY_BUDGET_ENV = "LORA_FORGE_ACTIVATION_MEMORY_BUDGET"
# compile modes that use CUDA graphs, which have a history of bugs with expandable segments
CUDA_GRAPH_COMPILE_MODES = {"reduce-overhead", "max-autotune"}


def apply_activation_memory_budget(config: dict, model_train_type: str, environment: dict, gpu_count: int) -> None:
    """activation_memory_budget (Anima with compile): instead of gradient checkpointing, which runs
    every block's forward twice, torch.compile keeps up to this fraction of each block's activations
    and recomputes only the rest (cheap element-wise work first). `auto` picks the fraction from the
    free VRAM when training starts. run_entry.py applies it and turns checkpointing off then.

    Not used without compile (only compiled blocks have the planner) and not with blocks_to_swap or
    the offload options, which are there to save VRAM.
    """
    budget = str(config.pop("activation_memory_budget", "")).strip().lower()
    if budget in ("", "off", "none"):
        return
    if budget != "auto":
        try:
            value = float(budget)
        except ValueError:
            raise TrainingConfigError(t("run.memoryBudgetRange"))
        if not 0 < value <= 1:
            raise TrainingConfigError(t("run.memoryBudgetRange"))
        budget = repr(value)
    if model_train_type != "anima-lora" or not config.get("compile"):
        return
    offload = [name for name in ("cpu_offload_checkpointing", "unsloth_offload_checkpointing") if config.get(name)]
    if float(config.get("blocks_to_swap") or 0) > 0:
        offload.insert(0, "blocks_to_swap")
    if offload:
        log.info(f"activation_memory_budget is not used with {', '.join(offload)}; gradient checkpointing stays as set")
        return

    environment[MEMORY_BUDGET_ENV] = budget
    log.info(f"activation_memory_budget = {budget}: in place of gradient checkpointing once training starts")
    # Bounded compile gives the CUDA caching allocator a different tensor size almost every step;
    # its cached blocks then fragment and push the budget past the VRAM. Expandable segments avoid
    # that. Only set here: over the VRAM they fail the run ("CUDA driver error: device not ready")
    # where Windows would otherwise spill into system RAM. Kept away from settings of the user's own,
    # multi-GPU runs and CUDA graphs.
    if (
        not any(os.environ.get(name) for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"))
        and gpu_count <= 1
        and config.get("compile_mode") not in CUDA_GRAPH_COMPILE_MODES
    ):
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


FP16_ACCUMULATION_ENV = "LORA_FORGE_FP16_ACCUMULATION"

# full_fp16 keeps the trained weights, their gradients and most optimizers' state in fp16. Measured with
# sd-scripts' get_optimizer on real LoRA weights (fp16 vs fp32 weights, same gradients): these turn the
# weights into NaN (AdamW, RAdamScheduleFree, CAME: an epsilon below fp16's range) or stop learning
# (Prodigy / ProdigyPlus / DAdaptation: their step size estimate rounds away).
FULL_FP16_BROKEN = {
    "adamw", "radamschedulefree", "pytorch_optimizer.came", "prodigy", "prodigyplus.prodigyplusschedulefree",
    "dadaptation", "dadaptadam", "dadaptadagrad", "dadaptadanip", "dadaptlion", "dadaptsgd",
}
# these train, but lose small updates in fp16 weights
FULL_FP16_DEGRADED = {"lion", "sgdnesterov", "sgdnesterov8bit"}
FULL_FP16_WORKS = "AdamW8bit, PagedAdamW8bit, Lion8bit, PagedLion8bit, AdaFactor"


def check_full_fp16_optimizer(config: dict) -> None:
    if not config.get("full_fp16"):
        return
    optimizer = str(config.get("optimizer_type") or "AdamW")
    if optimizer.lower() in FULL_FP16_BROKEN:
        raise TrainingConfigError(t("run.fullFp16Optimizer", optimizer=optimizer, works=FULL_FP16_WORKS))
    if optimizer.lower() in FULL_FP16_DEGRADED:
        log.warning(f"full_fp16 with {optimizer}: small updates are lost in fp16 weights; {FULL_FP16_WORKS} do not have this problem")


def apply_fp16_accumulation(config: dict, environment: dict) -> None:
    """fp16_accumulation (Anima, with training_precision fp16 / full-fp16): fp16 matrix multiplies also add up
    in fp16, which GeForce cards run at about twice the rate of fp32 accumulation. run_entry.py turns
    it on; sd-scripts never sees the key."""
    if not config.pop("fp16_accumulation", False):
        return
    if config.get("mixed_precision") != "fp16":
        log.info("fp16_accumulation is only used with mixed_precision = fp16; ignored")
        return
    environment[FP16_ACCUMULATION_ENV] = "1"
    log.info("fp16_accumulation: fp16 matrix multiplies accumulate in fp16")


SDXL_COMPILE_ENV = "LORA_FORGE_SDXL_COMPILE"
# tried with SDXL compile: lycoris.kohya fails to compile (Triton "@jit functions should be defined in a
# Python file"), networks.dylora too (its rank changes every step)
SDXL_COMPILE_NETWORKS = {"networks.lora", "networks.oft"}


def apply_sdxl_speed(config: dict, model_train_type: str, environment: dict) -> None:
    """SDXL (run_entry.py, sdxl_speed.py): the SDXL LoRA page's compile compiles each transformer and resnet
    block. sd-scripts' SDXL trainers have no compile option of their own."""
    if model_train_type not in ("sdxl-lora", "sdxl-finetune"):
        return
    if config.pop("compile", False) and model_train_type == "sdxl-lora":
        network = config.get("network_module", "networks.lora")
        if network in SDXL_COMPILE_NETWORKS:
            environment[SDXL_COMPILE_ENV] = "1"
            log.info("compile: each SDXL transformer and resnet block is compiled")
        else:
            log.info(f"compile is not used with {network}: it fails under torch.compile; training uncompiled")


def resolve_attention(config: dict) -> None:
    """PyTorch's SDPA attention runs on flash kernels since PyTorch 2.15, so LoRA Forge no longer installs
    flash-attn or xformers and its pages no longer offer them. A config from before (attn_mode = auto / flash /
    xformers on Anima, xformers on the SD pages) uses SDPA unless that package was installed by hand: sd-scripts
    would fail without it."""
    attn_mode = str(config.get("attn_mode", "")).strip().lower()
    package = {"flash": "flash_attn", "xformers": "xformers"}.get(attn_mode)
    if attn_mode == "auto" or (package and importlib.util.find_spec(package) is None):
        config["attn_mode"] = "torch"
        if attn_mode != "auto":
            log.info(f"attn_mode {attn_mode}: {package} is not installed, using PyTorch's SDPA attention")
    if config.get("xformers") and importlib.util.find_spec("xformers") is None:
        config["xformers"] = False
        config["sdpa"] = True
        log.info("xformers is not installed, using PyTorch's SDPA attention")


def check_step_counts(config: dict) -> None:
    """sd-scripts and musubi-tuner read lr_warmup_steps / lr_decay_steps as a step count when it is an
    int and as a fraction of all steps when it is a float, so 1.5 would mean 150% of the run."""
    for key in ("lr_warmup_steps", "lr_decay_steps"):
        value = config.get(key)
        if value is None or value == "" or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise TrainingConfigError(t("run.stepsOrFraction", option=key, value=value)) from None
        if number < 0 or (number >= 1 and not number.is_integer()):
            raise TrainingConfigError(t("run.stepsOrFraction", option=key, value=value))
        config[key] = int(number) if number >= 1 or number == 0 else number


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
    check_step_counts(config)
    subset_timestep_sampling_offsets = config.pop("subset_timestep_sampling_offsets", {})
    gpu_ids = config.pop("gpu_ids", None)

    model_train_type = config.pop("model_train_type", "sd-lora")
    trainer_file = TRAINERS.get(model_train_type)
    if trainer_file is None:
        raise TrainingConfigError(t("run.unknownType", type=model_train_type))
    if model_train_type not in (krea2.MODEL_TRAIN_TYPE, moss_image.MODEL_TRAIN_TYPE):  # those resolve their own
        resolve_attention(config)
    # compile_shapes = bounded (LoRA Forge's own Anima mode before PyTorch 2.15, which rejects it) becomes
    # torch.compile's automatic dynamic shapes, which 2.15 fixed
    if config.pop("compile_dynamic_seq", False) and model_train_type == "anima-lora":
        config["compile_dynamic"] = "auto"
        log.info("compile_shapes bounded: using auto, which replaces it since PyTorch 2.15")

    # LoRA-Forge's own options, applied by run_entry.py; sd-scripts never sees them.
    environment = {}
    apply_activation_memory_budget(config, model_train_type, environment, gpu_count=len(gpu_ids or []))
    if model_train_type != moss_image.MODEL_TRAIN_TYPE:  # LoRA Forge's MOSS-Image trainer reads it itself
        apply_fp16_accumulation(config, environment)
    apply_sdxl_speed(config, model_train_type, environment)

    if model_train_type == krea2.MODEL_TRAIN_TYPE:
        return _prepare_krea2(config, timestamp, autosave_dir, toml_file, trainer_file, gpu_ids, environment)
    if model_train_type == moss_image.MODEL_TRAIN_TYPE:
        return _prepare_moss_image(config, timestamp, autosave_dir, toml_file, trainer_file, gpu_ids, environment)

    resolve_dataloader_workers(config)

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

    check_full_fp16_optimizer(config)

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


def _prepare_krea2(config: dict, timestamp: str, autosave_dir: str, toml_file: str, trainer_file: str, gpu_ids, environment: dict) -> PreparedRun:
    """Krea 2 through musubi-tuner: its dataset TOML, its prompt format, its `src` on PYTHONPATH."""
    if gpu_ids and len(gpu_ids) > 1:
        raise TrainingConfigError(t("run.krea2SingleGpu"))
    train_data_dir = config.get("train_data_dir", "")
    validated, message = validation.validate_data_dir(train_data_dir)
    if not validated:
        raise TrainingConfigError(message)
    cpu_threads = 8 if count_images(train_data_dir) > 200 else 2

    prompt_file = str(config.pop("prompt_file", "")).strip()
    if prompt_file:
        if not os.path.exists(prompt_file):
            raise TrainingConfigError(t("run.promptFileMissing", path=prompt_file))
        config["sample_prompts"] = prompt_file
    else:
        prompts = krea2.sample_prompts(config)
        if prompts and config.pop("enable_preview", True):
            sample_prompts_file = os.path.join(autosave_dir, f"{timestamp}-prompt.txt")
            with open(sample_prompts_file, "w", encoding="utf-8") as f:
                f.write(prompts)
            config["sample_prompts"] = sample_prompts_file
    config.pop("enable_preview", None)

    try:
        krea2.prepare_config(config, os.path.join(autosave_dir, f"{timestamp}-dataset.toml"), environment)
    except krea2.Krea2ConfigError as error:
        raise TrainingConfigError(str(error)) from error
    environment["PYTHONPATH"] = krea2.musubi_pythonpath(str(musubi_tuner.src_dir()), os.environ.get("PYTHONPATH"))

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))
    return PreparedRun(
        toml_file=toml_file,
        trainer_file=trainer_file,
        gpu_ids=gpu_ids,
        cpu_threads=cpu_threads,
        run_id=timestamp,
        model_train_type=krea2.MODEL_TRAIN_TYPE,
        config=config,
        preview=False,
        environment=environment,
    )


def _prepare_moss_image(config: dict, timestamp: str, autosave_dir: str, toml_file: str, trainer_file: str, gpu_ids, environment: dict) -> PreparedRun:
    """MOSS-Image through LoRA Forge's own trainer (single GPU, batch 1)."""
    if gpu_ids and len(gpu_ids) > 1:
        raise TrainingConfigError(t("run.mossSingleGpu"))
    train_data_dir = config.get("train_data_dir", "")
    validated, message = validation.validate_data_dir(train_data_dir)
    if not validated:
        raise TrainingConfigError(message)
    try:
        moss_image.prepare_config(config, os.path.join(autosave_dir, f"{timestamp}-prompt.txt"))
    except moss_image.MossConfigError as error:
        raise TrainingConfigError(str(error)) from error

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))
    return PreparedRun(
        toml_file=toml_file,
        trainer_file=trainer_file,
        gpu_ids=gpu_ids,
        cpu_threads=2,
        run_id=timestamp,
        model_train_type=moss_image.MODEL_TRAIN_TYPE,
        config=config,
        preview=False,
        environment=environment,
    )
