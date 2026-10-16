"""MOSS-Image LoRA training: LoRA Forge's own trainer (no upstream trainer supports MOSS-Image yet).

`moss_train_network.py --config_file <toml>`, launched like sd-scripts' trainers (accelerate ->
run_entry.py), so the Runs page gets the same progress, epochs and metrics. The recipe follows the
MOSS-Image technical report: velocity target (noise - x0), timestep input 1000 * sigma, its
logit-normal sigma sampling, the text encoder frozen. Runs at batch 1 (each sample keeps its own
text length, as in OpenMOSS's pipeline); gradient_accumulation_steps gives larger effective batches.

Before training, the latents and the text-encoder outputs are cached next to the images (moss_data.py).
"""

import argparse
import ast
import gc
import importlib
import json
import math
import os
import random
import sys
import time

import toml
import torch
from torch.nn import functional as F

import moss_data
import moss_lora
import moss_model
import moss_sampling

DEFAULTS = {
    "pretrained_model_name_or_path": "",
    "text_encoder": "",
    "vae": "",
    "tokenizer": "",
    "train_data_dir": "",
    "resolution": "1024,1024",
    "enable_bucket": True,
    "bucket_no_upscale": True,
    "caption_extension": ".txt",
    "output_dir": "./output",
    "output_name": "moss_image_lora",
    "save_precision": "fp16",
    "save_every_n_epochs": None,
    "save_every_n_steps": None,
    "max_train_epochs": None,
    "max_train_steps": 1600,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "network_dim": 32,
    "network_alpha": 32,
    "network_dropout": 0.0,
    "network_weights": "",
    "optimizer_type": "AdamW",
    "learning_rate": 1e-4,
    "optimizer_args": [],
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1.0,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "compile": True,
    "compile_cache_size_limit": 64,
    "attn_mode": "auto",
    "seed": 42,
    "logit_mean": -1.09,
    "logit_std": 0.95,
    "sample_prompts": "",
    "sample_every_n_epochs": None,
    "sample_every_n_steps": None,
    "sample_at_first": False,
    "log_with": None,
    "logging_dir": None,
    "log_prefix": None,
    "log_tracker_name": None,
    "wandb_api_key": None,
    "wandb_run_name": None,
}
SAVE_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "float": torch.float32, "fp32": torch.float32}


def read_config(argv) -> dict:
    parser = argparse.ArgumentParser(description="MOSS-Image LoRA training")
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args(argv)
    config = toml.load(args.config_file)
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        print(f"LoRA-Forge: not used for MOSS-Image: {', '.join(unknown)}", flush=True)
    return {**DEFAULTS, **{key: value for key, value in config.items() if key in DEFAULTS}}


def resolution_of(value) -> tuple:
    parts = [int(part) for part in (value if isinstance(value, (list, tuple)) else str(value).split(",")) if str(part).strip()]
    return (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])


def optimizer_kwargs(args) -> dict:
    kwargs = {}
    for arg in args or []:
        key, _, value = str(arg).partition("=")
        value = value.strip()
        try:
            kwargs[key.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key.strip()] = value
    return kwargs


def make_optimizer(config, params):
    name = str(config["optimizer_type"])
    lr = float(config["learning_rate"])
    kwargs = optimizer_kwargs(config["optimizer_args"])
    lower = name.lower()
    if lower == "adamw":
        optimizer_class = torch.optim.AdamW
        kwargs.setdefault("fused", True)
    elif lower == "adamw8bit":
        import bitsandbytes

        optimizer_class = bitsandbytes.optim.AdamW8bit
    elif lower == "adafactor":
        from transformers.optimization import Adafactor

        optimizer_class = Adafactor
        for key in ("scale_parameter", "relative_step", "warmup_init"):
            kwargs.setdefault(key, False)
    elif "." in name:
        module, class_name = name.rsplit(".", 1)
        optimizer_class = getattr(importlib.import_module(module), class_name)
    else:
        raise SystemExit(f"unknown optimizer_type {name}: use AdamW, AdamW8bit, AdaFactor or an import path")
    print(f"LoRA-Forge: optimizer {optimizer_class.__module__}.{optimizer_class.__name__} lr {lr:g} {kwargs or ''}", flush=True)
    return optimizer_class(params, lr=lr, **kwargs)


def set_optimizer_mode(optimizer, train: bool):
    """Schedule-free optimizers (ProdigyPlusScheduleFree) keep training weights apart from the
    averaged ones that are saved and sampled."""
    method = getattr(optimizer, "train" if train else "eval", None)
    if callable(method):
        method()


def make_tracker_config(config) -> dict:
    values = {}
    for key, value in config.items():
        if key == "wandb_api_key":
            continue
        values[key] = value if isinstance(value, (int, float, str, bool)) else json.dumps(value)
    return values


class LossRecorder:
    """sd-scripts' moving average: the mean loss over the last epoch's worth of steps."""

    def __init__(self, size: int):
        self.size = max(1, size)
        self.values = []
        self.position = 0

    def add(self, value: float) -> float:
        if len(self.values) < self.size:
            self.values.append(value)
        else:
            self.values[self.position] = value
        self.position = (self.position + 1) % self.size
        return sum(self.values) / len(self.values)


class Trainer:
    def __init__(self, config: dict):
        from accelerate import Accelerator

        self.config = config
        if int(config["train_batch_size"] or 1) != 1:
            raise SystemExit("MOSS-Image trains at train_batch_size 1; use gradient_accumulation_steps for larger batches")
        for key in ("pretrained_model_name_or_path", "text_encoder", "vae", "train_data_dir"):
            if not str(config[key]).strip() or not os.path.exists(config[key]):
                raise SystemExit(f"{key} not found: {config[key]!r}")
        log_with = config["log_with"]
        logging_dir = None
        if config["logging_dir"]:
            logging_dir = os.path.join(config["logging_dir"], (config["log_prefix"] or "") + time.strftime("%Y%m%d%H%M%S"))
        if log_with in ("wandb", "all") and config["wandb_api_key"]:
            import wandb

            wandb.login(key=config["wandb_api_key"])
        trackers = None if not log_with else (["tensorboard", "wandb"] if log_with == "all" else [log_with])
        self.accelerator = Accelerator(log_with=trackers, project_dir=logging_dir)
        self.device = self.accelerator.device
        seed = int(config["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        self.generator = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------ preparation
    def prepare_data(self):
        config = self.config
        items = moss_data.scan_dataset(config["train_data_dir"], config["caption_extension"] or ".txt")
        if not items:
            raise SystemExit(f"no images in N_name folders under {config['train_data_dir']}")
        moss_data.assign_buckets(items, resolution_of(config["resolution"]), bool(config["enable_bucket"]), bool(config["bucket_no_upscale"]))
        self.items = items
        buckets = {}
        for item in items:
            buckets[item.bucket] = buckets.get(item.bucket, 0) + 1
        print(f"LoRA-Forge: {len(items)} images in {len(buckets)} buckets, "
              f"{sum(item.repeats for item in items)} samples per epoch with repeats", flush=True)

        self.prompts = []
        if config["sample_prompts"]:
            self.prompts = moss_sampling.read_prompt_file(config["sample_prompts"])
        encoded = moss_data.cache_latents(items, config["vae"], self.device)
        if encoded:
            print(f"LoRA-Forge: cached latents of {encoded} images", flush=True)
        self.free_memory()
        texts = [p["prompt"] for p in self.prompts] + [p["negative"] for p in self.prompts]
        self.prompt_embeddings = moss_data.cache_text(items, texts, config["text_encoder"], self.device, config["tokenizer"])
        self.free_memory()

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def prepare_model(self):
        config = self.config
        attn_mode = str(config["attn_mode"] or "auto").lower()
        if attn_mode in ("auto", "flash", "flash_attn"):
            if moss_model.flash_attn_func is None and attn_mode != "auto":
                raise SystemExit("attn_mode flash_attn needs flash-attn (uv sync --extra flash-attn)")
            attn_mode = "flash_attn" if moss_model.flash_attn_func is not None else "sdpa"
        moss_model.ATTENTION["mode"] = attn_mode
        print(f"LoRA-Forge: loading the MOSS-Image DiT, attention {attn_mode}", flush=True)
        model = moss_model.load_dit(config["pretrained_model_name_or_path"], self.device)
        self.network = moss_lora.LoRANetwork(model, int(config["network_dim"]), float(config["network_alpha"] or config["network_dim"]),
                                             float(config["network_dropout"] or 0))
        if config["network_weights"]:
            self.network.load_weights(config["network_weights"])
            print(f"LoRA-Forge: continuing from {config['network_weights']}", flush=True)
        model.gradient_checkpointing = bool(config["gradient_checkpointing"])
        if config["compile"]:
            import torch._dynamo

            torch._dynamo.config.recompile_limit = int(config["compile_cache_size_limit"] or 64)
            model.block_fn = torch.compile(moss_model.Block.forward)
        self.model = model
        params = self.network.parameters()
        print(f"LoRA-Forge: LoRA rank {self.network.rank}, alpha {self.network.alpha:g}, "
              f"{sum(p.numel() for p in params) / 1e6:.1f}M parameters", flush=True)
        self.optimizer = make_optimizer(config, params)
        self.params = params

    def plan_steps(self):
        config = self.config
        accumulation = max(1, int(config["gradient_accumulation_steps"] or 1))
        samples = sum(item.repeats for item in self.items)
        steps_per_epoch = math.ceil(samples / accumulation)
        if config["max_train_epochs"]:
            epochs = int(config["max_train_epochs"])
            total = epochs * steps_per_epoch
        else:
            total = int(config["max_train_steps"] or steps_per_epoch)
            epochs = math.ceil(total / steps_per_epoch)
        self.accumulation, self.steps_per_epoch, self.epochs, self.total_steps = accumulation, steps_per_epoch, epochs, total

    def make_scheduler(self):
        from diffusers.optimization import get_scheduler

        config = self.config
        return get_scheduler(str(config["lr_scheduler"] or "constant"), self.optimizer,
                             num_warmup_steps=int(config["lr_warmup_steps"] or 0), num_training_steps=self.total_steps,
                             num_cycles=int(config["lr_scheduler_num_cycles"] or 1), power=float(config["lr_scheduler_power"] or 1.0))

    # ------------------------------------------------------------ output
    def metadata(self, epoch: int, step: int) -> dict:
        config = self.config
        return {
            "ss_base_model_version": "moss_image",
            "ss_sd_model_name": os.path.basename(os.path.normpath(config["pretrained_model_name_or_path"])),
            "ss_vae_name": os.path.basename(os.path.normpath(config["vae"])),
            "ss_network_module": "networks.lora",
            "ss_network_dim": self.network.rank,
            "ss_network_alpha": self.network.alpha,
            "ss_network_dropout": config["network_dropout"],
            "ss_output_name": config["output_name"],
            "ss_learning_rate": config["learning_rate"],
            "ss_optimizer": config["optimizer_type"],
            "ss_lr_scheduler": config["lr_scheduler"],
            "ss_lr_warmup_steps": config["lr_warmup_steps"],
            "ss_num_train_images": len(self.items),
            "ss_num_batches_per_epoch": self.steps_per_epoch,
            "ss_num_epochs": self.epochs,
            "ss_max_train_steps": self.total_steps,
            "ss_epoch": epoch,
            "ss_steps": step,
            "ss_resolution": json.dumps(list(resolution_of(config["resolution"]))),
            "ss_gradient_checkpointing": bool(config["gradient_checkpointing"]),
            "ss_gradient_accumulation_steps": self.accumulation,
            "ss_mixed_precision": "bf16",
            "ss_seed": config["seed"],
            "ss_timestep_sampling": "moss_logit_normal",
            "ss_logit_mean": config["logit_mean"],
            "ss_logit_std": config["logit_std"],
            "ss_training_started_at": self.started_at,
            "ss_training_finished_at": time.time(),
        }

    def save(self, name: str, epoch: int, step: int):
        config = self.config
        os.makedirs(config["output_dir"], exist_ok=True)
        path = os.path.join(config["output_dir"], name + ".safetensors")
        set_optimizer_mode(self.optimizer, False)
        self.network.save(path, SAVE_DTYPES.get(str(config["save_precision"]), torch.float16), self.metadata(epoch, step))
        set_optimizer_mode(self.optimizer, True)
        print(f"LoRA-Forge: saved {path}", flush=True)

    def sample(self, epoch=None, step=None):
        if not self.prompts:
            return
        from PIL import Image
        from tqdm import tqdm

        config = self.config
        folder = os.path.join(config["output_dir"], "sample")
        os.makedirs(folder, exist_ok=True)
        set_optimizer_mode(self.optimizer, False)
        self.network.train(False)
        latents = []
        for index, prompt in enumerate(tqdm(self.prompts, desc="Sampling", leave=False)):
            seed = prompt["seed"] if prompt["seed"] is not None else int(config["seed"]) + index
            positive = self.prompt_embeddings[prompt["prompt"]][None].to(self.device)
            negative = self.prompt_embeddings[prompt["negative"]][None].to(self.device)
            latents.append((index, seed, moss_sampling.generate(self.model, positive, negative, prompt["width"], prompt["height"],
                                                                prompt["steps"], prompt["cfg"], seed).cpu()))
        self.network.train(True)
        set_optimizer_mode(self.optimizer, True)
        self.free_memory()
        vae = moss_data.load_vae(config["vae"], self.device)
        suffix = f"e{epoch:06d}" if epoch is not None else f"{step:06d}"
        stamp = time.strftime("%Y%m%d%H%M%S")
        for index, seed, latent in latents:
            pixels = moss_data.decode_latent(vae, latent.to(self.device))[0]
            array = ((pixels.float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype("uint8")
            name = f"{config['output_name']}_{suffix}_{index:02d}_{stamp}_{seed}.png"
            Image.fromarray(array).save(os.path.join(folder, name))
        del vae
        self.free_memory()

    # ------------------------------------------------------------ training
    def train(self):
        config = self.config
        self.prepare_data()
        self.prepare_model()
        self.plan_steps()
        scheduler = self.make_scheduler()
        name = config["log_tracker_name"] or "moss_image_lora"
        self.accelerator.init_trackers(name, config=make_tracker_config(config),
                                       init_kwargs={"wandb": {"name": config["wandb_run_name"]}} if config["wandb_run_name"] else {})
        from tqdm import tqdm

        print(f"LoRA-Forge: {self.total_steps} steps, {self.epochs} epochs, {self.steps_per_epoch} steps per epoch, "
              f"gradient accumulation {self.accumulation}", flush=True)
        self.started_at = time.time()
        set_optimizer_mode(self.optimizer, True)
        self.network.train(True)
        if config["sample_at_first"]:
            self.sample(step=0)
        epoch_samples = [item for item in self.items for _ in range(item.repeats)]
        recorder = LossRecorder(self.steps_per_epoch)
        progress = tqdm(total=self.total_steps, smoothing=0, desc="steps")
        max_norm = float(config["max_grad_norm"] or 0)
        save_steps = int(config["save_every_n_steps"] or 0)
        save_epochs = int(config["save_every_n_epochs"] or 0)
        sample_steps = int(config["sample_every_n_steps"] or 0)
        sample_epochs = int(config["sample_every_n_epochs"] or 0)
        mean, std = float(config["logit_mean"]), float(config["logit_std"])
        step = 0
        for epoch in range(1, self.epochs + 1):
            self.accelerator.print(f"\nepoch {epoch}/{self.epochs}")
            order = list(epoch_samples)
            random.shuffle(order)
            epoch_losses = []
            accumulated = 0.0
            for index, item in enumerate(order):
                latent, hidden = moss_data.load_cached(item, random.randrange(len(item.captions)))
                x0 = latent[None].to(self.device, torch.float32)
                context = hidden[None].to(self.device)
                sigma = moss_sampling.sample_sigma(self.generator, mean, std)
                noise = torch.randn(x0.shape, generator=self.generator).to(self.device)
                prediction = self.model((1 - sigma) * x0 + sigma * noise, sigma * moss_sampling.TRAIN_STEPS, context)
                loss = F.mse_loss(prediction.float(), noise - x0)
                (loss / self.accumulation).backward()
                accumulated += loss.item() / self.accumulation
                if (index + 1) % self.accumulation and index + 1 != len(order):
                    continue
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.params, max_norm)
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                average = recorder.add(accumulated)
                epoch_losses.append(accumulated)
                progress.update(1)
                progress.set_postfix(avr_loss=f"{average:.4f}")
                if step >= self.total_steps:
                    progress.close()  # before the last save and previews, so the step rate stays the training's
                self.accelerator.log(self.step_logs(accumulated, average, scheduler), step=step)
                accumulated = 0.0
                if save_steps and step % save_steps == 0 and step < self.total_steps:
                    self.save(f"{config['output_name']}-step{step:08d}", epoch, step)
                if sample_steps and step % sample_steps == 0:
                    self.sample(step=step)
                if step >= self.total_steps:
                    break
            if epoch_losses:
                self.accelerator.log({"loss/epoch_average": sum(epoch_losses) / len(epoch_losses)}, step=epoch)
            if step >= self.total_steps:
                break
            if save_epochs and epoch % save_epochs == 0:
                self.save(f"{config['output_name']}-{epoch:06d}", epoch, step)
            if sample_epochs and epoch % sample_epochs == 0:
                self.sample(epoch=epoch)
        progress.close()
        self.save(config["output_name"], self.epochs, step)
        sampled_last_step = sample_steps and step % sample_steps == 0
        if (sample_epochs or sample_steps) and not sampled_last_step:
            self.sample(epoch=self.epochs)
        self.accelerator.end_training()
        print(f"LoRA-Forge: training finished, peak VRAM {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB", flush=True)

    def step_logs(self, loss: float, average: float, scheduler) -> dict:
        logs = {"loss/current": loss, "loss/average": average}
        for group, lr in zip(self.optimizer.param_groups, scheduler.get_last_lr()):
            logs["lr/unet"] = lr
            if "d" in group:
                logs["lr/d*lr/unet"] = group["d"] * group["lr"]
                if "effective_lr" in group:
                    logs["lr/d*eff_lr/unet"] = group["d"] * group["effective_lr"]
        return logs


def main():
    config = read_config(sys.argv[1:])
    Trainer(config).train()


if __name__ == "__main__":
    main()
