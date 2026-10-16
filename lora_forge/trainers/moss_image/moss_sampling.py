"""Timestep sampling for training and the preview sampler, both as OpenMOSS describes them.

Training: the MOSS-Image technical report's final pretraining interval (Eq. 4): Z ~ N(mu, s^2),
S = 1 - sigmoid(Z) truncated at the cutoff and rescaled onto the 1000-step grid, mixed with uniform
sampling. Inference: OpenMOSS's MOSSImageScheduler (shift 5 snapped to the training grid, Euler),
CFG with block 22 skipped in the negative pass.
"""

import math
import re

import torch

TRAIN_STEPS = 1000


def sample_sigma(generator, mean=-1.09, std=0.95, cutoff=0.95, uniform=0.01) -> float:
    if torch.rand((), generator=generator).item() < uniform:
        step = int(torch.randint(1, TRAIN_STEPS + 1, (), generator=generator).item())
    else:
        while True:
            z = mean + std * torch.randn((), generator=generator).item()
            s = 1 / (1 + math.exp(z))  # 1 - sigmoid(z)
            if s <= cutoff:
                break
        step = min(TRAIN_STEPS, max(1, math.ceil(TRAIN_STEPS * s / cutoff)))
    return step / TRAIN_STEPS


def inference_sigmas(steps: int, shift: float = 5.0) -> list:
    def grid(count, value):
        lower = value / (TRAIN_STEPS + value - 1)
        values = torch.linspace(1, lower, count)
        return value * values / (1 + (value - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(TRAIN_STEPS, 1.0).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=TRAIN_STEPS - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    return torch.unique_consecutive(training[nearest]).tolist() + [0.0]


@torch.no_grad()
def generate(model, positive, negative, width, height, steps=30, cfg=3.0, seed=0, uncond_skip=(22,), shift=5.0):
    device = positive.device
    generator = torch.Generator(device).manual_seed(seed)
    latents = torch.randn((1, model.latent_channels, height // 16, width // 16), generator=generator, device=device, dtype=torch.float32)
    sigmas = inference_sigmas(steps, shift)
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        prediction = model(latents, sigma * TRAIN_STEPS, positive).float()
        if cfg != 1.0:
            uncond = model(latents, sigma * TRAIN_STEPS, negative, skip_layers=uncond_skip).float()
            prediction = uncond + cfg * (prediction - uncond)
        latents = latents + (next_sigma - sigma) * prediction
    return latents


OPTION = re.compile(r"\s+--(\w+)(?=\s|$)")


def parse_prompt_line(line: str) -> dict:
    """sd-scripts' prompt line: `text --n negative --w 1024 --h 1024 --d seed --s steps --l cfg`
    (an empty `--n` is fine)."""
    parts = OPTION.split(" " + line.strip())
    prompt = {"prompt": parts[0].strip(), "negative": "", "width": 1024, "height": 1024, "seed": None, "steps": 30, "cfg": 3.0}
    names = {"n": "negative", "w": "width", "h": "height", "d": "seed", "s": "steps", "l": "cfg"}
    for name, value in zip(parts[1::2], parts[2::2]):
        key = names.get(name)
        if key is None:
            continue
        value = value.strip()
        prompt[key] = value if key == "negative" else (float(value) if key == "cfg" else int(value))
    prompt["width"] = max(16, prompt["width"] // 16 * 16)
    prompt["height"] = max(16, prompt["height"] // 16 * 16)
    return prompt


def read_prompt_file(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    return [parse_prompt_line(line) for line in lines if line and not line.startswith("#")]
