"""LoRA on the MOSS-Image DiT's block Linears, saved with kohya-style keys.

`lora_unet_layers_{i}_proj_{name}` with lora_down / lora_up / alpha: ComfyUI loads these as they are,
mapping q / k / v onto slices of its fused qkv weight.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
PREFIX = "lora_unet"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.dropout = dropout
        device = base.weight.device
        self.down = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.multiplier = 1.0

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        if self.multiplier == 0:
            return out
        hidden = F.linear(x, self.down.to(x.dtype))
        if self.dropout and self.training:
            hidden = F.dropout(hidden, self.dropout)
        return out + F.linear(hidden, self.up.to(x.dtype)) * (self.scale * self.multiplier)


class LoRANetwork:
    def __init__(self, model, rank: int, alpha: float, dropout: float = 0.0):
        self.modules = {}
        for index, layer in enumerate(model.layers):
            for name in TARGETS:
                wrapped = LoRALinear(layer.proj[name], rank, alpha, dropout)
                layer.proj[name] = wrapped
                self.modules[f"{PREFIX}_layers_{index}_proj_{name}"] = wrapped
        self.rank, self.alpha = rank, alpha

    def parameters(self):
        return [p for module in self.modules.values() for p in (module.down, module.up)]

    def set_multiplier(self, value: float):
        for module in self.modules.values():
            module.multiplier = value

    def train(self, mode: bool = True):
        for module in self.modules.values():
            module.train(mode)

    def state_dict(self, dtype):
        state = {}
        for key, module in self.modules.items():
            state[f"{key}.lora_down.weight"] = module.down.detach().to(dtype).cpu().contiguous()
            state[f"{key}.lora_up.weight"] = module.up.detach().to(dtype).cpu().contiguous()
            state[f"{key}.alpha"] = torch.tensor(float(module.alpha), dtype=dtype)
        return state

    def save(self, path: str, dtype, metadata: dict):
        from safetensors.torch import save_file

        save_file(self.state_dict(dtype), path, metadata={k: str(v) for k, v in metadata.items() if v is not None})

    def load_weights(self, path: str):
        """Continue from a saved LoRA of the same rank (network_weights)."""
        from safetensors.torch import load_file

        state = load_file(path)
        for key, module in self.modules.items():
            down, up = state.get(f"{key}.lora_down.weight"), state.get(f"{key}.lora_up.weight")
            if down is None or up is None:
                raise ValueError(f"{path} has no weights for {key}")
            if tuple(down.shape) != tuple(module.down.shape):
                raise ValueError(f"{path} has rank {down.shape[0]}, network_dim is {module.rank}")
            alpha = state.get(f"{key}.alpha")
            # keep the saved LoRA's effect when its alpha differs from network_alpha
            factor = float(alpha) / module.alpha if alpha is not None else module.rank / module.alpha
            module.down.data.copy_(down.float())
            module.up.data.copy_(up.float() * factor)
