"""MOSS-Image DiT for LoRA training.

Ported from OpenMOSS's `pipeline_moss_image.py` (https://github.com/OpenMOSS/MOSS-Image,
Copyright OpenMOSS, Apache-2.0): same layers, state-dict names and numerics. Changes: flash-attn
when available, gradient checkpointing per block, an optional compiled block function, and one
sample per call (the training loop runs batch 1; each sample keeps its true text length).
"""

import glob
import os

import torch
from torch import nn
from torch.nn import functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

# set by the trainer: "flash_attn" or "sdpa"
ATTENTION = {"mode": "sdpa"}


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope)
    k = rotate(k.reshape(shape), rope)
    v = v.reshape(shape)
    q, k = q.to(v.dtype), k.to(v.dtype)
    if ATTENTION["mode"] == "flash_attn":
        out = flash_attn_func(q, k, v)
    else:
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    return out.flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](situ(hidden, gate)) * gate_f[:, None]


def rope_table(height, width, text_length, device, head_dim=128, base=10000.0):
    """(r, f, h, w) coordinates: the image sits after the text on f, its rows and columns centered."""
    num_bands = head_dim // 8
    bands = (1 / (base ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))).to(device)
    h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
    text[:, 1] = torch.arange(text_length, device=device)
    phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageDiT(nn.Module):
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42, text_dim=4096, freq_dim=256, latent_channels=128):
        super().__init__()
        self.freq_dim = freq_dim
        self.latent_channels = latent_channels
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.gradient_checkpointing = False
        # optional replacement for Block.forward (the compiled one), called as block_fn(block, x, rope, mod)
        self.block_fn = None

    def forward(self, latents, timestep, context, skip_layers=()):
        """latents [1, 128, H, W] (patchified, normalized FLUX.2 latents); timestep = 1000 * sigma;
        context [1, L, 4096]: the text encoder's last hidden state at its true length."""
        _, _, height, width = latents.shape
        times = torch.as_tensor(timestep, device=latents.device, dtype=torch.float32).reshape(-1)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        dtype = self.time_embedding[0].weight.dtype
        time = self.time_embedding(embedding.to(dtype)).float()
        mod = self.shared_block_modulation(time.to(dtype)).view(1, 6, -1)
        head = self.video_head_modulation(time.to(dtype)).view(1, 2, -1)
        video = self.video_embedder(latents.flatten(2).transpose(1, 2).to(dtype).contiguous())
        rope = rope_table(height, width, context.shape[1], latents.device).to(video.dtype)
        text = self.text_embedder(context, rope[:, height * width:])
        x = torch.cat((video, text), 1)
        for index, layer in enumerate(self.layers):
            if index in skip_layers:
                continue
            if self.block_fn is not None:
                call = lambda x_, rope_, mod_, layer=layer: self.block_fn(layer, x_, rope_, mod_)
            else:
                call = layer
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(call, x, rope, mod, use_reentrant=False)
            else:
                x = call(x, rope, mod)
        x = self.final_norm_video(x[:, :height * width])
        x = x * (1 + head[:, 1, None]) + head[:, 0, None]
        x = self.final_linear_video(x)
        return x.transpose(1, 2).reshape(1, -1, height, width)


def _weight_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {path}")
        return files
    return [path]


def load_dit(path, device, dtype=torch.bfloat16):
    """The DiT from ComfyUI's single file (q/k/v fused as `layers.N.proj.qkv`, split here) or from
    OpenMOSS's `transformer` folder (sharded, separate q/k/v)."""
    from safetensors import safe_open

    with torch.device("meta"):
        model = MOSSImageDiT()
    state = {}
    for file in _weight_files(path):
        with safe_open(file, framework="pt", device=str(device)) as f:
            for key in f.keys():
                if key == "rope.bands":  # derived constants, recomputed in fp32
                    continue
                name = key[len("diffusion_model."):] if key.startswith("diffusion_model.") else key
                tensor = f.get_tensor(key).to(dtype)
                if name.endswith(".proj.qkv.weight"):
                    for part_name, part in zip(("q", "k", "v"), tensor.chunk(3, 0)):
                        state[name.replace(".qkv.", f".{part_name}.")] = part.contiguous()
                else:
                    state[name] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise ValueError(f"{path} is not a MOSS-Image DiT: missing {missing[:3]}, unexpected {unexpected[:3]}")
    model.requires_grad_(False)
    return model.eval()
