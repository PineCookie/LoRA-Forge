"""Faster SDXL steps for sd-scripts' sdxl_original_unet, installed by run_entry.py before the trainer runs.

- flash attention (LORA_FORGE_SDXL_FLASH=1, set with sdpa when flash-attn is installed): CrossAttention's
  SDPA path calls flash_attn_func. PyTorch's Windows builds have no flash kernels for SDPA, so it falls back
  to the memory-efficient ones; flash-attn's are faster. Same attention, other summation order.
- per-block compile (LORA_FORGE_SDXL_COMPILE=1, GUI option compile): torch.compile of each transformer
  block's and resnet block's forward_body, inside sd-scripts' gradient checkpoint. Only the token count
  (transformer) or the height and width (resnet) are dynamic, so all bucket sizes share one graph per block
  shape. Resnets compile only their no-grad pass (under checkpointing, the first forward): with gradients,
  torch 2.14's compiled GroupNorm backward pins height and width to the first size seen.
On an SDXL LoRA (batch 2, 1024 px, a 16 GB card) the two took a step from 1.17 to 0.81 s with the same losses.

Kept free of lora_forge imports: run_entry.py loads it in the training process.
"""

import math

import torch

# Inductor guards each tensor below 2**31 elements; the widest tensors are a transformer's GEGLU input
# (tokens x 10240 at 1280 channels) and a resnet's input (up to 2560 channels)
INT32 = 2**31 - 1
RECOMPILE_LIMIT = 64


def install(flash: bool, compile_blocks: bool, batch_size: int) -> None:
    from library import sdxl_original_unet as unet

    if flash:
        install_flash(unet)
    if compile_blocks:
        install_compile(unet, max(1, batch_size))


def install_flash(unet) -> None:
    from flash_attn import flash_attn_func

    sdpa = unet.CrossAttention.forward_sdpa

    def forward_flash(self, x, context=None, mask=None):
        q = self.to_q(x)
        if mask is not None or q.dtype not in (torch.float16, torch.bfloat16):
            return sdpa(self, x, context, mask)
        context = (context if context is not None else x).to(x.dtype)
        k = self.to_k(context)
        v = self.to_v(context)
        batch, tokens, _ = q.shape
        h = self.heads
        out = flash_attn_func(q.view(batch, tokens, h, -1), k.view(batch, k.shape[1], h, -1), v.view(batch, v.shape[1], h, -1))
        return self.to_out[0](out.reshape(batch, tokens, -1))

    unet.CrossAttention.forward_sdpa = forward_flash
    print("LoRA-Forge: SDXL's SDPA attention runs on flash-attn", flush=True)


def install_compile(unet, batch_size: int) -> None:
    import torch._dynamo

    max_tokens = min(32768, INT32 // (batch_size * 10240))
    max_side = min(512, math.isqrt(INT32 // (batch_size * 2560)))

    def configure():
        # the checkpoint's recompute runs in the autograd thread, where Dynamo's config has its defaults
        config = torch._dynamo.config
        if config.recompile_limit != RECOMPILE_LIMIT:
            config.recompile_limit = RECOMPILE_LIMIT
            # only the dimensions marked below may become dynamic
            config.automatic_dynamic_shapes = False

    configure()
    transformer_body = torch.compile(unet.BasicTransformerBlock.forward_body, dynamic=None)
    resnet_body = torch.compile(unet.ResnetBlock2D.forward_body, dynamic=None)
    eager_resnet_body = unet.ResnetBlock2D.forward_body

    def transformer_forward_body(self, hidden_states, context=None, timestep=None):
        configure()
        # outside the range the shape compiles a graph of its own
        if 16 <= hidden_states.shape[1] <= max_tokens:
            torch._dynamo.mark_dynamic(hidden_states, 1, min=16, max=max_tokens)
        return transformer_body(self, hidden_states, context, timestep)

    def resnet_forward_body(self, x, emb):
        if torch.is_grad_enabled():
            return eager_resnet_body(self, x, emb)
        configure()
        if max(x.shape[2], x.shape[3]) <= max_side:
            torch._dynamo.mark_dynamic(x, 2, min=2, max=max_side)
            torch._dynamo.mark_dynamic(x, 3, min=2, max=max_side)
        return resnet_body(self, x, emb)

    unet.BasicTransformerBlock.forward_body = transformer_forward_body
    unet.ResnetBlock2D.forward_body = resnet_forward_body
    print(f"LoRA-Forge: SDXL compiles each transformer and resnet block (up to {max_tokens} tokens, "
          f"{max_side * 8} px a side in one graph)", flush=True)
