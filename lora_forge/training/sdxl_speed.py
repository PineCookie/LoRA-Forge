"""Faster SDXL steps for sd-scripts' sdxl_original_unet, installed by run_entry.py before the trainer runs.

Per-block compile (LORA_FORGE_SDXL_COMPILE=1, GUI option compile): torch.compile of each transformer block's
and resnet block's forward_body, inside sd-scripts' gradient checkpoint. torch.compile's automatic dynamic
shapes let the bucket sizes share graphs: a second size makes the shapes dynamic. Resnets compile only their
no-grad pass (under checkpointing, the first forward): with gradients, the compiled GroupNorm backward pinned
height and width to the first size seen (torch 2.14).
On an SDXL LoRA (batch 2, 1024 px, a 16 GB card, flash attention) it took a step from 0.97 to 0.81 s with the
same losses (measured on torch 2.14).

Kept free of lora_forge imports: run_entry.py loads it in the training process.
"""

import torch
import torch._dynamo

RECOMPILE_LIMIT = 64


def install_compile() -> None:
    from library import sdxl_original_unet as unet

    def configure():
        # the checkpoint's recompute runs in the autograd thread, where Dynamo's config has its defaults
        if torch._dynamo.config.recompile_limit != RECOMPILE_LIMIT:
            torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT

    configure()
    transformer_body = torch.compile(unet.BasicTransformerBlock.forward_body, dynamic=None)
    resnet_body = torch.compile(unet.ResnetBlock2D.forward_body, dynamic=None)
    eager_resnet_body = unet.ResnetBlock2D.forward_body

    def transformer_forward_body(self, hidden_states, context=None, timestep=None):
        configure()
        return transformer_body(self, hidden_states, context, timestep)

    def resnet_forward_body(self, x, emb):
        if torch.is_grad_enabled():
            return eager_resnet_body(self, x, emb)
        configure()
        return resnet_body(self, x, emb)

    unet.BasicTransformerBlock.forward_body = transformer_forward_body
    unet.ResnetBlock2D.forward_body = resnet_forward_body
    print("LoRA-Forge: SDXL compiles each transformer and resnet block", flush=True)
