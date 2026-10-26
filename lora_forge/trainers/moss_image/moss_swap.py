"""Block swap for MOSS-Image LoRA training: the frozen Linear weights of some blocks live in RAM and are copied
to the GPU just before their block runs, never back (the base model does not change).

The design follows kohya-ss sd-scripts' / Musubi Tuner's H2D-only block swap (`LoRAStreamOffloader`,
Apache-2.0): the swapped ("streaming") blocks are spread evenly over the model, each block's weights are one
flat CPU buffer (pinned for fast asynchronous copies), and they are copied into a ring of two GPU buffers, one
being computed on while the next one loads. The LoRA weights and everything else stay on the GPU. It needs
gradient checkpointing: a ring slot is overwritten as soon as its block is done, and the backward pass reads
the weights again when it recomputes the block.
"""

import torch

RING = 2
ALIGN = 256  # byte offset of each weight in a block's flat buffer, so every dtype view is valid


def _swap_tensors(block):
    """(module, attribute name) of every frozen Linear weight in a block: nn.Linear (a parameter, also inside a
    LoRALinear) or Fp16Linear (a buffer). The LoRA's own weights are not among them."""
    jobs = []
    for linear in block.proj.values():
        base = getattr(linear, "base", linear)
        if "weight" in base._parameters or "weight" in base._buffers:
            jobs.append((base, "weight"))
    return jobs


def _wrap(jobs, tensors):
    """The tensors as the modules hold them: frozen Parameters for parameters, plain tensors for buffers."""
    return [torch.nn.Parameter(t, requires_grad=False) if name in module._parameters else t
            for (module, name), t in zip(jobs, tensors)]


def _bind(jobs, tensors):
    for (module, name), tensor in zip(jobs, tensors):
        if name in module._parameters:
            module._parameters[name] = tensor
        else:
            module._buffers[name] = tensor


class BlockSwap:
    def __init__(self, blocks, blocks_to_swap: int, device, pinned: bool = True):
        self.blocks = blocks
        count = len(blocks)
        # evenly spaced streaming blocks (midpoint formula, distinct for blocks_to_swap <= count)
        self.stream = sorted({((2 * i + 1) * count) // (2 * blocks_to_swap) for i in range(blocks_to_swap)})
        self.rank = {block: k for k, block in enumerate(self.stream)}
        self.ring = min(RING, len(self.stream))
        self.device = torch.device(device)
        self.pinned = pinned
        self.jobs = {index: _swap_tensors(blocks[index]) for index in self.stream}
        self.copy_stream = torch.cuda.Stream(self.device)
        self.in_slot = [None] * self.ring  # slot -> block index whose weights it holds
        self.ready = {}  # block index -> event: its copy into the ring finished
        self.free = [None] * self.ring  # slot -> event: compute is done with the slot's block

    def place(self, model):
        """Called once on a model loaded in RAM: moves everything but the streaming blocks' frozen Linear weights to
        the GPU and keeps those as one flat (pinned) CPU buffer per block, with a ring of GPU buffers for them."""
        self.masters, self.flat = {}, {}
        layout = None
        for index in self.stream:
            tensors = [getattr(module, name) for module, name in self.jobs[index]]
            if layout is None:
                offsets, total = [], 0
                for tensor in tensors:
                    total = (total + ALIGN - 1) // ALIGN * ALIGN
                    offsets.append(total)
                    total += tensor.numel() * tensor.element_size()
                layout = (offsets, total, [(t.shape, t.dtype) for t in tensors])
            if [(t.shape, t.dtype) for t in tensors] != layout[2]:
                raise ValueError(f"block {index} has differently shaped weights than the other blocks")
            flat = torch.empty(layout[1], dtype=torch.uint8)
            if self.pinned:
                flat = flat.pin_memory()
            views = self._views(flat, layout)
            for view, tensor in zip(views, tensors):
                view.copy_(tensor.detach().cpu())
            # empty placeholders, so that moving the model to the GPU does not take these weights along
            _bind(self.jobs[index], _wrap(self.jobs[index], [torch.empty(0, dtype=t.dtype) for t in tensors]))
            self.masters[index], self.flat[index] = _wrap(self.jobs[index], views), flat
        model.to(self.device)
        for index in self.stream:
            _bind(self.jobs[index], self.masters[index])
        self.layout = layout
        template = self.jobs[self.stream[0]]
        self.ring_flat = [torch.empty(layout[1], dtype=torch.uint8, device=self.device) for _ in range(self.ring)]
        self.ring_views = [_wrap(template, self._views(flat, layout)) for flat in self.ring_flat]
        self.start()

    @staticmethod
    def _views(flat, layout):
        offsets, _, specs = layout
        return [flat[o:o + torch.Size(shape).numel() * dtype.itemsize].view(dtype).view(shape)
                for o, (shape, dtype) in zip(offsets, specs)]

    def _load(self, rank: int):
        """Copies the streaming block at `rank` into its ring slot (after compute is done with the slot)."""
        index, slot = self.stream[rank], rank % self.ring
        if self.in_slot[slot] == index:
            _bind(self.jobs[index], self.ring_views[slot])
            return
        previous = self.in_slot[slot]
        if previous is not None:
            _bind(self.jobs[previous], self.masters[previous])  # free: the CPU copy is identical
            self.ready.pop(previous, None)
        with torch.cuda.stream(self.copy_stream):
            if self.free[slot] is not None:
                self.copy_stream.wait_event(self.free[slot])
            self.ring_flat[slot].copy_(self.flat[index], non_blocking=self.pinned)
            self.ready[index] = self.copy_stream.record_event()
        _bind(self.jobs[index], self.ring_views[slot])
        self.in_slot[slot] = index

    def start(self):
        """Before every forward pass: the first streaming blocks into the ring (no copies after a backward pass or
        a preview, which end with them loaded)."""
        for rank in range(self.ring):
            self._load(rank)

    def before(self, index: int):
        """Before block `index` runs: waits on the GPU (not the CPU) until its weights are in place."""
        if index not in self.rank:
            return
        rank = self.rank[index]
        if self.in_slot[rank % self.ring] != index:
            self._load(rank)
        event = self.ready.get(index)
        if event is not None:
            torch.cuda.current_stream(self.device).wait_event(event)

    def after_forward(self, index: int, backward_follows: bool):
        """After block `index` ran forward: frees its slot and starts loading the block `ring` places ahead."""
        if index not in self.rank:
            return
        rank = self.rank[index]
        self.free[rank % self.ring] = torch.cuda.current_stream(self.device).record_event()
        if rank + self.ring < len(self.stream):
            self._load(rank + self.ring)
        elif not backward_follows and rank == len(self.stream) - 1:
            self.start()  # previews: the next pass begins with the first streaming blocks again

    def after_backward(self, index: int):
        """After block `index`'s backward pass: frees its slot, loads the block `ring` places behind, and makes
        sure the previous block is ready for its own backward pass."""
        if index in self.rank:
            rank = self.rank[index]
            self.free[rank % self.ring] = torch.cuda.current_stream(self.device).record_event()
            if rank - self.ring >= 0:
                self._load(rank - self.ring)
        if index > 0:
            self.before(index - 1)


class _BackwardHook(torch.autograd.Function):
    """Identity on a block's input; its backward runs once the block's own backward (and recompute) is done."""

    @staticmethod
    def forward(ctx, x, swap, index):
        ctx.swap, ctx.index = swap, index
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        ctx.swap.after_backward(ctx.index)
        return grad, None, None


def hook_input(x, swap, index):
    return _BackwardHook.apply(x, swap, index)
