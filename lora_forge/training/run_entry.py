"""Runs an sd-scripts trainer and records its progress for the web UI.

Launched by accelerate in place of the trainer: `run_entry.py <trainer.py> <trainer args...>`.
The trainer itself runs unchanged; this only adds observers, writing JSON lines to the file
named by LORA_FORGE_EVENTS:

- metrics: an accelerate tracker. sd-scripts logs loss, learning rates, norms and validation
  loss to every tracker it has, and logs nothing without one, so this works without TensorBoard.
- progress: tqdm bars (training steps, caching, validation), at most one event per second each.
- epoch: the "epoch N/M" line sd-scripts prints at each epoch start.
- compile: how many graphs torch.compile has built, and a notice when it hits its recompile
  limit (from then on new shapes run uncompiled, silently except for one warning).

For Anima it also patches two torch.compile problems of sd-scripts' per-block compile (set
LORA_FORGE_PATCHES=0 to run sd-scripts exactly as it is, e.g. to compare):
- block 0 gets an input without requires_grad (the embedders are frozen), blocks 1+ one with it,
  so every bucket shape compiled two graphs and hit the recompile limit twice as fast;
- with unsloth_offload_checkpointing the compiled block only calls the dynamo-disabled
  checkpoint function, so nothing was compiled; the block's inner _forward is compiled instead.
And with LORA_FORGE_COMPILE_DYNAMIC_SEQ=1 (GUI option compile_dynamic_seq) it compiles one graph
for all bucket sizes: only the token count is dynamic (see bounded_block_call). On the user's
Anima set (20 buckets, batch 2) that is 2 graphs instead of 28 with matching loss, and it avoids
pytorch#193194, which only hits when every dimension may become dynamic.

Kept free of lora_forge imports: it runs in the training process, with sd-scripts' directory
first on sys.path.
"""

import json
import os
import re
import runpy
import sys
import threading
import time

EVENTS_ENV = "LORA_FORGE_EVENTS"
PATCHES_ENV = "LORA_FORGE_PATCHES"
# set by the runner from the GUI option compile_dynamic_seq (a LoRA-Forge option, not sd-scripts')
DYNAMIC_SEQ_ENV = "LORA_FORGE_COMPILE_DYNAMIC_SEQ"
PROGRESS_INTERVAL = 1.0


class EventWriter:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.file = None

    def write(self, kind: str, **fields):
        event = {"t": round(time.time(), 3), "type": kind, **fields}
        line = json.dumps(event, ensure_ascii=False, default=str, allow_nan=False) + "\n"
        with self.lock:
            if self.file is None:
                self.file = open(self.path, "a", encoding="utf-8")
            self.file.write(line)
            self.file.flush()


def _number(value):
    """A finite float for JSON, or None (tensors, NaN, inf and non-numbers are dropped)."""
    try:
        value = float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError, RuntimeError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def install_tracker(events: EventWriter):
    import accelerate
    from accelerate.tracking import GeneralTracker

    class EventTracker(GeneralTracker):
        name = "lora_forge"
        requires_logging_directory = False
        main_process_only = True

        def __init__(self):
            super().__init__()

        @property
        def tracker(self):
            return self

        def log(self, values: dict, step=None, **kwargs):
            numbers = {key: _number(value) for key, value in values.items()}
            numbers = {key: value for key, value in numbers.items() if value is not None}
            if numbers:
                events.write("metrics", step=step, values=numbers)

    original_init = accelerate.Accelerator.__init__

    def init(self, *args, **kwargs):
        log_with = kwargs.get("log_with")
        if log_with is None:
            log_with = []
        elif not isinstance(log_with, (list, tuple)):
            log_with = [log_with]
        kwargs["log_with"] = [*log_with, EventTracker()]
        original_init(self, *args, **kwargs)

    accelerate.Accelerator.__init__ = init

    epoch_line = re.compile(r"^\s*epoch (\d+)/(\d+)\s*$")
    original_print = accelerate.Accelerator.print

    def print_(self, *args, **kwargs):
        original_print(self, *args, **kwargs)
        match = epoch_line.match(" ".join(str(arg) for arg in args))
        if match:
            events.write("epoch", epoch=int(match.group(1)), total=int(match.group(2)))

    accelerate.Accelerator.print = print_


class CompileMonitor:
    """Reports torch.compile's graph count when it changes and when it hits the recompile limit."""

    def __init__(self, events: EventWriter):
        self.events = events
        self.graphs = 0
        self.limit_reported = False

    def install(self):
        import logging

        monitor = self

        class LimitHandler(logging.Handler):
            def emit(self, record):
                try:
                    message = record.getMessage()
                except Exception:
                    return
                if "recompile_limit" in message or "cache_size_limit" in message:
                    monitor.limit_hit(message)

        logging.getLogger("torch._dynamo").addHandler(LimitHandler(level=logging.WARNING))

    def limit_hit(self, message: str):
        if self.limit_reported:
            return
        self.limit_reported = True
        self.check()
        import torch._dynamo.config as dynamo_config

        limit = getattr(dynamo_config, "recompile_limit", None) or getattr(dynamo_config, "cache_size_limit", None)
        self.events.write("notice", code="recompile_limit", limit=limit, graphs=self.graphs, text=message.splitlines()[0])

    def check(self):
        # only once torch._dynamo is in use; reading the counter is a dict lookup
        utils = sys.modules.get("torch._dynamo.utils")
        if utils is None:
            return
        graphs = utils.counters["stats"]["unique_graphs"]
        if graphs != self.graphs:
            self.graphs = graphs
            self.events.write("compile", graphs=graphs)


def bounded_block_call(fn):
    """Call `fn` (a compiled Anima block, or its _forward) with the token grid flattened to one
    sequence axis that torch.compile treats as dynamic, so every bucket shows the same shape.

    Block._forward flattens (B, T, H, W, D) to (B, T*H*W, D) for attention and back anyway, and the
    positions come precomputed in rope_emb_L_1_1_D, so (B, 1, L, 1, D) is the same computation in
    the same token order. Each call marks its inputs again: a block's input is the previous block's
    output, which Dynamo would otherwise see as static.
    """
    import torch

    mark = torch._dynamo.mark_dynamic

    def call(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, attn_params, use_fp32=False,
             rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None):
        B, T, H, W, D = x_B_T_H_W_D.shape
        L = T * H * W
        # the batch is dynamic too (2 .. train_batch_size), so smaller last batches of a bucket share
        # the graph; batch 1 always gets its own (PyTorch never treats size 1 as dynamic)
        dynamic_batch = BATCH_MAX > 2 and 2 <= B <= BATCH_MAX
        seq_max = seq_max_for(BATCH_MAX if dynamic_batch else B, D)
        # outside the range (a huge bucket at a large batch) the shape is compiled statically instead
        dynamic = SEQ_MIN <= L <= seq_max
        if not dynamic:
            report_outside_range(L, B, seq_max)
        x = x_B_T_H_W_D.reshape(B, 1, L, 1, D)
        if dynamic:
            mark(x, 2, min=SEQ_MIN, max=seq_max)
        if rope_emb_L_1_1_D is not None and dynamic:
            mark(rope_emb_L_1_1_D, 0, min=SEQ_MIN, max=seq_max)
        if extra_per_block_pos_emb is not None:
            extra_per_block_pos_emb = extra_per_block_pos_emb.reshape(extra_per_block_pos_emb.shape[0], 1, L, 1, D)
            if dynamic:
                mark(extra_per_block_pos_emb, 2, min=SEQ_MIN, max=seq_max)
        if dynamic and dynamic_batch:
            for tensor in (x, emb_B_T_D, crossattn_emb, adaln_lora_B_T_3D, extra_per_block_pos_emb):
                if tensor is not None and tensor.shape[0] == B:
                    mark(tensor, 0, min=2, max=BATCH_MAX)
        out = fn(x, emb_B_T_D, crossattn_emb, attn_params, use_fp32, rope_emb_L_1_1_D, adaln_lora_B_T_3D, extra_per_block_pos_emb)
        return out.reshape(B, T, H, W, D)

    return call


# token counts the dynamic graph covers: a 64x64 px bucket up to 2048x2048 px (16 px per token)
SEQ_MIN, SEQ_MAX = 16, 16384
# train_batch_size, set by the runner; the upper bound of the dynamic batch dimension
BATCH_MAX = int(os.environ.get("LORA_FORGE_BATCH_SIZE", "0") or 0)


# set by main(): writes a notice to the run's events (the Runs page shows it)
notify = None
_reported_shapes = set()


def report_outside_range(tokens: int, batch: int, seq_max: int):
    """Once per shape: it is compiled separately, one more graph."""
    if (tokens, batch) in _reported_shapes:
        return
    _reported_shapes.add((tokens, batch))
    print(f"LoRA-Forge: {tokens} tokens at batch {batch} is outside compile_dynamic_seq's range "
          f"({SEQ_MIN}..{seq_max}); this shape is compiled separately", flush=True)
    if notify is not None:
        notify("notice", code="dynamic_seq_outside", tokens=tokens, batch=batch, min=SEQ_MIN, max=seq_max)


def bounded_patch_problem(anima_models) -> "str | None":
    """Why bounded_block_call cannot be used with this sd-scripts version, or None. It relies on the
    block's argument order and on Block._forward flattening (b t h w d) to (b (t h w) d) itself."""
    import inspect

    expected = ["x_B_T_H_W_D", "emb_B_T_D", "crossattn_emb", "attn_params", "use_fp32",
                "rope_emb_L_1_1_D", "adaln_lora_B_T_3D", "extra_per_block_pos_emb"]
    for method in ("forward", "_forward"):
        params = list(inspect.signature(getattr(anima_models.Block, method)).parameters)[1:]
        if params != expected:
            return f"Block.{method} takes ({', '.join(params)})"
    try:
        source = inspect.getsource(anima_models.Block._forward)
    except (OSError, TypeError):
        return "the source of Block._forward is not available"
    if source.count('"b t h w d -> b (t h w) d"') < 2:
        return "Block._forward no longer flattens the token grid for attention"
    return None


def seq_max_for(batch: int, dim: int) -> int:
    """Largest token count to mark dynamic. Inductor guards that its largest tensor, the MLP's
    hidden activation (batch, L, 4 * dim), stays 32-bit indexable over the whole range:
    batch * 4 * dim * L < 2**31 (the guard a too large bound failed on, on the GPU)."""
    return max(SEQ_MIN, min(SEQ_MAX, (2**31 - 1) // (max(1, batch) * dim * 4)))


def install_anima_patches(dynamic_seq: bool = False):
    import torch
    from library import anima_models, compile_utils

    if dynamic_seq:
        problem = bounded_patch_problem(anima_models)
        if problem:
            print(f"LoRA-Forge: compile_dynamic_seq is off for this sd-scripts version ({problem}); "
                  "compiling with static shapes", flush=True)
            dynamic_seq = False

    # one graph per shape: the block-stack input requires grad like every later block's input
    prepare = anima_models.Anima.prepare_embedded_sequence

    def prepare_embedded_sequence(self, *args, **kwargs):
        x, *rest = prepare(self, *args, **kwargs)
        if self.training and torch.is_grad_enabled() and not x.requires_grad:
            x = x.requires_grad_()
        return (x, *rest)

    anima_models.Anima.prepare_embedded_sequence = prepare_embedded_sequence

    # unsloth offload checkpointing: compile what runs inside the checkpoint, not the wrapper
    compile_transformer = compile_utils.compile_transformer

    class BoundedBlock(torch.nn.Module):
        """An eager shell around a compiled block that flattens and marks its input (see
        bounded_block_call). The block stays a child, so its parameters move with .to()."""

        def __init__(self, compiled):
            super().__init__()
            self.compiled = compiled
            self._call = bounded_block_call(compiled)

        def forward(self, *args, **kwargs):
            return self._call(*args, **kwargs)

    def compile_inner(args, transformer, target_blocks, disable_linear):
        blocks = [block for group in target_blocks for block in group]
        unsloth = any(getattr(block, "unsloth_offload_checkpointing", False) and getattr(block, "gradient_checkpointing", False) for block in blocks)
        # block swap indexes the block list itself; keep sd-scripts' static compile there
        bounded = dynamic_seq and not disable_linear
        if dynamic_seq and disable_linear:
            print("LoRA-Forge: compile_dynamic_seq is not used with blocks_to_swap; compiling with static shapes", flush=True)
        if not unsloth and not bounded:
            return compile_transformer(args, transformer, target_blocks, disable_linear)

        # sd-scripts' settings (cache size limit, static nn.Module properties) without compiling blocks
        compile_transformer(args, transformer, [], False)
        if disable_linear:
            for block in blocks:
                compile_utils.disable_linear_from_compile(block)
        if bounded:
            # only the marked sequence axis is dynamic: no automatic dynamic shapes (pytorch#193194)
            torch._dynamo.config.automatic_dynamic_shapes = False
            dynamic = None
            print(f"LoRA-Forge: compile_dynamic_seq: one graph for every token count from {SEQ_MIN} to {SEQ_MAX}", flush=True)
        else:
            dynamic = None if args.compile_dynamic is None else {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]
        options = dict(backend=args.compile_backend, mode=args.compile_mode, dynamic=dynamic, fullgraph=args.compile_fullgraph)

        if unsloth:
            # compile what runs inside the dynamo-disabled checkpoint, not the wrapper
            print("LoRA-Forge: unsloth offload checkpointing is on, so each block's inner _forward is compiled", flush=True)
            for block in blocks:
                compiled = torch.compile(block._forward, **options)
                block._forward = bounded_block_call(compiled) if bounded else compiled
            return transformer

        for group in target_blocks:
            for i, block in enumerate(group):
                group[i] = BoundedBlock(torch.compile(block, **options))
        return transformer

    compile_utils.compile_transformer = compile_inner


def install_progress(events: EventWriter, compile_monitor: "CompileMonitor"):
    import tqdm as tqdm_module

    base = tqdm_module.tqdm

    class ProgressBar(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._lf_reported = 0.0
            if not self.disable and self.total:
                self._lf_report()

        def _lf_report(self, final: bool = False):
            info = self.format_dict
            elapsed = _number(info.get("elapsed"))
            rate = _number(info.get("rate"))
            # sd-scripts' step bar uses smoothing=0, where tqdm leaves the rate to be the average
            if rate is None and elapsed:
                rate = (self.n - (info.get("initial") or 0)) / elapsed or None
            events.write(
                "progress",
                desc=(self.desc or "").rstrip(": "),
                n=self.n,
                total=self.total,
                rate=rate,
                elapsed=elapsed,
                done=final,
            )
            self._lf_reported = time.monotonic()

        def update(self, n=1):
            result = super().update(n)
            if not self.disable and self.total and time.monotonic() - self._lf_reported >= PROGRESS_INTERVAL:
                self._lf_report()
                compile_monitor.check()
            return result

        def close(self):
            if not self.disable and self.total and not getattr(self, "_lf_closed", False):
                self._lf_closed = True
                self._lf_report(final=True)
            super().close()

    # sd-scripts uses `from tqdm import tqdm`, so the package attribute is the one to replace
    tqdm_module.tqdm = ProgressBar


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_entry.py <trainer.py> [trainer args...]")
    trainer = os.path.abspath(sys.argv[1])
    sys.argv = [trainer, *sys.argv[2:]]
    # as if the trainer were started directly: its folder first on the path, not this one
    sys.path[0] = os.path.dirname(trainer)

    events_path = os.environ.get(EVENTS_ENV)
    # with several GPUs only the main process reports (trackers are main-process-only anyway)
    if events_path and os.environ.get("RANK", "0") == "0":
        events = EventWriter(events_path)
        events.write("start", pid=os.getpid())
        compile_monitor = CompileMonitor(events)
        compile_monitor.install()
        global notify
        notify = events.write
        install_tracker(events)
        install_progress(events, compile_monitor)

    if os.path.basename(trainer) == "anima_train_network.py" and os.environ.get(PATCHES_ENV, "1") != "0":
        install_anima_patches(dynamic_seq=os.environ.get(DYNAMIC_SEQ_ENV) == "1")

    runpy.run_path(trainer, run_name="__main__")


if __name__ == "__main__":
    main()
