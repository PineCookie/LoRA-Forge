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


def install_anima_patches():
    import torch
    from library import anima_models, compile_utils

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

    def compile_inner(args, transformer, target_blocks, disable_linear):
        blocks = [block for group in target_blocks for block in group]
        if not any(getattr(block, "unsloth_offload_checkpointing", False) and getattr(block, "gradient_checkpointing", False) for block in blocks):
            return compile_transformer(args, transformer, target_blocks, disable_linear)
        # sd-scripts' settings (cache size limit, static nn.Module properties) without compiling blocks
        compile_transformer(args, transformer, [], False)
        if disable_linear:
            for block in blocks:
                compile_utils.disable_linear_from_compile(block)
        dynamic = None if args.compile_dynamic is None else {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]
        print("LoRA-Forge: unsloth offload checkpointing is on, so each block's inner _forward is compiled", flush=True)
        for block in blocks:
            block._forward = torch.compile(
                block._forward, backend=args.compile_backend, mode=args.compile_mode, dynamic=dynamic, fullgraph=args.compile_fullgraph
            )
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
        install_tracker(events)
        install_progress(events, compile_monitor)

    if os.path.basename(trainer) == "anima_train_network.py" and os.environ.get(PATCHES_ENV, "1") != "0":
        install_anima_patches()

    runpy.run_path(trainer, run_name="__main__")


if __name__ == "__main__":
    main()
