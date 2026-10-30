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
With LORA_FORGE_ACTIVATION_MEMORY_BUDGET (GUI option activation_memory_budget, a fraction or `auto`)
torch.compile's recompute planner replaces gradient checkpointing (install_memory_budget), with
flash-attn's forward made visible to that planner (plan_flash_attention). A run that fails for lack of
VRAM gets an out_of_memory notice; on Windows, a run that spills VRAM into system RAM (much slower,
but no error) gets a vram_spill notice (SpillMonitor).
For every trainer, ProdigyPlusScheduleFree gets a batched step with the same math
(batched_ppsf.py); for SDXL, per-block compile (sdxl_speed.py). LORA_FORGE_PATCHES=0
turns these off too.

The output goes to the GUI through a pipe: install_console_output() keeps tqdm bars full width and
rich's colors, and install_log_filter() shortens network_reg_lrs / network_reg_dims logging.

Kept free of lora_forge imports: it runs in the training process, with sd-scripts' directory
first on sys.path.
"""

import json
import math
import os
import re
import runpy
import sys
import threading
import time

EVENTS_ENV = "LORA_FORGE_EVENTS"
PATCHES_ENV = "LORA_FORGE_PATCHES"
# set by the runner when its console shows ANSI colors
ANSI_ENV = "LORA_FORGE_ANSI"
# GUI option activation_memory_budget (Anima with compile), in place of gradient checkpointing
MEMORY_BUDGET_ENV = "LORA_FORGE_ACTIVATION_MEMORY_BUDGET"
# GUI option fp16_accumulation (Anima, fp16 training precisions): fp16 matrix multiplies accumulate in fp16
FP16_ACCUMULATION_ENV = "LORA_FORGE_FP16_ACCUMULATION"
# SDXL (sdxl_speed.py): per-block compile (GUI option compile)
SDXL_COMPILE_ENV = "LORA_FORGE_SDXL_COMPILE"
SDXL_TRAINERS = ("sdxl_train_network.py", "sdxl_train.py")
# Krea 2 (musubi-tuner): JSON with the caching scripts' arguments, see run_krea2_caching
KREA2_CACHE_ENV = "LORA_FORGE_KREA2_CACHE"
KREA2_TURBO_LORA_ENV = "LORA_FORGE_KREA2_TURBO_LORA"
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


# set by main(): writes a notice to the run's events (the Runs page shows it)
notify = None


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
        unsloth = any(getattr(block, "unsloth_offload_checkpointing", False) and getattr(block, "gradient_checkpointing", False) for block in blocks)
        if not unsloth:
            return compile_transformer(args, transformer, target_blocks, disable_linear)

        # sd-scripts' settings (cache size limit, static nn.Module properties) without compiling blocks
        compile_transformer(args, transformer, [], False)
        if disable_linear:
            for block in blocks:
                compile_utils.disable_linear_from_compile(block)
        dynamic = None if args.compile_dynamic is None else {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]
        options = dict(backend=args.compile_backend, mode=args.compile_mode, dynamic=dynamic, fullgraph=args.compile_fullgraph)
        print("LoRA-Forge: unsloth offload checkpointing is on, so each block's inner _forward is compiled", flush=True)
        for block in blocks:
            block._forward = torch.compile(block._forward, **options)
        return transformer

    compile_utils.compile_transformer = compile_inner

GIB = 2**30
# activation_memory_budget = auto, measured on Anima (rank-32 LoRA, 16 GB card): the peak VRAM above
# what is allocated when training starts grows with the tokens of the largest batch (batch size x
# px / 256). Per token: gradient checkpointing's own part, and what each 1.0 of budget adds (the
# upper end of what was measured; the planner saves tensors in steps, not smoothly).
CHECKPOINTING_GIB_PER_TOKEN = 3.9e-4
BUDGET_GIB_PER_TOKEN = 2.4e-3
# left free for what the estimate does not cover (kernel workspaces, preview images)
BUDGET_MARGIN_GIB = 0.75
# below this the gain is small: gradient checkpointing stays. Above the maximum the speed barely
# grows (keeping everything adds memory faster than BUDGET_GIB_PER_TOKEN for ~1% speed)
BUDGET_MIN, BUDGET_MAX = 0.1, 0.7
# the budget in use, for the out_of_memory notice
applied_budget = None


def largest_batch_tokens(dataset) -> int:
    """Anima tokens in the largest batch a dataset makes: images in its largest filled bucket (up to
    the batch size) x (width / 16) x (height / 16), from the 8x VAE and the 2x2 patches."""
    largest = 0
    for part in getattr(dataset, "datasets", [dataset]):
        manager = getattr(part, "bucket_manager", None)
        if manager is None:
            continue
        batch_size = int(getattr(part, "batch_size", 1) or 1)
        for (width, height), images in zip(manager.resos, manager.buckets):
            if images:
                largest = max(largest, min(len(images), batch_size) * (width // 16) * (height // 16))
    return largest


def estimate_memory_budget(free_gib: float, trainable_gib: float, tokens: int) -> float:
    """The largest budget (a multiple of 0.05, at most BUDGET_MAX) whose estimated peak fits in the free
    VRAM; trainable_gib is what gradients and optimizer state will take (allocated after the first step)."""
    if tokens <= 0:
        return 0.0
    spare = free_gib - BUDGET_MARGIN_GIB - trainable_gib - CHECKPOINTING_GIB_PER_TOKEN * tokens
    budget = spare / (BUDGET_GIB_PER_TOKEN * tokens)
    return round(max(0.0, min(BUDGET_MAX, math.floor(budget * 20) / 20)), 2)


def install_memory_budget(setting: str):
    """activation_memory_budget: each compiled block keeps at most this fraction of its activations for
    the backward and recomputes the rest, in place of gradient checkpointing. Applied when the
    training loop asks for its first batch: by then the model is on the GPU and the datasets and
    optimizer exist, and nothing has been compiled for training yet (graphs are partitioned when
    first called). `auto` estimates the budget there; with too little VRAM, checkpointing stays."""
    import accelerate
    import torch
    import torch._functorch.config as functorch_config
    import torch.utils.data.dataloader as dataloader
    from library import anima_models

    checkpointed = []  # blocks sd-scripts turned gradient checkpointing on for (compile wraps them later)
    enable_checkpointing = anima_models.Block.enable_gradient_checkpointing

    def record_checkpointing(self, *args, **kwargs):
        checkpointed.append(self)
        return enable_checkpointing(self, *args, **kwargs)

    anima_models.Block.enable_gradient_checkpointing = record_checkpointing

    seen = {"datasets": [], "trainable_bytes": 0}
    prepare = accelerate.Accelerator.prepare

    def prepare_and_record(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.utils.data.DataLoader):
                seen["datasets"].append(arg.dataset)
            optimizer = getattr(arg, "optimizer", arg)
            if isinstance(optimizer, torch.optim.Optimizer):
                # gradients, plus optimizer state counted as two fp32 moments (an upper bound for most)
                seen["trainable_bytes"] = sum(param.numel() * (param.element_size() + 8) for group in optimizer.param_groups
                                              for param in group["params"] if param.requires_grad)
        return prepare(self, *args, **kwargs)

    accelerate.Accelerator.prepare = prepare_and_record

    next_batch = dataloader._BaseDataLoaderIter.__next__

    def first_batch(self):
        dataloader._BaseDataLoaderIter.__next__ = next_batch
        apply_budget(seen["datasets"] or [self._dataset])
        return next_batch(self)

    dataloader._BaseDataLoaderIter.__next__ = first_batch

    def apply_budget(datasets):
        if setting == "auto":
            if not checkpointed:
                print("LoRA-Forge: activation_memory_budget = auto: gradient checkpointing is off, nothing to replace", flush=True)
                return
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            tokens = max(largest_batch_tokens(dataset) for dataset in datasets)
            budget = estimate_memory_budget(free / GIB, seen["trainable_bytes"] / GIB, tokens)
            print(f"LoRA-Forge: activation_memory_budget = auto: {free / GIB:.1f} GiB free, largest batch "
                  f"{tokens} tokens -> {budget:g}", flush=True)
            if budget < BUDGET_MIN:
                print("LoRA-Forge: too little free VRAM for activation_memory_budget; gradient checkpointing stays on", flush=True)
                try:
                    # config.py set them for the budget; a checkpointed run keeps spilling to RAM instead of failing
                    torch.cuda.memory._set_allocator_settings("expandable_segments:False")
                except (AttributeError, RuntimeError):
                    pass
                return
        else:
            budget = float(setting)
        functorch_config.activation_memory_budget = budget
        plan_flash_attention()
        global applied_budget
        applied_budget = f"{budget:g}"
        for block in checkpointed:
            block.disable_gradient_checkpointing()
        print(f"LoRA-Forge: activation_memory_budget = {budget:g} in place of gradient checkpointing", flush=True)


def plan_flash_attention():
    """Lets the activation_memory_budget planner weigh flash-attn's forward like any other costly op. Without
    this torch 2.14 recomputes it under every budget below 1: ops from outside aten are only considered with
    is_non_builtin_to_include, and the planner's cost estimate (FLOPs) has no formula for them. With both,
    Anima at budget 0.3 kept half of its attention outputs and stepped about 3% faster at slightly lower peak
    memory. sdpa needs nothing: aten's attention ops are already counted."""
    try:
        import flash_attn.flash_attn_interface  # noqa: F401  (registers torch.ops.flash_attn.*)
    except ImportError:
        return
    import torch
    import torch._functorch.config as functorch_config
    from torch.utils.flop_counter import flop_registry, register_flop_formula

    ops = torch.ops.flash_attn
    if not hasattr(functorch_config, "is_non_builtin_to_include") or not hasattr(ops, "_flash_attn_forward"):
        return

    # q / k are shapes here: (batch, length, heads, head_dim) and, for varlen, (tokens, heads, head_dim)
    def forward_flops(q, k, v, *args, out_shape=None, **kwargs):
        batch, length_q, heads, head_dim = q
        return 4 * batch * heads * length_q * k[1] * head_dim

    def varlen_forward_flops(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, out_shape=None, **kwargs):
        tokens_q, heads, head_dim = q
        return 4 * tokens_q * heads * max_seqlen_k * head_dim

    # a later torch or flash-attn may bring its own formula (registering twice raises)
    if ops._flash_attn_forward not in flop_registry:
        register_flop_formula(ops._flash_attn_forward)(forward_flops)
    if hasattr(ops, "_flash_attn_varlen_forward") and ops._flash_attn_varlen_forward not in flop_registry:
        register_flop_formula(ops._flash_attn_varlen_forward)(varlen_forward_flops)
    functorch_config.is_non_builtin_to_include = True


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


def install_console_output():
    """The output reaches the GUI's console through a pipe, where tqdm cannot read the console
    width (a 10-column bar) and rich turns colors off. Use COLUMNS for tqdm's width, and with
    LORA_FORGE_ANSI=1 (the GUI's console shows ANSI colors; the runner also sets FORCE_COLOR) stop
    rich from taking the pipe for an old Windows console, which it colors with console API calls
    that a pipe drops."""
    columns = int(os.environ.get("COLUMNS") or 0)
    if columns:
        import tqdm.std as tqdm_std

        screen_shape_wrapper = tqdm_std._screen_shape_wrapper

        def wrapper():
            measure = screen_shape_wrapper()

            def screen_shape(fp):
                cols, rows = measure(fp) if measure else (None, None)
                return (cols, rows) if cols else (columns - 1, rows)

            return screen_shape

        tqdm_std._screen_shape_wrapper = wrapper

    if os.environ.get(ANSI_ENV) == "1":
        try:
            import rich.console
        except ImportError:
            return
        rich.console.detect_legacy_windows = lambda: False


# sd-scripts' network modules log one line per LoRA module for network_reg_lrs / network_reg_dims
# (hundreds of lines for Anima); one line per pattern says the same
PER_MODULE_LINE = re.compile(r"^Module \S+ matched (?:with )?regex '(.*)' -> (LR|dim:) (.*)$")


def install_rich_logging():
    """musubi-tuner's modules call logging.basicConfig, which prints plain `INFO:name:message`
    lines; sd-scripts logs through rich. A rich handler on the root logger first turns those calls
    into no-ops, so Krea 2 runs look like the others (same handler as sd-scripts' setup_logging)."""
    import logging

    if logging.root.handlers:
        return
    try:
        from rich.console import Console
        from rich.logging import RichHandler
    except ImportError:
        return
    handler = RichHandler(console=Console(stderr=True))
    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)


def install_log_filter():
    import logging

    # httpx logs every Hugging Face request at INFO (the tokenizers are checked on each load)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    seen = set()
    handle = logging.Logger.handle

    def filtered_handle(self, record):
        if record.name.startswith("networks."):
            match = PER_MODULE_LINE.match(record.getMessage())
            if match:
                if match.groups() in seen:
                    return None
                seen.add(match.groups())
                pattern, kind, value = match.groups()
                option = "network_reg_lrs" if kind == "LR" else "network_reg_dims"
                record.msg = f"{option}: modules matching '{pattern}' -> {kind.rstrip(':')} {value}"
                record.args = ()
        return handle(self, record)

    logging.Logger.handle = filtered_handle


def install_fp16_accumulation():
    """fp16_accumulation: fp16 matrix multiplies add up in fp16 instead of fp32, which
    GeForce tensor cores run at about twice the rate (data-center cards gain nothing). Measured on
    Anima: about a quarter less time per step, and gradients at least as close to fp32 as bf16's."""
    import torch

    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        print("LoRA-Forge: this PyTorch has no fp16 accumulation (needs 2.7 or newer); training without it", flush=True)
        return
    matmul.allow_fp16_accumulation = True
    print("LoRA-Forge: fp16_accumulation is on: fp16 matrix multiplies accumulate in fp16", flush=True)


def load_sibling(name: str):
    """A module next to this file, loaded by path: sys.path[0] is the trainer's folder, and lora_forge
    is not imported here."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"lora_forge_{name}", os.path.join(os.path.dirname(__file__), f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_batched_ppsf():
    """ProdigyPlusScheduleFree with a batched step (batched_ppsf.py): same math, far fewer GPU launches.
    About 7% less time per step on an Anima rank-32 LoRA."""
    try:
        from prodigyplus import core_optimiser, prodigy_plus_schedulefree
    except ImportError:
        return
    module = load_sibling("batched_ppsf")
    if not module.install(prodigy_plus_schedulefree.ProdigyPlusScheduleFree, core_optimiser.CoreOptimiser):
        print("LoRA-Forge: this prodigy-plus-schedule-free version is not checked for the batched step; "
              "using its own", flush=True)


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
        install_spill_monitor()

    install_console_output()
    if "musubi_tuner" in trainer.replace("\\", "/").split("/"):
        install_rich_logging()
    install_log_filter()

    patches = os.environ.get(PATCHES_ENV, "1") != "0"
    if os.path.basename(trainer) == "anima_train_network.py" and patches:
        install_anima_patches()
    if patches:
        install_batched_ppsf()
    if patches and os.environ.get(SDXL_COMPILE_ENV) == "1" and os.path.basename(trainer) in SDXL_TRAINERS:
        load_sibling("sdxl_speed").install_compile()

    budget = os.environ.get(MEMORY_BUDGET_ENV)
    if budget and os.path.basename(trainer) == "anima_train_network.py":
        install_memory_budget(budget)

    if os.environ.get(FP16_ACCUMULATION_ENV) == "1":
        install_fp16_accumulation()

    if os.environ.get(KREA2_TURBO_LORA_ENV):
        install_turbo_lora_conversion(os.environ[KREA2_TURBO_LORA_ENV])

    try:
        if os.environ.get(KREA2_CACHE_ENV):
            run_krea2_caching(trainer, json.loads(os.environ[KREA2_CACHE_ENV]))
        runpy.run_path(trainer, run_name="__main__")
    except Exception as error:
        if notify is not None and is_out_of_memory(error):
            notify("notice", code="out_of_memory", budget=applied_budget, text=str(error).strip().splitlines()[0][:300])
        raise


class SpillMonitor:
    """Windows does not fail when a training needs more VRAM than the card has: the driver moves part of
    it to system RAM (WDDM sysmem fallback) and every step gets much slower. This watches the process's
    GPU memory counters and reports how much spilled (notice `vram_spill`, the largest amount seen).

    Pinned RAM (block swap, staging buffers) is counted as shared GPU memory too, so the spill is the
    shared usage minus PyTorch's pinned memory minus what was shared at the start, and only counts while
    the card's dedicated memory is full (a paged optimizer's managed memory is shared as well). Reads
    performance counters only and never calls CUDA from this thread."""

    INTERVAL = 10.0
    # the card counts as full within this much of its size (Windows keeps about 0.5 GiB back)
    FULL_MARGIN = 1.0 * 2 ** 30
    # report from this much, seen twice in a row
    REPORT_FROM = 0.25 * 2 ** 30

    def __init__(self, notify):
        import ctypes
        from ctypes import wintypes

        class FmtValue(ctypes.Structure):
            _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]

        class FmtItem(ctypes.Structure):
            _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", FmtValue)]

        self.ctypes, self.wintypes, self.item_type = ctypes, wintypes, FmtItem
        self.pdh = ctypes.WinDLL("pdh.dll")
        self.notify = notify
        self.query = wintypes.HANDLE()
        if self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)) != 0:
            raise OSError("PdhOpenQueryW failed")
        process = f"\\GPU Process Memory(pid_{os.getpid()}_*)"
        self.dedicated = self.add(process + "\\Dedicated Usage")
        self.shared = self.add(process + "\\Shared Usage")
        self.adapters = self.add("\\GPU Adapter Memory(*)\\Dedicated Usage")
        self.baseline = None
        self.reported = 0
        self.pending = False

    def add(self, path):
        handle = self.wintypes.HANDLE()
        status = self.pdh.PdhAddEnglishCounterW(self.query, path, None, self.ctypes.byref(handle))
        if status != 0:
            raise OSError(f"PdhAddEnglishCounterW {path}: {status & 0xFFFFFFFF:#x}")
        return handle

    def values(self, handle) -> dict:
        ctypes = self.ctypes
        size, count = self.wintypes.DWORD(0), self.wintypes.DWORD(0)
        self.pdh.PdhGetFormattedCounterArrayW(handle, 0x400, ctypes.byref(size), ctypes.byref(count), None)
        if not size.value:
            return {}
        buffer = (ctypes.c_byte * size.value)()
        if self.pdh.PdhGetFormattedCounterArrayW(handle, 0x400, ctypes.byref(size), ctypes.byref(count), buffer) != 0:
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(self.item_type))
        return {items[i].szName: items[i].FmtValue.largeValue for i in range(count.value)}

    def sample(self, total: int):
        import torch

        def pinned_now():
            return torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)

        # skip a sample taken while memory was being pinned or freed: the counters and the
        # pinned total would not match, and the baseline would stay too low for the rest of the run
        pinned = pinned_now()
        if self.pdh.PdhCollectQueryData(self.query) != 0 or pinned_now() != pinned:
            return
        dedicated = self.values(self.dedicated)
        if not dedicated:
            return
        # this process's card: the adapter it has the most dedicated memory on
        luid = max(dedicated, key=dedicated.get).split("_luid_")[-1]
        adapter = sum(value for name, value in self.values(self.adapters).items() if luid in name)
        unexplained = sum(self.values(self.shared).values()) - pinned
        full = adapter >= total - self.FULL_MARGIN
        # the shared memory a process has without any spill (CUDA's own, ~0.06 GiB), taken only while
        # the card is not full; a run that overflows right away is measured from 0
        if not full:
            self.baseline = unexplained if self.baseline is None else min(self.baseline, unexplained)
        spill = unexplained - (self.baseline or 0)
        if not full or spill < max(self.REPORT_FROM, self.reported + self.REPORT_FROM):
            self.pending = False
            return
        if self.pending:
            self.reported = spill
            self.notify("notice", code="vram_spill", gib=round(spill / 2 ** 30, 1), total_gib=round(total / 2 ** 30, 1))
        self.pending = not self.pending

    def run(self):
        total = None
        while True:
            time.sleep(self.INTERVAL)
            try:
                torch = sys.modules.get("torch")
                # never the first to touch CUDA: the trainer picks the device and initializes it
                if torch is None or not torch.cuda.is_initialized():
                    continue
                if total is None:
                    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                self.sample(total)
            except Exception:
                return


def install_spill_monitor():
    if sys.platform != "win32" or notify is None:
        return
    try:
        monitor = SpillMonitor(notify)
    except Exception:
        return
    threading.Thread(target=monitor.run, name="lora-forge-spill-monitor", daemon=True).start()


def run_krea2_caching(trainer: str, cache: dict):
    """musubi-tuner caches latents and text-encoder outputs with two scripts before training
    (`--skip_existing`: a second run on the same dataset only encodes new images and changed
    captions). They run here, in the training process, so the Runs page shows them as part of the run."""
    import gc

    import torch

    scripts = (
        ("krea2_cache_latents.py", ["--vae", cache["vae"]]),
        ("krea2_cache_text_encoder_outputs.py", ["--text_encoder", cache["text_encoder"]]),
    )
    folder = os.path.dirname(trainer)
    trainer_argv = list(sys.argv)
    install_caption_check()
    for name, args in scripts:
        if os.environ.get("RANK", "0") != "0":
            break
        print(f"LoRA-Forge: caching with {name}", flush=True)
        # the scripts' own bars have no total, so they report nothing: one bar per script for the Runs page
        desc = "caching latents" if "latents" in name else "caching text encoder outputs"
        if notify is not None:
            notify("progress", desc=desc, n=0, total=1, rate=None, elapsed=0, done=False)
        sys.argv = [os.path.join(folder, name), "--dataset_config", cache["dataset_config"], "--skip_existing", *args]
        runpy.run_path(sys.argv[0], run_name="__main__")
        if notify is not None:
            notify("progress", desc=desc, n=1, total=1, rate=None, elapsed=0, done=True)
        # the script's models are module locals of that run; give their VRAM back before the DiT loads
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sys.argv = trainer_argv


def install_caption_check():
    """With `--skip_existing` musubi-tuner keeps a text-encoder cache whenever its file exists, so
    an edited caption would train with the old one. Each cache records its caption (`caption1`
    metadata): a cache counts as current only when that still matches."""
    from musubi_tuner import cache_text_encoder_outputs
    from safetensors import safe_open

    original = cache_text_encoder_outputs.process_text_encoder_batches

    def cache_is_current(item) -> bool:
        path = item.text_encoder_output_cache_path
        if not os.path.exists(path):
            return False
        try:
            with safe_open(path, "pt") as f:
                return (f.metadata() or {}).get("caption1") == item.caption
        except Exception:
            return False

    def process_text_encoder_batches(*args, **kwargs):
        if kwargs.get("cache_is_current") is None:
            kwargs["cache_is_current"] = cache_is_current
        return original(*args, **kwargs)

    cache_text_encoder_outputs.process_text_encoder_batches = process_text_encoder_batches


def install_turbo_lora_conversion(path: str):
    """ComfyUI's Krea 2 Turbo LoRA (`krea2_turbo_lora_rank_64_bf16`) names its weights
    `diffusion_model.<module>.lora_down.weight` without alphas; musubi-tuner only reads its own
    `lora_unet_<module>` names. That one file is converted with musubi-tuner's own converter as it
    is loaded (musubi-tuner imports load_file after this runs, so it gets the wrapper)."""
    import safetensors.torch

    original = safetensors.torch.load_file
    target = os.path.normcase(os.path.abspath(path))

    def load_file(filename, *args, **kwargs):
        weights = original(filename, *args, **kwargs)
        if os.path.normcase(os.path.abspath(str(filename))) == target and any(key.startswith("diffusion_model.") for key in weights):
            from musubi_tuner.convert_lora import convert_from_diffusers

            weights = convert_from_diffusers("lora_unet_", weights)
        return weights

    safetensors.torch.load_file = load_file


def is_out_of_memory(error: BaseException) -> bool:
    """CUDA out of memory, also as it shows with expandable segments on Windows (an allocation that
    does not fit fails with "CUDA driver error: device not ready")."""
    message = str(error)
    return (type(error).__name__ == "OutOfMemoryError" or "out of memory" in message.lower()
            or "CUDA driver error: device not ready" in message)


if __name__ == "__main__":
    main()
