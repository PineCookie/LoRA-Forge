"""Run records for the web UI's monitor: one folder per training run under logs/runs.

Each run folder holds
- run.json: what was run (model type, output name and folder, start / end time, final status)
- events.jsonl: progress, epochs and metrics, written by run_entry.py in the training process
- output.log: everything the training process printed (console output, with progress bars)

Records stay after the GUI exits, so past runs can be looked at and compared.
"""

import codecs
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from lora_forge.launcher import base_dir_path
from lora_forge.log import log

RUNS_DIR = base_dir_path() / "logs" / "runs"
RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}(-[0-9]+)?$")

# a page of events per request: a long run catches up in a few requests instead of one huge one
MAX_EVENT_BYTES = 2 * 1024 * 1024
OUTPUT_TAIL_BYTES = 256 * 1024
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt"}
SAMPLE_NAME = re.compile(r"_(e?)(\d{6})_(\d{2})_")


class RunNotFoundError(LookupError):
    pass


@dataclass
class RunInfo:
    id: str
    model_train_type: str
    output_name: str
    output_dir: str
    preview: bool
    started: float
    task_id: str = ""
    ended: Optional[float] = None
    # running, finished, failed, stopped; "running" without a live task means the GUI exited first
    status: str = "running"
    returncode: Optional[int] = None
    config_file: str = ""
    # learning rate group names sd-scripts logs ("reg_lr_0", ...) -> the network_reg_lrs pattern
    lr_groups: dict = field(default_factory=dict)


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise RunNotFoundError(run_id)
    path = RUNS_DIR / run_id
    if not (path / "run.json").is_file():
        raise RunNotFoundError(run_id)
    return path


def events_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "events.jsonl"


def output_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "output.log"


def _write_info(info: RunInfo):
    path = RUNS_DIR / info.id / "run.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(info), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_info(run_id: str) -> RunInfo:
    data = json.loads((_run_dir(run_id) / "run.json").read_text(encoding="utf-8"))
    known = RunInfo.__dataclass_fields__
    return RunInfo(**{key: value for key, value in data.items() if key in known})


def reg_lr_groups(config: dict) -> dict[str, str]:
    """The patterns of network_reg_lrs by the group name sd-scripts gives them in its learning rate
    logs (reg_lr_<n>, in order), parsed the way its LoRA modules parse the argument."""
    args = config.get("network_args") or []
    if isinstance(args, str):
        args = [args]
    value = next((str(arg).split("=", 1)[1] for arg in args if str(arg).split("=", 1)[0].strip() == "network_reg_lrs"), None)
    if not value:
        return {}
    patterns = []
    for pair in value.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        pattern, lr = (part.strip() for part in pair.split("=", 1))
        try:
            float(lr)
        except ValueError:
            continue
        if pattern not in patterns:
            patterns.append(pattern)
    return {f"reg_lr_{i}": pattern for i, pattern in enumerate(patterns)}


def create_run(run_id: str, config: dict, model_train_type: str, preview: bool, config_file: str) -> RunInfo:
    """A new run folder; `run_id` is the timestamp of the saved config (a suffix keeps it unique)."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    unique, n = run_id, 1
    while (RUNS_DIR / unique).exists():
        n += 1
        unique = f"{run_id}-{n}"
    (RUNS_DIR / unique).mkdir()
    info = RunInfo(
        id=unique,
        model_train_type=model_train_type,
        output_name=str(config.get("output_name") or ""),
        output_dir=str(config.get("output_dir") or ""),
        preview=preview,
        started=time.time(),
        config_file=config_file,
        lr_groups=reg_lr_groups(config),
    )
    _write_info(info)
    return info


def set_task(run_id: str, task_id: str):
    info = read_info(run_id)
    info.task_id = task_id
    _write_info(info)


def finish_run(run_id: str, status: str, returncode: Optional[int]):
    try:
        info = read_info(run_id)
    except (RunNotFoundError, OSError, ValueError) as e:
        log.warning(f"Cannot record the end of run {run_id}: {e}")
        return
    info.status, info.returncode, info.ended = status, returncode, time.time()
    _write_info(info)


def list_runs(limit: int = 100) -> list[RunInfo]:
    if not RUNS_DIR.is_dir():
        return []
    runs = []
    for path in sorted(RUNS_DIR.iterdir(), reverse=True):
        if len(runs) >= limit:
            break
        if not RUN_ID.match(path.name):
            continue
        try:
            runs.append(read_info(path.name))
        except (RunNotFoundError, OSError, ValueError, TypeError):
            continue
    return runs


def delete_run(run_id: str):
    path = _run_dir(run_id)
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


# --- events ------------------------------------------------------------------------------------

def read_events(run_id: str, offset: int = 0) -> tuple[list[dict], int]:
    """Events from byte `offset` on, and the offset to continue from. Only complete lines are
    read, so a line the trainer is still writing comes with the next request."""
    _run_dir(run_id)
    path = events_path(run_id)
    if not path.is_file():
        return [], 0
    with path.open("rb") as f:
        f.seek(max(0, offset))
        chunk = f.read(MAX_EVENT_BYTES)
    end = chunk.rfind(b"\n") + 1
    events = []
    for line in chunk[:end].splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events, offset + end


@dataclass
class Summary:
    """The latest state of a run, kept up to date incrementally for the task list."""
    offset: int = 0
    # the training-steps bar, else the latest other bar (caching, validation)
    step: Optional[dict] = None
    phase: Optional[dict] = None
    epoch: Optional[dict] = None
    loss: Optional[float] = None
    # torch.compile: graphs built so far; notices (e.g. the recompile limit was hit), one per code
    graphs: Optional[int] = None
    notices: dict = field(default_factory=dict)


_summaries: dict[str, Summary] = {}
_summary_lock = threading.Lock()


def summary(run_id: str) -> dict:
    with _summary_lock:
        state = _summaries.setdefault(run_id, Summary())
        try:
            while True:
                events, offset = read_events(run_id, state.offset)
                for event in events:
                    kind = event.get("type")
                    if kind == "progress":
                        bar = {key: event.get(key) for key in ("desc", "n", "total", "rate", "elapsed", "done")}
                        if bar["desc"] == "steps":
                            state.step = bar
                            state.phase = None
                        else:
                            state.phase = None if bar["done"] else bar
                    elif kind == "epoch":
                        state.epoch = {"n": event.get("epoch"), "total": event.get("total")}
                    elif kind == "metrics":
                        loss = event.get("values", {}).get("loss/average")
                        if loss is not None:
                            state.loss = loss
                    elif kind == "compile":
                        state.graphs = event.get("graphs")
                    elif kind == "notice" and event.get("code"):
                        notice = {key: value for key, value in event.items() if key not in ("t", "type")}
                        state.notices.setdefault(event["code"], notice)
                moved = offset != state.offset
                state.offset = offset
                if not moved or not events:
                    break
        except (RunNotFoundError, OSError):
            pass
        return {
            "step": state.step,
            "phase": state.phase,
            "epoch": state.epoch,
            "loss": state.loss,
            "graphs": state.graphs,
            "notices": list(state.notices.values()),
        }


# --- console output ------------------------------------------------------------------------------

def tail_output(run_id: str, max_lines: int = 400) -> list[str]:
    """The last lines the training printed, as a terminal would show them: colors removed and
    progress bars (redrawn with \\r) reduced to their latest state."""
    _run_dir(run_id)
    path = output_path(run_id)
    if not path.is_file():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(max(0, size - OUTPUT_TAIL_BYTES))
        data = f.read()
    text = ANSI.sub("", data.decode("utf-8", errors="replace")).replace("\r\n", "\n")
    lines = text.split("\n")
    if size > OUTPUT_TAIL_BYTES:
        lines = lines[1:]  # the first line was cut
    shown = []
    for line in lines:
        parts = [part for part in line.split("\r") if part.strip()]
        if parts:
            shown.append(parts[-1].rstrip())
    return shown[-max_lines:]


class OutputPump:
    """Copies the training process's output to the GUI console and to output.log."""

    def __init__(self, stream, run_id: str, console):
        self.stream = stream
        self.path = output_path(run_id)
        self.console = console
        self.thread = threading.Thread(target=self._run, name=f"output-{run_id}", daemon=True)

    def start(self):
        self.thread.start()

    def join(self, timeout: Optional[float] = None):
        self.thread.join(timeout)

    def _run(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with self.path.open("ab") as out:
            while True:
                try:
                    chunk = self.stream.read1(65536)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                out.write(chunk)
                out.flush()
                if self.console is not None:
                    try:
                        self.console.write(decoder.decode(chunk))
                        self.console.flush()
                    except (OSError, ValueError):
                        self.console = None


# --- files the run produced ----------------------------------------------------------------------

def _output_dir(info: RunInfo) -> Optional[Path]:
    if not info.output_dir:
        return None
    path = Path(info.output_dir)
    if not path.is_absolute():
        path = base_dir_path() / path
    return path if path.is_dir() else None


def _produced(folder: Path, info: RunInfo, extensions: set[str]) -> list[os.DirEntry]:
    # files of this run: its output name, written after it started (a little slack for clocks)
    since = info.started - 5
    until = (info.ended + 60) if info.ended else float("inf")
    # samples are "<name>_<step>_...", checkpoints "<name>-<step>" and "<name>" for the last one
    prefixes = (f"{info.output_name}_", f"{info.output_name}-") if info.output_name else ("",)
    found = []
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                name = entry.name
                stem, extension = os.path.splitext(name)
                if not entry.is_file() or extension.lower() not in extensions:
                    continue
                if not (name.startswith(prefixes) or stem == info.output_name):
                    continue
                mtime = entry.stat().st_mtime
                if since <= mtime <= until:
                    found.append(entry)
    except OSError:
        return []
    return sorted(found, key=lambda entry: entry.stat().st_mtime)


def samples(info: RunInfo) -> list[dict]:
    folder = _output_dir(info)
    if folder is None or not (folder / "sample").is_dir():
        return []
    result = []
    for entry in _produced(folder / "sample", info, IMAGE_EXTENSIONS):
        match = SAMPLE_NAME.search(entry.name)
        item = {"name": entry.name, "time": entry.stat().st_mtime}
        if match:
            item["epoch" if match.group(1) else "step"] = int(match.group(2))
            item["prompt"] = int(match.group(3))
        result.append(item)
    return result


def sample_file(info: RunInfo, name: str) -> Optional[Path]:
    """A sample image of this run by file name; None for anything else."""
    if any(item["name"] == name for item in samples(info)):
        return _output_dir(info) / "sample" / name
    return None


def checkpoints(info: RunInfo) -> list[dict]:
    folder = _output_dir(info)
    if folder is None:
        return []
    return [
        {"name": entry.name, "size": entry.stat().st_size, "time": entry.stat().st_mtime}
        for entry in _produced(folder, info, MODEL_EXTENSIONS)
    ]
