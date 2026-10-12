"""Start sd-scripts training runs as background tasks."""

import os
import re
import shutil
import sys
import threading

from lora_forge.i18n import t
from lora_forge.launcher import base_dir_path
from lora_forge.log import log
from lora_forge.training import runs
from lora_forge.training.config import PreparedRun
from lora_forge.training.run_entry import EVENTS_ENV
from lora_forge.training.tasks import TaskStatus, tm

# accelerate's launch settings (single machine, no distributed setup unless several GPUs are picked)
ACCELERATE_CONFIG = base_dir_path() / "config" / "accelerate.yaml"
# runs the trainer and records progress and metrics for the monitor (see run_entry.py)
RUN_ENTRY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_entry.py")


class TaskNotStartedError(RuntimeError):
    """The task manager refused a new task (one is already running); the message is user-facing."""


def _physical_devices(indices: list[int]) -> list[str]:
    # The GUI lists GPUs as torch sees them under its own CUDA_VISIBLE_DEVICES, so "GPU n" is the n-th
    # visible device. Map it back through that list, otherwise the child process would get physical GPU n.
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not visible:
        return [str(i) for i in indices]
    return [visible[i] if i < len(visible) else str(i) for i in indices]


def start_training(prepared: PreparedRun) -> tuple[str, str]:
    """Launch the prepared run through accelerate; returns the task id and the run id."""
    toml_path, gpu_ids = prepared.toml_file, prepared.gpu_ids
    log.info(f"Training started with config file: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--config_file", str(ACCELERATE_CONFIG),
        "--num_cpu_threads_per_process", str(prepared.cpu_threads),
        "--quiet",  # silence accelerate error message
        RUN_ENTRY,
        prepared.trainer_file,
        "--config_file", toml_path,
    ]

    environ = os.environ.copy()
    environ["ACCELERATE_DISABLE_RICH"] = "1"
    environ["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "win32":
        environ["PYTHONUTF8"] = "0"
    # the output goes through a pipe (copied to this console and the run's output.log): UTF-8 on
    # every platform, and logs as wide as the console instead of rich's 80-column default
    environ["PYTHONIOENCODING"] = "utf-8"
    environ.setdefault("COLUMNS", str(max(120, shutil.get_terminal_size((160, 40)).columns)))
    environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
    environ.update(prepared.environment)

    if gpu_ids:
        # "GPU 0: <name>" from the picker, or just "GPU 0" from an imported config
        gpu_indices = [int(match.group(1)) for gpu in gpu_ids if (match := re.search(r"GPU\s+(\d+)", str(gpu)))]
        if not gpu_indices:
            gpu_indices = [0]
        log.info(f"Using GPU(s): {gpu_ids}")
    else:
        # With several GPUs visible, sd-scripts adds InitProcessGroupKwargs and accelerate then treats a
        # single-process run as distributed, failing with "Default process group has not been initialized".
        # Pin to the first GPU unless the user picked some.
        gpu_indices = [0]
        log.info("No GPU selected, using the first GPU")

    environ["CUDA_VISIBLE_DEVICES"] = ",".join(_physical_devices(gpu_indices))
    log.info(f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']}")

    if len(gpu_indices) > 1:
        args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_indices))]
        if sys.platform == "win32":
            environ["USE_LIBUV"] = "0"
            args[3:3] = ["--rdzv_backend", "c10d"]

    run = runs.create_run(prepared.run_id, prepared.config, prepared.model_train_type, prepared.preview, toml_path)
    environ[EVENTS_ENV] = str(runs.events_path(run.id))

    task = tm.create_task(args, environ, run_id=run.id)
    if task is None:
        runs.delete_run(run.id)
        raise TaskNotStartedError(t("run.alreadyRunning"))
    runs.set_task(run.id, task.task_id)

    def watch():
        status, returncode = "failed", None
        try:
            if not task.execute():
                log.info("Training cancelled before it started")
                status = "stopped"
                return
            returncode = task.wait()
            if task.status == TaskStatus.TERMINATED:
                status = "stopped"
                log.info("Training terminated")
            elif returncode != 0:
                log.error(f"Training failed (exit code {returncode})")
            else:
                status = "finished"
                log.info("Training finished")
        except Exception as e:
            log.error(f"An error occurred when training: {e}")
        finally:
            runs.finish_run(run.id, status, returncode)

    threading.Thread(target=watch, name=f"training-{task.task_id}", daemon=True).start()
    return task.task_id, run.id
