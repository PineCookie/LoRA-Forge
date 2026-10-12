"""Start sd-scripts training runs as background tasks."""

import os
import re
import sys
import threading
from typing import Optional

from lora_forge.launcher import base_dir_path
from lora_forge.log import log
from lora_forge.training.tasks import TaskStatus, tm

# accelerate's launch settings (single machine, no distributed setup unless several GPUs are picked)
ACCELERATE_CONFIG = base_dir_path() / "config" / "accelerate.yaml"


class TaskNotStartedError(RuntimeError):
    """The task manager refused a new task (one is already running); the message is user-facing."""


def _physical_devices(indices: list[int]) -> list[str]:
    # The GUI lists GPUs as torch sees them under its own CUDA_VISIBLE_DEVICES, so "GPU n" is the n-th
    # visible device. Map it back through that list, otherwise the child process would get physical GPU n.
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not visible:
        return [str(i) for i in indices]
    return [visible[i] if i < len(visible) else str(i) for i in indices]


def start_training(toml_path: str,
                   trainer_file: str,
                   gpu_ids: Optional[list] = None,
                   cpu_threads: int = 2) -> str:
    """Launch `trainer_file` with `toml_path` through accelerate; returns the task id."""
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--config_file", str(ACCELERATE_CONFIG),
        "--num_cpu_threads_per_process", str(cpu_threads),
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    environ = os.environ.copy()
    environ["ACCELERATE_DISABLE_RICH"] = "1"
    environ["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "win32":
        environ["PYTHONUTF8"] = "0"
    environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        gpu_indices = [int(re.search(r"GPU\s+(\d+):", gpu).group(1)) for gpu in gpu_ids]
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")
    else:
        # With several GPUs visible, sd-scripts adds InitProcessGroupKwargs and accelerate then treats a
        # single-process run as distributed, failing with "Default process group has not been initialized".
        # Pin to the first GPU unless the user picked some.
        gpu_indices = [0]
        log.info("No GPU selected, using the first GPU / 未选择显卡，默认使用第一张显卡")

    environ["CUDA_VISIBLE_DEVICES"] = ",".join(_physical_devices(gpu_indices))
    log.info(f"CUDA_VISIBLE_DEVICES={environ['CUDA_VISIBLE_DEVICES']}")

    if len(gpu_indices) > 1:
        args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_indices))]
        if sys.platform == "win32":
            environ["USE_LIBUV"] = "0"
            args[3:3] = ["--rdzv_backend", "c10d"]

    task = tm.create_task(args, environ)
    if task is None:
        raise TaskNotStartedError("已有训练任务正在运行，无法创建新任务 / A training task is already running")

    def run():
        try:
            if not task.execute():
                log.info("Training cancelled before it started / 训练在启动前已取消")
                return
            result = task.communicate()
            if task.status == TaskStatus.TERMINATED:
                log.info("Training terminated / 训练已终止")
            elif result.returncode != 0:
                log.error("Training failed / 训练失败")
            else:
                log.info("Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    threading.Thread(target=run, name=f"training-{task.task_id}", daemon=True).start()
    return task.task_id
