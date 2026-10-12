
import asyncio
import os
import sys
import re
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import TaskStatus, tm


def _physical_devices(indices: list[int]) -> list[str]:
    # The GUI lists GPUs as torch sees them under its own CUDA_VISIBLE_DEVICES, so "GPU n" is the n-th
    # visible device. Map it back through that list, otherwise the child process would get physical GPU n.
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    if not visible:
        return [str(i) for i in indices]
    return [visible[i] if i < len(visible) else str(i) for i in indices]


def run_train(toml_path: str,
              trainer_file: str = "./scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "win32":
        customize_env["PYTHONUTF8"] = "0"
    customize_env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"

    if gpu_ids:
        gpu_indices = [int(re.search(r"GPU\s+(\d+):", gpu).group(1)) for gpu in gpu_ids]
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")
    else:
        # With several GPUs visible, sd-scripts adds InitProcessGroupKwargs and accelerate then treats a
        # single-process run as distributed, failing with "Default process group has not been initialized".
        # Pin to the first GPU unless the user picked some.
        gpu_indices = [0]
        log.info("No GPU selected, using the first GPU / 未选择显卡，默认使用第一张显卡")

    customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(_physical_devices(gpu_indices))
    log.info(f"CUDA_VISIBLE_DEVICES={customize_env['CUDA_VISIBLE_DEVICES']}")

    if len(gpu_indices) > 1:
        args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_indices))]
        if sys.platform == "win32":
            customize_env["USE_LIBUV"] = "0"
            args[3:3] = ["--rdzv_backend", "c10d"]

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="fail", message="Failed to create task / 无法创建训练任务", data=None)

    def _run():
        try:
            if not task.execute():
                log.info("Training cancelled before it started / 训练在启动前已取消")
                return
            result = task.communicate()
            if task.status == TaskStatus.TERMINATED:
                log.info("Training terminated / 训练已终止")
            elif result.returncode != 0:
                log.error(f"Training failed / 训练失败")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}", data=None)
