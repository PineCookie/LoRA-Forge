"""HTTP API used by the web UI (mounted under /api)."""

import asyncio
import hashlib
import json
import os
from pathlib import Path

from fastapi import APIRouter, Request

from lora_forge import launcher, sd_scripts
from lora_forge.app.models import APIResponse, APIResponseFail, APIResponsePending, APIResponseSuccess
from lora_forge.devices import printable_devices
from lora_forge.file_dialogs import open_directory_selector, open_file_selector
from lora_forge.log import log
from lora_forge.training import runner
from lora_forge.training.config import TrainingConfigError, prepare_training, redact_config
from lora_forge.training.dataset import count_images, subset_folders
from lora_forge.training.tasks import tm

router = APIRouter()

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schema"


def load_schemas() -> list[dict]:
    """Read lora_forge/schema/*.ts on every request, so editing a schema only needs a page
    refresh instead of a backend restart. A file that cannot be read is skipped."""
    schemas = []
    for path in sorted(SCHEMA_DIR.glob("*.ts")):
        if path.name.endswith(".d.ts"):
            continue  # editor type declarations, not a schema
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning(f"Failed to read schema {path.name}: {e}")
            continue
        schemas.append({"name": path.stem, "schema": content, "hash": hashlib.md5(content.encode()).hexdigest()})
    return schemas


@router.post("/run")
async def run_training(request: Request):
    try:
        config: dict = json.loads((await request.body()).decode("utf-8"))
        log.info(f"Training config received: {redact_config(config)}")
        prepared = prepare_training(config)
        task_id = runner.start_training(prepared.toml_file, prepared.trainer_file, prepared.gpu_ids, prepared.cpu_threads)
    except (TrainingConfigError, runner.TaskNotStartedError) as e:
        return APIResponseFail(message=str(e))
    except Exception as e:
        log.error(f"Error in /run endpoint: {e}", exc_info=True)
        return APIResponseFail(message=f"Server error: {e}")
    return APIResponseSuccess(message=f"Training started / 训练开始 ID: {task_id}")


@router.get("/pick_file")
async def pick_file(picker_type: str):
    if picker_type == "folder":
        result = await asyncio.to_thread(open_directory_selector, "")
    elif picker_type == "model-file":
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt"), ("all files", "*.*")]
        result = await asyncio.to_thread(open_file_selector, "", "Select file", file_types)
    else:
        return APIResponseFail(message=f"不支持的 picker_type: {picker_type}")

    if result == "":
        return APIResponseFail(message="用户取消选择")
    return APIResponseSuccess(data={"path": result})


@router.get("/dataset_info")
async def get_dataset_info(train_data_dir: str = "") -> APIResponse:
    """Image counts per `repeats_name` subfolder of a dataset, with and without repeats."""
    if not train_data_dir or not os.path.isdir(train_data_dir):
        return APIResponseFail(message="训练数据集路径不存在或无效")

    def scan():
        subdirs = []
        for name, repeat, concept in subset_folders(train_data_dir):
            image_count = count_images(os.path.join(train_data_dir, name))
            subdirs.append({"name": name, "concept": concept, "repeat": repeat, "image_count": image_count})
        return subdirs

    try:
        # walking a large dataset can take a while; keep it off the event loop
        subdirs = await asyncio.to_thread(scan)
    except OSError:
        return APIResponseFail(message="无法读取训练数据集目录")

    return APIResponseSuccess(data={
        "subdirs": subdirs,
        "total_images": sum(subdir["image_count"] for subdir in subdirs),
        "total_images_with_repeats": sum(subdir["image_count"] * subdir["repeat"] for subdir in subdirs),
    })


@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    return APIResponseSuccess(data={"tasks": tm.dump()})


@router.get("/tasks/terminate/{task_id}", response_model_exclude_none=True)
async def terminate_task(task_id: str):
    tm.terminate_task(task_id)
    return APIResponseSuccess()


@router.get("/graphic_cards")
async def list_available_cards() -> APIResponse:
    if not printable_devices:
        return APIResponsePending()
    return APIResponseSuccess(data={"cards": printable_devices})


@router.get("/runtime")
async def get_runtime() -> APIResponse:
    versions = {}
    try:
        versions["sd_scripts"] = launcher.git_tag(str(launcher.sd_scripts_path()))
    except Exception:
        versions["sd_scripts"] = "unknown"

    # whether the installed sd-scripts can apply the options this GUI exposes
    try:
        versions["sd_scripts_status"] = sd_scripts.compatibility_status()
    except Exception:
        versions["sd_scripts_status"] = "unknown"

    try:
        import torch
        versions["pytorch"] = torch.__version__
        versions["cuda"] = (torch.version.cuda or "unknown") if torch.cuda.is_available() else "not available"
    except Exception:
        versions["pytorch"] = "unknown"
        versions["cuda"] = "unknown"

    try:
        import triton
        versions["triton"] = getattr(triton, "__version__", "installed")
    except Exception:
        versions["triton"] = "not installed"

    return APIResponseSuccess(data={
        "services": {
            "tensorboard": os.environ.get("LORA_FORGE_ENABLE_TENSORBOARD", "0") == "1",
        },
        "versions": versions,
    })


@router.get("/schemas/all")
async def get_all_schemas() -> APIResponse:
    return APIResponseSuccess(data={"schemas": await asyncio.to_thread(load_schemas)})
