"""HTTP API used by the web UI (mounted under /api)."""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from lora_forge import launcher, sd_scripts
from lora_forge.app.models import APIResponse, APIResponseFail, APIResponsePending, APIResponseSuccess
from lora_forge.devices import gpu_info, printable_devices
from lora_forge.file_dialogs import open_directory_selector, open_file_selector
from lora_forge.i18n import t
from lora_forge.log import log
from lora_forge.training import model_files, runner, runs
from lora_forge.training.config import TrainingConfigError, prepare_training, redact_config
from lora_forge.training.dataset import count_images, subset_folders
from lora_forge.training.tasks import TaskStatus, tm

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


def load_schema_translations() -> dict[str, dict[str, str]]:
    """lora_forge/schema/locales/<locale>.json: the schema text (English) -> its translation."""
    translations = {}
    for path in sorted((SCHEMA_DIR / "locales").glob("*.json")):
        try:
            translations[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning(f"Failed to read schema translations {path.name}: {e}")
    return translations


@router.post("/run")
async def run_training(request: Request):
    try:
        config: dict = json.loads((await request.body()).decode("utf-8"))
        log.info(f"Training config received: {redact_config(config)}")
        prepared = prepare_training(config)
        task_id, run_id = runner.start_training(prepared)
    except (TrainingConfigError, runner.TaskNotStartedError) as e:
        return APIResponseFail(message=str(e))
    except Exception as e:
        log.error(f"Error in /run endpoint: {e}", exc_info=True)
        return APIResponseFail(message=t("run.serverError", error=e))
    return APIResponseSuccess(message=t("run.started"), data={"task_id": task_id, "run_id": run_id})


@router.get("/pick_file")
async def pick_file(picker_type: str):
    if picker_type == "folder":
        result = await asyncio.to_thread(open_directory_selector, "")
    elif picker_type == "model-file":
        file_types = [("checkpoints", "*.safetensors;*.ckpt;*.pt"), ("all files", "*.*")]
        result = await asyncio.to_thread(open_file_selector, "", "Select file", file_types)
    else:
        return APIResponseFail(message=t("picker.unsupported", type=picker_type))

    if result == "":
        return APIResponseFail(message=t("picker.cancelled"))
    return APIResponseSuccess(data={"path": result})


@router.get("/dataset_info")
async def get_dataset_info(train_data_dir: str = "") -> APIResponse:
    """Image counts per `repeats_name` subfolder of a dataset, with and without repeats."""
    if not train_data_dir or not os.path.isdir(train_data_dir):
        return APIResponseFail(message=t("dataset.invalid"))

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
        return APIResponseFail(message=t("dataset.unreadable"))

    return APIResponseSuccess(data={
        "subdirs": subdirs,
        "total_images": sum(subdir["image_count"] for subdir in subdirs),
        "total_images_with_repeats": sum(subdir["image_count"] * subdir["repeat"] for subdir in subdirs),
    })


@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    def collect():
        tasks = tm.dump()
        for task in tasks:
            if task["run_id"] and task["status"] in ("CREATED", "RUNNING"):
                task["progress"] = runs.summary(task["run_id"])
        return tasks

    return APIResponseSuccess(data={"tasks": await asyncio.to_thread(collect)})


# --- runs (the monitor) ------------------------------------------------------------------------

def _run_status(info: runs.RunInfo) -> str:
    """The recorded status, corrected for a run whose GUI process exited while it ran."""
    if info.status != "running":
        return info.status
    task = tm.find_by_run(info.id)
    if task is None:
        return "interrupted"
    return "starting" if task.status == TaskStatus.CREATED else "running"


def _run_data(info: runs.RunInfo) -> dict:
    data = asdict(info)
    data["status"] = _run_status(info)
    # the folder and the config path are local paths; the UI shows names only
    data["config_file"] = os.path.basename(info.config_file)
    return data


def _find_run(run_id: str) -> runs.RunInfo:
    try:
        return runs.read_info(run_id)
    except (runs.RunNotFoundError, OSError, ValueError, TypeError):
        raise runs.RunNotFoundError(run_id)


@router.get("/runs")
async def list_runs() -> APIResponse:
    def collect():
        result = []
        for info in runs.list_runs():
            data = _run_data(info)
            if data["status"] in ("starting", "running"):
                data["progress"] = runs.summary(info.id)
            result.append(data)
        return result

    return APIResponseSuccess(data={"runs": await asyncio.to_thread(collect)})


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> APIResponse:
    def collect():
        info = _find_run(run_id)
        return {
            "run": _run_data(info),
            "progress": runs.summary(run_id),
            "samples": runs.samples(info),
            "checkpoints": runs.checkpoints(info),
        }

    try:
        return APIResponseSuccess(data=await asyncio.to_thread(collect))
    except runs.RunNotFoundError:
        return APIResponseFail(message=t("runs.notFound"))


@router.get("/runs/{run_id}/events")
async def get_run_events(run_id: str, offset: int = 0) -> APIResponse:
    try:
        events, next_offset = await asyncio.to_thread(runs.read_events, run_id, offset)
    except runs.RunNotFoundError:
        return APIResponseFail(message=t("runs.notFound"))
    return APIResponseSuccess(data={"events": events, "offset": next_offset})


@router.get("/runs/{run_id}/output")
async def get_run_output(run_id: str, lines: int = 400) -> APIResponse:
    try:
        output = await asyncio.to_thread(runs.tail_output, run_id, max(1, min(lines, 2000)))
    except runs.RunNotFoundError:
        return APIResponseFail(message=t("runs.notFound"))
    return APIResponseSuccess(data={"lines": output})


@router.get("/runs/{run_id}/samples/{name}")
async def get_run_sample(run_id: str, name: str):
    try:
        path = await asyncio.to_thread(runs.sample_file, _find_run(run_id), name)
    except runs.RunNotFoundError:
        path = None
    if path is None:
        raise HTTPException(status_code=404)
    return FileResponse(path)


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str) -> APIResponse:
    task = tm.find_by_run(run_id)
    if task is None:
        return APIResponseFail(message=t("runs.notFound"))
    tm.terminate_task(task.task_id)
    return APIResponseSuccess()


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> APIResponse:
    try:
        info = _find_run(run_id)
        if _run_status(info) in ("starting", "running"):
            return APIResponseFail(message=t("runs.stillRunning"))
        await asyncio.to_thread(runs.delete_run, run_id)
    except (runs.RunNotFoundError, OSError):
        return APIResponseFail(message=t("runs.notFound"))
    return APIResponseSuccess()


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
    # a release zip has no .git, so this falls back to pyproject.toml's version
    versions["lora_forge"] = launcher.git_tag(str(launcher.base_dir_path()))
    try:
        versions["sd_scripts"] = launcher.git_tag(str(launcher.sd_scripts_path()))
    except Exception:
        versions["sd_scripts"] = "unknown"

    # whether the installed sd-scripts can apply the options this GUI exposes
    try:
        versions["sd_scripts_state"], versions["sd_scripts_problem"] = sd_scripts.compatibility()
    except Exception:
        versions["sd_scripts_state"], versions["sd_scripts_problem"] = "unknown", ""

    try:
        musubi = launcher.musubi_tuner_path()
        versions["musubi_tuner"] = launcher.git_tag(str(musubi)) if musubi.exists() else "not installed"
    except Exception:
        versions["musubi_tuner"] = "unknown"

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
        "gpus": await asyncio.to_thread(gpu_info),
        # download links on the trainer pages use the mirror when the launcher set one (--cn)
        "hf_endpoint": (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/"),
    })


@router.get("/model_file")
async def get_model_file_state(model: str, field: str, path: str = "") -> APIResponse:
    """Whether a model file a trainer page asks for is there and is the right one."""
    return APIResponseSuccess(data={"state": await asyncio.to_thread(model_files.check, model, field, path)})


@router.get("/schemas/all")
async def get_all_schemas() -> APIResponse:
    return APIResponseSuccess(data={
        "schemas": await asyncio.to_thread(load_schemas),
        "translations": await asyncio.to_thread(load_schema_translations),
    })
