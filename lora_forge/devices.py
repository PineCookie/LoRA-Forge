from lora_forge.log import log
from packaging.version import Version

available_devices = []
printable_devices = []


def check_torch_gpu():
    try:
        import torch
        log.info(f'Torch {torch.__version__}')
        if not torch.cuda.is_available():
            log.error("Torch is not able to use GPU, please check your torch installation.")
            if "cpu" in torch.__version__:
                log.error("You are using torch CPU, please install the GPU build: run `uv sync` in the LoRA-Forge folder.")
            return

        if Version(torch.__version__) < Version("2.3.0"):
            log.warning("Torch version is lower than 2.3.0, which may not be able to train FLUX model properly. Run `uv sync` in the LoRA-Forge folder to upgrade Torch.")

        if torch.version.cuda:
            log.info(
                f'Torch backend: nVidia CUDA {torch.version.cuda} cuDNN {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"}')
        elif torch.version.hip:
            log.info(f'Torch backend: AMD ROCm HIP {torch.version.hip}')

        devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

        for pos, device in enumerate(devices):
            name = torch.cuda.get_device_name(device)
            memory = torch.cuda.get_device_properties(device).total_memory
            available_devices.append(device)
            printable_devices.append(f"GPU {pos}: {name} ({round(memory / (1024**3))} GB)")
            log.info(
                f'Torch detected GPU: {name} VRAM {round(memory / 1024 / 1024)} Arch {torch.cuda.get_device_capability(device)} Cores {torch.cuda.get_device_properties(device).multi_processor_count}')
        for gpu in gpu_info():
            if gpu["support"] == "unsupported":
                log.error(f'{gpu["name"]} (compute capability {gpu["capability"]}) is too old for this PyTorch build, which needs '
                          f'{gpu["minimum"]} or newer (RTX 20 series and up). Training on it will fail.')
            elif gpu["support"] == "limited":
                log.warning(f'{gpu["name"]} (compute capability {gpu["capability"]}) has no bf16 and no flash-attn support: '
                            'choose fp16 instead of bf16 on the trainer pages.')
    except Exception as e:
        log.error(f'Could not load torch: {e}')


# bf16 tensor cores and flash-attn start with Ampere
FULL_SUPPORT = (8, 0)


def gpu_info() -> list[dict]:
    """Each GPU with its compute capability and how well this PyTorch build supports it: `ok`, `limited`
    (Turing: no bf16, no flash-attn) or `unsupported` (older than the oldest architecture the build has
    kernels for; CUDA 13 builds start at 7.5). Reads device properties only, runs nothing on the GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        built = [int(arch[3:]) for arch in torch.cuda.get_arch_list() if arch.startswith("sm_") and arch[3:].isdigit()]
        minimum = divmod(min(built), 10) if built else None
        result = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = (properties.major, properties.minor)
            if minimum and capability < minimum:
                support = "unsupported"
            elif capability < FULL_SUPPORT:
                support = "limited"
            else:
                support = "ok"
            result.append({
                "name": properties.name,
                "memory_gb": round(properties.total_memory / 1024**3),
                "capability": "%d.%d" % capability,
                "minimum": "%d.%d" % minimum if minimum else "",
                "support": support,
            })
        return result
    except Exception:
        return []
