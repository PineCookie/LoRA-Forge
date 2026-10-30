#!/usr/bin/env bash
# LoRA-Forge launcher for Linux. Needs uv (https://docs.astral.sh/uv/): `uv run` installs Python 3.14
# and the dependencies on the first start and keeps them in sync, so there is no separate install step.
#
#   bash run_gui.sh                                  start the GUI (the first start installs PyTorch etc.)
#   bash run_gui.sh --cn                             use mirrors for mainland China (PyPI: TUNA, Hugging Face: hf-mirror)
#   bash run_gui.sh update-sd-scripts [--branch X]   update scripts/sd-scripts
#   bash run_gui.sh update-musubi-tuner [--branch X] update scripts/musubi-tuner (Krea 2)
#
# Any other arguments go to main.py (see README "Program arguments").

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export HF_HOME="$script_dir/huggingface"
export PYTHONUTF8=1

forwarded=()
for arg in "$@"; do
    if [[ "$arg" == "--cn" ]]; then
        export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
        export HF_ENDPOINT="https://hf-mirror.com"
        echo "Using mirrors in China (PyPI: TUNA, Hugging Face: hf-mirror). PyTorch still comes from the index set in pyproject.toml."
    else
        forwarded+=("$arg")
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found. Install it with the official installer, then open a new terminal and run this again:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "(other ways: https://docs.astral.sh/uv/getting-started/installation/)" >&2
    exit 1
fi

if [[ ! -d "$script_dir/.venv" ]]; then
    echo "First start: installing dependencies (PyTorch etc.), this can take a while..."
fi

# Since PyTorch 2.15 its own attention runs on flash kernels, so flash-attn and xformers are no longer used.
# `uv run` keeps packages it did not install: remove builds of them made for an older PyTorch.
stale=()
for package in flash_attn xformers; do
    if compgen -G "$script_dir/.venv/lib/python3*/site-packages/$package" >/dev/null; then
        stale+=("${package//_/-}")
    fi
done
if (( ${#stale[@]} )); then
    echo "Removing ${stale[*]}, which PyTorch's own attention replaces"
    uv pip uninstall --python "$script_dir/.venv" "${stale[@]}"
fi

exec uv run --project "$script_dir" python "$script_dir/main.py" "${forwarded[@]}"
