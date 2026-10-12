#!/usr/bin/env bash
# LoRA-Forge launcher for Linux. Installs uv when it is missing; `uv run` then installs and syncs
# the Python dependencies before starting, so there is no separate install step.
#
#   bash run_gui.sh                                  start the GUI (the first start installs PyTorch etc.)
#   bash run_gui.sh --cn                             use mirrors for mainland China (PyPI: TUNA, Hugging Face: hf-mirror)
#   bash run_gui.sh update-sd-scripts [--branch X]   update scripts/sd-scripts
#
# Any other arguments go to main.py (see README "Program arguments").

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export HF_HOME="$script_dir/huggingface"
export PYTHONUTF8=1

forwarded=()
for arg in "$@"; do
    if [[ "$arg" == "--cn" ]]; then
        export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
        export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
        export HF_ENDPOINT="https://hf-mirror.com"
        echo "Using mirrors in China (PyPI: TUNA, Hugging Face: hf-mirror). PyTorch still comes from the index set in pyproject.toml."
    else
        forwarded+=("$arg")
    fi
done

uv=(uv)
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing it with pip..."
    if python3 -m pip install --user --upgrade uv; then
        uv=(python3 -m uv)
    else
        echo "Installing uv with pip failed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi
fi

if [[ ! -d "$script_dir/.venv" ]]; then
    echo "First start: installing dependencies (PyTorch etc.), this can take a while..."
fi

exec "${uv[@]}" run --project "$script_dir" python "$script_dir/main.py" "${forwarded[@]}"
