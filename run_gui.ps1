# LoRA-Forge launcher for Windows. Installs uv when it is missing; `uv run` then installs and syncs
# the Python dependencies before starting, so there is no separate install step.
#
#   .\run_gui.ps1                                  start the GUI (the first start installs PyTorch etc.)
#   .\run_gui.ps1 --cn                             use mirrors for mainland China (PyPI: TUNA, Hugging Face: hf-mirror)
#   .\run_gui.ps1 update-sd-scripts [--branch X]   update scripts/sd-scripts
#
# Any other arguments go to main.py (see README "Program arguments").

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Env:HF_HOME = Join-Path $ScriptDir "huggingface"
$Env:PYTHONUTF8 = "1"

$forwarded = @($args | Where-Object { $_ -ne "--cn" })
if ($args -contains "--cn") {
    $Env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
    $Env:PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
    $Env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host -ForegroundColor Cyan "使用国内镜像 (PyPI: TUNA, Hugging Face: hf-mirror)。PyTorch 仍从 pyproject.toml 配置的 PyTorch 源下载。"
}
$isGui = -not ($forwarded.Count -gt 0 -and $forwarded[0] -eq "update-sd-scripts")

function Exit-WithError([string]$Message) {
    Write-Host -ForegroundColor Red $Message
    # keeps a window opened with "Run with PowerShell" open long enough to read the error
    Read-Host "Press Enter to exit / 按回车退出" | Out-Null
    exit 1
}

# A portable Python in .\python takes precedence over the system one.
$PortablePython = Join-Path $ScriptDir "python"
if (Test-Path (Join-Path $PortablePython "python.exe")) {
    $Env:PATH = "$PortablePython;$Env:PATH"
}

$UseUvModule = $false
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found, installing it with pip... / 未找到 uv，正在用 pip 安装..."
    python -m pip install --upgrade uv
    if ($LASTEXITCODE -ne 0) {
        Exit-WithError "Installing uv failed. Install Python 3.12 or uv (https://docs.astral.sh/uv/) and try again. / uv 安装失败，请安装 Python 3.12 或 uv 后重试。"
    }
    $UseUvModule = $true
}

function Invoke-Uv {
    if ($UseUvModule) { python -m uv @args } else { uv @args }
}
# --- Auto-detect any installed Visual Studio (2022/2026/...) for torch.compile support ---
function Enable-MSVCEnvironment {
    # Prefer vswhere so the year-based install path does not matter; fall back to the
    # well-known VS 2022 paths when vswhere is unavailable.
    $vswhere = $null
    foreach ($base in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if ($base) {
            $candidate = Join-Path $base "Microsoft Visual Studio\Installer\vswhere.exe"
            if (Test-Path $candidate) { $vswhere = $candidate; break }
        }
    }

    $installationPath = $null
    if ($vswhere) {
        $installArgs = @("-latest", "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath")
        $installationPath = & $vswhere @installArgs 2>$null
        if (-not $installationPath) {
            # Also look at prerelease installs (e.g. Insiders) if no stable one matched.
            $installationPath = & $vswhere @installArgs -prerelease 2>$null
        }
        $installationPath = $installationPath | Select-Object -First 1
    }

    if (-not $installationPath) {
        $fallbacks = @(
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional",
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise"
        )
        $installationPath = $fallbacks |
            Where-Object { Test-Path (Join-Path $_ "VC\Auxiliary\Build\vcvars64.bat") } |
            Select-Object -First 1
    }

    if (-not $installationPath) {
        Write-Host -ForegroundColor Yellow "MSVC (Visual Studio C++ build tools) not found. --compile_dynamic=true will not work."
        Write-Host -ForegroundColor Yellow "  Install Visual Studio Build Tools and select 'Desktop development with C++': https://visualstudio.microsoft.com/downloads/"
        return $false
    }

    $vcvars = Join-Path $installationPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        Write-Host -ForegroundColor Yellow "Found Visual Studio at $installationPath but vcvars64.bat is missing (C++ workload not installed). --compile_dynamic=true will not work."
        return $false
    }

    Write-Host -ForegroundColor Cyan "Found Visual Studio at: $installationPath"
    Write-Host -ForegroundColor Cyan "Loading MSVC environment for torch.compile --compile_dynamic support..."
    # Run vcvars64.bat in cmd and capture the resulting environment
    cmd /c "`"$vcvars`" >NUL 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1]
            $val = $matches[2]
            if ($key -ne '_OLD_VIRTUAL_PATH' -and $key -ne 'PROMPT') {
                [Environment]::SetEnvironmentVariable($key, $val, 'Process')
            }
        }
    }
    Write-Host -ForegroundColor Green "MSVC environment loaded successfully"
    return $true
}

if ($isGui) {
    $vsLoaded = Enable-MSVCEnvironment
    if (-not (Test-Path (Join-Path $ScriptDir ".venv"))) {
        Write-Host -ForegroundColor Cyan "First start: installing dependencies (PyTorch etc.), this can take a while... / 首次启动：正在安装依赖（PyTorch 等），可能需要较长时间..."
    }
}

Invoke-Uv run --project $ScriptDir python (Join-Path $ScriptDir "main.py") @forwarded
if ($isGui -and $LASTEXITCODE -ne 0) {
    Exit-WithError "LoRA-Forge exited with an error (code $LASTEXITCODE). / LoRA-Forge 异常退出。"
}
exit $LASTEXITCODE