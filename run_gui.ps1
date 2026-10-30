# LoRA-Forge launcher for Windows. Needs uv (https://docs.astral.sh/uv/): `uv run` installs Python 3.14
# and the dependencies on the first start and keeps them in sync, so there is no separate install step.
#
#   .\run_gui.ps1                                  start the GUI (the first start installs PyTorch etc.)
#   .\run_gui.ps1 --cn                             use mirrors for mainland China (PyPI: TUNA, Hugging Face: hf-mirror)
#   .\run_gui.ps1 update-sd-scripts [--branch X]   update scripts/sd-scripts
#   .\run_gui.ps1 update-musubi-tuner [--branch X] update scripts/musubi-tuner (Krea 2)
#
# Any other arguments go to main.py (see README "Program arguments").

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Env:HF_HOME = Join-Path $ScriptDir "huggingface"
$Env:PYTHONUTF8 = "1"

$forwarded = @($args | Where-Object { $_ -ne "--cn" })
if ($args -contains "--cn") {
    $Env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
    $Env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host -ForegroundColor Cyan "Using mirrors in China (PyPI: TUNA, Hugging Face: hf-mirror). PyTorch still comes from the index set in pyproject.toml."
}
$isGui = -not ($forwarded.Count -gt 0 -and $forwarded[0] -in @("update-sd-scripts", "update-musubi-tuner"))

function Exit-WithError([string]$Message) {
    Write-Host -ForegroundColor Red $Message
    # keeps a window opened with "Run with PowerShell" open long enough to read the error
    Read-Host "Press Enter to exit" | Out-Null
    exit 1
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Exit-WithError ("uv was not found. Install it with the official installer, then open a new terminal and run this again:`n" +
        "  powershell -ExecutionPolicy ByPass -c `"irm https://astral.sh/uv/install.ps1 | iex`"`n" +
        "(other ways: https://docs.astral.sh/uv/getting-started/installation/)")
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
        Write-Host -ForegroundColor Cyan "First start: installing dependencies (PyTorch etc.), this can take a while..."
    }
}

# Since PyTorch 2.15 its own attention runs on flash kernels, so flash-attn and xformers are no longer used.
# `uv run` keeps packages it did not install: remove builds of them made for an older PyTorch.
$sitePackages = Join-Path $ScriptDir ".venv\Lib\site-packages"
$stale = @("flash_attn", "xformers") | Where-Object { Test-Path (Join-Path $sitePackages $_) }
if ($stale) {
    Write-Host -ForegroundColor Cyan "Removing $($stale -join ' and '), which PyTorch's own attention replaces"
    uv pip uninstall --python (Join-Path $ScriptDir ".venv") ($stale -replace "_", "-")
}

uv run --project $ScriptDir python (Join-Path $ScriptDir "main.py") @forwarded
if ($isGui -and $LASTEXITCODE -ne 0) {
    Exit-WithError "LoRA-Forge exited with an error (code $LASTEXITCODE)."
}
exit $LASTEXITCODE