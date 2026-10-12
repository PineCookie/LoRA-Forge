import os
import shutil
import subprocess
import sys
import socket
from pathlib import Path
import tomllib

from lora_forge.log import log


def base_dir_path():
    return Path(__file__).parents[1].absolute()


def find_windows_git():
    possible_paths = ["git\\bin\\git.exe", "git\\cmd\\git.exe",
                      "Git\\mingw64\\libexec\\git-core\\git.exe", "C:\\Program Files\\Git\\cmd\\git.exe"]
    for path in possible_paths:
        if os.path.exists(path):
            return path


def prepare_git():
    if shutil.which("git"):
        return True

    log.info("Finding git...")

    if sys.platform == "win32":
        git_path = find_windows_git()

        if git_path is not None:
            log.info(f"Git not found, but found git in {git_path}, add it to PATH")
            os.environ["PATH"] += os.pathsep + os.path.dirname(git_path)
            return True
        else:
            return False
    else:
        log.error("git not found, please install git first")
        return False


SD_SCRIPTS_REMOTE = "https://github.com/kohya-ss/sd-scripts.git"


def sd_scripts_path() -> Path:
    return base_dir_path() / "scripts" / "sd-scripts"


def ensure_sd_scripts(ref: str = "main"):
    """Clone kohya-ss/sd-scripts at `ref` (branch or tag) if it is missing.

    An existing checkout is left exactly as it is: updating is explicit (`main.py
    update-sd-scripts`), so starting the GUI never moves sd-scripts away from a tested version.
    """
    path = sd_scripts_path()

    if (path / ".git").exists():
        return

    if path.exists() and any(path.iterdir()):
        log.warning(f"sd-scripts path already exists and is not empty: {path}")
        return

    if not prepare_git():
        log.error("git not found, please install git first")
        sys.exit(1)

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Cloning kohya-ss/sd-scripts ({ref}) to {path}...")
    result = subprocess.run(["git", "clone", "--branch", ref, SD_SCRIPTS_REMOTE, str(path)])
    if result.returncode != 0:
        # Fallback: clone then checkout the ref (handles tags that aren't branches)
        log.info("Branch clone failed, trying full clone + checkout...")
        result = subprocess.run(["git", "clone", SD_SCRIPTS_REMOTE, str(path)])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone kohya-ss/sd-scripts")
        subprocess.run(
            ["git", "-C", str(path), "checkout", ref],
            check=True
        )


def update_sd_scripts(ref: str = "main", remote: str = SD_SCRIPTS_REMOTE):
    """Update scripts/sd-scripts: fast-forward to the latest commit of branch `ref`, or switch to
    tag `ref`. Clones it when missing. Stops (non-zero exit) instead of overwriting local changes."""
    path = sd_scripts_path()
    if not (path / ".git").exists():
        if path.exists() and any(path.iterdir()):
            raise SystemExit(f"{path} exists but is not a Git checkout.")
        ensure_sd_scripts(ref)
        return
    if not prepare_git():
        raise SystemExit("git was not found on PATH. Install Git before updating sd-scripts.")

    def git(*args: str, quiet: bool = False) -> subprocess.CompletedProcess:
        result = subprocess.run(["git", "-C", str(path), *args], capture_output=quiet, text=True)
        if result.returncode != 0 and not quiet:
            raise SystemExit(result.returncode)
        return result

    if git("remote", "get-url", "origin", quiet=True).stdout.strip() != remote:
        git("remote", "set-url", "origin", remote)

    git("fetch", "origin", "--tags")
    if git("rev-parse", "--verify", "--quiet", f"refs/tags/{ref}", quiet=True).returncode == 0:
        git("checkout", f"refs/tags/{ref}")
    else:
        git("fetch", "origin", f"refs/heads/{ref}:refs/remotes/origin/{ref}")
        if git("show-ref", "--verify", "--quiet", f"refs/heads/{ref}", quiet=True).returncode == 0:
            git("checkout", ref)
        else:
            git("checkout", "--track", f"origin/{ref}")
        git("pull", "--ff-only", "origin", ref)

    head = git("rev-parse", "--short", "HEAD", quiet=True).stdout.strip()
    log.info(f"sd-scripts is now at {git_tag(str(path))} ({head})")


def git_tag(path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", path, "describe", "--tags"],
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        pyproject_path = Path(path) / "pyproject.toml"
        if pyproject_path.exists():
            try:
                with pyproject_path.open("rb") as f:
                    return tomllib.load(f).get("project", {}).get("version", "<none>")
            except Exception:
                pass
        return "<none>"


def catch_exception(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            log.error(f"An error occurred: {e}")
    return wrapper


def check_port_available(port: int):
    try:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.close()
        return True
    except:
        return False


def find_available_ports(port_init: int, port_range: int):
    server_ports = range(port_init, port_range)

    for p in server_ports:
        if check_port_available(p):
            return p

    log.error(f"error finding available ports in range: {port_init} -> {port_range}")
    return None
