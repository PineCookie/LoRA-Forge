"""Keep frontend/dist in step with the Vue source in frontend/.

The built UI is not committed. Release zips ship it prebuilt; a git checkout builds it on launch
when Node.js is available. `frontend/dist/.source-hash` records which source the build came
from, so the launcher only rebuilds after the source changed (e.g. after `git pull`).

`python -m lora_forge.frontend stamp` writes that record; the release workflow runs it after
building. This module only uses the standard library so CI can run it without the app's
dependencies.
"""

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_DIR / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
STAMP_FILE = DIST_DIR / ".source-hash"

# Everything the build reads. The schema files are fetched at runtime, so they are not part of it.
SOURCE_FILES = ("index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts")
SOURCE_DIRS = ("src",)
EXTRA_FILES = (REPO_DIR / "assets" / "logo.svg",)


def _source_paths():
    paths = [FRONTEND_DIR / name for name in SOURCE_FILES]
    for directory in SOURCE_DIRS:
        paths.extend(path for path in (FRONTEND_DIR / directory).rglob("*") if path.is_file())
    paths.extend(EXTRA_FILES)
    return sorted((path for path in paths if path.is_file()), key=lambda path: path.relative_to(REPO_DIR).as_posix())


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in _source_paths():
        digest.update(path.relative_to(REPO_DIR).as_posix().encode("utf-8") + b"\0")
        # line endings differ between a Windows checkout and the CI build
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def is_built() -> bool:
    return (DIST_DIR / "index.html").is_file()


def is_current() -> bool:
    try:
        return is_built() and STAMP_FILE.read_text(encoding="utf-8").strip() == source_hash()
    except OSError:
        return False


def write_stamp():
    STAMP_FILE.write_text(source_hash() + "\n", encoding="utf-8")


def _dependencies_installed() -> bool:
    # npm writes node_modules/.package-lock.json on install; a newer package-lock.json means
    # the dependencies changed since then
    installed = FRONTEND_DIR / "node_modules" / ".package-lock.json"
    lock = FRONTEND_DIR / "package-lock.json"
    return installed.is_file() and installed.stat().st_mtime >= lock.stat().st_mtime


def build(npm: str):
    if not _dependencies_installed():
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=FRONTEND_DIR, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)
    write_stamp()


def ensure_frontend():
    """Build the UI if it is missing or older than its source; never raises."""
    from lora_forge.log import log

    if is_current():
        return

    npm = shutil.which("npm")
    if npm is None:
        if is_built():
            log.warning("The web UI may be out of date with frontend/, and Node.js (npm) was not found to rebuild it. "
                        "Install Node.js 20+ to rebuild it automatically.")
        else:
            log.error("The web UI is not built and Node.js (npm) was not found. Install Node.js 20+ and restart, "
                      "or use the ready-to-run zip from GitHub Releases.")
        return

    log.info("Building the web UI (frontend/dist)...")
    try:
        build(npm)
    except (OSError, subprocess.CalledProcessError) as e:
        log.error(f"Building the web UI failed: {e}. Run `npm ci && npm run build` in frontend/ to see the details.")
        return
    log.info("Web UI built.")


if __name__ == "__main__":
    if sys.argv[1:] == ["stamp"]:
        write_stamp()
        print(f"{STAMP_FILE}: {source_hash()}")
    elif sys.argv[1:] == ["check"]:
        sys.exit(0 if is_current() else 1)
    else:
        sys.exit("usage: python -m lora_forge.frontend stamp|check")
