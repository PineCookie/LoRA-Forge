"""LoRA-Forge entry point.

    main.py [options]                         start the GUI (see README "Program arguments")
    main.py update-sd-scripts [--branch REF]  update scripts/sd-scripts to the latest REF (branch or tag)
    main.py update-musubi-tuner [--branch REF]  the same for scripts/musubi-tuner (Krea 2)

run_gui.ps1 / run_gui.sh forward their arguments here through `uv run`.
"""

import argparse
import os
import platform
import subprocess
import sys

from lora_forge.frontend import ensure_frontend
from lora_forge.launcher import (
    MUSUBI_TUNER_REMOTE,
    SD_SCRIPTS_REMOTE,
    base_dir_path,
    catch_exception,
    git_tag,
    check_port_available,
    ensure_musubi_tuner,
    ensure_sd_scripts,
    find_available_ports,
    musubi_tuner_path,
    sd_scripts_path,
    update_musubi_tuner,
    update_sd_scripts,
)
from lora_forge.log import log
from lora_forge.musubi_tuner import MUSUBI_TUNER_TESTED_TAG
from lora_forge.sd_scripts import SD_SCRIPTS_TESTED_TAG

parser = argparse.ArgumentParser(
    description="GUI for stable diffusion training",
    epilog="commands: update-sd-scripts, update-musubi-tuner (run `main.py <command> -h` for details)",
)
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=28000, help="Port to run the server on")
parser.add_argument("--listen", action="store_true")
parser.add_argument("--enable-tensorboard", action="store_true")
parser.add_argument("--skip-prepare-sd-scripts", action="store_true")
parser.add_argument("--skip-frontend-build", action="store_true", help="Do not (re)build the web UI in frontend/ on launch")
parser.add_argument("--sd-scripts-branch", type=str, default=SD_SCRIPTS_TESTED_TAG, help=f"Branch or tag to clone from kohya-ss/sd-scripts when it is missing (default: {SD_SCRIPTS_TESTED_TAG}, the tested version)")
parser.add_argument("--skip-prepare-musubi-tuner", action="store_true", help="Do not clone musubi-tuner (Krea 2 training)")
parser.add_argument("--musubi-tuner-branch", type=str, default=MUSUBI_TUNER_TESTED_TAG, help=f"Branch or tag to clone from kohya-ss/musubi-tuner when it is missing (default: {MUSUBI_TUNER_TESTED_TAG}, the tested version)")
parser.add_argument("--tensorboard-host", type=str, default="127.0.0.1", help="Port to run the tensorboard")
parser.add_argument("--tensorboard-port", type=int, default=6006, help="Port to run the tensorboard")
parser.add_argument("--dev", action="store_true")
parser.add_argument("--open-browser", action="store_true", help="Open the browser after the server starts")


@catch_exception
def run_tensorboard():
    log.info("Starting tensorboard...")
    subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", "logs",
                     "--host", args.tensorboard_host, "--port", str(args.tensorboard_port)])


def launch():
    # the app resolves its folders (config, output, scripts, ...) relative to the repository
    os.chdir(base_dir_path())

    log.info("Starting LoRA-Forge GUI...")
    log.info(f"Base directory: {base_dir_path()}, Working directory: {os.getcwd()}")
    log.info(f"{platform.system()} Python {platform.python_version()} {sys.executable}")

    if not args.skip_prepare_sd_scripts:
        ensure_sd_scripts(args.sd_scripts_branch)
    log.info(f"sd-scripts: {git_tag(str(sd_scripts_path()))} (not updated automatically; run `update-sd-scripts` to update)")
    if not args.skip_prepare_musubi_tuner:
        ensure_musubi_tuner(args.musubi_tuner_branch)
    log.info(f"musubi-tuner: {git_tag(str(musubi_tuner_path()))} (Krea 2; run `update-musubi-tuner` to update)")

    if not args.skip_frontend_build:
        ensure_frontend()

    if not check_port_available(args.port):
        available = find_available_ports(30000, 30000+20)
        if available:
            args.port = available
        else:
            log.error("port finding fallback error")

    log.info(f"LoRA-Forge Version: {git_tag(str(base_dir_path()))}")

    if args.listen:
        args.host = "0.0.0.0"
        args.tensorboard_host = "0.0.0.0"

    os.environ["LORA_FORGE_HOST"] = args.host
    os.environ["LORA_FORGE_PORT"] = str(args.port)
    os.environ["LORA_FORGE_TENSORBOARD_HOST"] = args.tensorboard_host
    os.environ["LORA_FORGE_TENSORBOARD_PORT"] = str(args.tensorboard_port)
    os.environ["LORA_FORGE_ENABLE_TENSORBOARD"] = "1" if args.enable_tensorboard else "0"
    os.environ["LORA_FORGE_OPEN_BROWSER"] = "1" if args.open_browser else "0"

    if args.enable_tensorboard:
        run_tensorboard()

    import uvicorn
    log.info(f"Server started at http://{args.host}:{args.port}")
    uvicorn.run("lora_forge.app:app", host=args.host, port=args.port, log_level="error", reload=args.dev)


def command_update_sd_scripts(argv):
    command = argparse.ArgumentParser(prog="main.py update-sd-scripts", description="Update scripts/sd-scripts: fast-forward a branch or switch to a tag. Stops instead of overwriting local changes.")
    command.add_argument("--branch", default="main", help="Branch or tag to follow (default: main; Anima needs a recent main)")
    command.add_argument("--remote", default=SD_SCRIPTS_REMOTE, help=f"Git remote URL (default: {SD_SCRIPTS_REMOTE})")
    options = command.parse_args(argv)
    update_sd_scripts(options.branch, options.remote)


def command_update_musubi_tuner(argv):
    command = argparse.ArgumentParser(prog="main.py update-musubi-tuner", description="Update scripts/musubi-tuner (Krea 2 training): fast-forward a branch or switch to a tag. Stops instead of overwriting local changes.")
    command.add_argument("--branch", default="main", help="Branch or tag to follow (default: main)")
    command.add_argument("--remote", default=MUSUBI_TUNER_REMOTE, help=f"Git remote URL (default: {MUSUBI_TUNER_REMOTE})")
    options = command.parse_args(argv)
    update_musubi_tuner(options.branch, options.remote)


COMMANDS = {
    "update-sd-scripts": command_update_sd_scripts,
    "update-musubi-tuner": command_update_musubi_tuner,
}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        COMMANDS[sys.argv[1]](sys.argv[2:])
    else:
        args, _ = parser.parse_known_args()
        launch()
