<div align="center">

<img src="assets/logo.svg" width="200" height="200" alt="LoRA Forge" style="border-radius: 25px">

# LoRA Forge

_✨ Enjoy LoRA training! ✨_

</div>

<p align="center">
  <a href="https://github.com/PineCookie/lora-forge" style="margin: 2px;">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PineCookie/lora-forge">
  </a>
  <a href="https://github.com/PineCookie/lora-forge" style="margin: 2px;">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/PineCookie/lora-forge">
  </a>
  <a href="https://raw.githubusercontent.com/PineCookie/lora-forge/main/LICENSE" style="margin: 2px;">
    <img src="https://img.shields.io/github/license/PineCookie/lora-forge" alt="license">
  </a>
  <a href="https://github.com/PineCookie/lora-forge/releases" style="margin: 2px;">
    <img src="https://img.shields.io/github/v/release/PineCookie/lora-forge?color=blueviolet&include_prereleases" alt="release">
  </a>
</p>

<p align="center">
  <a href="https://github.com/PineCookie/lora-forge/releases">Download</a>
  ·
  <a href="https://github.com/PineCookie/lora-forge/blob/main/README.md">Documents</a>
  ·
  <a href="https://github.com/PineCookie/lora-forge/blob/main/README-zh.md">中文README</a>
</p>

LoRA & Dreambooth training GUI and one-click training environment for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git)

**WARNING: This project is mainly maintained for personal use. It aims to continue the easy-to-use UI and workflow from [lora-scripts](https://github.com/Akegarasu/lora-scripts), and may currently contain many bugs. Suggestions and PRs are welcome!**

## ✨ New: Anima LoRA Training

- [x] Refactor the UI to support model-specific training parameters, including Anima (Vue 3 + TypeScript)
- [x] Use uv to install and manage the environment instead of relying on requirements.txt or legacy install scripts
- [ ] Support Torch installation for different GPUs
- [ ] Add missing features from the original frontend


# Usage

### Required Dependencies

Python 3.12 and Git. When running from a git clone, also [Node.js](https://nodejs.org/) 20 or newer: the launcher uses it to build the web UI on first start and after updates.

### Get the code

Either download the ready-to-run zip from [Releases](https://github.com/PineCookie/lora-forge/releases) (the web UI is prebuilt, no Node.js needed), or clone the repository:

```sh
git clone https://github.com/PineCookie/lora-forge
```

## ✨ LoRA Forge GUI

Start the GUI with the launcher for your system, then open [http://127.0.0.1:28000](http://127.0.0.1:28000) (add `--open-browser` to open it automatically):

```powershell
.\run_gui.ps1          # Windows (or right-click → Run with PowerShell)
```

```sh
bash run_gui.sh        # Linux
```

There is no separate install step: the launcher installs [uv](https://docs.astral.sh/uv/) if it is missing, and `uv run` installs the dependencies from `pyproject.toml` into `.venv` on the first start (this downloads PyTorch and takes a while) and keeps them in sync after updates. In mainland China add `--cn` to use the TUNA PyPI mirror and hf-mirror.

`run_gui.ps1 update-sd-scripts` (`bash run_gui.sh update-sd-scripts` on Linux) updates `scripts/sd-scripts`; see [Updating sd-scripts](#updating-sd-scripts).

### Optional: flash-attn

`flash-attn` is not installed by default. If you have a compatible local wheel, keep it outside Git (for example in the ignored `wheelhouse/` folder), then include the optional extra:

```sh
uv sync -U --extra flash-attn --find-links ./wheelhouse --no-build-package flash-attn
```

The `-U` flag lets uv update your local lockfile to use the wheel, and `--no-build-package flash-attn` prevents uv from falling back to building `flash-attn` from source.

### Web UI development

The web UI is a Vue 3 + TypeScript app in `frontend/`. To work on it with hot reload, start the GUI with `--skip-frontend-build`, then:

```sh
cd frontend
npm ci
npm run dev        # serves the UI and forwards /api to the GUI on port 28000
npm test           # unit tests (they evaluate the schemas in lora_forge/schema)
npm run typecheck
```

Releases are built by GitHub Actions: pushing a `v*` tag publishes a release with the ready-to-run zip.

### Languages

The web UI is in English, Chinese and Japanese. It follows the browser language and can be switched at the bottom of the sidebar. English is the source language; translations live in three places:

| Text | Source (English) | Translations |
|---|---|---|
| Buttons, pages, notices | `frontend/src/locales/en.ts` | `zh.ts`, `ja.ts` next to it (the type check fails on a missing key) |
| Training options and section titles | the schema files in `lora_forge/schema/` | `lora_forge/schema/locales/{zh,ja}.json`, keyed by the English text |
| Messages from the backend | `lora_forge/i18n.py` | the same file |

When you add or reword an option, add its English text to both JSON files; `npm test` fails while a translation is missing or left over. Terminal output and exported TOML files stay in English.

### TensorBoard

Start the GUI with `--enable-tensorboard` (for example `run_gui.ps1 --enable-tensorboard`) to open TensorBoard from the TensorBoard page. It reads the `logs` folder and listens on http://127.0.0.1:6006/.

## Program arguments

| Parameter Name                | Type  | Default Value | Description                                      |
|-------------------------------|-------|---------------|--------------------------------------------------|
| `--host`                      | str   | "127.0.0.1"   | Hostname for the server                          |
| `--port`                      | int   | 28000         | Port to run the server                           |
| `--listen`                    | bool  | false         | Enable listening mode for the server             |
| `--skip-prepare-sd-scripts`   | bool  | false         | Skip cloning kohya-ss/sd-scripts                |
| `--skip-frontend-build`       | bool  | false         | Do not (re)build the web UI on launch            |
| `--sd-scripts-branch`         | str   | "main"        | Branch or tag to clone when sd-scripts is missing |
| `--enable-tensorboard`        | bool  | false         | Enable TensorBoard                               |
| `--tensorboard-host`          | str   | "127.0.0.1"   | Host to run TensorBoard                          |
| `--tensorboard-port`          | int   | 6006          | Port to run TensorBoard                          |
| `--dev`                       | bool  | false         | Developer mode to disable some checks            |
| `--open-browser`              | bool  | false         | Open the browser after the server starts         |

## Updating sd-scripts

`scripts/sd-scripts` is cloned as its own Git checkout from `kohya-ss/sd-scripts`, not as a parent-repo submodule. The launcher clones it when missing, but never updates an existing checkout, so starting the GUI does not move sd-scripts away from the version you trained with. It is ignored by this repository (see `.gitignore`), so do not remove it with `git clean`; if it is missing, start the launcher again to re-clone it.

To update it to the latest commit of `main`:

```powershell
.\run_gui.ps1 update-sd-scripts
```

```sh
bash run_gui.sh update-sd-scripts
```

Anima training and the acceleration options the UI exposes (`--compile`, `--qwen_image_vae_2d`,
per-subset timestep offsets, `--show_timesteps`) require a recent `main`; older branches such as
`sd3` do not have them and silently ignore those settings.

`--branch` follows another branch or switches to a tag, for example `update-sd-scripts --branch v0.12.0`.
Branches are updated with `git pull --ff-only`, so the update stops instead of overwriting local changes.

## Anima training notes

Which Anima options actually take effect, the sd-scripts version each one needs, the speed/memory
recipes and the known silent-failure traps are documented in [docs/anima.md](docs/anima.md)
(Chinese). The About page shows the sd-scripts compatibility of the installed checkout.

## Project layout

| Path | What it is |
|---|---|
| `main.py` | Entry point (GUI and the `update-sd-scripts` command) |
| `run_gui.ps1`, `run_gui.sh` | Launchers: set up uv and the dependencies, then run `main.py` |
| `lora_forge/` | Python backend: web server and API (`app/`), training config, validation and runs (`training/`), sd-scripts compatibility checks |
| `lora_forge/schema/` | Training form definitions, read by the web UI |
| `frontend/` | Vue 3 + TypeScript web UI |
| `scripts/sd-scripts` | kohya-ss/sd-scripts checkout, cloned on first start (not part of this repository) |

The backend package was named `mikazuki` in upstream lora-scripts. It was renamed to `lora_forge` in October 2026, together with its environment variables (`MIKAZUKI_*` → `LORA_FORGE_*`).

## License and credits

LoRA-Forge is a modified version of [lora-scripts](https://github.com/Akegarasu/lora-scripts) by Akegarasu and contributors. It was forked in May 2026 and has been substantially changed since (Vue web UI, Anima training, reworked backend and launchers). Like the original, it is licensed under the **GNU Affero General Public License v3.0**; see [LICENSE](LICENSE).

- If you run a modified version as a network service (for example with `--listen`), AGPL section 13 requires offering its source code to the users. The web UI links to this repository for that; point the link at your own source if you host a modified version.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (Apache-2.0) is downloaded into `scripts/sd-scripts` on first start and is not distributed with this repository.
- The built web UI bundles Vue and vue-router (MIT). Their license notices are in `frontend/dist/THIRD-PARTY-LICENSES.md`, which is included in the release zip.

Thanks to Akegarasu and the original contributors for building the foundation.
