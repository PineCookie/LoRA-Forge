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
  <a href="docs/anima.md">Anima guide</a>
  ·
  <a href="docs/krea2.md">Krea 2 guide</a>
  ·
  <a href="README-zh.md">中文</a>
</p>

LoRA Forge is a web UI for training LoRAs with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and, for Krea 2, [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner). You fill in a form in the browser, press Start, and follow the training on a live page; LoRA Forge sets up Python, PyTorch and the trainers for you. It continues [lora-scripts](https://github.com/Akegarasu/lora-scripts).

- **Models**: Anima, Krea 2, SDXL (Illustrious, NoobAI, Pony, ...), FLUX, SD3.5 and SD1.5, plus full fine-tuning for SDXL and SD1.5 (Dreambooth). Each model has its own page with sensible defaults.
- **Easy forms**: an *Essentials* view with the options most people change, all options one click away, search, and import / export of configs as JSON or TOML.
- **Runs page**: progress and time left, loss and learning rate charts, sample images, the saved LoRA files and the console output, also for past runs. No TensorBoard needed.
- **Faster training**: LoRA Forge patches sd-scripts where it is slow, for example one compiled graph for all image sizes on Anima, per-block compile and flash-attn on SDXL, and a much faster ProdigyPlus optimizer step. An SDXL LoRA trains about twice as fast as with plain sd-scripts ([details](#speed)).
- **Krea 2 on 16 GB cards**: int8 base weights and block swap by default ([details](docs/krea2.md)).
- English, 中文 and 日本語, light and dark mode.

> [!NOTE]
> LoRA Forge is maintained mainly for personal use, so expect rough edges. Bug reports, suggestions and PRs are welcome.

## Quick start

### What you need

- Windows or Linux with an **NVIDIA GPU** and a driver that supports **CUDA 13** (LoRA Forge uses PyTorch 2.14 for CUDA 13.0). Other GPUs are not supported yet.
- [Git](https://git-scm.com/downloads).
- [uv](https://docs.astral.sh/uv/), which installs Python and everything else. Install it with the official installer, then **open a new terminal**:

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
  ```

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux
  ```

### Install and start

1. Download the latest `lora-forge-<version>.zip` from [Releases](https://github.com/PineCookie/lora-forge/releases) and extract it to a folder with enough free space (the environment and models take several GB).
2. Start it:
   - **Windows**: right-click `run_gui.ps1` → *Run with PowerShell*, or run `.\run_gui.ps1` in a terminal.
   - **Linux**: `bash run_gui.sh`.

   In mainland China, add `--cn` (`.\run_gui.ps1 --cn`) to download through the TUNA PyPI mirror and hf-mirror.
3. Open [http://127.0.0.1:28000](http://127.0.0.1:28000) (or start with `--open-browser`).

The first start downloads Python, PyTorch, sd-scripts and musubi-tuner and takes a while; later starts are quick. There is no separate install step.

## Training your first LoRA

**1. Prepare the dataset.** Put your images in subfolders named `<repeats>_<name>` inside one dataset folder. The number is how often each image is used per epoch. Each image can have a caption in a `.txt` file with the same name:

```
train/my_character/          ← choose this folder in the GUI
├── 10_my_character/
│   ├── 001.png
│   ├── 001.txt              ← "my_character, 1girl, smile, ..."
│   └── ...
└── 5_my_character_outfit/
    └── ...
```

Images directly in the dataset folder are not used; the GUI tells you when no `<repeats>_<name>` subfolder is found.

**2. Get the base model.** Download the model you want to train on (for Anima and Krea 2 also their text encoder and VAE). The model page shows which files it needs.

**3. Fill in the form.** Open the page for your model in the sidebar. In the *Essentials* view, set at least the base model file, the dataset folder and the output name; the other values are reasonable defaults. The step estimate next to the dataset folder shows how many steps the settings give.

**4. Start and watch.** Press **Start training**. The status in the header shows the progress; click it to open the run on the **Runs** page, where you see the loss, samples and console output. The LoRA files are saved to the output folder (`output/` by default) and listed on the run.

Each page keeps what you entered while the browser tab is open, also when you switch to the Runs page or another model and back, but not after a reload. To keep a config, use **Export JSON** or **Export TOML**, and **Import…** to load it again. Every training you start also saves its config in `config/autosave/`, which you can import as well.

## Speed

LoRA Forge runs sd-scripts and musubi-tuner unchanged, but patches them when a training starts, where they are slow. What is marked *on* below needs nothing from you. To run sd-scripts exactly as it is (for example to compare), set the environment variable `LORA_FORGE_PATCHES=0`.

| What | Models | Default | Gain per step |
|---|---|---|---|
| One compiled graph for all bucket and batch sizes (`compile_shapes = bounded`) instead of one per size | Anima | on | 2 graphs instead of about 28: an epoch 11% faster, 55% with an empty compile cache |
| Per-block `torch.compile` of the transformer and resnet blocks, one graph for all bucket sizes | SDXL | on | −21% |
| flash-attn for the attention (PyTorch's Windows builds have no flash kernels) | SDXL | on, when [flash-attn](#flash-attn-optional) is installed | −17% |
| SDPA attention by default (sd-scripts' plain attention ran when it was off) | SDXL, SD1.5 | on | about 2× faster, and less VRAM |
| ProdigyPlusScheduleFree updates tensors of the same shape together instead of one by one (same math) | all | on | −7% on Anima, −23% on SDXL |
| fp16 training with fp16 accumulation (`training_precision = fp16`), as precise as bf16 | Anima, SDXL, SD1.5 | opt-in | −24% on Anima, −8% on SDXL |
| `activation_memory_budget` in place of gradient checkpointing: keeps what fits in VRAM instead of recomputing it | Anima | opt-in (`auto`) | −8% to −11%, uses most of the VRAM |
| flash-attn for the attention | Krea 2 | on, when installed | −22% |
| Pinned, one-way block swap: swapping costs almost nothing and frees about 6 GB | Krea 2 | on | same speed with far more blocks swapped |
| No DataLoader worker processes when the latents are cached | all | on | the first step after about 7 s instead of 80 s on Windows |

Together: an SDXL LoRA step went from 1.51 s (sd-scripts with SDPA) to 0.81 s, or 0.72 s with fp16; an Anima LoRA step (already compiled) from 1.73 s to 1.17 s with fp16 and `activation_memory_budget = auto`. Every change was checked against the original for the same losses. Measured on an RTX 5070 Ti (16 GB) under Windows with rank-32 LoRAs at batch 2 and about 1 megapixel; other cards gain differently, and fp16 accumulation gains nothing on data-center or RTX PRO cards. The [Anima guide](docs/anima.md) and [Krea 2 guide](docs/krea2.md) explain the options.

## Updating

- **LoRA Forge**: download the new release and extract it over the old folder (or into a new one and copy your `train/`, `output/` and `config/` folders), or `git pull` if you cloned the repository.
- **sd-scripts** is never updated automatically, so a working setup stays as it is. A fresh install gets the version LoRA Forge was tested with. To move to the latest sd-scripts:

  ```powershell
  .\run_gui.ps1 update-sd-scripts           # Windows
  ```

  ```sh
  bash run_gui.sh update-sd-scripts         # Linux
  ```

  `--branch <name or tag>` picks another branch or a tag (`update-sd-scripts --branch v0.12.0`). The update stops instead of overwriting changes you made in `scripts/sd-scripts`. The **System info** panel on the home page shows whether the installed sd-scripts supports every option of the GUI.
- **musubi-tuner** (Krea 2) works the same way: `update-musubi-tuner [--branch <name or tag>]` updates `scripts/musubi-tuner`.

## Options

### flash-attn (optional)

[flash-attn](https://github.com/Dao-AILab/flash-attention) is a faster attention implementation; on the Anima page you choose it with `attn_mode = flash` (the default, `torch`, needs nothing extra), and the Krea 2 and SDXL pages use it automatically when it is installed (SDXL with `sdpa` on, its default). It is not installed by default. To install it, run this once in the LoRA Forge folder (Windows, or Linux on x86_64 or ARM):

```sh
uv sync --extra flash-attn
```

This downloads a prebuilt wheel (about 230 MB) from [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) that matches LoRA Forge's PyTorch, so nothing is compiled. From then on the launcher keeps it installed and updates it together with PyTorch. If you run `uv sync` yourself later, add `--extra flash-attn`, or `uv sync` removes it again.

### TensorBoard (optional)

The Runs page covers what most people need. To use TensorBoard as well, start with `--enable-tensorboard` and open the TensorBoard page; it reads the `logs` folder and listens on http://127.0.0.1:6006/.

### Program arguments

<details>
<summary>All launcher arguments</summary>

| Argument | Default | Description |
|---|---|---|
| `--cn` | off | Use mirrors for mainland China (launcher only) |
| `--host` | `127.0.0.1` | Address the server listens on |
| `--port` | `28000` | Port of the web UI (if it is taken, a free port from 30000 is used) |
| `--listen` | off | Listen on all network interfaces, so other devices can open the UI |
| `--open-browser` | off | Open the browser after starting |
| `--enable-tensorboard` | off | Start TensorBoard as well |
| `--tensorboard-host` | `127.0.0.1` | Address TensorBoard listens on |
| `--tensorboard-port` | `6006` | Port of TensorBoard |
| `--sd-scripts-branch` | `v0.12.0` | Branch or tag to clone when sd-scripts is missing (the tested version) |
| `--skip-prepare-sd-scripts` | off | Do not clone sd-scripts |
| `--musubi-tuner-branch` | `v0.3.6` | Branch or tag to clone when musubi-tuner is missing (the tested version) |
| `--skip-prepare-musubi-tuner` | off | Do not clone musubi-tuner |
| `--skip-frontend-build` | off | Do not (re)build the web UI on start |
| `--dev` | off | Developer mode |

</details>

## Troubleshooting

| Problem | What to do |
|---|---|
| `uv was not found` | Install uv as shown in [What you need](#what-you-need), then open a **new** terminal |
| Windows: *running scripts is disabled on this system* | Start it with right-click → *Run with PowerShell*, or run `powershell -ExecutionPolicy Bypass -File .\run_gui.ps1` |
| The first start takes very long | It downloads PyTorch (several GB); in mainland China use `--cn` |
| A training fails | Open it on the Runs page: the console output shows the error |
| An option seems to have no effect | Check *sd-scripts compatibility* in the **System info** panel on the home page |
| Anima: compile errors or slow first steps | See the [Anima guide](docs/anima.md) |
| SDXL: compile errors | Turn `compile` off on the SDXL LoRA page; training then runs uncompiled |
| Krea 2: minutes per step | The model does not fit in VRAM; raise `blocks_to_swap` (see the [Krea 2 guide](docs/krea2.md)) |

## For developers

<details>
<summary>Web UI development, translations, releases and the project layout</summary>

### Web UI development

The web UI is a Vue 3 + TypeScript app in `frontend/`. When you run from a git clone, the launcher builds it with [Node.js](https://nodejs.org/) 20 or newer on the first start and after updates. To work on it with hot reload, start the GUI with `--skip-frontend-build`, then:

```sh
cd frontend
npm ci
npm run dev        # serves the UI and forwards /api to the GUI on port 28000
npm test           # unit tests (they evaluate the schemas in lora_forge/schema)
npm run typecheck
```

### Translations

English is the source language; translations live in three places:

| Text | Source (English) | Translations |
|---|---|---|
| Buttons, pages, notices | `frontend/src/locales/en.ts` | `zh.ts`, `ja.ts` next to it (the type check fails on a missing key) |
| Training options and section titles | the schema files in `lora_forge/schema/` | `lora_forge/schema/locales/{zh,ja}.json`, keyed by the English text |
| Messages from the backend | `lora_forge/i18n.py` | the same file |

When you add or reword an option, add its English text to both JSON files; `npm test` fails while a translation is missing or left over. Terminal output and exported TOML files stay in English.

### Releases

GitHub Actions builds the releases: pushing a `v*` tag publishes a release with the ready-to-run zip (tags with a `-`, like `v1.0.0-rc1`, as pre-releases). An annotated tag's message becomes the release notes: `git tag -a v1.0.0 --cleanup=verbatim -F notes.md` (`--cleanup=verbatim` keeps Markdown headings).

### Project layout

| Path | What it is |
|---|---|
| `main.py` | Entry point (GUI and the `update-sd-scripts` / `update-musubi-tuner` commands) |
| `run_gui.ps1`, `run_gui.sh` | Launchers: check for uv, then run `main.py` through `uv run` |
| `lora_forge/` | Python backend: web server and API (`app/`), training config, validation and runs (`training/`), sd-scripts compatibility checks |
| `lora_forge/schema/` | Training form definitions, read by the web UI |
| `frontend/` | Vue 3 + TypeScript web UI |
| `docs/` | User guides (Anima, Krea 2) |
| `scripts/sd-scripts`, `scripts/musubi-tuner` | kohya-ss/sd-scripts and musubi-tuner checkouts, cloned on first start (not part of this repository; do not remove them with `git clean`) |

The backend package was named `mikazuki` in upstream lora-scripts. It was renamed to `lora_forge` in October 2026, together with its environment variables (`MIKAZUKI_*` → `LORA_FORGE_*`).

</details>

## License and credits

LoRA-Forge is a modified version of [lora-scripts](https://github.com/Akegarasu/lora-scripts) by Akegarasu and contributors. It was forked in May 2026 and has been substantially changed since (Vue web UI, Anima training, reworked backend and launchers). Like the original, it is licensed under the **GNU Affero General Public License v3.0**; see [LICENSE](LICENSE).

- If you run a modified version as a network service (for example with `--listen`), AGPL section 13 requires offering its source code to the users. The web UI links to this repository for that; point the link at your own source if you host a modified version.
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) and [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) (Apache-2.0) are downloaded into `scripts/` on first start and are not distributed with this repository.
- The built web UI bundles Vue and vue-router (MIT). Their license notices are in `frontend/dist/THIRD-PARTY-LICENSES.md`, which is included in the release zip.

Thanks to Akegarasu and the original contributors for building the foundation.
