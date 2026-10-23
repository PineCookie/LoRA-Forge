<div align="center">

<img src="assets/logo.svg" width="200" height="200" alt="LoRA Forge" style="border-radius: 25px">

# LoRA Forge

_✨ 享受 LoRA 训练！ ✨_

</div>

<p align="center">
  <a href="https://github.com/PineCookie/lora-forge" style="margin: 2px;">
    <img alt="GitHub 仓库星标" src="https://img.shields.io/github/stars/PineCookie/lora-forge">
  </a>
  <a href="https://github.com/PineCookie/lora-forge" style="margin: 2px;">
    <img alt="GitHub 仓库分支" src="https://img.shields.io/github/forks/PineCookie/lora-forge">
  </a>
  <a href="https://raw.githubusercontent.com/PineCookie/lora-forge/main/LICENSE" style="margin: 2px;">
    <img src="https://img.shields.io/github/license/PineCookie/lora-forge" alt="许可证">
  </a>
  <a href="https://github.com/PineCookie/lora-forge/releases" style="margin: 2px;">
    <img src="https://img.shields.io/github/v/release/PineCookie/lora-forge?color=blueviolet&include_prereleases" alt="发布版本">
  </a>
</p>

<p align="center">
  <a href="https://github.com/PineCookie/lora-forge/releases">下载</a>
  ·
  <a href="docs/anima-zh.md">Anima 训练说明</a>
  ·
  <a href="docs/krea2-zh.md">Krea 2 训练说明</a>
  ·
  <a href="docs/moss-image-zh.md">MOSS-Image 训练说明</a>
  ·
  <a href="README.md">English</a>
</p>

LoRA Forge 是一个训练 LoRA 的网页界面。选好模型，在浏览器里填好表单、点击开始，就能在实时页面上跟踪训练；Python、PyTorch 和训练脚本都由 LoRA Forge 自动准备。底层使用 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)、[kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)（Krea 2）和 LoRA Forge 自带的 MOSS-Image 训练器，并在它们较慢的地方做了加速。本项目延续自 [lora-scripts](https://github.com/Akegarasu/lora-scripts)。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/runs-dark.png">
  <img src="docs/images/runs-light.png" alt="运行记录页面：一次已完成的 Anima LoRA 训练的进度、loss 与学习率曲线">
</picture>

- **模型**：Anima、Krea 2、MOSS-Image、SDXL（Illustrious、NoobAI、Pony 等）、FLUX、SD3.5 和 SD1.5，另有 SDXL 与 SD1.5（Dreambooth）的全量微调。每个模型都有自己的页面和合适的默认值。
- **简洁的表单**：先显示大多数人会改的选项，其余选项一键展开；子选项在适用时才出现；实时提示（步数、预热、文件检查），开始前就显示问题；支持搜索，配置可导入/导出为 JSON 或 TOML。Anima、Krea 2 和 MOSS-Image 页面会列出所需的文件及下载链接，并检查你选择的文件。
- **运行记录页面**：进度与剩余时间、loss 与学习率曲线、样图、已保存的 LoRA 文件和控制台输出，以往的训练也能查看。不需要 TensorBoard。
- **更快的训练**：SDXL LoRA 的训练速度约为原版 sd-scripts 的两倍，Anima、Krea 2 和 MOSS-Image 也更快。大部分加速默认开启（[详情](#训练速度)）。
- **16 GB 显卡也能训练 Krea 2**：默认使用 int8 底模权重和 block 交换（[详情](docs/krea2-zh.md)）。
- **MOSS-Image**：OpenMOSS 的 5B 模型，文字渲染出色，用 LoRA Forge 自带的训练器在 16 GB 显卡上训练（[详情](docs/moss-image-zh.md)）。
- 界面支持 English、中文、日本語，以及浅色和深色模式。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/home-dark.png">
  <img src="docs/images/home-light.png" alt="首页：每个模型一个页面，以及最近的训练和系统信息">
</picture>

> [!NOTE]
> LoRA Forge 主要为个人使用而维护，可能还有不少粗糙之处。欢迎反馈问题、提出建议或提交 PR。

## 快速开始

### 准备

- Windows 或 Linux，**RTX 20 系列或更新的 NVIDIA 显卡**（计算能力 7.5），**驱动 580 或更新**（LoRA Forge 使用 CUDA 13.0 版的 PyTorch 2.14）。CUDA 13 不支持 GTX 10 系列及更早的显卡，暂不支持其他品牌的显卡。推荐 RTX 30 系列或更新的显卡：bf16 训练和 flash-attn 都需要它，RTX 20 系列请改用 fp16。首页的 **系统信息** 面板会显示你的显卡是否受支持。
- **显存**：默认设置下，批大小 2、1024×1024 时 Anima 和 SDXL LoRA 约需 9 GB。MOSS-Image 和 Krea 2 按 16 GB 显卡设置；Krea 2 调高 `blocks_to_swap` 后 12 GB 也能训练。
- [Git](https://git-scm.com/downloads)，启动脚本用它下载 sd-scripts 和 musubi-tuner。
- [uv](https://docs.astral.sh/uv/)，由它安装 Python 和其余所有依赖。请用官方安装脚本安装，然后**打开新的终端**：

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
  ```

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux
  ```

### 安装与启动

1. 从 [Releases](https://github.com/PineCookie/lora-forge/releases) 下载最新的 `lora-forge-<版本>.zip`，解压到空间充足的文件夹（运行环境和模型需要数 GB）。
2. 启动：
   - **Windows**：右键 `run_gui.ps1` → *使用 PowerShell 运行*，或在终端中运行 `.\run_gui.ps1`。
   - **Linux**：`bash run_gui.sh`。

   国内用户请加上 `--cn`（`.\run_gui.ps1 --cn`），通过清华 PyPI 镜像和 hf-mirror 下载。
3. 打开 [http://127.0.0.1:28000](http://127.0.0.1:28000)（或加上 `--open-browser` 自动打开）。

首次启动会下载 Python、PyTorch、sd-scripts 和 musubi-tuner，耗时较长；之后启动会很快。不需要单独的安装步骤。

## 训练第一个 LoRA

**1. 准备数据集。** 在一个数据集文件夹中建立名为 `<重复次数>_<名称>` 的子文件夹，把图片放进去。数字表示每个 epoch 中每张图片使用的次数。每张图片可以配一个同名的 `.txt` 标注文件：

```
train/my_character/          ← 在 GUI 中选择这个文件夹
├── 10_my_character/
│   ├── 001.png
│   ├── 001.txt              ← "my_character, 1girl, smile, ..."
│   └── ...
└── 5_my_character_outfit/
    └── ...
```

直接放在数据集文件夹里的图片不会被使用；找不到 `<重复次数>_<名称>` 子文件夹时，GUI 会给出提示。上面这种标签式标注适合 Anima 和 SDXL；Krea 2 和 MOSS-Image 更适合用句子描述，MOSS-Image 的标注文件还可以每行写一条标注（[详情](docs/moss-image-zh.md#标注)）。

**2. 准备底模。** 下载要在其上训练的模型。Anima、Krea 2 和 MOSS-Image 还需要文本编码器和 VAE：它们的页面上有 **模型文件** 面板，给出每个文件的下载链接。SDXL、FLUX、SD3.5 和 SD1.5 可以使用该模型的任意 checkpoint。

**3. 填写表单。** 在侧边栏打开对应模型的页面。至少填写底模文件、数据集文件夹和输出名称，其他值都有合理的默认值，每个分组的「+ 另外 N 个参数」可以展开其余选项。右侧面板显示当前设置下的训练步数和需要处理的问题；必需的文件填好后才能 **开始训练**。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/anima-dark.png">
  <img src="docs/images/anima-light.png" alt="Anima 页面：已检查的模型文件、显示步数与时间的训练概览，以及带下载链接的模型文件面板">
</picture>

**4. 开始并查看。** 点击 **开始训练**。页头的状态会显示进度，点击它即可在 **运行记录** 页面打开这次训练，查看 loss、样图和控制台输出。LoRA 文件保存在输出文件夹（默认 `output/`），并列在这次运行中，可直接在 ComfyUI 等界面中加载。

在浏览器标签页打开期间，每个页面都会保留已填写的内容，切换到「运行记录」或其他模型再回来也不会丢失，但刷新页面后会恢复默认。要保存配置，请用 **导出 JSON** 或 **导出 TOML**，再用 **导入配置…** 重新载入。每次开始训练时，配置也会保存到 `config/autosave/`，同样可以导入。

## 训练速度

LoRA Forge 不修改 sd-scripts 和 musubi-tuner 的代码，而是在训练开始时为它们加速。每项改动都与原版对比过，loss 一致。下表数字为每步耗时，测试环境为 Windows 下的 RTX 5070 Ti（16 GB），rank 32 的 LoRA，批大小 2，约 100 万像素；其他显卡的提升会不同。

**默认开启**，无需设置：

| 内容 | 模型 | 每步耗时 |
|---|---|---|
| 编译模型的各个 block（`torch.compile`），所有图像尺寸共用编译结果，换尺寸不会重新编译 | Anima、SDXL、MOSS-Image | SDXL −21%，MOSS-Image −29%；Anima 一个 epoch −11%（首次 −55%） |
| 安装 [flash-attn](#flash-attn可选) 后，注意力使用 flash-attn | SDXL、Krea 2、MOSS-Image | SDXL −17%，Krea 2 −22%，MOSS-Image −29% |
| 使用 PyTorch 的 SDPA 注意力，而不是 sd-scripts 的普通注意力 | SDXL、SD1.5 | 约减半，显存也更少 |
| 更快的 ProdigyPlusScheduleFree 优化器步骤（结果相同） | 全部 | Anima −7%，SDXL −23% |
| 几乎不影响速度的 block 交换，可以把更多模型放在内存里 | Krea 2 | 交换更多 block 而速度不变 |
| 缓存 latent 时不启动 DataLoader 工作进程，训练更快开始 | 全部 | Windows 上第一步约提前一分钟 |

**可选**，在模型页面中设置：

| 设置 | 模型 | 每步耗时 | 代价 |
|---|---|---|---|
| `training_precision = fp16`（配合默认开启的 `fp16_accumulation`） | Anima、MOSS-Image、SDXL、SD1.5 | Anima −24%，MOSS-Image −27%，SDXL −8% | 测试中精度与 bf16 相当。只在 GeForce 显卡上有提升，RTX PRO 和数据中心卡没有 |
| `activation_memory_budget = auto` | Anima | −8% 到 −11% | 会用掉大部分显存 |

合计：SDXL LoRA 每步从 1.51 秒（启用 SDPA 的 sd-scripts）在默认设置下降到 0.81 秒，使用 fp16 时为 0.72 秒；Anima LoRA 每步在使用 fp16 和 `activation_memory_budget = auto` 时从 1.73 秒降到 1.17 秒。各选项的说明见 [Anima](docs/anima-zh.md)、[Krea 2](docs/krea2-zh.md) 和 [MOSS-Image](docs/moss-image-zh.md) 指南。如果想原样运行 sd-scripts（例如做对比），请设置环境变量 `LORA_FORGE_PATCHES=0`。

## 更新

- **LoRA Forge**：下载新版本解压覆盖旧文件夹（或解压到新文件夹，再把 `train/`、`output/` 和 `config/` 复制过去）；如果是克隆的仓库，执行 `git pull`。
- **sd-scripts** 从不自动更新，可用的环境会保持原样。全新安装会获取 LoRA Forge 测试过的版本。要更新到最新的 sd-scripts：

  ```powershell
  .\run_gui.ps1 update-sd-scripts           # Windows
  ```

  ```sh
  bash run_gui.sh update-sd-scripts         # Linux
  ```

  `--branch <分支或标签>` 可选择其他分支或标签（`update-sd-scripts --branch v0.12.2`）。如果你修改过 `scripts/sd-scripts` 中的文件，更新会停止而不会覆盖。首页的 **系统信息** 面板会显示已安装的 sd-scripts 是否支持 GUI 的所有选项。
- **musubi-tuner**（Krea 2）同理：`update-musubi-tuner [--branch <分支或标签>]` 更新 `scripts/musubi-tuner`。

## 可选功能

### flash-attn（可选）

[flash-attn](https://github.com/Dao-AILab/flash-attention) 是更快的注意力实现（见 [训练速度](#训练速度)）。它默认不会安装。安装后，Anima、SDXL、Krea 2 和 MOSS-Image 页面会自动使用它。要安装它，请在 LoRA Forge 文件夹中运行一次（适用于 Windows，以及 x86_64 或 ARM 上的 Linux）：

```sh
uv sync --extra flash-attn
```

这会从 [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) 下载与 LoRA Forge 的 PyTorch 匹配的预编译 wheel（约 230 MB），无需编译。之后启动脚本会保留它，并在 PyTorch 更新时一起更新。如果之后你自己运行 `uv sync`，请加上 `--extra flash-attn`，否则 `uv sync` 会把它移除。

### TensorBoard（可选）

运行记录页面已能满足大多数需求。如果还想用 TensorBoard，启动时加上 `--enable-tensorboard`，然后打开 TensorBoard 页面；它读取 `logs` 文件夹，监听 http://127.0.0.1:6006/。

### 程序参数

<details>
<summary>全部启动参数</summary>

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--cn` | 关 | 使用国内镜像（仅启动脚本） |
| `--host` | `127.0.0.1` | 服务器监听的地址 |
| `--port` | `28000` | 网页界面的端口（被占用时会改用 30000 起的空闲端口） |
| `--listen` | 关 | 监听所有网络接口，让其他设备也能打开界面 |
| `--open-browser` | 关 | 启动后自动打开浏览器 |
| `--enable-tensorboard` | 关 | 同时启动 TensorBoard |
| `--tensorboard-host` | `127.0.0.1` | TensorBoard 监听的地址 |
| `--tensorboard-port` | `6006` | TensorBoard 的端口 |
| `--sd-scripts-branch` | `v0.12.2` | sd-scripts 不存在时克隆的分支或标签（已测试的版本） |
| `--skip-prepare-sd-scripts` | 关 | 不克隆 sd-scripts |
| `--musubi-tuner-branch` | `v0.3.6` | musubi-tuner 不存在时克隆的分支或标签（已测试的版本） |
| `--skip-prepare-musubi-tuner` | 关 | 不克隆 musubi-tuner |
| `--skip-frontend-build` | 关 | 启动时不（重新）构建网页界面 |
| `--dev` | 关 | 开发者模式 |

</details>

## 常见问题

| 问题 | 处理 |
|---|---|
| `uv was not found` | 按 [准备](#准备) 中的方法安装 uv，然后打开**新的**终端 |
| Windows：*在此系统上禁止运行脚本* | 右键 → *使用 PowerShell 运行*，或运行 `powershell -ExecutionPolicy Bypass -File .\run_gui.ps1` |
| 首次启动非常慢 | 它在下载 PyTorch（数 GB）；国内请使用 `--cn` |
| 训练失败 | 在运行记录页面打开它：控制台输出中有错误信息 |
| 每步要几分钟而不是几秒 | 训练放不进显存，Windows 会借用系统内存而不报错。请降低批大小或分辨率，或保持 `gradient_checkpointing` 开启 |
| **模型文件** 中某一行显示 *模型类型不对* | 该文件不是这个页面训练的模型（例如在 Anima 页面选了 SDXL checkpoint）。同一模型的微调版和融合模型没有问题 |
| 某个选项似乎不起作用 | 查看首页的 **系统信息** 面板：已安装的 sd-scripts 过旧、不支持某些选项时，这里会给出提示 |
| Anima：编译报错或最初几步很慢 | 见 [Anima 训练说明](docs/anima-zh.md) |
| SDXL：编译报错 | 在 SDXL LoRA 页面关闭 `compile`，训练会以未编译的方式运行 |
| Krea 2：每步要几分钟 | 模型放不进显存；请调高 `blocks_to_swap`（见 [Krea 2 训练说明](docs/krea2-zh.md)） |

## 开发者

<details>
<summary>网页界面开发、翻译、发布与项目结构</summary>

### 网页界面开发

网页界面是位于 `frontend/` 的 Vue 3 + TypeScript 应用。从 git 克隆运行时，启动脚本会在首次启动和更新后用 [Node.js](https://nodejs.org/) 20 或更高版本构建它。需要热重载开发时，先用 `--skip-frontend-build` 启动 GUI，然后：

```sh
cd frontend
npm ci
npm run dev        # 启动界面开发服务器，并把 /api 转发到 28000 端口的 GUI
npm test           # 单元测试（会读取 lora_forge/schema 中的 schema）
npm run typecheck
```

### 翻译

源语言是英文，翻译分布在三处：

| 文本 | 源文本（英文） | 翻译 |
|---|---|---|
| 按钮、页面、提示 | `frontend/src/locales/en.ts` | 同目录下的 `zh.ts`、`ja.ts`（缺少键时类型检查会失败） |
| 训练参数说明与分组标题 | `lora_forge/schema/` 中的 schema 文件 | `lora_forge/schema/locales/{zh,ja}.json`，以英文原文为键 |
| 后端返回的消息 | `lora_forge/i18n.py` | 同一文件 |

新增或修改参数说明时，请把英文原文同时加入两个 JSON 文件；缺少或残留翻译时 `npm test` 会失败。终端输出和导出的 TOML 文件始终为英文。

### 发布

发布由 GitHub Actions 完成：推送 `v*` 标签即会发布带完整压缩包的 Release（带 `-` 的标签，如 `v1.0.0-rc1`，作为预发布）。附注标签的说明会成为发布说明：`git tag -a v1.0.0 --cleanup=verbatim -F notes.md`（`--cleanup=verbatim` 会保留 Markdown 标题）。

### 项目结构

| 路径 | 说明 |
|---|---|
| `main.py` | 程序入口（GUI 与 `update-sd-scripts` / `update-musubi-tuner` 命令） |
| `run_gui.ps1`、`run_gui.sh` | 启动脚本：检查 uv，然后通过 `uv run` 运行 `main.py` |
| `lora_forge/` | Python 后端：网页服务与 API（`app/`）、训练配置/校验/运行（`training/`）、sd-scripts 兼容性检查 |
| `lora_forge/schema/` | 训练表单定义，由网页界面读取 |
| `frontend/` | Vue 3 + TypeScript 网页界面 |
| `docs/` | 使用说明（Anima、Krea 2、MOSS-Image）以及 MOSS-Image 训练器的原理 |
| `scripts/sd-scripts`、`scripts/musubi-tuner` | kohya-ss/sd-scripts 与 musubi-tuner，首次启动时克隆（不属于本仓库；请勿用 `git clean` 删除） |

后端包在上游 lora-scripts 中名为 `mikazuki`，已于 2026 年 10 月改名为 `lora_forge`，环境变量也随之改为 `LORA_FORGE_*`（原 `MIKAZUKI_*`）。

</details>

## 许可证与鸣谢

LoRA-Forge 是 Akegarasu 及贡献者的 [lora-scripts](https://github.com/Akegarasu/lora-scripts) 的修改版本，于 2026 年 5 月分叉，此后有大幅修改（Vue 网页界面、Anima 训练、重写的后端与启动脚本）。与原项目相同，采用 **GNU Affero General Public License v3.0** 许可，见 [LICENSE](LICENSE)。

- 如果把修改后的版本作为网络服务运行（例如使用 `--listen`），AGPL 第 13 条要求向使用者提供源代码。网页界面为此链接到本仓库；托管修改版时请把链接改为你自己的源代码。
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 与 [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)（Apache-2.0）在首次启动时下载到 `scripts/`，不随本仓库分发。
- MOSS-Image 训练器的模型代码移植自 OpenMOSS 的 [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) 推理流程（Apache-2.0）；`lora_forge/trainers/moss_image/configs` 中的模型配置来自 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)。
- 构建后的网页界面包含 Vue 和 vue-router（MIT），其许可声明位于 `frontend/dist/THIRD-PARTY-LICENSES.md`，并包含在发布压缩包中。

感谢 Akegarasu 和原项目贡献者打下的基础。
