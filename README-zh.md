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
  <a href="https://github.com/PineCookie/lora-forge/blob/main/README.md">文档</a>
  ·
  <a href="https://github.com/PineCookie/lora-forge/blob/main/README-zh.md">中文README</a>
</p>

LoRA & Dreambooth 训练图形界面与一键训练环境，用于 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts.git)

**WARNING: 本项目主要为个人使用，目的是延续 [lora-scripts](https://github.com/Akegarasu/lora-scripts) 的易用 UI 和工作流。当前可能存在大量 Bug，欢迎提出意见或 PR！**

## ✨ 新增：Anima LoRA 训练

- [x] 重构界面以支持不同模型的训练参数，包括 Anima（Vue 3 + TypeScript）
- [x] 使用 uv 安装和管理环境，不再依赖 requirements.txt 和旧安装脚本
- [ ] 支持不同显卡的 Torch 安装
- [ ] 加入缺失的原前端功能

# 使用方法

### 必要依赖

Python 3.12 和 Git。通过 git clone 运行时还需要 [Node.js](https://nodejs.org/) 20 或更高版本：启动器会在首次启动和更新后用它构建网页界面。

### 获取代码

可以从 [Releases](https://github.com/PineCookie/lora-forge/releases) 下载完整压缩包（网页界面已预先构建，无需 Node.js），或克隆仓库：

```sh
git clone https://github.com/PineCookie/lora-forge
```

## ✨ LoRA Forge GUI

用对应系统的启动脚本启动 GUI，然后在浏览器中打开 [http://127.0.0.1:28000](http://127.0.0.1:28000)（加上 `--open-browser` 可自动打开）。国内用户建议加上 `--cn`，使用清华 PyPI 镜像和 hf-mirror：

```powershell
.\run_gui.ps1 --cn     # Windows（或右键 → 使用 PowerShell 运行）
```

```sh
bash run_gui.sh --cn   # Linux
```

不需要单独的安装步骤：启动脚本会在缺少 [uv](https://docs.astral.sh/uv/) 时自动安装，`uv run` 会在首次启动时根据 `pyproject.toml` 把依赖安装到 `.venv`（需要下载 PyTorch，耗时较长），并在更新后自动同步。

更新 `scripts/sd-scripts` 请运行 `run_gui.ps1 update-sd-scripts`（Linux：`bash run_gui.sh update-sd-scripts`，`--branch` 可指定分支或 tag）。启动 GUI 时不会自动更新 sd-scripts。

### 可选：flash-attn

`flash-attn` 默认不会安装。如果你有兼容的本地 wheel，请将它放在 Git 忽略的目录中，例如 `wheelhouse/`，然后启用可选依赖：

```sh
uv sync -U --extra flash-attn --find-links ./wheelhouse --no-build-package flash-attn
```

`-U` 会让 uv 更新本地 lockfile 以使用该 wheel，`--no-build-package flash-attn` 可以避免 uv 回退到从源码构建 `flash-attn`。

### 网页界面开发

网页界面是位于 `frontend/` 的 Vue 3 + TypeScript 应用。需要热重载开发时，先用 `--skip-frontend-build` 启动 GUI，然后：

```sh
cd frontend
npm ci
npm run dev        # 启动界面开发服务器，并把 /api 转发到 28000 端口的 GUI
npm test           # 单元测试（会读取 lora_forge/schema 中的 schema）
npm run typecheck
```

发布由 GitHub Actions 完成：推送 `v*` 标签即会发布带完整压缩包的 Release。

### 界面语言

网页界面支持英文、中文和日文，默认跟随浏览器语言，也可以在侧边栏底部切换。源语言是英文，翻译分布在三处：

| 文本 | 源文本（英文） | 翻译 |
|---|---|---|
| 按钮、页面、提示 | `frontend/src/locales/en.ts` | 同目录下的 `zh.ts`、`ja.ts`（缺少键时类型检查会失败） |
| 训练参数说明与分组标题 | `lora_forge/schema/` 中的 schema 文件 | `lora_forge/schema/locales/{zh,ja}.json`，以英文原文为键 |
| 后端返回的消息 | `lora_forge/i18n.py` | 同一文件 |

新增或修改参数说明时，请把英文原文同时加入两个 JSON 文件；缺少或残留翻译时 `npm test` 会失败。终端输出和导出的 TOML 文件始终为英文。

### 运行记录

「运行记录」页面会跟踪每次训练：进度、速度与剩余时间、loss 与学习率曲线、样图、已保存的 LoRA 文件以及控制台输出。训练失败时会直接显示错误。记录保存在 `logs/runs` 中，关闭 GUI 后仍会保留，方便对比以往的训练；删除记录不会影响训练出的文件。此功能不需要 TensorBoard。

### TensorBoard

TensorBoard 为可选功能。启动 GUI 时加上 `--enable-tensorboard`（例如 `run_gui.ps1 --enable-tensorboard`），即可在 TensorBoard 页面中查看。它读取 `logs` 文件夹，监听 http://127.0.0.1:6006/。

## 程序参数

| 参数名称                     | 类型  | 默认值       | 描述                                            |
|------------------------------|-------|--------------|-------------------------------------------------|
| `--host`                     | str   | "127.0.0.1"  | 服务器的主机名                                  |
| `--port`                     | int   | 28000        | 运行服务器的端口                                |
| `--listen`                   | bool  | false        | 启用服务器的监听模式                            |
| `--skip-prepare-sd-scripts`  | bool  | false        | 跳过克隆 kohya-ss/sd-scripts                    |
| `--skip-frontend-build`      | bool  | false        | 启动时不（重新）构建网页界面                     |
| `--sd-scripts-branch`        | str   | "main"        | sd-scripts 不存在时克隆所用的分支或标签           |
| `--enable-tensorboard`       | bool  | false        | 启用 TensorBoard                                |
| `--tensorboard-host`         | str   | "127.0.0.1"  | 运行 TensorBoard 的主机                         |
| `--tensorboard-port`         | int   | 6006         | 运行 TensorBoard 的端口                          |
| `--dev`                      | bool  | false        | 开发者模式，用于禁用某些检查                     |
| `--open-browser`             | bool  | false        | 服务器启动后自动打开浏览器                      |

## 文档

- [Anima LoRA 训练说明](docs/anima.md)：哪些选项真正生效、每个功能需要的 sd-scripts 版本、加速/显存组合，以及常见的"静默失效"陷阱。
- 「关于」页会显示当前 sd-scripts 的**兼容性**状态。

## 项目结构

| 路径 | 说明 |
|---|---|
| `main.py` | 程序入口（GUI 与 `update-sd-scripts` 命令） |
| `run_gui.ps1`、`run_gui.sh` | 启动脚本：准备 uv 和依赖，然后运行 `main.py` |
| `lora_forge/` | Python 后端：网页服务与 API（`app/`）、训练配置/校验/运行（`training/`）、sd-scripts 兼容性检查 |
| `lora_forge/schema/` | 训练表单定义，由网页界面读取 |
| `frontend/` | Vue 3 + TypeScript 网页界面 |
| `scripts/sd-scripts` | kohya-ss/sd-scripts，首次启动时克隆（不属于本仓库） |

后端包在上游 lora-scripts 中名为 `mikazuki`，已于 2026 年 10 月改名为 `lora_forge`，环境变量也随之改为 `LORA_FORGE_*`（原 `MIKAZUKI_*`）。

## 许可证与鸣谢

LoRA-Forge 是 Akegarasu 及贡献者的 [lora-scripts](https://github.com/Akegarasu/lora-scripts) 的修改版本，于 2026 年 5 月分叉，此后有大幅修改（Vue 网页界面、Anima 训练、重写的后端与启动脚本）。与原项目相同，采用 **GNU Affero General Public License v3.0** 许可，见 [LICENSE](LICENSE)。

- 如果把修改后的版本作为网络服务运行（例如使用 `--listen`），AGPL 第 13 条要求向使用者提供源代码。网页界面为此链接到本仓库；托管修改版时请把链接改为你自己的源代码。
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)（Apache-2.0）在首次启动时下载到 `scripts/sd-scripts`，不随本仓库分发。
- 构建后的网页界面包含 Vue 和 vue-router（MIT），其许可声明位于 `frontend/dist/THIRD-PARTY-LICENSES.md`，并包含在发布压缩包中。

感谢 Akegarasu 和原项目贡献者打下的基础。
