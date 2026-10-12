# Anima LoRA 训练说明（lora-scripts / LoRA-Forge）

这份文档记录 **本 GUI 与 sd-scripts 的 Anima 训练对接**中容易踩的点：哪些选项真正生效、依赖什么版本、什么情况下会被静默忽略。
底层参数语义以 sd-scripts 上游的 [`docs/anima_train_network.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md) 与 [`docs/anima_torch_compile.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_torch_compile.md) 为准。

## 1. sd-scripts 版本要求

GUI 把选项写成配置键交给 sd-scripts，而 sd-scripts **对未知配置键不报错、直接忽略**。所以版本太旧时选项会"看起来生效、实际没生效"。

| 功能 | 最低版本 | 说明 |
|---|---|---|
| Anima LoRA 训练本体 | 含 `anima_train_network.py` 的版本 | |
| 逐 block `torch.compile`、`--cuda_allow_tf32`、`--cuda_cudnn_benchmark` | v0.11.1 | 需要 Triton |
| `--qwen_image_vae_2d`（2D VAE） | v0.11.1 | |
| `--show_timesteps` / `--show_timesteps_resolution`（时间步预览） | v0.11.1 | |
| 子文件夹时间步偏移、`--show_timesteps_offset` | main（> v0.11.1） | 不在任何 tag 里 |

- 当前 GUI 已对照验证的版本：**v0.12.0**（`690ea7f`）。v0.12.0 起上游把 `transformers` 升到 5.17、`diffusers` 0.40、`huggingface-hub` 1.32，本仓库的 `pyproject.toml` 已同步；升级 sd-scripts 时请一并 `uv sync`，不要只更新其中一边。
- 更新 sd-scripts：`.\run_gui.ps1 update-sd-scripts`（默认跟 `main`；`--branch` 可指定其他分支或 tag）。启动 GUI 不会自动更新 sd-scripts。
- GUI 的「关于」页会显示 **sd-scripts 兼容性**：`兼容（已测试版本）` / `兼容` / `过旧：缺少 …`。
- 提交训练时，如果所选功能超出已安装版本，GUI 会**直接报错**而不是静默跑一个无效配置。
- 已知的上游陷阱：`sd3` 分支没有上述任何新功能，且它的 `anima_train_network.py` 不处理 `--show_timesteps`——在该分支上点"预览时间步分布"会**真的开始一轮训练**。

## 2. 必备文件

| 参数 | 内容 | 备注 |
|---|---|---|
| `pretrained_model_name_or_path` | Anima DiT（`.safetensors`） | ComfyUI 命名（`net.` 前缀）也可 |
| `qwen3` | Qwen3-0.6B 文本编码器 | 可填 safetensors 文件、目录或 HF repo id；**必填** |
| `vae` | Qwen-Image VAE | **必填** |
| `llm_adapter_path` | 单独的 LLM Adapter | 不填时从 DiT 文件里读 |
| `t5_tokenizer_path` | T5 tokenizer 目录 | 不填用 sd-scripts 内置 `configs/t5_old/` |

GUI 会在提交时校验 `qwen3` / `vae`（缺失或路径不存在会直接提示，而不是丢进训练日志里报 traceback）。

## 3. 哪些选项真正生效

| 选项 | 是否生效 | 说明 |
|---|---|---|
| `attn_mode` | ✅ | `torch`（= PyTorch SDPA）/ `xformers` / `flash`（需 flash-attn） |
| `split_attn` | ✅（仅省显存） | Anima 全程不带 attention mask，xformers **不强制**需要它；与 `compile_fullgraph` 互斥 |
| `xformers` / `sdpa` 复选框 | ❌ | 会被 `--attn_mode` 覆盖，GUI 已隐藏这两个开关 |
| `discrete_flow_shift` | ⚠️ | **只有** `timestep_sampling=sigma/shift` 会用到；`sigmoid`/`uniform`/`flux_shift` 下被忽略 |
| `timestep_sampling` | ✅ | `sigma` / `uniform` / `sigmoid`(默认) / `shift` / `flux_shift`；`logit_normal`、`mode` 等权重方案只在 `sigma` 下走密度采样 |
| `weighting_scheme` | ⚠️ | Anima 只实现 `sigma_sqrt` / `cosmap`，其余按等权处理 |
| `weighted_captions` | ❌ | Anima/FLUX/SD3 的文本策略没有实现加权分词：开着缓存会被静默忽略，不开缓存会 `NotImplementedError`；GUI 已隐藏并对提交做拦截 |
| `vae_disable_cache` | ⚠️ | `qwen_image_vae_2d` 下**无效**（2D VAE 没有时间维缓存） |
| `vae_chunk_size` | ⚠️ | 2D VAE 下对峰值显存影响很小（峰值由全分辨率激活与 mid-block attention 主导） |
| `llm_adapter_lr` | ⚠️ | 主要给全量微调用；LoRA 场景填 0（冻结） |
| `show_timesteps*` | ⚠️ | 只影响"预览时间步分布"按钮；正常训练会忽略这些字段 |

## 4. 加速与显存

推荐顺序（省显存优先 → 加速优先）：

1. **latent / 文本编码器缓存**：`cache_latents`、`cache_text_encoder_outputs`（可 `_to_disk`）。默认开启。
   - 启用文本编码器缓存后**无法**同时训练文本编码器，因此 `network_train_unet_only` 会被自动打开。
   - 改了 `caption_dropout_rate`（或其它 caption 设置）**必须删除并重建缓存**。
2. **2D VAE**（`qwen_image_vae_2d`）：官方 VAE 权重在加载时把 Conv3d 转成等价的 Conv2d，单图 latent 与 3D VAE 数值一致；约 2 倍速、峰值显存约 1/3。**latent 缓存场景强烈推荐**（缓存是同一份，切换 2D/3D 不需要重算）。
3. **`torch.compile`**（`compile`）：逐 block 编译，首次迭代慢、之后明显加速；多分辨率数据集会按桶重新编译，建议 `compile_cache_size_limit=32`。与旧版 `--torch_compile` 互斥；`compile_fullgraph` 与 `split_attn` 互斥。Windows 上 `compile_dynamic=true` 需要 VS2022 C++ 编译器。⚠ **`compile_dynamic` 默认 `false`**，原因见第 9 节。
4. **`cuda_allow_tf32` / `cuda_cudnn_benchmark`**：与 `compile` 无关，可单独开启（Ampere+）。
5. **省显存**：`blocks_to_swap`（交换到 CPU 的 block 数）、`gradient_checkpointing`、`unsloth_offload_checkpointing`、`cpu_offload_checkpointing`。
   - `blocks_to_swap` 与两种 offload **互斥**；两种 offload 之间也互斥。GUI 在勾选时会自动清掉冲突项。
6. **混合精度**：`bf16` + `full_bf16` 是推荐组合；`full_fp16` 要求 `mixed_precision=fp16`，`full_bf16` 要求 `bf16`（GUI 会自动同步）。

## 5. 子文件夹时间步偏移（timestep offset）

用途：让不同内容粒度的子集在不同噪声区间上训练——**负值偏向低噪声（细节）**、**正值偏向高噪声（结构）**。

- 生效范围：`timestep_sampling` = `sigmoid` / `shift` / `flux_shift`；`sigma` / `uniform` 下无效。
- 只影响**训练**，不影响验证（验证用无偏采样，便于对比 loss）。
- 建议范围 `-0.5 ~ 0.5`；`shift`/`flux_shift` 本身已偏向高噪声，正偏移要更保守（从 ±0.25 起步）。
- 用法：在「数据集设置」的统计面板里给每个 `重复次数_名称` 子文件夹填偏移，非 0 才会写入；提交时会生成一份带 `custom_attributes.timestep_sampling.offset` 的 dataset TOML。
- 预览：先用「预览时间步分布」+ `show_timesteps_offset` 看分布，再正式训练。
- 与缓存无关：偏移只改变采样，不需要重建 latent / 文本编码器缓存。
- 只支持一层子文件夹（`数字_名称`），命名不符的目录会被跳过并在日志里给出 warning。

## 6. 常见坑与排查

| 现象 | 原因 / 处理 |
|---|---|
| 训练在数据集构建阶段报 `extra keys not allowed` | 生成的 dataset 配置里出现了 sd-scripts 数据集 schema 不认识的键（历史 bug：`weighted_captions`）；升级到含该修复的版本 |
| 勾了加速项但速度没变 | sd-scripts 版本过旧、选项被静默忽略；看「关于」页的兼容性状态 |
| 改分辨率/子文件夹后 `下列时间步偏移对应的子文件夹不存在` | 旧偏移值残留；GUI 会自动清理，若仍出现请刷新页面重填 |
| loss 变 NaN | 确认 PyTorch 版本足够新（Anima 需要较新的 2.x） |
| `full_fp16 requires mixed precision='fp16'` / `blocks_to_swap is not supported with ...` | 非法的选项组合；GUI 已做联动，用自定义 TOML/预设导入时需自行避免 |
| `compile_fullgraph cannot be used with --split_attn` | 两者互斥 |
| 训练第一步报 `AssertionError: lift_tracked_freevar_to_input should not be called on root SubgraphTracer` | torch 2.14（含 2.14.1）的 Dynamo bug（PyTorch #193194，动态形状 + 非重入 checkpoint）；保持 `compile_dynamic=false`（GUI 默认），或等 torch 2.15（修复 PR #193254）。见第 9 节 |
| 训练时刷屏 `remark: <file>:<line>:0: N instructions in function` | triton-windows `3.8.0.post28` 的 Windows 回归（[triton-windows #54](https://github.com/triton-lang/triton-windows/issues/54)），**与本节的 Dynamo bug 无关**；升级到 `triton-windows>=3.8.0.post29` 即默认隐藏。见第 10 节 |
| 预览"没输出" | 预览结果输出在训练日志（控制台）里；GUI 只使用 `console`，`image` 需要 matplotlib，且会在服务器上弹出窗口 |

## 7. 元数据与出图

- LoRA 文件里会写入 `ss_timestep_sampling`、`ss_sigmoid_scale`、`ss_discrete_flow_shift`、`ss_weighting_scheme` 等，方便复现训练设置。
- 默认只训练 DiT（`network_train_unet_only`），这样的 LoRA 可以直接在 ComfyUI 中使用。只有同时训练了 Qwen3 文本编码器时，ComfyUI 才可能无法加载文本编码器部分；这时可用 sd-scripts 自带的 `networks/convert_anima_lora_to_comfy.py` 转换（在 `scripts/sd-scripts` 目录运行 `uv run python -m networks.convert_anima_lora_to_comfy 源 目标`）。
- 预览图 CFG 建议 4~5、步数 30~50。

## 8. 给维护者的备注

- GUI 的 schema 与预设**每次请求都从磁盘重读**，改完刷新页面即可，不需要重启后端。
- 上游 sd-scripts 跟 `main`，未固定提交；新增/改名的上游参数会让 GUI 的选项静默失效。`lora_forge/sd_scripts.py` 记录了每个功能对应的上游提交（以及无 git 时的文件特征回退），`/api/runtime` 会给出兼容性结论。
- 建议在升级 sd-scripts 后跑一次「schema 字段 ↔ `--help`」的对照检查（见 `lora_forge/sd_scripts.py` 的能力表）。

## 9. torch.compile 动态形状已知问题（PyTorch #193194）

- **现象**：`compile=true` 且 `compile_dynamic` 为 `true`/`auto` 时，训练在第一步就崩，报
  `AssertionError: lift_tracked_freevar_to_input should not be called on root SubgraphTracer`。
- **原因**：Dynamo 在追踪「非重入 checkpoint 的 HOP 子图」且启用动态形状（或 automatic-dynamic 在第二个形状重新编译）时，跨子图复用的代理无法提升为根图输入。Anima 的「同一 block 一次带 cross-attn context、一次不带」的双次遍历结构正好命中。属 **PyTorch 侧 bug，与 sd-scripts 无关**。
  - issue：https://github.com/pytorch/pytorch/issues/193194
  - 修复 PR（2026-08-22 合并到 main）：https://github.com/pytorch/pytorch/pull/193254
  - **torch 2.14.0 与 2.14.1 均不含该修复**（2.14.1 是 MPS/CUDA 二进制补丁，未 cherry-pick；已核对两版的 `builder.py`/`output_graph.py` 仍为旧代码）。修复预计进入 2.15。
- **处理**：
  - **当前（默认）**：保持 `compile_dynamic=false`（GUI 已默认）。代价是多分辨率数据集会按 bucket 各自编译一次；把 `compile_cache_size_limit` 提到 `32` 可减少缓存抖动。这是 issue 给出的 `automatic_dynamic_shapes=False` 等价 workaround。
  - **恢复动态形状**：等 torch 2.15（含 PR #193254）后改回 `auto`/`true`。
  - **应急**：设环境变量 `TORCHDYNAMO_AUTOMATIC_DYNAMIC_SHAPES=0`，等价于 `dynamic=false`。
- **版本背景**：该 bug 在 torch 2.9.1 / 2.13.0 / 2.14.x 上均可复现，并非 2.14 独有；是否触发还取决于模型结构（双次遍历 + 非重入 checkpoint + 动态形状）。
- **易混淆**：若在 `compile_dynamic=false`（不触发本节 bug）下仍看到大量 `remark: ... N instructions in function`，那是 triton-windows 的独立回归（见第 10 节），**不需要**动 `compile_dynamic`。

## 10. 运行时依赖（torch / Triton）

Anima 的 `compile` 依赖 Triton，torch / triton-windows 的搭配比 sd-scripts 版本更敏感。要点：

- **torch**：`pyproject.toml` 约束 `torch>=2.9,<2.15`；2.14.0 / 2.14.1 仍不含 #193194 修复（见第 9 节）。
- **triton-windows（Windows）**：约束 `<3.9`，即 3.8.x；`3.8.0.post28` 有 `remark: ... N instructions in function` 刷屏的回归（[triton-windows #54](https://github.com/triton-lang/triton-windows/issues/54)），请使用 `>=3.8.0.post29`。官方说明 Triton 3.8 只保证配 torch 2.14 / 2.15。
- **attn 后端**：xformers、flash-attn 需安装与 torch 版本匹配的 wheel（flash-attn 尤其按 torch 版本分别发布）。
- **缓存**：升级 torch 或 Triton 会让 Inductor / Triton 的磁盘缓存（默认 `~/.cache/torch/inductor`、`~/.triton/cache`，可用 `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` 改变位置）失效，下一次训练会重新编译一轮，之后恢复命中；这些缓存不会自动清理。
- **升级到 torch 2.15 时**：需重建 flash-attn（并确认 xformers 有对应版本），此后可把 `compile_dynamic` 改回 `auto`/`true`。
