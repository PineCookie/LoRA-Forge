# Anima LoRA 训练说明

[English](anima.md)

用 LoRA-Forge 训练 Anima LoRA 时需要知道的事：哪些选项真正生效、各自依赖什么，以及常见的坑。
选项本身的含义请参阅 sd-scripts 的
[`anima_train_network.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md)
与 [`anima_torch_compile.md`](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_torch_compile.md)。

## sd-scripts 版本

sd-scripts 会忽略它不认识的配置键，所以在旧版本上，某个选项可能看起来设好了、实际却不起作用。GUI 会替你检查：
首页的 **系统信息** 面板会显示已安装版本的兼容性；如果某次训练需要更新的版本，提交时会直接报错说明。

| 功能 | 需要 |
|---|---|
| 逐 block `torch.compile`、`cuda_allow_tf32`、`cuda_cudnn_benchmark`、`qwen_image_vae_2d`、时间步预览 | v0.11.1 |
| 子文件夹时间步偏移、`show_timesteps_offset` | main（v0.11.1 之后） |

已在 v0.12.0 上测试。更新方法：`.\run_gui.ps1 update-sd-scripts`（默认跟随 `main`，`--branch`
可指定其他分支或 tag），之后执行一次 `uv sync`。启动 GUI 不会自动更新 sd-scripts。

## 模型文件

三个文件都在 [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) 的 `split_files/` 下。页面上的 **模型文件** 面板给出每个文件的下载链接，并检查你填写的路径：文件是否存在、是否是正确的文件（例如不是 Qwen-Image 2.1 VAE）。

| 选项 | 文件 | 备注 |
|---|---|---|
| `pretrained_model_name_or_path` | `diffusion_models/anima-base-v1.0.safetensors`（4.2 GB） | 请在 Anima-Base 上训练；ComfyUI 的文件（`net.` 前缀）也可以 |
| `qwen3` | `text_encoders/qwen_3_06b_base.safetensors`（1.2 GB），Qwen3-0.6B | 文件、文件夹或 Hugging Face repo id；**必填** |
| `vae` | `vae/qwen_image_vae.safetensors`（254 MB），Qwen-Image VAE | **必填**；不是 Qwen-Image 2.1 VAE |
| `llm_adapter_path` | 单独的 LLM Adapter | 不填时从 DiT 文件中读取 |

## 有限制的选项

| 选项 | |
|---|---|
| `attn_mode` | `torch`（PyTorch SDPA）、`xformers` 或 `flash`（需要 flash-attn，[安装方法](../README-zh.md#flash-attn可选)）。取代其他模型的 `xformers` / `sdpa` 开关 |
| `split_attn` | 只省显存；不能与 `compile_fullgraph` 同时使用 |
| `timestep_sampling` | `sigma`、`uniform`、`sigmoid`（默认）、`shift`、`flux_shift` |
| `discrete_flow_shift` | 只在 `timestep_sampling = sigma` 或 `shift` 时使用 |
| `weighting_scheme` | Anima 只实现了 `sigma_sqrt` 和 `cosmap`，其余都按等权处理 |
| `vae_disable_cache`、`vae_chunk_size` | 开启 `qwen_image_vae_2d` 时无效（缓存）或影响很小（分块） |
| `network_reg_lrs` | 按正则为各模块设置学习率（`.*self_attn.*=1e-4,.*mlp.*=5e-5`）。sd-scripts 的 `self_attn_lr`、`cross_attn_lr`、`mlp_lr`、`mod_lr` 和 `llm_adapter_lr` 只用于全量微调，因此 LoRA 页面不提供这些选项 |

## 加速与显存

- **缓存**：`cache_latents` 和 `cache_text_encoder_outputs`（可写入磁盘）默认开启。缓存文本编码器输出会同时打开
  `network_train_unet_only`。修改 `caption_dropout_rate` 等 caption 设置后，请删除缓存让它重建。
- **2D VAE**（`qwen_image_vae_2d`）：把 VAE 的 3D 卷积换成等价的 2D 卷积；latent 相同，约 2 倍速，峰值显存约 1/3。
  缓存是同一份，切换时不需要重新缓存。
- **`compile`**（默认开启）：逐 block 的 `torch.compile`。编译期间最初几步较慢，之后明显加速；编译结果会缓存，
  之后的训练启动更快。不需要 C++ 编译器（只有 `compile_shapes = dynamic` 需要）。如果编译失败，请关闭 `compile`，训练会以未编译的方式运行。
  - `compile_shapes` 决定不同分桶尺寸和批大小如何编译：
    - **`bounded`**（默认，LoRA-Forge 的模式）：所有分桶尺寸和批大小共用一个编译图，只有 token 数量和批大小是动态的。
      在一个 20 个分桶的数据集上，编译图从 28 个降到 2 个，loss 不变；一个 epoch 在编译缓存已就绪时快 11%，
      缓存为空时快 55%（批大小 4）。它覆盖最大 2048x2048 px 的分桶，超出范围的形状会单独编译，「运行记录」页面会提示。
      与 `blocks_to_swap` 同时使用时等同于 `static`。
    - **`static`**：每个分桶尺寸（以及每个较小的最后一批）都要编译自己的图，上限为 `compile_cache_size_limit`（64）；
      超过后，新的尺寸将不经编译运行。「运行记录」页面会显示已编译的图数，并在达到上限时提示。
    - **`auto`** / **`dynamic`**：torch.compile 自带的动态形状（即 sd-scripts 的 `compile_dynamic`）。在 torch 2.14 上会失败，
      见下文 [PyTorch #193194](#pytorch-193194)。Windows 上 `dynamic` 需要 VS 2022 C++ 编译器。
  - 由于 `bounded` 下多出来的分桶尺寸不再带来额外编译，`bucket_reso_steps = 16`（默认）可以放心使用：在一个 36 张图的数据集上，
    平均裁剪面积从 1.32%（步长 64）降到 0.34%，每个 epoch 只多 2 个 batch。
  - `activation_memory_budget`（默认 `off`）用空余显存换速度，代替梯度检查点（它让每个 block 的前向跑两遍）：
    每个已编译的 block 保留最多这一比例的激活，只重算其余部分，loss 不变。数值越大越快，但需要更多显存。
    `auto` 在训练开始时根据空闲显存和最大的 batch 选出比例（所选的值会打印在控制台输出中）；空闲显存太少时保留梯度检查点。
    只要用上了比例，该次训练就会关闭梯度检查点；保存的配置仍保留你的 `gradient_checkpointing` 设置。每个比例第一次使用时会编译一次。这类训练还会启用 PyTorch 的可扩展显存段（expandable segments），
    防止显存碎片化（设置了 `PYTORCH_CUDA_ALLOC_CONF`、多卡训练或 `compile_mode` 使用 CUDA graphs 时不启用）。
    显存放不下时，训练会在第一步失败，而不是借用系统内存变慢，「运行记录」页面会提示：请选更小的比例。
    与 `blocks_to_swap` 或 offload 选项同时使用时不生效。
  - 不能与旧的 `torch_compile` 选项同时使用。
- **`cuda_allow_tf32`、`cuda_cudnn_benchmark`**：与 `compile` 无关，可单独开启（Ampere 及更新的显卡）。
- **ProdigyPlusScheduleFree**：LoRA-Forge 以批量方式执行它的优化器步骤，把形状相同的 LoRA 张量放在一起更新，而不是逐个更新。
  计算相同，保存的状态也相同；每步耗时约少 7%，与 AdamW8bit 差不多快。使用它的 `use_bias_correction`、`use_adopt`、
  `use_cautious`、`use_grams`、`use_orthograd`、`use_focus` 或 `use_speed` 选项时，仍走原来的步骤。
  `LORA_FORGE_PATCHES=0` 会连同 LoRA-Forge 的其他补丁一起关闭它。
- **省显存**：`blocks_to_swap`、`gradient_checkpointing`、`unsloth_offload_checkpointing`、`cpu_offload_checkpointing`。
  `blocks_to_swap` 与两种 offload 选项互斥；打开其中一个时，GUI 会自动清掉冲突的选项。
- **精度**（`training_precision`）：同时决定两件事：训练计算的精度（sd-scripts 的 `mixed_precision`）和训练中
  LoRA 权重的精度（`full_bf16` / `full_fp16`，否则为 fp32）。保存的文件按 `save_precision`，与此无关。

  | 选项 | 计算 | LoRA 权重 | 用途 |
  |---|---|---|---|
  | `bf16` | bf16 | fp32 | |
  | `full-bf16`（默认） | bf16 | bf16（sd-scripts 的 `full_bf16`） | 标准选项（RTX 30 系及更新） |
  | `fp16` | fp16 | fp32 | 不支持 bf16 的显卡，或配合 `fp16_accumulation` 提速；显存比 bf16 略多 |
  | `full-fp16` | fp16 | fp16（sd-scripts 的 `full_fp16`） | 显存比 `fp16` 略少；仅适用于 AdamW8bit、PagedAdamW8bit、Lion8bit、PagedLion8bit 或 AdaFactor |
  | `fp32` | fp32 | fp32 | 最慢，仅用于调试 |

  选 `fp16` 或 `full-fp16` 时，**`fp16_accumulation`**（默认开启）让矩阵乘法也用 fp16 累加：在 GeForce RTX 显卡上
  明显更快（数据中心卡和 RTX PRO 卡没有提升），精度仍与 bf16 相当。它是 LoRA-Forge 自己写入导出 TOML 的键，sd-scripts
  不会读到。权重为 fp16 时（所有页面的
  `full_fp16`）其他优化器会出问题：AdamW、RAdamScheduleFree、CAME 会让权重变成 NaN，Prodigy、ProdigyPlus、
  DAdaptation 会停止学习，因此 GUI 拒绝启动这些组合。Lion 和 SGDNesterov 能训练，但会丢失较小的更新（控制台会警告）。

## 子文件夹时间步偏移

让不同子集在不同的噪声区间上训练：**负值偏向低噪声（细节），正值偏向高噪声（结构）**。

- 适用于 `timestep_sampling = sigmoid`、`shift` 或 `flux_shift`；`sigma`、`uniform` 下无效。
- 只影响训练；验证仍使用无偏采样，便于对比 loss。
- 建议范围 -0.5 ~ 0.5。`shift` / `flux_shift` 本身已偏向高噪声，正偏移请从 +0.25 或更小开始。
- 在数据集面板中为每个 `重复次数_名称` 子文件夹设置；先用 **预览时间步分布** 检查分布。偏移不需要重建缓存。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 开了加速选项但速度没变 | sd-scripts 版本过旧，选项被忽略；查看首页的 **系统信息** |
| loss 变成 NaN | 更新 PyTorch；Anima 需要较新的 2.x |
| 第一步报 Inductor 或 Triton 错误 | 关闭 `compile`，并反馈该错误 |
| 第一步报 `AssertionError: lift_tracked_freevar_to_input should not be called on root SubgraphTracer` | `compile_shapes` 为 `dynamic` 或 `auto`；见下文 |
| 第一步报 `CUDA driver error: device not ready` | 使用 `activation_memory_budget` 时显存不足：请选更小的比例或 `off` |
| 大量 `remark: <file>:<line>:0: N instructions in function` | 旧版 triton-windows（3.8.0.post28，[#54](https://github.com/triton-lang/triton-windows/issues/54)）；锁文件安装的是 post29，执行 `uv sync` 即可 |
| 时间步预览没有输出 | 结果在「运行记录」页面该次运行的控制台输出中 |

### PyTorch #193194

`compile_shapes = dynamic` 或 `auto`（即 sd-scripts 的 `compile_dynamic = true` / `auto`）时，torch 2.14（及更早版本）
会在第一步报上面的 `lift_tracked_freevar_to_input` 断言：这是 Dynamo 在动态形状 + 梯度检查点下的 bug
（[issue](https://github.com/pytorch/pytorch/issues/193194)，由 [PR #193254](https://github.com/pytorch/pytorch/pull/193254) 修复，
预计进入 torch 2.15）。请使用 `bounded`（默认）：它能获得动态形状的好处，又不会触发它。

## 运行时依赖

- `pyproject.toml` 把 torch 限制在 2.15 以下；triton-windows 为 3.8.x（3.8 支持 torch 2.14 和 2.15）。
- xformers 和 flash-attn 需要与已安装 torch 版本对应的 wheel。
- 升级 torch 或 Triton 会让编译缓存（`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`）失效，下一次训练会重新编译一次。

## 使用训练好的 LoRA

- 文件中记录了训练设置（`ss_timestep_sampling`、`ss_sigmoid_scale`、`ss_discrete_flow_shift`、`ss_weighting_scheme` 等）。
- 只训练 DiT 的 LoRA（默认，`network_train_unet_only`）可以直接在 ComfyUI 中加载。如果同时训练了 Qwen3 文本编码器，
  请用 sd-scripts 的工具转换：在 `scripts/sd-scripts` 目录执行
  `uv run python -m networks.convert_anima_lora_to_comfy <源文件> <目标文件>`。
- 预览图建议 CFG 4-5、30-50 步。
