# Krea 2 LoRA 训练说明

[English](krea2.md)

Krea 2 LoRA 由 [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)（其
`krea2_train_network.py`）训练，LoRA Forge 首次启动时会把它克隆到 `scripts/musubi-tuner`，与 sd-scripts
并列。页面上的选项就是 musubi-tuner 的选项，含义请参阅它的
[`docs/krea2.md`](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md)。Krea 2 支持在那里标注为实验性。

## 模型文件

Krea 2 是 12.9B 的文生图模型，使用 Qwen3-VL-4B 文本编码器和 Qwen-Image VAE（与 Anima 相同的 VAE）。从
[Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) 和
[Comfy-Org/Qwen3-VL](https://huggingface.co/Comfy-Org/Qwen3-VL) 下载：

| 选项 | 文件 | 备注 |
|---|---|---|
| `pretrained_model_name_or_path` | `krea2_raw_int8_convrot.safetensors`（13.5 GB） | RAW 模型，已预先量化；配合 `convrot_int8`（默认）可直接训练。`fp8_base` 或 bf16 训练需要 `krea2_raw_bf16`（26 GB） |
| `vae` | `qwen_image_vae.safetensors` | **必填** |
| `text_encoder` | `text_encoders/qwen3vl_4b_bf16.safetensors` | **必填**：训练前用它编码标注；生成预览图也需要 |
| `turbo_dit` | `krea2_turbo_*.safetensors` | 可选，只用于预览图；目前不能与 `convrot_int8` 同用 |

**用 RAW 训练，LoRA 用在 Turbo 上。** Turbo 是蒸馏出的少步数、无 CFG 的模型，在它上面训练会与蒸馏相冲突；
musubi-tuner 只在 RAW 上训练。训练出的 LoRA 在 ComfyUI 中两个模型都能加载。

## 一次训练的流程

1. **缓存。** 训练开始前，musubi-tuner 的两个缓存脚本先缓存 latent（VAE）和文本编码器输出；「运行记录」页面会显示
   「缓存 latent」和「缓存文本编码器输出」。缓存文件写在图片旁边（与 sd-scripts 的 `.npz` 一样），同一数据集再次训练时只编码新图片。
2. **训练。** 以页面上的选项运行 `krea2_train_network.py`，监控方式与其他页面相同：进度、loss 曲线、预览图和保存的 LoRA 文件。

数据集文件夹的结构与其他模型相同（`<重复次数>_<名称>` 子文件夹，内含图片和 `.txt` 标注）；LoRA Forge 据此生成
musubi-tuner 的数据集 TOML。只支持单卡训练（上游未测试多卡）。

## 显存与速度

- **`convrot_int8`**（默认开启）：冻结的底模权重以 int8 保存，并通过 Triton（LoRA Forge 已安装）运行真正的 int8 矩阵乘法。
  显存为 bf16 的一半，且在 Krea 2 上比 fp8 更快（fp8 路径每次前向都要反量化）。LoRA 本身仍以全精度训练。
- **`blocks_to_swap`**（默认 18）：放在系统内存、用到时再搬进显存的 block 数。开启 `use_pinned_memory_for_block_swap`
  和 `block_swap_h2d_only`（默认都开启）时，复制与计算同时进行，即使换出全部 26 个可换出的 block 也几乎不耗时。
  默认值让 16 GB 显卡在 1024x1024 下留有充足显存；训练更大的图片或批次时可调高。0 表示整个模型都在显存里。
- **锁页内存**（`use_pinned_memory_for_block_swap`）不能被换页：每个换出的 block 约 0.5 GB，默认约 9 GB。
  Windows 任务管理器会把它显示为共享 GPU 内存，这并不表示模型溢出了显存。内存较少的电脑请减少换出的 block。
- **`gradient_checkpointing`**（默认开启）：这个模型规模下必需，`block_swap_h2d_only` 也需要它。
- **`compile`**（默认开启）：用 torch.compile 编译 28 个主 block；量化层和换出的 block 会自动排除。最初几步用于编译。
- **`attn_mode`**（默认 `auto`）：已安装 flash-attn 时使用 FlashAttention，否则使用 PyTorch 的 SDPA。在 Krea 2 上
  FlashAttention 让每步快约五分之一，结果相同，建议安装（见 [flash-attn](../README-zh.md#flash-attn可选)）。
  不提供 SageAttention：它没有反向传播，无法用于训练。
- 显存放不下时，Windows 会借用系统内存，训练变得极慢（每步数分钟）而不是报错：请调高 `blocks_to_swap`。

## 训练设置

默认值沿用 musubi-tuner 的 Krea 2 示例：rank 与 alpha 32、AdamW8bit、学习率 1e-4、16 个 epoch、
`timestep_sampling = krea2_shift`（Krea 2 自带的、随分辨率变化的时间步偏移；`shift` 加 `discrete_flow_shift = 2.5`
是 1024x1024 下的固定等价设置）。LoRA 默认覆盖 DiT 的全部 264 个 Linear 层，可用 `network_args` 缩小范围（见 musubi-tuner 文档）。
上游尚未确定最优设置。

## 预览图

预览图需要 `text_encoder`。默认用 RAW 模型生成（28 步、CFG 5.5）；设置了 `turbo_dit` 时改用 Turbo 模型（8 步、无 CFG），
这也是 LoRA 通常的使用方式。

## 更新 musubi-tuner

与 sd-scripts 一样，musubi-tuner 克隆的是测试过的版本，不会自动更新：

```powershell
.\run_gui.ps1 update-musubi-tuner           # Windows
```

```sh
bash run_gui.sh update-musubi-tuner         # Linux
```

`--branch <分支或标签>` 可选择其他分支或标签。
