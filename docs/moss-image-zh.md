# MOSS-Image LoRA 训练

[English](moss-image.md)

[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS，Apache-2.0）是一个 5B 的文生图模型，文字渲染和版式控制能力很强。目前还没有上游训练器支持它，所以 LoRA Forge 用自带的训练器（`lora_forge/trainers/moss_image`）训练。它遵循 [MOSS-Image 技术报告](https://github.com/OpenMOSS/MOSS-Image/tree/main/paper) 的做法：相同的损失、相同的时间步采样，文本编码器保持冻结。训练器的内部原理见 [moss-image-trainer.md](moss-image-trainer.md)（英文）。

页面上的 **模型文件** 面板列出下面三个文件及其下载链接，并检查你填写的每个路径：文件是否存在、是否是正确的文件（例如不是 int8 文本编码器）。

## 模型文件

| 选项 | 文件 | 说明 |
|---|---|---|
| `pretrained_model_name_or_path` | [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) 的 `diffusion_models/moss_image_base_bf16.safetensors`（9.5 GB） | Base 模型；也可以用 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) 中 OpenMOSS 的 `transformer` 文件夹。int8 文件无法用于训练 |
| `text_encoder` | [Comfy-Org/Qwen3.5](https://huggingface.co/Comfy-Org/Qwen3.5) 的 `text_encoders/qwen3.5_9b_bf16.safetensors`（19.3 GB） | **必填**，需要 bf16 文件：训练前用它编码标注，预览图的提示词也用它 |
| `vae` | [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/main/vae) 的 `vae/diffusion_pytorch_model.safetensors`（250 MB） | **必填**。这是模型训练时用的 VAE。ComfyUI 的 `flux2-vae.safetensors` 也可以，latent 相差不到 1% |

分词器（几 MB）会在第一次训练时从 Hugging Face（`Qwen/Qwen3.5-9B`）下载。

**在 Base 上训练，生成时两者都可以用。** MOSS-Image-SFT 是在 Base 上做的轻量微调（权重相差约 1%），在 Base 上训练的 LoRA 两个模型都能用。OpenMOSS 的后训练建议也是针对 Base 写的。在我们的测试中，同一个 LoRA 用 SFT 生成的图更好看。

## 标注

MOSS-Image 是用自然语言描述训练的，所以请写句子，而不是标签列表。图中的文字会按标注里写的样子渲染。

**标注文件的每一行是该图片的一条标注。** 每一步随机选用其中一条训练，就像 OpenMOSS 用同一张图的短、中、长三种标注训练一样。这一点对角色很重要：如果每条标注都把完整外观写出来，模型学到的是画出这段描述，单靠名字几乎唤不回外观。请再加一行简短的，写角色名和这张图里会变化的内容：

```
An anime-style illustration of my_character from the chest up in her school uniform: a navy blazer over a white shirt with a striped tie, a club badge on the lapel, ... Plain white background.
my_character in her school uniform with a club badge, smiling at the viewer.
```

每一行都要写出这张图与其他图不同的东西（徽章、眼镜、发带），只把固定不变的外观留给名字。短句里没写的内容，模型会归到名字上，然后在每张图里混着出现：徽章跑到别的服装上、多出来的发夹。

只有一行的标注文件照常训练。OpenMOSS 的[用户指南](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE_ZH.md)里有他们推荐的视觉模型标注提示词（长、中、短三种标注）。

## 训练流程

1. **缓存。** 训练开始前先缓存 latent（VAE）和文本编码器输出，训练页面上显示为“缓存 latent”和“缓存文本编码器输出”。缓存文件写在图片旁边（`*_moss_latent.safetensors`、`*_moss_te.safetensors`），并记录它们由什么生成：图片、标注或模型文件改动后会重新编码。文本编码器需要约 20 GB 可用内存，显存放不下的部分在内存中运行。
2. **训练**，批大小为 1：每张图保留自己的标注长度，与 OpenMOSS 的流程一致。想要更大的批次请用 `gradient_accumulation_steps`。每张图按自身宽高比得到一个分桶，像素数等于训练分辨率，边长为 16 的倍数。
3. **预览图**使用 OpenMOSS 的采样方式：Euler、shift 5 加 CFG，负面推理跳过第 22 个 block。预览图用训练所用的模型（Base）生成，它比 SFT 画得朴素：只给名字时可能画出别的角色，训练中途的检查点也比最终 LoRA 在 SFT 上的效果差。请在 ComfyUI 中用 SFT 评判 LoRA，不要只看预览图。

数据集文件夹的结构与其他模型相同（`<重复次数>_<名称>` 子文件夹，里面放图片和 `.txt` 标注）。只支持单卡训练。

## 显存与速度

在 16 GB 显卡上、约 100 万像素、rank 32：默认设置下**每步 2.1 秒**，`training_precision = fp16` 时 **1.5 秒**，不用 flash-attn 时 2.9 秒。显存峰值约 12-13 GB。

- **`training_precision`**（默认 `bf16`）：`fp16` 让 block 的矩阵乘法以 fp16 运行并使用损失缩放；配合 `fp16_accumulation`（选择 `fp16` 时默认开启），累加也用 fp16，GeForce RTX 显卡上约快一倍：每步少用约四分之一的时间。在测试中，它的梯度精度与 bf16 相当，也没有因溢出而跳过的步数。两种设置下 LoRA 权重都以 fp32 训练。在有更多完整训练与 bf16 对比之前，它需要手动开启。
- **`gradient_checkpointing`**（默认开启）：16 GB 显存请保持开启，仅 DiT 就占约 10 GB。
- **`blocks_to_swap`**（默认 0）：用于显存不足 16 GB 的显卡。这么多个 block（最多 42 个中的 40 个）的权重留在系统内存中，用到之前才复制到显卡，每个约 0.2 GB；训练出的 LoRA 完全相同。在 16 GB 显卡上：20 个 block 约 9 GB 显存、速度不变；30 个约 7 GB、慢 6%；40 个约 5 GB、慢 13%。PCIe 更慢的机器损失更多。它需要 `gradient_checkpointing`；`use_pinned_memory_for_block_swap`（默认开启）让复制与计算同时进行，关闭时 40 个 block 慢 53%。
- **`compile`**（默认开启）：逐 block 的 torch.compile，约快 30%。所有分桶尺寸只需要构建少数几个图；编译期间最初几步较慢。
- **`attn_mode`**（默认 `auto`）：安装了 [flash-attn](../README-zh.md#flash-attn可选) 时用 FlashAttention，否则用 PyTorch 的 SDPA。

## 训练设置

默认：rank 和 alpha 为 32，覆盖全部 42 个 block 的注意力层和前馈层（约 7400 万参数），AdamW 学习率 1e-4，使用 cosine 调度，10 个 epoch，时间步采样使用报告中的值（`logit_mean` -1.09、`logit_std` 0.95）。

- **cosine 调度**：学习率在训练过程中逐渐降到 0，后期的步数对 LoRA 的改动更小。ProdigyPlus 是 schedule-free 优化器，始终使用恒定学习率。默认不预热：`lr_warmup_steps` 填整数表示步数，小于 1 表示占总步数的比例（0.05 = 前 5%，会随训练长度变化），与 sd-scripts 相同。
- **标注丢弃**（`caption_dropout_rate`，默认关闭）：一部分步数用空标注训练。使用 CFG 时，每个采样步还会用负面提示词（默认为空）再跑一次模型，结果会放大两者之间的差异；标注丢弃让 LoRA 也针对这次空提示词的推理训练，学到"空提示词 → 像数据集的图"。这有助于画风在所有图中生效（常用 0.1）；训练角色时，可能导致提示词里不写角色名也会出现该角色。空提示词的编码只缓存一次，放在数据集文件夹中（`_moss_empty_prompt_te.safetensors`）。

我们的角色训练（28 张图，每个 epoch 约 100 步；当时学习率恒定、没有标注丢弃）得到的经验：

- **在我们的训练中它很容易练过头。** 学习率 1e-4、rank 32 时，角色约 200 步就学会了，500-600 步左右效果最好；约 700 步之后，覆盖全部层的 LoRA 开始破坏图像（姿势僵硬、颜色错误、人体变形）。请每个 epoch 都保存（`save_every_n_epochs = 1`），按图片效果挑选检查点；10 个 epoch 是上限，不是目标。
- **`lora_targets = attention`** 只训练注意力层（q、k、v、o；rank 32 时 3300 万参数），决定画法的前馈层保持底模原样。在我们的训练中它练过头的速度慢得多（900 步仍未崩坏），保留了底模的画风，角色也照样学到了。训练角色时值得先试（目前只测试过一个数据集，并非普遍结论）。
- **模型本身就能从文字画出很多东西。** 不加任何 LoRA，SFT 仅凭对发型和服装的文字描述就几乎准确画出了我们的测试角色。所以 LoRA 的任务是把名字和这个外观绑定、补上文字说不清的细节；强度 1.0 时它往往做得过多（头发纹理重复、服装被覆盖）。效果最干净的是：描述性提示词加上强度 0.5-0.7 的 LoRA，在 SFT 上生成；只给名字、强度 1.0 的组合最容易出现瑕疵。
- fp16 配合 `fp16_accumulation` 完整训练了两次，没有溢出；但没有在完全相同的设置下与 bf16 比较过图片效果。

## 在 ComfyUI 中使用 LoRA

ComfyUI 原生支持 MOSS-Image（请先更新）；用 *LoraLoaderModelOnly* 加载 LoRA。OpenMOSS 的采样设置：50 步、CFG 3、euler / sgm_uniform、shift 5（ComfyUI 对 MOSS-Image 的默认值）、负面提示词留空，并在负面推理中跳过第 22 个 block：添加 *SkipLayerGuidanceDiTSimple*，`single_layers` 填 `22`，`double_layers` 留空。ComfyUI 自己的示例工作流用 25 步、CFG 2.5、er_sde / simple，只在前一半步数跳过（`end_percent` 0.5）。
