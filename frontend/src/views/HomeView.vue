<script setup lang="ts">
import { RouterLink } from "vue-router";
import VersionInfo from "../components/VersionInfo.vue";

const modes = [
  { path: "/lora/anima.html", title: "Anima", text: "Anima DiT LoRA，支持逐 block torch.compile 与子文件夹时间步偏移。" },
  { path: "/lora/flux.html", title: "Flux", text: "FLUX.1 LoRA 训练。" },
  { path: "/lora/master.html", title: "Stable Diffusion", text: "SD1.5 / SDXL LoRA 全参数模式。" },
  { path: "/lora/sd3.html", title: "SD3.5", text: "Stable Diffusion 3.5 LoRA 训练。" },
  { path: "/lora/basic.html", title: "SD1.5", text: "精简参数的 SD1.5 LoRA 训练。" },
  { path: "/dreambooth/index.html", title: "Dreambooth", text: "Dreambooth / 全量微调。" },
];
</script>

<template>
  <div class="home">
    <section class="panel">
      <h2>LoRA-Forge 训练 UI</h2>
      <p class="muted">选择一个训练模式开始配置参数。配置可以导出为 JSON / TOML，并随时导入继续编辑。</p>
    </section>
    <div class="mode-grid">
      <RouterLink v-for="mode in modes" :key="mode.path" :to="mode.path" class="panel mode-card">
        <strong>{{ mode.title }}</strong>
        <span>{{ mode.text }}</span>
      </RouterLink>
    </div>
    <VersionInfo />
  </div>
</template>
