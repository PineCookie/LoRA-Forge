<script setup lang="ts">
import { computed } from "vue";
import { store } from "../store";

const rows = computed(() => {
  const versions = store.runtime?.versions ?? {};
  if (!Object.keys(versions).length) return [];
  return [
    ["sd-scripts", versions.sd_scripts || "unknown"],
    ...(versions.sd_scripts_status ? [["sd-scripts 兼容性", versions.sd_scripts_status]] : []),
    ["PyTorch", versions.pytorch || "unknown"],
    ["Triton", versions.triton || "unknown"],
    ["CUDA", versions.cuda || "unknown"],
  ];
});
</script>

<template>
  <section class="system-info">
    <h2>系统信息</h2>
    <div class="version-info">
      <div v-for="[label, value] in rows" :key="label" class="version-row">
        <span class="version-label">{{ label }}</span>
        <span class="version-value" :title="value">{{ value }}</span>
      </div>
      <span v-if="!rows.length" class="version-label">加载中...</span>
    </div>
  </section>
</template>
