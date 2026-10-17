<script setup lang="ts">
import { computed } from "vue";
import { t } from "../i18n";
import { store } from "../store";

const gpus = computed(() => store.runtime?.gpus ?? []);

const rows = computed(() => {
  const versions = store.runtime?.versions ?? {};
  if (!Object.keys(versions).length) return [];
  return [
    ...gpus.value.map((gpu, index) => [
      gpus.value.length > 1 ? `GPU ${index}` : "GPU",
      `${gpu.name.replace(/^NVIDIA /, "")} (${gpu.memory_gb} GB)`,
    ]),
    ["LoRA Forge", versions.lora_forge || "unknown"],
    ["sd-scripts", `${versions.sd_scripts || "unknown"}${versions.sd_scripts_state === "tested" ? ` (${t("system.tested")})` : ""}`],
    ["musubi-tuner", versions.musubi_tuner || "unknown"],
    ["PyTorch", versions.pytorch || "unknown"],
    ["Triton", versions.triton || "unknown"],
    ["CUDA", versions.cuda || "unknown"],
  ];
});

// cards this PyTorch build cannot run, or runs without bf16 / flash-attn; an sd-scripts too old for some options
const warnings = computed(() => [
  ...gpus.value
    .filter((gpu) => gpu.support !== "ok")
    .map((gpu) => ({
      level: gpu.support === "unsupported" ? "danger" : "warning",
      text: t(gpu.support === "unsupported" ? "system.gpuUnsupported" : "system.gpuLimited", {
        name: gpu.name.replace(/^NVIDIA /, ""),
        capability: gpu.capability,
        minimum: gpu.minimum,
      }),
    })),
  ...(store.runtime?.versions.sd_scripts_problem ? [{ level: "warning", text: store.runtime.versions.sd_scripts_problem }] : []),
]);
</script>

<template>
  <section class="panel system-info">
    <h2>{{ t("system.title") }}</h2>
    <dl v-if="rows.length">
      <div v-for="[label, value] in rows" :key="label">
        <dt>{{ label }}</dt>
        <dd :title="value">{{ value }}</dd>
      </div>
    </dl>
    <p v-else class="muted">{{ t("system.loading") }}</p>
    <p v-for="warning in warnings" :key="warning.text" class="gpu-warning" :class="warning.level">{{ warning.text }}</p>
  </section>
</template>

<style scoped>
dl {
  display: grid;
  gap: 4px;
  margin: 0;
}

dl div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 13px;
}

dt {
  color: var(--muted);
}

/* a narrow panel wraps the label, not the value */
dd {
  flex: 0 0 auto;
  max-width: 70%;
  margin: 0;
  overflow: hidden;
  font-family: var(--font-mono);
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gpu-warning {
  margin: 10px 0 0;
  padding: 8px 10px;
  border: 1px solid var(--warning-border);
  border-radius: var(--radius);
  background: var(--warning-soft);
  color: var(--warning);
  font-size: 13px;
  line-height: 1.45;
}

.gpu-warning.danger {
  border-color: var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
}
</style>
