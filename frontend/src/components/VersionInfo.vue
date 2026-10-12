<script setup lang="ts">
import { computed } from "vue";
import { t } from "../i18n";
import { store } from "../store";

const rows = computed(() => {
  const versions = store.runtime?.versions ?? {};
  if (!Object.keys(versions).length) return [];
  return [
    ["sd-scripts", versions.sd_scripts || "unknown"],
    ...(versions.sd_scripts_status ? [[t("system.sdScriptsStatus"), versions.sd_scripts_status]] : []),
    ["PyTorch", versions.pytorch || "unknown"],
    ["Triton", versions.triton || "unknown"],
    ["CUDA", versions.cuda || "unknown"],
  ];
});
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
  flex: 0 0 auto;
  color: var(--muted);
}

dd {
  margin: 0;
  overflow: hidden;
  font-family: var(--font-mono);
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
