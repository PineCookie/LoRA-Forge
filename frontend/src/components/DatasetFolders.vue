<script setup lang="ts">
import type { DatasetInfo } from "../composables/useDatasetInfo";
import { t } from "../i18n";

// The dataset folder's <repeats>_<name> subfolders, under its path: repeats x images = images per epoch.
defineProps<{ info: DatasetInfo | null; loading: boolean }>();
</script>

<template>
  <div v-if="loading" class="state">{{ t("dataset.loading") }}</div>
  <ul v-else-if="info?.subdirs.length" class="folders" :aria-label="t('dataset.folders')">
    <li v-for="subdir in info.subdirs" :key="subdir.name">
      <span class="repeat">× {{ subdir.repeat }}</span>
      <span class="name" :title="subdir.name">{{ subdir.concept || subdir.name }}</span>
      <span class="count">{{ t("dataset.folderImages", { count: subdir.image_count }) }}</span>
      <span class="total">= {{ (subdir.image_count * subdir.repeat).toLocaleString() }}</span>
    </li>
  </ul>
</template>

<style scoped>
.state {
  color: var(--muted);
  font-size: 12.5px;
}

.folders {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto auto;
  gap: 2px 14px;
  margin: 0;
  padding: 6px 10px;
  border-radius: var(--radius-sm);
  background: var(--surface-sunken);
  list-style: none;
  font-size: 12.5px;
}

li {
  display: contents;
}

.repeat,
.count,
.total {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.repeat {
  font-family: var(--font-mono);
}

.name {
  overflow: hidden;
  color: var(--text-strong);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.count,
.total {
  text-align: right;
}

.total {
  color: var(--text);
  font-weight: 600;
}
</style>
