<script setup lang="ts">
import { computed, toRef, watch } from "vue";
import type { DatasetInfo } from "../composables/useDatasetInfo";
import type { SubsetOffsets } from "../config/offsets";
import { t } from "../i18n";
import RichText from "./RichText.vue";

const offsets = defineModel<SubsetOffsets>("offsets", { required: true });
const props = defineProps<{
  dir: string;
  /** the folder's image counts (the page reads them once for the hints and the run summary) */
  info: DatasetInfo | null;
  loading: boolean;
  error: string;
  batchSize: number;
  gradAcc: number;
  supportsOffsets: boolean;
  timestepSampling: string;
}>();

const info = toRef(props, "info");

// Offsets belong to one dataset folder: drop them for another folder or for subfolders that no
// longer exist (/api/run rejects unknown folders, and their inputs would be gone from the table).
watch(info, (result: DatasetInfo | null) => {
  const dir = props.dir.trim();
  if (!result) {
    if (!dir) offsets.value = { dir: null, values: {} };
    return;
  }
  const names = new Set(result.subdirs.map((subdir) => subdir.name));
  const kept = offsets.value.dir === dir ? Object.entries(offsets.value.values).filter(([name]) => names.has(name)) : [];
  offsets.value = { dir, values: Object.fromEntries(kept) };
});

function setOffset(name: string, value: string) {
  offsets.value = { ...offsets.value, values: { ...offsets.value.values, [name]: value } };
}

const effectiveBatch = computed(() => props.batchSize * props.gradAcc);
// Lower bound: all images in one bucket -> ceil(total / batch). ARB buckets batch separately,
// so the real count is a little higher.
const stepsPerEpoch = computed(() => (info.value && effectiveBatch.value > 0 ? Math.ceil(info.value.total_images_with_repeats / effectiveBatch.value) : 0));
const suggestions = computed(() =>
  [500, 1000, 1500, 2000, 3000].map((steps) => t("dataset.suggestion", { steps: steps.toLocaleString(), epochs: Math.max(1, Math.round(steps / stepsPerEpoch.value)) })),
);

const advice = computed(() => {
  if (["sigma", "uniform"].includes(props.timestepSampling)) return t("dataset.adviceSigmaUniform");
  if (["shift", "flux_shift"].includes(props.timestepSampling)) return t("dataset.adviceShift");
  return t("dataset.adviceDefault");
});
</script>

<template>
  <div class="dataset-stats wide">
    <span v-if="loading" class="state">{{ t("dataset.loading") }}</span>
    <div v-else-if="info && !error" class="panel-inner">
      <div class="title">{{ t("dataset.title") }}</div>

      <template v-if="info.subdirs.length">
        <table>
          <thead>
            <tr>
              <th>{{ t("dataset.subfolder") }}</th>
              <th>{{ t("dataset.repeats") }}</th>
              <th>{{ t("dataset.images") }}</th>
              <th>{{ t("dataset.effectiveImages") }}</th>
              <th v-if="supportsOffsets">{{ t("dataset.offset") }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="subdir in info.subdirs" :key="subdir.name">
              <td>{{ subdir.concept || subdir.name }}</td>
              <td>×{{ subdir.repeat }}</td>
              <td>{{ subdir.image_count }}</td>
              <td>{{ subdir.image_count * subdir.repeat }}</td>
              <td v-if="supportsOffsets">
                <input
                  class="offset"
                  type="number"
                  step="0.01"
                  :value="offsets.values[subdir.name] ?? 0"
                  :title="t('dataset.offsetTitle')"
                  @input="setOffset(subdir.name, ($event.target as HTMLInputElement).value)"
                />
              </td>
            </tr>
          </tbody>
        </table>
        <template v-if="supportsOffsets">
          <RichText class="note" tag="p" :text="t('dataset.offsetNote')" />
          <RichText class="advice" tag="div" :text="advice" />
        </template>
      </template>
      <RichText v-else class="note" tag="p" :text="t('dataset.noSubsets')" />

      <template v-if="info.total_images_with_repeats > 0 && stepsPerEpoch > 0">
        <RichText class="bucket-note" tag="div" :text="t('dataset.bucketNote')" />
        <div class="suggestions">
          <div class="suggest-title">{{ t("dataset.suggestTitle") }}</div>
          {{ suggestions.join(" · ") }}
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.dataset-stats {
  margin: 4px 16px 14px;
}

.dataset-stats:empty {
  display: none;
}

.state {
  color: var(--muted);
  font-size: 13px;
  font-style: italic;
}

.panel-inner {
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-sunken);
  font-size: 13px;
  line-height: 1.5;
}

.title {
  margin-bottom: 8px;
  color: var(--text-strong);
  font-size: 14px;
  font-weight: 600;
}

table {
  width: 100%;
  margin-bottom: 8px;
  border-collapse: collapse;
  font-size: 12px;
}

th,
td {
  padding: 3px 8px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}

th {
  color: var(--muted);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
}

.offset {
  width: 5.5rem;
  padding: 4px 6px;
}

.note {
  margin: 0 0 6px;
  color: var(--muted);
}

.advice {
  margin: 8px 0;
  padding: 7px 10px;
  border: 1px solid var(--info-border);
  border-radius: var(--radius-sm);
  background: var(--info-soft);
  color: var(--info);
  font-size: 12px;
}

.bucket-note {
  margin-bottom: 8px;
  padding: 6px 10px;
  border: 1px solid var(--warning-border);
  border-radius: var(--radius-sm);
  background: var(--warning-soft);
  color: var(--warning);
  font-size: 12px;
}

.suggestions {
  padding-top: 6px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 12px;
}

.suggest-title {
  margin-bottom: 2px;
  color: var(--text);
  font-weight: 600;
}
</style>
