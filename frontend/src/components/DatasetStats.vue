<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { api } from "../api";
import type { SubsetOffsets } from "../config/offsets";
import { t } from "../i18n";

interface Subdir {
  name: string;
  concept: string;
  repeat: number;
  image_count: number;
}

interface DatasetInfo {
  subdirs: Subdir[];
  total_images: number;
  total_images_with_repeats: number;
}

const props = defineProps<{
  dir: string;
  batchSize: number;
  gradAcc: number;
  epochs: number;
  supportsOffsets: boolean;
  timestepSampling: string;
  offsets: SubsetOffsets;
  /** bump to force a reload, e.g. after importing a config */
  refreshKey: number;
}>();

const info = ref<DatasetInfo | null>(null);
const loading = ref(false);
const error = ref("");

let timer: ReturnType<typeof setTimeout> | undefined;
let requestId = 0;
let loadedDir: string | null = null;
let loadedKey = -1;

function clearOffsets() {
  props.offsets.values = {};
  props.offsets.dir = null;
}

// Drop offsets for another dataset or for subfolders that no longer exist: /api/run rejects
// unknown folders, and the input to clear them would be gone from the UI.
function pruneOffsets(dir: string, result: DatasetInfo) {
  if (dir !== props.offsets.dir) {
    props.offsets.values = {};
    props.offsets.dir = dir;
  }
  const names = new Set(result.subdirs.map((subdir) => subdir.name));
  for (const name of Object.keys(props.offsets.values)) {
    if (!names.has(name)) delete props.offsets.values[name];
  }
}

watch(
  () => [props.dir.trim(), props.refreshKey] as const,
  ([dir, key]) => {
    clearTimeout(timer);
    if (!dir) {
      requestId += 1;
      info.value = null;
      error.value = "";
      loading.value = false;
      loadedDir = null;
      clearOffsets();
      return;
    }
    if (dir === loadedDir && key === loadedKey) return;

    loading.value = true;
    error.value = "";
    const id = (requestId += 1);
    timer = setTimeout(async () => {
      try {
        const result = await api<DatasetInfo>(`/api/dataset_info?train_data_dir=${encodeURIComponent(dir)}`);
        if (id !== requestId) return;
        info.value = result;
        loadedDir = dir;
        loadedKey = key;
        pruneOffsets(dir, result);
      } catch (err) {
        if (id !== requestId) return;
        info.value = null;
        loadedDir = null;
        error.value = err instanceof Error ? err.message : String(err);
      } finally {
        if (id === requestId) loading.value = false;
      }
    }, 400);
  },
  { immediate: true },
);

onBeforeUnmount(() => clearTimeout(timer));

const effectiveBatch = computed(() => props.batchSize * props.gradAcc);
// Lower bound: all images in one bucket -> ceil(total / batch). ARB buckets batch separately,
// so the real count is a little higher.
const stepsPerEpoch = computed(() => (info.value && effectiveBatch.value > 0 ? Math.ceil(info.value.total_images_with_repeats / effectiveBatch.value) : 0));
const totalSteps = computed(() => stepsPerEpoch.value * props.epochs);
const suggestions = computed(() =>
  [500, 1000, 1500, 2000, 3000].map((steps) => ({ steps, epochs: Math.max(1, Math.round(steps / stepsPerEpoch.value)) })),
);

const recommendation = computed(() => {
  if (["sigma", "uniform"].includes(props.timestepSampling)) return "sigma-uniform";
  if (["shift", "flux_shift"].includes(props.timestepSampling)) return "shift";
  return "default";
});

function setOffset(name: string, event: Event) {
  props.offsets.values[name] = (event.target as HTMLInputElement).value;
}
</script>

<template>
  <div class="dataset-stats">
    <span v-if="loading" class="dataset-stats-loading">{{ t("dataset.loading") }}</span>
    <span v-else-if="error" class="dataset-stats-error">{{ t("dataset.error", { error }) }}</span>
    <div v-else-if="info" class="dataset-stats-panel">
      <div class="dataset-stats-title">{{ t("dataset.title") }}</div>

      <template v-if="info.subdirs.length">
        <table class="dataset-stats-table">
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
                  class="subset-timestep-offset"
                  type="number"
                  step="0.01"
                  :value="offsets.values[subdir.name] ?? 0"
                  :title="t('dataset.offsetTitle')"
                  @input="setOffset(subdir.name, $event)"
                />
              </td>
            </tr>
          </tbody>
        </table>
        <template v-if="supportsOffsets">
          <p class="dataset-stats-note" v-html="t('dataset.offsetNote')" />
          <div v-if="recommendation === 'sigma-uniform'" class="timestep-offset-recommendation" v-html="t('dataset.adviceSigmaUniform')" />
          <div v-else-if="recommendation === 'shift'" class="timestep-offset-recommendation" v-html="t('dataset.adviceShift')" />
          <div v-else class="timestep-offset-recommendation" v-html="t('dataset.adviceDefault')" />
        </template>
      </template>
      <p v-else class="dataset-stats-note" v-html="t('dataset.noSubsets')" />

      <div class="dataset-stats-summary">
        <div class="stat-row"><span>{{ t("dataset.totalImages") }}</span><span>{{ info.total_images }}</span></div>
        <div class="stat-row"><span>{{ t("dataset.totalWithRepeats") }}</span><span>{{ info.total_images_with_repeats }}</span></div>
        <div class="stat-row"><span>{{ t("dataset.effectiveBatch") }}</span><span>{{ effectiveBatch }}</span></div>
        <div class="stat-row"><span>{{ t("dataset.stepsPerEpoch") }}</span><span>{{ stepsPerEpoch.toLocaleString() }}</span></div>
        <div class="stat-row stat-row-highlight">
          <span>{{ t("dataset.totalSteps", { epochs }) }}</span><span>{{ totalSteps.toLocaleString() }}</span>
        </div>
      </div>

      <template v-if="info.total_images_with_repeats > 0 && stepsPerEpoch > 0">
        <div class="dataset-stats-note-bucket" v-html="t('dataset.bucketNote')" />
        <div class="dataset-stats-suggestions">
          <div class="dataset-stats-suggest-title">{{ t("dataset.suggestTitle") }}</div>
          <span>{{ suggestions.map((item) => t("dataset.suggestion", { steps: item.steps.toLocaleString(), epochs: item.epochs })).join(" · ") }}</span>
        </div>
      </template>
    </div>
  </div>
</template>