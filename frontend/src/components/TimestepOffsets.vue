<script setup lang="ts">
import { computed } from "vue";
import type { DatasetInfo } from "../composables/useDatasetInfo";
import { OFFSET_NAME, type SubsetOffsets } from "../config/offsets";
import { t } from "../i18n";
import RichText from "./RichText.vue";

// FLUX / Anima: a timestep offset per dataset subfolder, laid out like an option row. sd-scripts reads it
// from its dataset config as each subset's custom_attributes.timestep_sampling.offset (config.py writes that
// file). Values belong to one dataset folder; TrainerView drops them for another folder.
const offsets = defineModel<SubsetOffsets>("offsets", { required: true });
const props = defineProps<{ info: DatasetInfo; timestepSampling: string }>();

function setOffset(name: string, value: string) {
  offsets.value = { ...offsets.value, values: { ...offsets.value.values, [name]: value } };
}

const advice = computed(() => {
  if (["sigma", "uniform"].includes(props.timestepSampling)) return t("dataset.adviceSigmaUniform");
  if (["shift", "flux_shift"].includes(props.timestepSampling)) return t("dataset.adviceShift");
  return t("dataset.adviceDefault");
});
</script>

<template>
  <div class="offsets-row">
    <div class="meta">
      <div class="name">{{ OFFSET_NAME }}</div>
      <RichText class="description" tag="div" :text="t('dataset.offsetNote')" />
    </div>
    <div class="control">
      <label v-for="subdir in info.subdirs" :key="subdir.name" class="folder">
        <span :title="subdir.name">{{ subdir.concept || subdir.name }}</span>
        <input
          type="number"
          step="0.01"
          :value="offsets.values[subdir.name] ?? 0"
          :title="t('dataset.offsetTitle')"
          @input="setOffset(subdir.name, ($event.target as HTMLInputElement).value)"
        />
      </label>
      <RichText class="advice" tag="div" :text="advice" />
    </div>
  </div>
</template>

<style scoped>
.offsets-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 320px);
  gap: 6px 28px;
  padding: 12px 18px;
  border-top: 1px solid var(--border);
}

.name {
  color: var(--text-strong);
  font-family: var(--font-mono);
  font-size: 13.5px;
  font-weight: 600;
}

.description {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}

.control {
  display: grid;
  align-content: start;
  gap: 6px;
}

.folder {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 96px;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}

.folder span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.folder input {
  padding: 5px 8px;
}

.advice {
  padding: 7px 10px;
  border-radius: var(--radius-sm);
  background: var(--info-soft);
  color: var(--info);
  font-size: 12px;
  line-height: 1.45;
}

@container (max-width: 620px) {
  .offsets-row {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
