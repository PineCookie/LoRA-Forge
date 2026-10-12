<script setup lang="ts">
import { isActive, type Run } from "../composables/useRuns";
import { locale, t } from "../i18n";
import { formatDuration } from "../monitor/metrics";
import { modelLabel } from "../monitor/labels";
import RunStatusBadge from "./RunStatusBadge.vue";

defineProps<{ runs: Run[]; selected: string | null }>();
const emit = defineEmits<{ select: [id: string] }>();

function startedText(run: Run) {
  const date = new Date(run.started * 1000);
  const today = new Date().toDateString() === date.toDateString();
  return new Intl.DateTimeFormat(locale.value, today ? { timeStyle: "short" } : { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function stepPercent(run: Run) {
  const step = run.progress?.step;
  return step?.total ? (step.n / step.total) * 100 : 0;
}
</script>

<template>
  <ul class="run-list">
    <li v-for="run in runs" :key="run.id">
      <button type="button" class="run" :class="{ selected: run.id === selected }" :aria-current="run.id === selected" @click="emit('select', run.id)">
        <span class="top">
          <strong class="name">{{ run.preview ? t("runs.previewName") : run.output_name || t("runs.unnamed") }}</strong>
          <RunStatusBadge :status="run.status" compact />
        </span>
        <span class="meta">
          <span>{{ modelLabel(run.model_train_type) }}</span>
          <span>{{ startedText(run) }}</span>
          <span v-if="run.ended">{{ formatDuration(run.ended - run.started) }}</span>
        </span>
        <span v-if="isActive(run.status)" class="bar" aria-hidden="true"><span :style="{ width: `${stepPercent(run)}%` }" /></span>
      </button>
    </li>
  </ul>
</template>

<style scoped>
.run-list {
  display: grid;
  gap: 4px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.run {
  position: relative;
  display: grid;
  /* the base button centres its content */
  justify-content: stretch;
  align-items: start;
  gap: 4px;
  width: 100%;
  overflow: hidden;
  padding: 10px 12px;
  border-color: transparent;
  background: transparent;
  box-shadow: none;
  text-align: left;
  white-space: normal;
}

.run:hover {
  border-color: var(--border);
  background: var(--surface-muted);
}

.run.selected {
  border-color: var(--primary);
  background: var(--primary-soft);
}

.top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.name {
  overflow: hidden;
  color: var(--text-strong);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 2px 10px;
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.bar {
  position: absolute;
  inset: auto 0 0 0;
  height: 2px;
  background: var(--border);
}

.bar span {
  display: block;
  height: 100%;
  background: var(--success);
  transition: width 0.4s ease;
}
</style>
