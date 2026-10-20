<script setup lang="ts">
import { computed } from "vue";
import type { CheckLevel } from "../config/checks";
import { t } from "../i18n";

export interface SummaryCheck {
  field: string;
  level: CheckLevel;
  text: string;
}

export interface SpeedEstimate {
  /** seconds for the whole run */
  seconds: number;
  secondsPerStep: number;
  /** the run it was measured on */
  from: string;
}

const props = defineProps<{
  totalSteps: number | null;
  stepsPerEpoch: number | null;
  epochs: string;
  images: number | null;
  imagesWithRepeats: number | null;
  datasetLoading: boolean;
  estimate: SpeedEstimate | null;
  checkpoints: number | null;
  previews: string;
  learningRate: string;
  checks: SummaryCheck[];
  changed: { name: string; value: string }[];
}>();
defineEmits<{ go: [field: string]; reset: [field: string] }>();

const ORDER: Record<CheckLevel, number> = { error: 0, todo: 1, warn: 2 };
const sortedChecks = computed(() => [...props.checks].sort((a, b) => ORDER[a.level] - ORDER[b.level]));
const blocking = computed(() => props.checks.some((check) => check.level !== "warn"));

function duration(seconds: number) {
  const minutes = Math.max(1, Math.round(seconds / 60));
  return minutes < 60 ? t("summary.minutes", { m: minutes }) : t("summary.hours", { h: Math.floor(minutes / 60), m: String(minutes % 60).padStart(2, "0") });
}
const fmt = (n: number) => n.toLocaleString();
</script>

<template>
  <aside class="run-summary" :aria-label="t('summary.title')">
    <section class="card">
      <h2>{{ t("summary.title") }}</h2>
      <template v-if="totalSteps !== null">
        <div class="big">{{ t("summary.steps", { steps: fmt(totalSteps) }) }}</div>
        <div v-if="estimate" class="sub">
          {{ t("summary.estimate", { time: duration(estimate.seconds), speed: estimate.secondsPerStep.toFixed(2), run: estimate.from }) }}
        </div>
        <div v-else class="sub">{{ t("summary.noEstimate") }}</div>
      </template>
      <div v-else class="sub">{{ t(datasetLoading ? "dataset.loading" : "summary.noDataset") }}</div>
      <dl>
        <template v-if="images !== null && imagesWithRepeats !== null">
          <dt>{{ t("summary.dataset") }}</dt>
          <dd>{{ t("summary.images", { images: fmt(images), repeats: fmt(imagesWithRepeats) }) }}</dd>
        </template>
        <template v-if="stepsPerEpoch !== null">
          <dt>{{ t("summary.perEpoch") }}</dt>
          <dd>{{ t("summary.perEpochValue", { steps: fmt(stepsPerEpoch), epochs }) }}</dd>
        </template>
        <template v-if="checkpoints !== null">
          <dt>{{ t("summary.checkpoints") }}</dt>
          <dd>{{ checkpoints }}</dd>
        </template>
        <dt>{{ t("summary.previews") }}</dt>
        <dd>{{ previews }}</dd>
        <template v-if="learningRate">
          <dt>{{ t("summary.learningRate") }}</dt>
          <dd>{{ learningRate }}</dd>
        </template>
      </dl>
    </section>

    <section class="card">
      <h2>{{ t("summary.checks") }}<span v-if="checks.length"> · {{ checks.length }}</span></h2>
      <ul v-if="checks.length" class="list">
        <li v-for="check in sortedChecks" :key="`${check.field}:${check.text}`">
          <span class="level" :class="check.level" />
          <span class="entry">
            <button type="button" class="key" @click="$emit('go', check.field)">{{ check.field }}</button>
            <span class="text">{{ check.text }}</span>
          </span>
        </li>
      </ul>
      <div v-else class="empty">{{ t("summary.noChecks") }}</div>
      <p v-if="blocking" class="blocking">{{ t("summary.blocking") }}</p>
    </section>

    <section class="card">
      <h2>{{ t("summary.changed") }} · {{ changed.length }}</h2>
      <ul v-if="changed.length" class="list">
        <li v-for="item in changed" :key="item.name">
          <button type="button" class="key" @click="$emit('go', item.name)">{{ item.name }}</button>
          <span class="value" :title="item.value">{{ item.value || t("field.empty") }}</span>
          <button type="button" class="reset" :title="t('field.reset')" :aria-label="t('field.reset')" @click="$emit('reset', item.name)">↺</button>
        </li>
      </ul>
      <div v-else class="empty">{{ t("summary.noChanges") }}</div>
    </section>
  </aside>
</template>

<style scoped>
.run-summary {
  display: grid;
  align-content: start;
  gap: 12px;
}

.card {
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}

h2 {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.big {
  color: var(--text-strong);
  font-size: 22px;
  font-weight: 700;
}

.sub {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

dl {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 3px 12px;
  margin: 10px 0 0;
  font-size: 13px;
}

dt {
  color: var(--muted);
}

dd {
  margin: 0;
  color: var(--text-strong);
  text-align: right;
  overflow-wrap: anywhere;
}

.list {
  display: grid;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.list li {
  display: flex;
  align-items: baseline;
  gap: 8px;
  min-width: 0;
  font-size: 13px;
}

.key {
  display: block;
  min-width: 0;
  overflow: hidden;
  padding: 0;
  border: 0;
  background: none;
  box-shadow: none;
  color: var(--text-strong);
  font-family: var(--font-mono);
  font-size: 12.5px;
  text-align: left;
  text-overflow: ellipsis;
}

.key:hover {
  color: var(--primary-text);
  text-decoration: underline;
}

.entry {
  display: grid;
  min-width: 0;
}

.entry .key {
  justify-self: start;
  max-width: 100%;
}

.text {
  color: var(--muted);
  font-size: 12px;
}

.level {
  flex: 0 0 auto;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  transform: translateY(-1px);
}

.level.error {
  background: var(--danger);
}

.level.todo {
  background: var(--primary);
}

.level.warn {
  background: var(--warning);
}

.value {
  max-width: 45%;
  margin-left: auto;
  overflow: hidden;
  color: var(--muted);
  font-family: var(--font-mono);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.reset {
  padding: 0 5px;
  border: 0;
  background: none;
  box-shadow: none;
  color: var(--faint);
  font-size: 13px;
}

.reset:hover {
  background: var(--surface-muted);
  color: var(--text-strong);
}

.empty {
  color: var(--faint);
  font-size: 13px;
}

.blocking {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: 12px;
}
</style>
