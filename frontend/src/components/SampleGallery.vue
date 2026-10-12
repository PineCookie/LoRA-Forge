<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import type { Sample } from "../composables/useRuns";
import { t } from "../i18n";

const props = defineProps<{ runId: string; samples: Sample[] }>();

const url = (sample: Sample) => `/api/runs/${encodeURIComponent(props.runId)}/samples/${encodeURIComponent(sample.name)}`;

const label = (sample: Sample) =>
  sample.epoch !== undefined
    ? t("runs.sampleEpoch", { n: sample.epoch })
    : sample.step !== undefined
      ? t("runs.sampleStep", { n: sample.step })
      : t("runs.sampleOther");

// the newest samples first, grouped by the step / epoch they were made at
const groups = computed(() => {
  const byLabel = new Map<string, Sample[]>();
  for (const sample of [...props.samples].sort((a, b) => b.time - a.time)) {
    const key = label(sample);
    if (!byLabel.has(key)) byLabel.set(key, []);
    byLabel.get(key)!.push(sample);
  }
  return [...byLabel].map(([title, items]) => ({ title, items: items.sort((a, b) => (a.prompt ?? 0) - (b.prompt ?? 0)) }));
});

const ordered = computed(() => groups.value.flatMap((group) => group.items));
const dialog = ref<HTMLDialogElement>();
const current = ref<Sample | null>(null);

async function open(sample: Sample) {
  current.value = sample;
  await nextTick();
  if (!dialog.value?.open) dialog.value?.showModal();
}

function step(delta: number) {
  if (!current.value) return;
  const index = ordered.value.findIndex((sample) => sample.name === current.value!.name);
  const next = ordered.value[index + delta];
  if (next) current.value = next;
}

function onKey(event: KeyboardEvent) {
  if (event.key === "ArrowLeft") step(-1);
  if (event.key === "ArrowRight") step(1);
}

function onDialogClick(event: MouseEvent) {
  if (event.target === dialog.value) dialog.value?.close();
}
</script>

<template>
  <section class="panel">
    <h2>{{ t("runs.samples") }} <span class="count">{{ samples.length }}</span></h2>
    <div v-for="group in groups" :key="group.title" class="group">
      <h3>{{ group.title }}</h3>
      <div class="thumbs">
        <button v-for="sample in group.items" :key="sample.name" type="button" class="thumb" :title="sample.name" @click="open(sample)">
          <img :src="url(sample)" :alt="sample.name" loading="lazy" />
        </button>
      </div>
    </div>

    <dialog ref="dialog" class="lightbox" @click="onDialogClick" @keydown="onKey" @close="current = null">
      <figure v-if="current">
        <img :src="url(current)" :alt="current.name" />
        <figcaption>
          <button type="button" class="ghost" :aria-label="t('runs.previous')" @click="step(-1)">‹</button>
          <span>{{ label(current) }} · {{ current.name }}</span>
          <button type="button" class="ghost" :aria-label="t('runs.next')" @click="step(1)">›</button>
          <button type="button" class="ghost" :aria-label="t('drawer.close')" @click="dialog?.close()">×</button>
        </figcaption>
      </figure>
    </dialog>
  </section>
</template>

<style scoped>
h2 {
  display: flex;
  align-items: center;
  gap: 8px;
}

.count {
  color: var(--faint);
  font-size: 13px;
  font-weight: 500;
}

h3 {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
}

.group + .group {
  margin-top: 14px;
}

.thumbs {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 8px;
}

.thumb {
  display: block;
  padding: 0;
  overflow: hidden;
  aspect-ratio: 1;
  border-color: var(--border);
  background: var(--surface-sunken);
}

.thumb img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.2s ease;
}

.thumb:hover img {
  transform: scale(1.04);
}

.lightbox {
  max-width: min(96vw, 1400px);
  max-height: 96vh;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-lg);
}

.lightbox::backdrop {
  background: var(--backdrop);
}

.lightbox figure {
  display: grid;
  margin: 0;
}

.lightbox img {
  display: block;
  max-width: 100%;
  max-height: calc(96vh - 52px);
  margin: 0 auto;
  object-fit: contain;
}

.lightbox figcaption {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 13px;
}

.lightbox figcaption span {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
