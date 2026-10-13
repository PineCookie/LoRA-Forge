<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { RouterLink, useRoute, useRouter } from "vue-router";
import { api } from "../api";
import PageHeader from "../components/PageHeader.vue";
import RunCharts from "../components/RunCharts.vue";
import RunList from "../components/RunList.vue";
import RunStatusBadge from "../components/RunStatusBadge.vue";
import SampleGallery from "../components/SampleGallery.vue";
import VersionInfo from "../components/VersionInfo.vue";
import { confirmAction } from "../composables/useConfirm";
import { isActive, useRun, useRunList, type RunNotice } from "../composables/useRuns";
import { useTasks } from "../composables/useTasks";
import { locale, t } from "../i18n";
import { modelLabel, phaseLabel } from "../monitor/labels";
import { formatBytes, formatDuration, formatRate, formatValue } from "../monitor/metrics";
import { notify } from "../toast";

const route = useRoute();
const router = useRouter();
const { runs, loaded, error: listError, refresh: refreshList } = useRunList();
const { refresh: refreshTasks } = useTasks();

const selectedId = computed(() => (typeof route.query.run === "string" ? route.query.run : null));
const consoleOpen = ref(false);
const { run, progress, samples, checkpoints, output, error, series, version, elapsed, remaining, refresh } = useRun(selectedId, consoleOpen);

function select(id: string) {
  void router.replace({ query: { ...route.query, run: id } });
}

// nothing picked: show the newest run
watch([loaded, runs], () => {
  if (loaded.value && !selectedId.value && runs.value.length) select(runs.value[0].id);
});

// a failed run or a timestep preview is mostly its console output
watch(
  () => run.value && (run.value.status === "failed" || run.value.preview),
  (show) => {
    if (show) consoleOpen.value = true;
  },
);
watch(selectedId, () => (consoleOpen.value = false));

const title = computed(() => {
  if (!run.value) return "";
  return run.value.preview ? t("runs.previewName") : run.value.output_name || t("runs.unnamed");
});

const startedText = computed(() =>
  run.value ? new Intl.DateTimeFormat(locale.value, { dateStyle: "medium", timeStyle: "medium" }).format(new Date(run.value.started * 1000)) : "",
);

function noticeText(notice: RunNotice) {
  if (notice.code === "recompile_limit") return t("runs.notice.recompileLimit", { limit: notice.limit ?? "?" });
  if (notice.code === "dynamic_seq_outside") {
    // a token is a 16x16 px patch; the side of a square image with that many tokens
    const side = Math.round(Math.sqrt(notice.tokens ?? 0) * 16);
    return t("runs.notice.dynamicSeqOutside", { tokens: notice.tokens ?? "?", side, batch: notice.batch ?? "?", max: notice.max ?? "?" });
  }
  if (notice.code === "out_of_memory") {
    return notice.budget ? t("runs.notice.outOfMemoryBudget", { budget: notice.budget }) : t("runs.notice.outOfMemory");
  }
  return notice.text ?? notice.code;
}

const step = computed(() => progress.value?.step ?? null);
const stepPercent = computed(() => (step.value?.total ? Math.min(100, (step.value.n / step.value.total) * 100) : 0));
const active = computed(() => isActive(run.value?.status));

// --- actions -------------------------------------------------------------------------------------

async function stop() {
  if (!run.value || !(await confirmAction(t("tasks.confirmTerminate"), t("tasks.terminate")))) return;
  try {
    await api(`/api/runs/${encodeURIComponent(run.value.id)}/stop`, { method: "POST" });
    notify(t("tasks.terminated"), "success");
  } catch (e) {
    notify(t("tasks.terminateFailed", { error: e instanceof Error ? e.message : String(e) }), "error");
  }
  await Promise.all([refresh(), refreshList(), refreshTasks()]);
}

async function remove() {
  if (!run.value || !(await confirmAction(t("runs.confirmDelete"), t("runs.delete")))) return;
  const id = run.value.id;
  try {
    await api(`/api/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (e) {
    notify(e instanceof Error ? e.message : String(e), "error");
    return;
  }
  await refreshList();
  const next = runs.value.find((item) => item.id !== id);
  void router.replace({ query: next ? { run: next.id } : {} });
}

// --- console -------------------------------------------------------------------------------------

const consoleBox = ref<HTMLElement>();
let stickToBottom = true;

function onConsoleScroll() {
  const box = consoleBox.value;
  if (box) stickToBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
}

watch(output, async () => {
  if (!stickToBottom) return;
  await nextTick();
  if (consoleBox.value) consoleBox.value.scrollTop = consoleBox.value.scrollHeight;
});
watch(consoleOpen, () => (stickToBottom = true));
</script>

<template>
  <PageHeader :title="t('runs.title')" />
  <div class="runs">
    <aside class="list">
      <section class="panel list-panel">
        <p v-if="listError" class="muted">{{ listError }}</p>
        <p v-else-if="loaded && !runs.length" class="muted">{{ t("runs.empty") }}</p>
        <RunList :runs="runs" :selected="selectedId" @select="select" />
      </section>
      <VersionInfo />
    </aside>

    <div class="detail">
      <section v-if="loaded && !runs.length" class="panel empty">
        <h2>{{ t("runs.emptyTitle") }}</h2>
        <p class="muted">{{ t("runs.emptyHint") }}</p>
        <RouterLink class="button" to="/lora/anima.html">{{ t("runs.goTrain") }}</RouterLink>
      </section>

      <p v-if="error" class="panel error">{{ error }}</p>

      <template v-if="run">
        <section class="panel overview">
          <div class="heading">
            <div>
              <h2>
                {{ title }}
                <RunStatusBadge :status="run.status" />
              </h2>
              <p class="meta">
                <span>{{ modelLabel(run.model_train_type) }}</span>
                <span>{{ startedText }}</span>
                <span v-if="run.config_file" :title="t('runs.configFile')"><code>{{ run.config_file }}</code></span>
                <span v-if="run.output_dir" class="path" :title="run.output_dir">{{ run.output_dir }}</span>
              </p>
            </div>
            <div class="actions">
              <button v-if="active" type="button" class="danger" @click="stop">{{ t("trainer.stop") }}</button>
              <button v-else type="button" class="ghost" @click="remove">{{ t("runs.delete") }}</button>
            </div>
          </div>

          <p v-if="run.status === 'failed'" class="alert danger">
            {{ t("runs.failedNote", { code: run.returncode ?? "?" }) }}
          </p>
          <p v-else-if="run.status === 'interrupted'" class="alert warning">{{ t("runs.interruptedNote") }}</p>
          <p v-for="notice in progress?.notices ?? []" :key="notice.code" class="alert warning" :title="notice.text">
            {{ noticeText(notice) }}
          </p>

          <template v-if="!run.preview">
            <div class="progress">
              <div class="progress-label">
                <span v-if="step">{{ t("runs.stepOf", { n: step.n.toLocaleString(), total: step.total.toLocaleString() }) }}</span>
                <span v-else>{{ active ? t("runs.preparing") : t("runs.noProgress") }}</span>
                <span v-if="step">{{ Math.floor(stepPercent) }}%</span>
              </div>
              <div class="track" :class="{ indeterminate: active && !step }">
                <span :style="{ width: `${stepPercent}%` }" />
              </div>
              <p v-if="progress?.phase" class="phase">
                {{ phaseLabel(progress.phase.desc) }} · {{ progress.phase.n.toLocaleString() }} / {{ progress.phase.total.toLocaleString() }}
              </p>
            </div>

            <dl class="stats">
              <div>
                <dt>{{ t("runs.stat.epoch") }}</dt>
                <dd>{{ progress?.epoch ? `${progress.epoch.n} / ${progress.epoch.total}` : "–" }}</dd>
              </div>
              <div>
                <dt>{{ t("runs.stat.speed") }}</dt>
                <dd>{{ formatRate(step?.rate) }}</dd>
              </div>
              <div>
                <dt>{{ t("runs.stat.elapsed") }}</dt>
                <dd>{{ elapsed === null ? "–" : formatDuration(elapsed) }}</dd>
              </div>
              <div>
                <dt>{{ t("runs.stat.remaining") }}</dt>
                <dd>{{ remaining === null ? "–" : formatDuration(remaining) }}</dd>
              </div>
              <div>
                <dt>{{ t("runs.stat.loss") }}</dt>
                <dd>{{ progress?.loss == null ? "–" : formatValue(progress.loss) }}</dd>
              </div>
              <div v-if="progress?.graphs != null" :title="t('runs.stat.graphsTitle')">
                <dt>{{ t("runs.stat.graphs") }}</dt>
                <dd>{{ progress.graphs }}</dd>
              </div>
            </dl>
          </template>
        </section>

        <RunCharts v-if="!run.preview" :series="series" :version="version" :lr-groups="run.lr_groups" />

        <SampleGallery v-if="samples.length" :run-id="run.id" :samples="samples" />

        <section v-if="checkpoints.length" class="panel">
          <h2>{{ t("runs.checkpoints") }}</h2>
          <table class="files">
            <tbody>
              <tr v-for="file in [...checkpoints].reverse()" :key="file.name">
                <td><code>{{ file.name }}</code></td>
                <td class="num">{{ formatBytes(file.size) }}</td>
                <td class="num">{{ new Date(file.time * 1000).toLocaleString(locale) }}</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section class="panel console">
          <button type="button" class="console-toggle" :aria-expanded="consoleOpen" @click="consoleOpen = !consoleOpen">
            <span class="chevron" :class="{ open: consoleOpen }" aria-hidden="true">›</span>
            {{ t("runs.console") }}
          </button>
          <pre v-if="consoleOpen" ref="consoleBox" class="code-block output" @scroll="onConsoleScroll">{{ output.length ? output.join("\n") : t("runs.noOutput") }}</pre>
        </section>
      </template>
    </div>
  </div>
</template>

<style scoped>
.runs {
  display: grid;
  grid-template-columns: 320px minmax(0, 1fr);
  gap: 20px;
  align-items: start;
  max-width: 1720px;
  margin: 0 auto;
  padding: 20px 24px 48px;
}

.list {
  position: sticky;
  top: calc(var(--header-height) + 20px);
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
  max-height: calc(100vh - var(--header-height) - 40px);
  overflow-y: auto;
}

.list-panel {
  padding: 8px;
}

.list-panel > .muted {
  padding: 8px;
}

.detail {
  display: grid;
  /* an explicit column, or the panels grow to their widest content (long output paths) */
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
  min-width: 0;
}

.empty {
  display: grid;
  justify-items: start;
  gap: 10px;
}

.empty h2 {
  margin: 0;
}

.error {
  margin: 0;
  border-color: var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
}

.heading {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.heading > div:first-child {
  flex: 1 1 320px;
  min-width: 0;
}

.heading h2 {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin: 0 0 6px;
  font-size: 20px;
  overflow-wrap: anywhere;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 14px;
  margin: 0;
  color: var(--muted);
  font-size: 13px;
}

.meta > span {
  min-width: 0;
}

.path {
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--font-mono);
  font-size: 12px;
}

.alert {
  margin: 14px 0 0;
  padding: 10px 12px;
  border: 1px solid;
  border-radius: var(--radius-sm);
  font-size: 14px;
}

.alert.danger {
  border-color: var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
}

.alert.warning {
  border-color: var(--warning-border);
  background: var(--warning-soft);
  color: var(--warning);
}

.progress {
  margin-top: 18px;
}

.progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
  color: var(--text-strong);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.track {
  position: relative;
  height: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--surface-sunken);
  box-shadow: inset 0 0 0 1px var(--border);
}

.track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--primary), var(--tone-data));
  transition: width 0.5s ease;
}

.track.indeterminate span {
  width: 30% !important;
  animation: sweep 1.4s ease-in-out infinite;
}

.phase {
  margin: 6px 0 0;
  color: var(--muted);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px;
  margin: 16px 0 0;
}

.stats div {
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface-muted);
}

.stats dt {
  color: var(--muted);
  font-size: 12px;
}

.stats dd {
  margin: 2px 0 0;
  color: var(--text-strong);
  font-size: 18px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

.files {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.files td {
  padding: 7px 4px;
  border-top: 1px solid var(--border);
}

.files tr:first-child td {
  border-top: 0;
}

.files .num {
  color: var(--muted);
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.console {
  padding: 10px 14px;
}

.console-toggle {
  width: 100%;
  justify-content: flex-start;
  padding: 6px 4px;
  border: 0;
  background: transparent;
  box-shadow: none;
  color: var(--text-strong);
  font-weight: 600;
}

.chevron {
  display: inline-block;
  width: 14px;
  color: var(--muted);
  transition: transform 0.15s ease;
}

.chevron.open {
  transform: rotate(90deg);
}

.output {
  max-height: 60vh;
  margin-top: 8px;
  white-space: pre;
  overflow-wrap: normal;
}

@keyframes sweep {
  from {
    transform: translateX(-100%);
  }
  to {
    transform: translateX(340%);
  }
}

@media (max-width: 1000px) {
  .runs {
    grid-template-columns: minmax(0, 1fr);
  }

  .list {
    position: static;
    max-height: none;
  }
}
</style>
