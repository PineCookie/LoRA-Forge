<script setup lang="ts">
import { computed } from "vue";
import { RouterLink } from "vue-router";
import PageHeader from "../components/PageHeader.vue";
import RichText from "../components/RichText.vue";
import RunStatusBadge from "../components/RunStatusBadge.vue";
import TaskStatus from "../components/TaskStatus.vue";
import VersionInfo from "../components/VersionInfo.vue";
import { isActive, useRunList, type Run } from "../composables/useRuns";
import { useTasks } from "../composables/useTasks";
import { locale, t, type MessageKey } from "../i18n";
import { modelLabel } from "../monitor/labels";
import { formatDuration } from "../monitor/metrics";
import { TRAINER_PAGES } from "../pages";
import { labelText } from "../router";

interface ModelInfo {
  text: MessageKey;
  /** a --tone-* color */
  tone: string;
  /** facts shown as chips on the featured cards: plain text or message keys */
  facts?: (string | { key: MessageKey })[];
}

const MODELS: Record<string, ModelInfo> = {
  "anima-lora": { text: "home.mode.anima", tone: "network", facts: ["2B", { key: "home.fact.anime" }] },
  "krea2-lora": { text: "home.mode.krea2", tone: "optimize", facts: ["12.9B", { key: "home.fact.16gb" }] },
  "moss-image-lora": { text: "home.mode.moss", tone: "data", facts: ["5B", { key: "home.fact.text" }, { key: "home.fact.new" }] },
  "sdxl-lora": { text: "home.mode.sdxl", tone: "model", facts: ["2.6B"] },
  "flux-lora": { text: "home.mode.flux", tone: "performance" },
  "sd3-lora": { text: "home.mode.sd3", tone: "observe" },
  "sd-lora": { text: "home.mode.sd15", tone: "general" },
  "sdxl-finetune": { text: "home.mode.sdxlFinetune", tone: "save" },
  "sd-dreambooth": { text: "home.mode.sdDreambooth", tone: "save" },
};

const GROUPS: { title?: MessageKey; schemas: string[]; featured?: boolean }[] = [
  { schemas: ["anima-lora", "krea2-lora", "moss-image-lora", "sdxl-lora"], featured: true },
  { title: "home.group.moreLora", schemas: ["flux-lora", "sd3-lora", "sd-lora"] },
  { title: "home.group.finetune", schemas: ["sdxl-finetune", "sd-dreambooth"] },
];

const groups = GROUPS.map((group) => ({
  ...group,
  pages: group.schemas
    .map((schema) => TRAINER_PAGES.find((page) => page.schemaName === schema))
    .filter((page) => page !== undefined)
    .map((page) => ({ path: page.path, title: labelText(page.title), mark: page.mark, ...MODELS[page.schemaName] })),
}));

const factText = (fact: string | { key: MessageKey }) => (typeof fact === "string" ? fact : t(fact.key));

const { activeTask } = useTasks();
const { runs, loaded } = useRunList();
const recent = computed(() => runs.value.filter((run) => !run.preview).slice(0, 5));

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
  <PageHeader :title="t('home.title')">
    <TaskStatus :task="activeTask" />
  </PageHeader>
  <div class="home">
    <div class="models">
      <p class="intro">{{ t("home.intro") }}</p>

      <section v-for="(group, index) in groups" :key="index" class="group">
        <h2 v-if="group.title">{{ t(group.title) }}</h2>
        <div v-if="group.featured" class="cards">
          <RouterLink v-for="page in group.pages" :key="page.path" :to="page.path" class="card" :style="{ '--tone': `var(--tone-${page.tone})` }">
            <span class="card-head">
              <span class="mark" aria-hidden="true">{{ page.mark }}</span>
              <strong>{{ page.title }}</strong>
              <span class="arrow" aria-hidden="true">→</span>
            </span>
            <span class="text">{{ t(page.text) }}</span>
            <span v-if="page.facts" class="facts">
              <span v-for="fact in page.facts" :key="factText(fact)" class="fact">{{ factText(fact) }}</span>
            </span>
          </RouterLink>
        </div>
        <div v-else class="rows">
          <RouterLink v-for="page in group.pages" :key="page.path" :to="page.path" class="row" :style="{ '--tone': `var(--tone-${page.tone})` }">
            <span class="mark small" aria-hidden="true">{{ page.mark }}</span>
            <strong>{{ page.title }}</strong>
            <span class="text">{{ t(page.text) }}</span>
            <span class="arrow" aria-hidden="true">→</span>
          </RouterLink>
        </div>
      </section>
    </div>

    <aside class="side">
      <section v-if="recent.length" class="panel runs">
        <div class="panel-head">
          <h2>{{ t("home.recent") }}</h2>
          <RouterLink to="/runs.html" class="more">{{ t("home.allRuns") }} →</RouterLink>
        </div>
        <ul>
          <li v-for="run in recent" :key="run.id">
            <RouterLink :to="{ path: '/runs.html', query: { run: run.id } }" class="run">
              <span class="run-top">
                <strong>{{ run.output_name || t("runs.unnamed") }}</strong>
                <RunStatusBadge :status="run.status" compact />
              </span>
              <span class="run-meta">
                <span>{{ modelLabel(run.model_train_type) }}</span>
                <span>{{ startedText(run) }}</span>
                <span v-if="run.ended">{{ formatDuration(run.ended - run.started) }}</span>
              </span>
              <span v-if="isActive(run.status)" class="bar" aria-hidden="true"><span :style="{ width: `${stepPercent(run)}%` }" /></span>
            </RouterLink>
          </li>
        </ul>
      </section>

      <section v-else-if="loaded" class="panel start">
        <h2>{{ t("home.start.title") }}</h2>
        <ol>
          <li><RichText :text="t('home.start.dataset')" /></li>
          <li><RichText :text="t('home.start.model')" /></li>
          <li><RichText :text="t('home.start.run')" /></li>
        </ol>
      </section>

      <VersionInfo />
    </aside>
  </div>
</template>

<style scoped>
.home {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  align-items: start;
  gap: 24px;
  max-width: 1320px;
  padding: 20px 24px 48px;
}

@media (max-width: 1100px) {
  .home {
    grid-template-columns: minmax(0, 1fr);
  }
}

.models {
  display: grid;
  gap: 22px;
  min-width: 0;
}

.intro {
  max-width: 760px;
  margin: 0;
  color: var(--muted);
  line-height: 1.55;
}

.group {
  display: grid;
  gap: 10px;
}

h2 {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.mark {
  display: inline-grid;
  flex: 0 0 auto;
  place-items: center;
  width: 38px;
  height: 38px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--tone) 14%, var(--surface));
  color: var(--tone);
  font-size: 14px;
  font-weight: 800;
  letter-spacing: -0.02em;
}

.mark.small {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  font-size: 12px;
}

.arrow {
  margin-left: auto;
  color: var(--faint);
  transition: color 0.16s ease, transform 0.16s ease;
}

/* --- featured models --------------------------------------------------------------------- */

.cards {
  display: grid;
  /* two columns on a desktop: the four featured models as a 2 x 2 block */
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 12px;
}

@media (min-width: 1500px) {
  .cards {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.card {
  display: grid;
  align-content: start;
  gap: 10px;
  padding: 16px 18px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
  transition: border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
}

.card:hover,
.card:focus-visible {
  border-color: color-mix(in srgb, var(--tone) 55%, var(--border));
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
  outline: none;
}

.card:hover .arrow,
.row:hover .arrow {
  color: var(--tone);
  transform: translateX(2px);
}

.card-head {
  display: flex;
  align-items: center;
  gap: 12px;
}

.card-head strong {
  color: var(--text-strong);
  font-size: 17px;
}

.text {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}

.facts {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.fact {
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--surface-sunken);
  color: var(--text);
  font-size: 11.5px;
  font-weight: 600;
}

/* --- other models ------------------------------------------------------------------------ */

.rows {
  display: grid;
  /* one name column for the whole list (rows are subgrids): at least as wide as in the other list,
     wider for a longer name */
  grid-template-columns: auto minmax(170px, max-content) minmax(0, 1fr) auto;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}

.row {
  display: grid;
  grid-column: 1 / -1;
  grid-template-columns: subgrid;
  align-items: center;
  column-gap: 16px;
  padding: 10px 16px;
  transition: background 0.16s ease;
}

@media (max-width: 640px) {
  .row .text {
    visibility: hidden;
  }
}

.row + .row {
  border-top: 1px solid var(--border);
}

.row:hover,
.row:focus-visible {
  background: var(--surface-muted);
  outline: none;
}

.row strong {
  color: var(--text-strong);
}

.row .text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* --- side column ------------------------------------------------------------------------- */

.side {
  display: grid;
  gap: 16px;
  min-width: 0;
}

.side .panel > h2,
.panel-head h2 {
  /* like the System info panel's heading */
  color: var(--text-strong);
  font-size: 17px;
  letter-spacing: 0;
  text-transform: none;
}

.panel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
}

.more {
  color: var(--primary-text);
  font-size: 13px;
  white-space: nowrap;
}

.more:hover {
  text-decoration: underline;
}

.runs ul {
  display: grid;
  gap: 2px;
  margin: 0 -8px;
  padding: 0;
  list-style: none;
}

.run {
  position: relative;
  display: grid;
  gap: 3px;
  overflow: hidden;
  padding: 8px;
  border-radius: var(--radius-sm);
}

.run:hover,
.run:focus-visible {
  background: var(--surface-muted);
  outline: none;
}

.run-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.run-top strong {
  overflow: hidden;
  color: var(--text-strong);
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.run-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 2px 10px;
  color: var(--muted);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.bar {
  position: absolute;
  inset: auto 8px 0 8px;
  height: 2px;
  background: var(--border);
}

.bar span {
  display: block;
  height: 100%;
  background: var(--success);
  transition: width 0.4s ease;
}

.start ol {
  display: grid;
  gap: 10px;
  margin: 0;
  padding-left: 20px;
  color: var(--text);
  font-size: 13px;
  line-height: 1.55;
}

.start li::marker {
  color: var(--primary);
  font-weight: 700;
}
</style>
