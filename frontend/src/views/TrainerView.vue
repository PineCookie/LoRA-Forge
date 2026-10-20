<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch, watchEffect } from "vue";
import { useRoute } from "vue-router";
import { api, postJson } from "../api";
import ConfigDrawer from "../components/ConfigDrawer.vue";
import DatasetFolders from "../components/DatasetFolders.vue";
import EpochTargets from "../components/EpochTargets.vue";
import FieldTree from "../components/FieldTree.vue";
import FormField, { type FieldMessage } from "../components/FormField.vue";
import FormSection from "../components/FormSection.vue";
import ModelFiles from "../components/ModelFiles.vue";
import PageHeader from "../components/PageHeader.vue";
import ResolutionPresets from "../components/ResolutionPresets.vue";
import RunSummary, { type SpeedEstimate, type SummaryCheck } from "../components/RunSummary.vue";
import SectionIndex from "../components/SectionIndex.vue";
import TaskStatus from "../components/TaskStatus.vue";
import TimestepOffsets from "../components/TimestepOffsets.vue";
import { confirmAction } from "../composables/useConfirm";
import { useConfigFile } from "../composables/useConfigFile";
import { useDatasetInfo } from "../composables/useDatasetInfo";
import { useModelFileStates } from "../composables/useModelFileStates";
import { useScrollSpy } from "../composables/useScrollSpy";
import { useTasks } from "../composables/useTasks";
import { useTrainerForm } from "../composables/useTrainerForm";
import { blocksStart, formChecks, type Check } from "../config/checks";
import { isEssential } from "../config/essentials";
import { formValueKey } from "../config/form";
import { fieldHint, resolutionOf, stepCount, stepsPerEpoch, totalSteps } from "../config/hints";
import { redactConfigForLog } from "../config/normalize";
import { OFFSET_NAME } from "../config/offsets";
import { RESOLUTIONS } from "../config/resolutions";
import { sectionSummary } from "../config/summaries";
import { conditionOf, isFieldVisible } from "../config/visibility";
import { locale, t, tr } from "../i18n";
import { guideUrl } from "../links";
import { TRAINER_PAGES } from "../pages";
import { labelText } from "../router";
import { fieldTree, type FieldNode } from "../schema/fieldTree";
import { MERGED_SECTIONS, mergedSectionOf, type Field, type Group, type MergedSectionId } from "../schema/groups";
import { store } from "../store";
import { notify } from "../toast";

const props = defineProps<{ schemaName: string }>();
const route = useRoute();

const form = useTrainerForm(props.schemaName);
const { groups, values, offsets, datasetDir, datasetRefresh } = form;
const { toml, generatedArgs, exportAs, importFile, copyToml } = useConfigFile(form);
const { activeTask, refresh: refreshTasks, terminate } = useTasks();
const { active: activeSection, refresh: refreshSpy, pin: pinSection } = useScrollSpy("[data-section]");
const dataset = useDatasetInfo(datasetDir, datasetRefresh);
const fileStates = useModelFileStates(props.schemaName, values);

const supportsTimestepPreview = ["flux-lora", "anima-lora"].includes(props.schemaName);
const resolutionSizes = RESOLUTIONS[props.schemaName]?.sizes ?? [];
const guide = TRAINER_PAGES.find((page) => page.schemaName === props.schemaName)?.guide;
const busy = ref(false);
const drawerOpen = ref(false);
const search = ref("");
const searchInput = ref<HTMLInputElement | null>(null);

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));
const allFields = groups.flatMap((group) => group.fields);

// --- the page's sections: each holds some layout groups, each group a tree of options --------------

interface PageGroup {
  group: Group;
  nodes: FieldNode[];
}

interface PageSection {
  id: MergedSectionId;
  groups: PageGroup[];
  fields: Field[];
}

const pageSections: PageSection[] = MERGED_SECTIONS.map(({ id }) => {
  const members = groups.filter((group) => mergedSectionOf(group.title) === id);
  return { id, groups: members.map((group) => ({ group, nodes: fieldTree(group.fields) })), fields: members.flatMap((group) => group.fields) };
}).filter((section) => section.groups.length);
const sectionOfField = new Map(pageSections.flatMap((section) => section.fields.map((field) => [field.name, section.id] as const)));

// --- which options are shown ---------------------------------------------------------------------

const query = computed(() => search.value.trim().toLowerCase());

function matchesSearch(field: Field) {
  if (!query.value) return false;
  const description = field.schema.meta.description ?? "";
  return [field.name, description, tr(description)].some((text) => text.toLowerCase().includes(query.value));
}
const hits = computed(() => new Set(allFields.filter(matchesSearch).map((field) => field.name)));

// an option stays once it was edited, so setting it back to its default does not make it vanish
// under the cursor; "Reset all" starts over
const revealed = reactive(new Set<string>());
watchEffect(() => {
  for (const field of allFields) if (form.isEdited(field.name)) revealed.add(field.name);
});

const sectionKey = (id: string) => `${route.path}:${id}`;
const isExpanded = (id: string) => store.expandedSections.has(sectionKey(id));

/** The options shown: visible for the current values, and common (config/essentials.ts), edited,
 * in a section opened with "+ more", or found by the search (with their parents for context). */
function shownSet(expandAll: boolean): Set<string> {
  const shown = new Set<string>();
  const visit = (node: FieldNode, expanded: boolean): boolean => {
    const name = node.field.name;
    const childShown = node.children.map((child) => visit(child, expanded)).some(Boolean);
    if (!isFieldVisible(name, values.value)) return false;
    const show = query.value ? hits.value.has(name) || childShown : isEssential(name) || revealed.has(name) || expanded || childShown;
    if (show) shown.add(name);
    return show;
  };
  for (const section of pageSections) {
    for (const { nodes } of section.groups) for (const node of nodes) visit(node, expandAll || isExpanded(section.id));
  }
  return shown;
}
const shown = computed(() => shownSet(false));
const shownExpanded = computed(() => shownSet(true));
const isShown = (name: string) => shown.value.has(name);

interface SectionView extends PageSection {
  title: string;
  summary: string;
  edited: number;
  /** options behind "+ more" */
  more: number;
  /** search results hidden by another option's value, with that condition */
  hiddenHits: { name: string; condition: string }[];
  visible: boolean;
}

function conditionText(name: string) {
  const condition = conditionOf(name);
  if (!condition) return "";
  const { controller, values: list, not } = condition;
  if (list.length === 1 && (list[0] === "true" || list[0] === "false")) {
    return t((list[0] === "true") !== not ? "trainer.whenOn" : "trainer.whenOff", { name: controller });
  }
  return t(not ? "trainer.whenNot" : "trainer.whenEquals", { name: controller, values: list.join(" / ") });
}

// --- timestep offsets per dataset subfolder (FLUX / Anima) --------------------------------------------

// Offsets belong to one dataset folder: drop them for another folder or for subfolders that no longer
// exist (/api/run rejects unknown folders, and their inputs would be gone)
watch(dataset.info, (result) => {
  const dir = datasetDir.value.trim();
  if (!result) {
    if (!dir) offsets.value = { dir: null, values: {} };
    return;
  }
  const names = new Set(result.subdirs.map((subdir) => subdir.name));
  const kept = offsets.value.dir === dir ? Object.entries(offsets.value.values).filter(([name]) => names.has(name)) : [];
  offsets.value = { dir, values: Object.fromEntries(kept) };
});
const offsetsAvailable = computed(() => supportsTimestepPreview && !!dataset.info.value?.subdirs.length);
const offsetsSet = computed(() => Object.values(offsets.value.values).some((value) => Number(value) !== 0));
// a row behind the section's "+ more", like the other tuning options; set offsets stay shown
const offsetsShown = computed(() => {
  if (!offsetsAvailable.value) return false;
  if (query.value) return [OFFSET_NAME, t("dataset.offsetNote")].some((text) => text.toLowerCase().includes(query.value));
  return isExpanded("timesteps") || offsetsSet.value;
});

const sections = computed<SectionView[]>(() =>
  pageSections.map((section) => {
    const names = section.fields.map((field) => field.name);
    const offsetsHere = section.id === "timesteps" && offsetsAvailable.value;
    return {
      ...section,
      title: t(`trainer.merged.${section.id}`),
      summary: sectionSummary(section.id, values.value),
      edited: names.filter(form.isEdited).length,
      more: names.filter((name) => shownExpanded.value.has(name) && !shown.value.has(name)).length + (offsetsHere && !offsetsShown.value ? 1 : 0),
      hiddenHits: query.value ? names.filter((name) => hits.value.has(name) && !isFieldVisible(name, values.value)).map((name) => ({ name, condition: conditionText(name) })) : [],
      visible: !query.value || names.some(isShown) || names.some((name) => hits.value.has(name)) || (offsetsHere && offsetsShown.value),
    };
  }),
);
const shownSections = computed(() => sections.value.filter((section) => section.visible));
const matchCount = computed(() => allFields.filter((field) => hits.value.has(field.name) && isShown(field.name)).length);
const groupShown = (group: PageGroup) => group.nodes.some((node) => isShown(node.field.name));
const multipleGroups = (section: SectionView) => section.groups.filter(groupShown).length > 1;

// Krea 2 and MOSS-Image do not train through sd-scripts
const SD_SCRIPTS_PAGE = !["krea2-lora", "moss-image-lora"].includes(props.schemaName);

function groupNote(group: Group) {
  if (group.title === "torch.compile") return SD_SCRIPTS_PAGE ? t("trainer.note.compile") : undefined;
  if (group.title !== "Timesteps and loss") return undefined;
  if (props.schemaName === "anima-lora") return t("trainer.note.lossAnima");
  const names = new Set(group.fields.map((field) => field.name));
  if (names.has("min_snr_gamma") && names.has("debiased_estimation_loss")) return t("trainer.note.lossMinSnr");
  return undefined;
}

// --- what the page knows about the values ------------------------------------------------------------

const imagesWithRepeats = computed(() => dataset.info.value?.total_images_with_repeats ?? null);
const perEpoch = computed(() => stepsPerEpoch(values.value, imagesWithRepeats.value));
const total = computed(() => totalSteps(values.value, imagesWithRepeats.value));

const checks = computed(() =>
  formChecks({
    schemaName: props.schemaName,
    values: values.value,
    datasetError: dataset.error.value || null,
    datasetSubfolders: dataset.loading.value ? null : (dataset.info.value?.subdirs.length ?? null),
    fileStates,
  }),
);
const checkText = (check: Check) => t(check.key, check.params);
const LEVEL_ORDER = { error: 0, todo: 1, warn: 2 } as const;
const checkByField = computed(() => {
  const map = new Map<string, Check>();
  for (const check of checks.value) {
    const current = map.get(check.field);
    if (!current || LEVEL_ORDER[check.level] < LEVEL_ORDER[current.level]) map.set(check.field, check);
  }
  return map;
});
const blocking = computed(() => checks.value.filter(blocksStart));

function fieldCheck(name: string): FieldMessage | null {
  const check = checkByField.value.get(name);
  return check ? { text: checkText(check), tone: check.level } : null;
}

function fieldHintMessage(name: string): FieldMessage | null {
  const hint = fieldHint(name, { values: values.value, imagesWithRepeats: imagesWithRepeats.value, fileStates });
  return hint ? { text: t(hint.key, hint.params), tone: hint.tone } : null;
}

const shortValue = (value: string, length = 40) => (value.length > length ? `${value.slice(0, length - 1)}…` : value);

function fieldNote(name: string) {
  const edit = form.coupled.value;
  if (!edit || edit.field !== name) return null;
  const list = edit.changed.map((other) => `${other} → ${shortValue(formValueKey(values.value[other]), 24) || t("field.empty")}`).join(", ");
  return t("field.alsoChanged", { list });
}

// --- the run summary -------------------------------------------------------------------------------

interface RunSpeed {
  output_name: string;
  seconds_per_step: number;
  train_batch_size?: number | null;
  resolution?: string | null;
}
const speed = ref<RunSpeed | null>(null);
onMounted(async () => {
  try {
    speed.value = (await api<{ speed: RunSpeed | null }>(`/api/runs/speed?model_train_type=${encodeURIComponent(props.schemaName)}`)).speed;
  } catch {
    speed.value = null;
  }
});

// the last run's speed, scaled by the batch size and the pixels per image (a rough estimate)
const estimate = computed<SpeedEstimate | null>(() => {
  if (!speed.value || total.value === null) return null;
  const batch = Number(values.value.train_batch_size) || 1;
  const runBatch = Number(speed.value.train_batch_size) || 1;
  const size = resolutionOf(values.value.resolution);
  const runSize = resolutionOf(speed.value.resolution ?? "") ?? size;
  const pixels = size && runSize ? (size[0] * size[1]) / (runSize[0] * runSize[1]) : 1;
  const secondsPerStep = speed.value.seconds_per_step * (batch / runBatch) * pixels;
  return { seconds: secondsPerStep * total.value, secondsPerStep, from: speed.value.output_name };
});

const summaryChecks = computed<SummaryCheck[]>(() => checks.value.map((check) => ({ field: check.field, level: check.level, text: checkText(check) })));
const changed = computed(() => allFields.filter((field) => form.isEdited(field.name)).map((field) => ({ name: field.name, value: shortValue(formValueKey(values.value[field.name])) })));
const checkpoints = computed(() => {
  const every = Number(values.value.save_every_n_epochs);
  const epochs = Number(values.value.max_train_epochs);
  return every > 0 && epochs > 0 ? Math.max(1, Math.floor(epochs / every)) : null;
});
const previews = computed(() => (values.value.enable_preview === true ? t("summary.everyEpochs", { count: String(values.value.sample_every_n_epochs || 1) }) : t("summary.off")));
const learningRate = computed(() => {
  if (!("lr_scheduler" in values.value)) return "";
  if (isFieldVisible("prodigyplus_d_coef", values.value)) return t("summary.scheduleFree");
  const warmup = total.value !== null && isFieldVisible("lr_warmup_steps", values.value) ? stepCount(values.value.lr_warmup_steps, total.value) : 0;
  return `${values.value.lr_scheduler}${warmup ? `, ${t("summary.warmup", { steps: warmup.toLocaleString() })}` : ""}`;
});

// --- sections: collapse, "+ more", jumping to an option ---------------------------------------------

const isCollapsed = (id: string) => !query.value && store.collapsedSections.has(sectionKey(id));

function toggleSection(id: string) {
  const key = sectionKey(id);
  if (store.collapsedSections.has(key)) store.collapsedSections.delete(key);
  else store.collapsedSections.add(key);
  refreshSpy();
}

function toggleMore(id: string) {
  const key = sectionKey(id);
  if (store.expandedSections.has(key)) store.expandedSections.delete(key);
  else store.expandedSections.add(key);
}

function setAllCollapsed(collapsed: boolean) {
  for (const { id } of pageSections) {
    if (collapsed) store.collapsedSections.add(sectionKey(id));
    else store.collapsedSections.delete(sectionKey(id));
  }
  refreshSpy();
}

async function jumpTo(id: string) {
  store.collapsedSections.delete(sectionKey(id));
  pinSection(id);
  await nextTick();
  document.getElementById(`section-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Scroll to an option (from the run summary), opening its section and "+ more" when needed. */
async function goToField(name: string) {
  const section = sectionOfField.get(name);
  if (!section) return;
  search.value = "";
  store.collapsedSections.delete(sectionKey(section));
  if (!isShown(name)) store.expandedSections.add(sectionKey(section));
  await nextTick();
  // after the rows have slid open
  setTimeout(() => {
    const row = document.getElementById(`option-${name}`);
    if (!row) return;
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    row.classList.remove("flash");
    void row.offsetWidth;
    row.classList.add("flash");
    document.getElementById(`field-${name}`)?.focus({ preventScroll: true });
  }, 240);
}

// "/" jumps to the search, as on many sites
function onKeydown(event: KeyboardEvent) {
  const target = event.target as HTMLElement | null;
  if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
  if (target && (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)) return;
  event.preventDefault();
  searchInput.value?.focus();
}
onMounted(() => window.addEventListener("keydown", onKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));

// a search shortens the page: start at its results, not somewhere below them
watch(query, () => window.scrollTo({ top: 0 }));

// another filter changes which sections exist: find the current one again
watch(
  () => shownSections.value.map((section) => section.id).join(),
  () => refreshSpy(),
  { flush: "post" },
);
const indexItems = computed(() =>
  shownSections.value.map(({ id, title, edited, fields }) => ({ id, title, edited, blocking: blocking.value.filter((check) => fields.some((field) => field.name === check.field)).length })),
);

// --- actions -------------------------------------------------------------------------------------

async function startRun(previewTimesteps: boolean) {
  if (busy.value || blocking.value.length) return;
  const runConfig = form.runConfig(previewTimesteps);
  console.log(previewTimesteps ? "Previewing timesteps with config:" : "Form submitted with config:", redactConfigForLog(runConfig));
  busy.value = true;
  try {
    notify(t(previewTimesteps ? "trainer.previewing" : "trainer.starting"));
    const result = await postJson("/api/run", runConfig);
    notify(result.message || t(previewTimesteps ? "trainer.previewSubmitted" : "trainer.submitted"), "success");
  } catch (error) {
    console.error("Error:", error);
    notify(errorText(error), "error");
  } finally {
    busy.value = false;
    void refreshTasks();
  }
}

async function pickPath(name: string, pickerType: string) {
  try {
    const data = await api<{ path: string }>(`/api/pick_file?picker_type=${encodeURIComponent(pickerType)}`);
    form.setValue(name, data.path);
  } catch (error) {
    notify(errorText(error), "error");
  }
}

function onImport(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) void importFile(file);
  input.value = ""; // allow importing the same file again
}

async function resetAll() {
  if (!(await confirmAction(t("trainer.confirmReset"), t("trainer.reset")))) return;
  form.reset();
  revealed.clear();
}

const startTitle = computed(() => (blocking.value.length ? t("trainer.startBlocked", { count: blocking.value.length }) : undefined));
</script>

<template>
  <section v-if="!form.schema" class="panel missing">{{ t("trainer.schemaMissing", { name: schemaName }) }}</section>

  <template v-else>
    <PageHeader :title="labelText(route.meta.title)">
      <template v-if="guide" #title>
        <a class="guide" :href="guideUrl(guide, locale)" target="_blank" rel="noopener">{{ t("trainer.guide") }} ↗</a>
      </template>
      <div class="button-group config-actions">
        <label class="button">
          {{ t("trainer.import") }}
          <input type="file" accept=".json,.toml,application/json,text/plain" @change="onImport" />
        </label>
        <button type="button" @click="exportAs('json')">{{ t("trainer.exportJson") }}</button>
        <button type="button" @click="exportAs('toml')">{{ t("trainer.exportToml") }}</button>
        <button type="button" @click="drawerOpen = true">{{ t("trainer.tomlPreview") }}</button>
      </div>
      <TaskStatus :task="activeTask" />
      <button v-if="supportsTimestepPreview" type="button" :disabled="busy || !!activeTask || blocking.length > 0" :title="startTitle" @click="startRun(true)">
        {{ t("trainer.previewTimesteps") }}
      </button>
      <button v-if="activeTask" type="button" class="danger" @click="terminate(activeTask)">{{ t("trainer.stop") }}</button>
      <button type="button" class="primary start" :disabled="busy || !!activeTask || blocking.length > 0" :title="startTitle" @click="startRun(false)">
        {{ t(busy ? "trainer.submitting" : "trainer.start") }}
      </button>
    </PageHeader>

    <div class="trainer">
      <SectionIndex
        class="index"
        :items="indexItems"
        :active="activeSection"
        @jump="jumpTo"
        @expand-all="setAllCollapsed(false)"
        @collapse-all="setAllCollapsed(true)"
      />

      <main class="sections">
        <div class="toolbar">
          <input
            ref="searchInput"
            v-model="search"
            class="search"
            type="search"
            :placeholder="t('trainer.search')"
            :aria-label="t('trainer.searchLabel')"
            @keydown.esc="search = ''"
          />
          <kbd v-if="!search" class="shortcut" aria-hidden="true">/</kbd>
          <span v-else class="count">{{ t("trainer.matchCount", { count: matchCount }) }}</span>
          <button type="button" class="ghost small" @click="resetAll">{{ t("trainer.reset") }}</button>
        </div>

        <FormSection
          v-for="section in shownSections"
          :key="section.id"
          :section-id="section.id"
          :title="section.title"
          :summary="section.summary"
          :edited-count="section.edited"
          :collapsed="isCollapsed(section.id)"
          @toggle="toggleSection(section.id)"
        >
          <template v-for="group in section.groups" :key="group.group.title">
            <div v-if="multipleGroups(section)" v-show="groupShown(group)" class="subhead">
              <span>{{ tr(group.group.title) }}</span>
              <small v-if="groupNote(group.group)">{{ groupNote(group.group) }}</small>
            </div>
            <FieldTree v-for="node in group.nodes" :key="node.field.name" :node="node" :shown="isShown">
              <template #field="{ field }">
                <FormField
                  :field="field"
                  :model-value="values[field.name]"
                  :edited="form.isEdited(field.name)"
                  :default-text="shortValue(form.initialText(field.name))"
                  :hint="fieldHintMessage(field.name)"
                  :check="fieldCheck(field.name)"
                  :note="fieldNote(field.name)"
                  :context="!!query && !hits.has(field.name)"
                  @update:model-value="form.setValue(field.name, $event)"
                  @reset="form.resetField(field.name)"
                  @pick="pickPath(field.name, $event)"
                  @undo="form.undoCoupled()"
                >
                  <DatasetFolders v-if="field.name === 'train_data_dir'" :info="dataset.info.value" :loading="dataset.loading.value" />
                  <ResolutionPresets
                    v-if="field.name === 'resolution' && resolutionSizes.length"
                    :sizes="resolutionSizes"
                    :value="String(values.resolution ?? '')"
                    @set="form.setValue('resolution', $event)"
                  />
                  <EpochTargets
                    v-if="field.name === 'max_train_epochs' && perEpoch"
                    :steps-per-epoch="perEpoch"
                    :epochs="String(values.max_train_epochs ?? '')"
                    @set="form.setValue('max_train_epochs', $event)"
                  />
                </FormField>
              </template>
            </FieldTree>
            <ModelFiles v-if="group.group.title === 'Model' && !query" :schema-name="schemaName" :values="values" :states="fileStates" />
            <TimestepOffsets
              v-if="group.group.title === 'Timesteps and loss' && offsetsShown && dataset.info.value"
              v-model:offsets="offsets"
              :info="dataset.info.value"
              :timestep-sampling="String(values.timestep_sampling ?? 'sigmoid')"
            />
          </template>

          <template #footer>
            <template v-if="query">
              <span v-if="section.hiddenHits.length" class="hidden-hits">
                {{ t("trainer.alsoMatches") }}
                <span v-for="(hit, index) in section.hiddenHits" :key="hit.name">
                  <code>{{ hit.name }}</code> ({{ hit.condition }}){{ index < section.hiddenHits.length - 1 ? "," : "" }}
                </span>
              </span>
            </template>
            <button v-else-if="section.more" type="button" class="more" @click="toggleMore(section.id)">
              {{ t(section.more === 1 ? "trainer.moreOption" : "trainer.moreOptions", { count: section.more }) }}
            </button>
            <button v-else-if="isExpanded(section.id)" type="button" class="more" @click="toggleMore(section.id)">{{ t("trainer.fewerOptions") }}</button>
          </template>
        </FormSection>

        <p v-if="!shownSections.length" class="panel empty">{{ t("trainer.noMatch", { query: search }) }}</p>
      </main>

      <RunSummary
        class="summary"
        :total-steps="total"
        :steps-per-epoch="perEpoch"
        :epochs="String(values.max_train_epochs ?? '')"
        :images="dataset.info.value?.total_images ?? null"
        :images-with-repeats="imagesWithRepeats"
        :dataset-loading="dataset.loading.value"
        :estimate="estimate"
        :checkpoints="checkpoints"
        :previews="previews"
        :learning-rate="learningRate"
        :checks="summaryChecks"
        :changed="changed"
        @go="goToField"
        @reset="form.resetField($event)"
      />
    </div>

    <ConfigDrawer
      v-model:open="drawerOpen"
      :toml="toml"
      :generated-args="generatedArgs"
      @copy="copyToml"
    />
  </template>
</template>

<style scoped>
.missing {
  margin: 24px;
  border-color: var(--danger-border);
  background: var(--danger-soft);
}

.trainer {
  display: grid;
  grid-template-columns: 200px minmax(0, 940px) 300px;
  justify-content: center;
  gap: 24px;
  align-items: start;
  /* room below the last section, so sections near the end can still scroll to the top */
  padding: 16px 24px calc(var(--viewport-height) * 0.4);
}

.sections {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 12px;
  min-width: 0;
}

.summary {
  position: sticky;
  top: calc(var(--header-height) + 16px);
}

.toolbar {
  position: sticky;
  top: var(--header-height);
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: -16px 0 0;
  padding: 12px 0 8px;
  background: var(--bg);
}

.search {
  flex: 1 1 auto;
  min-width: 0;
  padding-left: 34px;
  background: var(--surface)
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23667085' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.5-3.5'/%3E%3C/svg%3E")
    no-repeat 11px center;
}

.shortcut {
  padding: 1px 7px;
  border: 1px solid var(--border-strong);
  border-radius: 5px;
  color: var(--faint);
  font-family: var(--font-mono);
  font-size: 12px;
}

.count {
  color: var(--muted);
  font-size: 13px;
  white-space: nowrap;
}

.subhead {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 2px 12px;
  padding: 16px 18px 6px;
  border-top: 1px solid var(--border);
}

.subhead span {
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.subhead small {
  color: var(--faint);
  font-size: 12px;
}

/* the subheading's line is the first option's line */
.subhead + :deep(.reveal > .inner > .field) {
  border-top: 0;
}

.more {
  padding: 4px 0;
  border: 0;
  background: none;
  box-shadow: none;
  color: var(--primary-text);
  font-size: 13px;
  font-weight: 600;
}

.more:hover {
  text-decoration: underline;
}

.hidden-hits {
  color: var(--muted);
  font-size: 12.5px;
}

.guide {
  color: var(--primary-text);
  font-size: 14px;
  font-weight: 600;
  white-space: nowrap;
}

.guide:hover {
  text-decoration: underline;
}

.config-actions {
  margin-right: 6px;
}

.config-actions input {
  display: none;
}

.empty {
  color: var(--muted);
  text-align: center;
}

/* the section index goes first when space runs short, so the form keeps its width */
@container workspace (max-width: 1360px) {
  .trainer {
    grid-template-columns: minmax(0, 940px) 280px;
  }

  .index {
    display: none;
  }
}

@container workspace (max-width: 780px) {
  .trainer {
    grid-template-columns: minmax(0, 1fr);
    padding: 12px 16px calc(var(--viewport-height) * 0.4);
  }

  .summary {
    position: static;
    order: -1;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  }

  .toolbar {
    margin: 0;
  }
}
</style>
