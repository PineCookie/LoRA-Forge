<script setup lang="ts">
import { computed, nextTick, reactive, ref, watch, watchEffect } from "vue";
import { useRoute } from "vue-router";
import { api, postJson } from "../api";
import ConfigDrawer from "../components/ConfigDrawer.vue";
import DatasetStats from "../components/DatasetStats.vue";
import FormField from "../components/FormField.vue";
import FormSection from "../components/FormSection.vue";
import PageHeader from "../components/PageHeader.vue";
import SectionIndex from "../components/SectionIndex.vue";
import TaskStatus from "../components/TaskStatus.vue";
import { confirmAction } from "../composables/useConfirm";
import { useConfigFile } from "../composables/useConfigFile";
import { optionView, setOptionView, type OptionView } from "../composables/useOptionView";
import { useScrollSpy } from "../composables/useScrollSpy";
import { useTasks } from "../composables/useTasks";
import { useTrainerForm } from "../composables/useTrainerForm";
import { isEssential } from "../config/essentials";
import { redactConfigForLog } from "../config/normalize";
import { isFieldVisible } from "../config/visibility";
import { t, tr } from "../i18n";
import { labelText } from "../router";
import { groupTone, MERGED_SECTIONS, mergedSectionOf, type Field, type Group, type Tone } from "../schema/groups";
import { store } from "../store";
import { notify } from "../toast";

const props = defineProps<{ schemaName: string }>();
const route = useRoute();

const form = useTrainerForm(props.schemaName);
const { groups, values, offsets, datasetDir, datasetRefresh } = form;
const { toml, generatedArgs, exportAs, importFile, copyToml } = useConfigFile(form);
const { activeTask, refresh: refreshTasks, terminate } = useTasks();
const { active: activeSection, refresh: refreshSpy, pin: pinSection } = useScrollSpy("[data-section]");

const supportsTimestepPreview = ["flux-lora", "anima-lora"].includes(props.schemaName);
const busy = ref(false);
const drawerOpen = ref(false);
const search = ref("");
const changedOnly = ref(false);

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

// --- which fields and sections are shown ---------------------------------------------------

const query = computed(() => search.value.trim().toLowerCase());
const filtering = computed(() => Boolean(query.value) || changedOnly.value);

function matchesSearch(field: Field) {
  if (!query.value) return true;
  const description = field.schema.meta.description ?? "";
  return [field.name, description, tr(description)].some((text) => text.toLowerCase().includes(query.value));
}

// Essentials hides the less common options, but never one that differs from its default; a search
// always looks through every option
const essentialsOnly = computed(() => optionView.value === "essentials" && !query.value);
// an option stays once it was edited, so setting it back to its default does not make it vanish
// under the cursor; "Reset all" starts over
const revealed = reactive(new Set<string>());
watchEffect(() => {
  for (const group of groups) for (const field of group.fields) if (form.isEdited(field.name)) revealed.add(field.name);
});
const inView = (field: Field) => !essentialsOnly.value || isEssential(field.name) || revealed.has(field.name);

const fieldShown = (field: Field) =>
  isFieldVisible(field.name, values.value) && matchesSearch(field) && (!changedOnly.value || form.isEdited(field.name)) && inView(field);

const hiddenCount = computed(() =>
  essentialsOnly.value ? groups.flatMap((group) => group.fields).filter((field) => isFieldVisible(field.name, values.value) && !inView(field)).length : 0,
);

const VIEWS: { view: OptionView; label: "trainer.view.essentials" | "trainer.view.all" }[] = [
  { view: "essentials", label: "trainer.view.essentials" },
  { view: "all", label: "trainer.view.all" },
];

interface DisplaySection {
  id: string;
  /** shown as is (already translated) */
  title: string;
  tone: Tone;
  fields: Field[];
  edited: number;
  notes: string[];
  /** holds the dataset folder, so the dataset panel goes here */
  dataset: boolean;
}

function sectionNote(group: Group) {
  if (group.title === "Learning rate and optimizer") return t("trainer.note.optimizer");
  if (group.title === "torch.compile / CUDA performance") return t("trainer.note.compile");
  if (group.title !== "Loss") return undefined;
  if (props.schemaName === "anima-lora") return t("trainer.note.lossAnima");
  const names = new Set(group.fields.map((field) => field.name));
  if (names.has("min_snr_gamma") && names.has("debiased_estimation_loss")) return t("trainer.note.lossMinSnr");
  return undefined;
}

const notesOf = (groupsShown: Group[]) => groupsShown.map(sectionNote).filter((note): note is string => Boolean(note));

// All options: the schema's own sections. Essentials: their few options merged into six broader
// sections, so the page is not a stack of one-option sections.
const sections = computed<DisplaySection[]>(() => {
  const shown = groups.map((group) => ({ group, fields: group.fields.filter(fieldShown) })).filter((entry) => entry.fields.length);
  if (!essentialsOnly.value) {
    return shown.map(({ group, fields }) => ({
      id: `g${groups.indexOf(group)}`,
      title: tr(group.title),
      tone: groupTone(group.title),
      fields,
      edited: form.editedCount(group),
      notes: notesOf([group]),
      dataset: group.title === "Dataset",
    }));
  }
  return MERGED_SECTIONS.map(({ id, tone }) => {
    const members = shown.filter(({ group }) => mergedSectionOf(group.title) === id);
    return {
      id: `e-${id}`,
      title: t(`trainer.merged.${id}`),
      tone,
      fields: members.flatMap((member) => member.fields),
      edited: members.reduce((count, member) => count + form.editedCount(member.group), 0),
      notes: notesOf(members.map((member) => member.group)),
      dataset: members.some((member) => member.group.title === "Dataset"),
    };
  }).filter((section) => section.fields.length);
});
// another view or filter changes the section ids: find the current one again
watch(
  () => sections.value.map((section) => section.id).join(),
  () => refreshSpy(),
  { flush: "post" },
);
const shownCount = computed(() => sections.value.reduce((count, section) => count + section.fields.length, 0));
const indexItems = computed(() => sections.value.map(({ id, title, tone, edited }) => ({ id, title, tone, edited })));

// collapsed sections survive navigation between pages; filters always show everything they match
const sectionKey = (id: string) => `${route.path}:${id}`;
const isCollapsed = (id: string) => !filtering.value && store.collapsedSections.has(sectionKey(id));

function toggleSection(id: string) {
  const key = sectionKey(id);
  if (store.collapsedSections.has(key)) store.collapsedSections.delete(key);
  else store.collapsedSections.add(key);
  refreshSpy();
}

function setAllCollapsed(collapsed: boolean) {
  for (const { id } of sections.value) {
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

// --- actions -------------------------------------------------------------------------------

async function startRun(previewTimesteps: boolean) {
  if (busy.value) return;
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
</script>

<template>
  <section v-if="!form.schema" class="panel missing">{{ t("trainer.schemaMissing", { name: schemaName }) }}</section>

  <template v-else>
    <PageHeader :title="labelText(route.meta.title)">
      <div class="config-actions">
        <label class="button">
          {{ t("trainer.import") }}
          <input type="file" accept=".json,.toml,application/json,text/plain" @change="onImport" />
        </label>
        <button type="button" @click="exportAs('json')">{{ t("trainer.exportJson") }}</button>
        <button type="button" @click="exportAs('toml')">{{ t("trainer.exportToml") }}</button>
        <button type="button" @click="drawerOpen = true">{{ t("trainer.tomlPreview") }}</button>
      </div>
      <TaskStatus :task="activeTask" />
      <button v-if="supportsTimestepPreview" type="button" :disabled="busy || !!activeTask" @click="startRun(true)">
        {{ t("trainer.previewTimesteps") }}
      </button>
      <button v-if="activeTask" type="button" class="danger" @click="terminate(activeTask)">{{ t("trainer.stop") }}</button>
      <button type="button" class="primary start" :disabled="busy || !!activeTask" @click="startRun(false)">
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

      <div class="sections">
        <div class="toolbar">
          <div class="view-switch" role="group" :aria-label="t('trainer.view.label')">
            <button
              v-for="item in VIEWS"
              :key="item.view"
              type="button"
              :class="{ selected: optionView === item.view }"
              :aria-pressed="optionView === item.view"
              @click="setOptionView(item.view)"
            >
              {{ t(item.label) }}
            </button>
          </div>
          <input v-model="search" class="search" type="search" :placeholder="t('trainer.search')" :aria-label="t('trainer.searchLabel')" />
          <span v-if="filtering" class="count">{{ t("trainer.matchCount", { count: shownCount }) }}</span>
          <label class="changed-only">
            <input v-model="changedOnly" type="checkbox" />
            {{ t("trainer.changedOnly") }}
          </label>
          <button type="button" class="ghost small reset-all" @click="resetAll">{{ t("trainer.reset") }}</button>
        </div>

        <FormSection
          v-for="section in sections"
          :key="section.id"
          :section-id="section.id"
          :title="section.title"
          :tone="section.tone"
          :edited-count="section.edited"
          :notes="section.notes"
          :collapsed="isCollapsed(section.id)"
          @toggle="toggleSection(section.id)"
        >
          <FormField
            v-for="field in section.fields"
            :key="field.name"
            :field="field"
            :model-value="values[field.name]"
            :edited="form.isEdited(field.name)"
            @update:model-value="form.setValue(field.name, $event)"
            @reset="form.resetField(field.name)"
            @pick="pickPath(field.name, $event)"
          />
          <DatasetStats
            v-if="section.dataset"
            v-model:offsets="offsets"
            :dir="datasetDir"
            :batch-size="form.numberValue('train_batch_size')"
            :grad-acc="form.numberValue('gradient_accumulation_steps')"
            :epochs="form.numberValue('max_train_epochs')"
            :supports-offsets="supportsTimestepPreview"
            :timestep-sampling="String(values.timestep_sampling ?? 'sigmoid')"
            :refresh-key="datasetRefresh"
          />
        </FormSection>

        <p v-if="hiddenCount && !changedOnly" class="more">
          {{ t("trainer.hiddenCount", { count: hiddenCount }) }}
          <button type="button" class="small" @click="setOptionView('all')">{{ t("trainer.showAll") }}</button>
        </p>

        <p v-if="!sections.length" class="panel empty">
          {{ query ? t("trainer.noMatch", { query: search }) : t("trainer.noChanges") }}
        </p>
      </div>
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
  /* a readable form width on large screens: the index and form stay together, centred */
  grid-template-columns: 210px minmax(0, 1280px);
  justify-content: center;
  gap: 24px;
  align-items: start;
  /* room below the last section, so sections near the end can still scroll to the top */
  padding: 16px 24px 40vh;
}

.sections {
  display: grid;
  gap: 14px;
  min-width: 0;
}

.toolbar {
  position: sticky;
  top: var(--header-height);
  z-index: 10;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px 12px;
  margin: -16px 0 0;
  padding: 12px 0 8px;
  background: var(--bg);
}

.search {
  flex: 1 1 260px;
  min-width: 0;
  padding-left: 32px;
  background: var(--surface)
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%23667085' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.5-3.5'/%3E%3C/svg%3E")
    no-repeat 10px center;
}

.count {
  color: var(--muted);
  font-size: 13px;
}

.config-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-right: 10px;
  padding-right: 16px;
  border-right: 1px solid var(--border);
}

.reset-all {
  margin-left: auto;
}

.config-actions input {
  display: none;
}

.view-switch {
  display: inline-flex;
  padding: 2px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface-muted);
}

.view-switch button {
  padding: 5px 12px;
  border: 0;
  border-radius: 4px;
  background: transparent;
  box-shadow: none;
  color: var(--muted);
}

.view-switch button.selected {
  background: var(--surface);
  box-shadow: var(--shadow-sm);
  color: var(--text-strong);
  font-weight: 600;
}

.more {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 4px 0 0;
  padding: 14px;
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius);
  color: var(--muted);
}

.changed-only {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--muted);
  font-size: 13px;
  cursor: pointer;
  white-space: nowrap;
}

.empty {
  color: var(--muted);
  text-align: center;
}

@media (max-width: 1100px) {
  .trainer {
    grid-template-columns: minmax(0, 1fr);
  }

  .index {
    display: none;
  }
}
</style>
