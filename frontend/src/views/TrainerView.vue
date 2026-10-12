<script setup lang="ts">
import { computed, reactive, ref } from "vue";
import { useRoute } from "vue-router";
import { api, postJson } from "../api";
import DatasetStats from "../components/DatasetStats.vue";
import FormField from "../components/FormField.vue";
import FormSection from "../components/FormSection.vue";
import VersionInfo from "../components/VersionInfo.vue";
import { formValueKey, initialFormValues, readForm, toFormValue, type FormValue, type FormValues } from "../config/form";
import { expandImportedArgs } from "../config/import";
import { redactConfigForLog, type TrainingConfig } from "../config/normalize";
import type { SubsetOffsets } from "../config/offsets";
import { applyDependentValues, applyExclusiveRules } from "../config/rules";
import { fromToml, toToml } from "../config/toml";
import { isFieldVisible } from "../config/visibility";
import { t, tr } from "../i18n";
import { buildGroups, schemaOptions, type Field, type Group } from "../schema/groups";
import { getSchema, store } from "../store";
import { notify } from "../toast";

const props = defineProps<{ schemaName: string }>();
const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));
const route = useRoute();

const schema = getSchema(props.schemaName);
const groups: Group[] = schema ? buildGroups(schema, props.schemaName, store.cards) : [];
const fieldMap = new Map(groups.flatMap((group) => group.fields.map((field) => [field.name, field] as const)));
const supportsTimestepPreview = ["flux-lora", "anima-lora"].includes(props.schemaName);

const values = ref<FormValues>({});
const initialKeys = ref<Record<string, string>>({});
const offsets = reactive<SubsetOffsets>({ dir: null, values: {} });
const datasetRefresh = ref(0);
const busy = ref(false);
const search = ref("");

function resetForm() {
  const next = initialFormValues(groups);
  applyDependentValues(next);
  values.value = next;
  initialKeys.value = Object.fromEntries(Object.entries(next).map(([name, value]) => [name, formValueKey(value)]));
  offsets.dir = null;
  offsets.values = {};
  datasetRefresh.value += 1;
}
resetForm();

function onInput(name: string, value: FormValue) {
  values.value[name] = value;
  applyExclusiveRules(values.value, name);
  applyDependentValues(values.value);
}

const isEdited = (name: string) => formValueKey(values.value[name]) !== initialKeys.value[name];

// --- search and sections ---------------------------------------------------------------

const query = computed(() => search.value.trim().toLowerCase());

function matchesSearch(field: Field) {
  if (!query.value) return true;
  const description = field.schema.meta.description ?? "";
  return [field.name, description, tr(description)].some((text) => text.toLowerCase().includes(query.value));
}

const shownFields = (group: Group) => group.fields.filter((field) => isFieldVisible(field.name, values.value) && matchesSearch(field));
const sectionKey = (group: Group, index: number) => `${route.path}:${index}:${group.title}`;
const isCollapsed = (group: Group, index: number) => !query.value && store.collapsedSections.has(sectionKey(group, index));

function toggleSection(group: Group, index: number) {
  const key = sectionKey(group, index);
  if (store.collapsedSections.has(key)) store.collapsedSections.delete(key);
  else store.collapsedSections.add(key);
}

function setAllCollapsed(collapsed: boolean) {
  groups.forEach((group, index) => {
    if (collapsed) store.collapsedSections.add(sectionKey(group, index));
    else store.collapsedSections.delete(sectionKey(group, index));
  });
}

function sectionNote(group: Group) {
  if (group.title === "Learning rate and optimizer") return t("trainer.note.optimizer");
  if (group.title !== "Loss") return undefined;
  if (props.schemaName === "anima-lora") return t("trainer.note.lossAnima");
  const names = new Set(group.fields.map((field) => field.name));
  if (names.has("min_snr_gamma") && names.has("debiased_estimation_loss")) return t("trainer.note.lossMinSnr");
  return undefined;
}

const visibleCount = computed(() => groups.reduce((count, group) => count + shownFields(group).length, 0));

// --- previews --------------------------------------------------------------------------

const config = computed(() => readForm(values.value, groups));
const tomlPreview = computed(() => toToml(config.value));
const generatedArgs = computed(() =>
  (["optimizer_args", "network_args"] as const).map((name) => ({ name, args: (config.value[name] as string[] | undefined) ?? [] })).filter((section) => section.args.length),
);

const numberValue = (name: string) => parseInt(String(values.value[name] ?? ""), 10) || 1;
const datasetDir = computed(() => String(values.value.train_data_dir ?? ""));
const trainType = computed(() => String(values.value.model_train_type ?? ""));

// --- training --------------------------------------------------------------------------

function buildRunConfig(previewTimesteps: boolean): TrainingConfig {
  const runConfig = readForm(values.value, groups);
  // only submit offsets recorded for the dataset currently selected in the form
  if (offsets.dir === datasetDir.value.trim()) {
    const recorded = Object.fromEntries(Object.entries(offsets.values).filter(([, value]) => Number(value) !== 0));
    if (Object.keys(recorded).length) runConfig.subset_timestep_sampling_offsets = recorded;
  }
  if (previewTimesteps) {
    runConfig.show_timesteps ||= "console";
  } else {
    // sd-scripts exits after rendering --show_timesteps; a real run must never inherit it
    delete runConfig.show_timesteps;
    delete runConfig.show_timesteps_offset;
    delete runConfig.show_timesteps_resolution;
  }
  return runConfig;
}

async function startRun(previewTimesteps: boolean) {
  if (busy.value) return;
  const runConfig = buildRunConfig(previewTimesteps);
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
  }
}

async function stopTraining() {
  try {
    const data = await api<{ tasks: { id: string; status: string }[] }>("/api/tasks");
    const task = (data.tasks ?? []).find((item) => item.status === "RUNNING");
    if (!task) {
      notify(t("trainer.noRunningTask"), "warning");
      return;
    }
    if (!confirm(t("trainer.confirmStop", { id: task.id }))) return;
    await api(`/api/tasks/terminate/${encodeURIComponent(task.id)}`);
    notify(t("trainer.stopped", { id: task.id }), "success");
  } catch (error) {
    notify(t("trainer.stopFailed", { error: errorText(error) }), "error");
  }
}

async function pickPath(name: string, pickerType: string) {
  try {
    const data = await api<{ path: string }>(`/api/pick_file?picker_type=${encodeURIComponent(pickerType)}`);
    onInput(name, data.path);
  } catch (error) {
    notify(errorText(error), "error");
  }
}

// --- export / import -------------------------------------------------------------------

function download(text: string, type: string, extension: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${config.value.output_name || "training-config"}.${extension}`;
  link.click();
  URL.revokeObjectURL(url);
}

const exportJson = () => download(JSON.stringify(config.value, null, 2), "application/json", "json");
const exportToml = () => download(tomlPreview.value, "application/toml;charset=utf-8", "toml");

async function copyToml() {
  try {
    await navigator.clipboard.writeText(tomlPreview.value);
    notify(t("trainer.copied"), "success");
  } catch {
    notify(t("trainer.copyFailed"), "error");
  }
}

function allowedTrainTypes(): string[] {
  const field = fieldMap.get("model_train_type");
  const options = field ? (schemaOptions(field.schema) ?? []).map(String) : [];
  if (options.length) return options;
  const initial = initialKeys.value.model_train_type;
  return initial ? [initial] : [];
}

function applyImported(imported: TrainingConfig) {
  const importedType = imported.model_train_type;
  const allowed = allowedTrainTypes();
  if (importedType && allowed.length && !allowed.includes(importedType)) {
    throw new Error(t("trainer.wrongTrainType", { allowed: allowed.join(" / "), type: importedType }));
  }

  for (const [name, value] of Object.entries(expandImportedArgs(imported))) {
    const field = fieldMap.get(name);
    if (field && name in values.value) values.value[name] = toFormValue(name, field.schema, value, { imported: true });
  }
  applyExclusiveRules(values.value, null); // full pass: fix constrained combinations from the file
  applyDependentValues(values.value);
  datasetRefresh.value += 1;
}

async function importFile(event: Event, format: "json" | "toml") {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  try {
    const text = await file.text();
    applyImported(format === "json" ? JSON.parse(text) : fromToml(text));
    notify(t("trainer.imported", { format: format.toUpperCase() }), "success");
  } catch (error) {
    notify(t("trainer.importFailed", { format: format.toUpperCase(), error: errorText(error) }), "error");
  } finally {
    input.value = ""; // allow importing the same file again
  }
}
</script>

<template>
  <section v-if="!schema" class="panel error">{{ t("trainer.schemaMissing", { name: schemaName }) }}</section>

  <form v-else class="trainer" @submit.prevent="startRun(false)">
    <div class="form-grid">
      <div class="form-toolbar">
        <input v-model="search" class="form-search" type="search" :placeholder="t('trainer.search')" :aria-label="t('trainer.searchLabel')" />
        <span v-if="query" class="form-toolbar-count">{{ t("trainer.matchCount", { count: visibleCount }) }}</span>
        <template v-else>
          <button type="button" class="ghost-button" @click="setAllCollapsed(false)">{{ t("trainer.expandAll") }}</button>
          <button type="button" class="ghost-button" @click="setAllCollapsed(true)">{{ t("trainer.collapseAll") }}</button>
        </template>
      </div>

      <template v-for="(group, index) in groups" :key="`${index}:${group.title}`">
        <FormSection
          v-if="shownFields(group).length"
          :title="group.title"
          :note="sectionNote(group)"
          :collapsed="isCollapsed(group, index)"
          @toggle="toggleSection(group, index)"
        >
          <FormField
            v-for="field in shownFields(group)"
            :key="field.name"
            :field="field"
            :model-value="values[field.name]"
            :edited="isEdited(field.name)"
            @update:model-value="onInput(field.name, $event)"
            @pick="pickPath(field.name, $event)"
          />
          <DatasetStats
            v-if="group.title === 'Dataset'"
            :dir="datasetDir"
            :batch-size="numberValue('train_batch_size')"
            :grad-acc="numberValue('gradient_accumulation_steps')"
            :epochs="numberValue('max_train_epochs')"
            :supports-offsets="['flux-lora', 'anima-lora'].includes(trainType)"
            :timestep-sampling="String(values.timestep_sampling ?? 'sigmoid')"
            :offsets="offsets"
            :refresh-key="datasetRefresh"
          />
        </FormSection>
      </template>

      <p v-if="query && !visibleCount" class="panel empty-search">{{ t("trainer.noMatch", { query: search }) }}</p>
    </div>

    <aside class="actions">
      <section class="action-group action-train">
        <h2>{{ t("trainer.training") }}</h2>
        <button type="submit" class="primary" :disabled="busy">{{ t(busy ? "trainer.submitting" : "trainer.start") }}</button>
        <button v-if="supportsTimestepPreview" type="button" :disabled="busy" @click="startRun(true)">{{ t("trainer.previewTimesteps") }}</button>
        <button type="button" class="danger" @click="stopTraining">{{ t("trainer.stop") }}</button>
      </section>

      <section class="action-group action-config">
        <h2>{{ t("trainer.config") }}</h2>
        <button type="button" id="reset-form" @click="resetForm">{{ t("trainer.reset") }}</button>
        <div class="config-action-row">
          <button type="button" class="download-button" @click="exportJson">{{ t("trainer.downloadJson") }}</button>
          <label class="import-button import-config-button">
            {{ t("trainer.importJson") }}<input type="file" accept="application/json,.json" @change="importFile($event, 'json')" />
          </label>
        </div>
        <div class="config-action-row">
          <button type="button" class="download-button" @click="exportToml">{{ t("trainer.downloadToml") }}</button>
          <label class="import-button import-config-button">
            {{ t("trainer.importToml") }}<input type="file" accept=".toml,text/plain" @change="importFile($event, 'toml')" />
          </label>
        </div>
      </section>

      <section class="generated-args">
        <h2>{{ t("trainer.generatedArgs") }}</h2>
        <div class="args-preview">
          <div v-for="section in generatedArgs" :key="section.name" class="args-section">
            <div class="args-name">{{ section.name }}:</div>
            <div v-for="arg in section.args" :key="arg" class="args-value">{{ arg }}</div>
          </div>
          <span v-if="!generatedArgs.length" class="args-empty">{{ t("trainer.noGeneratedArgs") }}</span>
        </div>
      </section>

      <section class="config-preview">
        <h2>
          {{ t("trainer.tomlPreview") }}
          <button type="button" class="icon-button copy-button" :title="t('trainer.copyTitle')" @click="copyToml">{{ t("trainer.copy") }}</button>
        </h2>
        <pre id="preview">{{ tomlPreview }}</pre>
      </section>

      <VersionInfo />
    </aside>
  </form>
</template>
