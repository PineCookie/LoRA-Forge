<script setup lang="ts">
import { computed, onBeforeUnmount, reactive, watch } from "vue";
import { api } from "../api";
import type { FormValues } from "../config/form";
import { MODEL_FILES, type ModelFileState } from "../config/modelFiles";
import { t } from "../i18n";
import { huggingFaceFileUrl } from "../links";
import { store } from "../store";
import RichText from "./RichText.vue";

const props = defineProps<{ schemaName: string; values: FormValues }>();

const DEBOUNCE_MS = 400;
const specs = computed(() => MODEL_FILES[props.schemaName] ?? []);
const states = reactive<Record<string, ModelFileState | "checking">>({});
const timers = new Map<string, ReturnType<typeof setTimeout>>();
// a newer path for the same field wins over an answer still on its way
const latest = new Map<string, string>();

function check(field: string, path: string) {
  clearTimeout(timers.get(field));
  latest.set(field, path);
  if (!path) {
    states[field] = "empty";
    return;
  }
  states[field] = "checking";
  timers.set(
    field,
    setTimeout(async () => {
      try {
        const query = new URLSearchParams({ model: props.schemaName, field, path });
        const result = await api<{ state: ModelFileState }>(`/api/model_file?${query}`);
        if (latest.get(field) === path) states[field] = result.state;
      } catch {
        if (latest.get(field) === path) states[field] = "unknown";
      }
    }, DEBOUNCE_MS),
  );
}

watch(
  () => specs.value.map((spec) => String(props.values[spec.field] ?? "").trim()),
  (paths, previous) => specs.value.forEach((spec, index) => paths[index] !== previous?.[index] && check(spec.field, paths[index])),
  { immediate: true },
);
onBeforeUnmount(() => timers.forEach((timer) => clearTimeout(timer)));

const rows = computed(() =>
  specs.value.map((spec) => ({
    ...spec,
    name: spec.path.split("/").pop() ?? spec.path,
    url: huggingFaceFileUrl(store.runtime?.hf_endpoint, spec.repo, spec.path),
    state: states[spec.field] ?? "empty",
  })),
);
</script>

<template>
  <div v-if="rows.length" class="model-files wide">
    <div class="head">
      <span class="title">{{ t("modelFiles.title") }}</span>
      <span class="intro">{{ t("modelFiles.intro") }}</span>
    </div>
    <ul>
      <li v-for="row in rows" :key="row.field">
        <span class="role">{{ t(row.role) }}</span>
        <span class="file">
          <code>{{ row.name }}</code>
          <span class="where">{{ row.repo }} · {{ row.size }}</span>
          <RichText v-if="row.note" class="note" :text="t(row.note)" />
        </span>
        <span class="state" :class="row.state">{{ t(`modelFiles.state.${row.state}`) }}</span>
        <a class="download" :href="row.url" target="_blank" rel="noopener">{{ t("modelFiles.download") }} ↗</a>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.model-files {
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface-sunken);
  font-size: 13px;
  line-height: 1.5;
}

.head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 12px;
  margin-bottom: 8px;
}

.title {
  color: var(--text-strong);
  font-size: 14px;
  font-weight: 600;
}

.intro {
  color: var(--muted);
  font-size: 12px;
}

ul {
  display: grid;
  margin: 0;
  padding: 0;
  list-style: none;
}

li {
  display: grid;
  grid-template-columns: 140px minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 4px 14px;
  padding: 8px 0;
}

li + li {
  border-top: 1px solid var(--border);
}

.role {
  color: var(--text-strong);
  font-weight: 600;
}

.file {
  display: grid;
  min-width: 0;
}

.file code {
  justify-self: start;
  overflow: hidden;
  max-width: 100%;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.where,
.note {
  color: var(--muted);
  font-size: 12px;
}

.state {
  padding: 1px 8px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.state.ok {
  border-color: var(--success-border);
  background: var(--success-soft);
  color: var(--success);
}

.state.missing,
.state.wrong,
.state.quantized {
  border-color: var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
}

.download {
  color: var(--primary-text);
  font-weight: 600;
  white-space: nowrap;
}

.download:hover {
  text-decoration: underline;
}

@media (max-width: 720px) {
  li {
    grid-template-columns: minmax(0, 1fr) auto;
  }

  .file {
    grid-column: 1 / -1;
    grid-row: 2;
  }
}
</style>
