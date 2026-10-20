import { computed, onBeforeUnmount, reactive, watch, type Ref } from "vue";
import { api } from "../api";
import type { FormValues } from "../config/form";
import { MODEL_FILES, type ModelFileState } from "../config/modelFiles";

const DEBOUNCE_MS = 400;

export type FileStates = Record<string, ModelFileState | "checking">;

/** The check of each model file field of a page (/api/model_file), redone shortly after its path changes. */
export function useModelFileStates(schemaName: string, values: Ref<FormValues>) {
  const specs = computed(() => MODEL_FILES[schemaName] ?? []);
  const states = reactive<FileStates>({});
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
          const query = new URLSearchParams({ model: schemaName, field, path });
          const result = await api<{ state: ModelFileState }>(`/api/model_file?${query}`);
          if (latest.get(field) === path) states[field] = result.state;
        } catch {
          if (latest.get(field) === path) states[field] = "unknown";
        }
      }, DEBOUNCE_MS),
    );
  }

  watch(
    () => specs.value.map((spec) => String(values.value[spec.field] ?? "").trim()),
    (paths, previous) => specs.value.forEach((spec, index) => paths[index] !== previous?.[index] && check(spec.field, paths[index])),
    { immediate: true },
  );
  onBeforeUnmount(() => timers.forEach((timer) => clearTimeout(timer)));

  return states;
}
