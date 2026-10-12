import { computed, ref } from "vue";
import { formValueKey, initialFormValues, readForm, toFormValue, type FormValue, type FormValues } from "../config/form";
import { EXPANDED_KEYS, expandImportedArgs } from "../config/import";
import type { TrainingConfig } from "../config/normalize";
import type { SubsetOffsets } from "../config/offsets";
import { applyDependentValues, applyExclusiveRules } from "../config/rules";
import { t } from "../i18n";
import { buildGroups, schemaOptions, type Group } from "../schema/groups";
import { getSchema, store } from "../store";

export type TrainerForm = ReturnType<typeof useTrainerForm>;

/** The state of one trainer page: field values, what was edited, and the config they stand for. */
export function useTrainerForm(schemaName: string) {
  const schema = getSchema(schemaName);
  const groups: Group[] = schema ? buildGroups(schema, schemaName, store.cards) : [];
  const fieldMap = new Map(groups.flatMap((group) => group.fields.map((field) => [field.name, field] as const)));

  const values = ref<FormValues>({});
  const initialValues = ref<FormValues>({});
  const initialKeys = computed(() => Object.fromEntries(Object.entries(initialValues.value).map(([name, value]) => [name, formValueKey(value)])));
  const offsets = ref<SubsetOffsets>({ dir: null, values: {} });
  /** bumped whenever the dataset panel must reload (reset, import) */
  const datasetRefresh = ref(0);

  function reset() {
    const next = initialFormValues(groups);
    applyDependentValues(next);
    values.value = next;
    initialValues.value = structuredClone(next);
    offsets.value = { dir: null, values: {} };
    datasetRefresh.value += 1;
  }

  /** A user edit: also applies the rules that couple fields. */
  function setValue(name: string, value: FormValue) {
    values.value[name] = value;
    applyExclusiveRules(values.value, name);
    applyDependentValues(values.value);
  }

  /** Back to the value the page started with (coupled fields follow, as for an edit). */
  function resetField(name: string) {
    const initial = initialValues.value[name];
    if (initial !== undefined) setValue(name, structuredClone(initial));
  }

  const isEdited = (name: string) => formValueKey(values.value[name]) !== initialKeys.value[name];
  const editedCount = (group: Group) => group.fields.filter((field) => isEdited(field.name)).length;

  const config = computed(() => readForm(values.value, groups));
  const datasetDir = computed(() => String(values.value.train_data_dir ?? ""));

  function allowedTrainTypes(): string[] {
    const field = fieldMap.get("model_train_type");
    const options = field ? (schemaOptions(field.schema) ?? []).map(String) : [];
    if (options.length) return options;
    const initial = initialKeys.value.model_train_type;
    return initial ? [initial] : [];
  }

  /**
   * Load an imported config into the form; throws when it is for another training type. Returns
   * the keys of the file this page has no option for (they are not kept).
   */
  function applyImported(imported: TrainingConfig): string[] {
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
    // gpu_ids depends on the machine: with one GPU the page has no picker, which is not worth a warning
    return Object.keys(imported).filter((name) => !fieldMap.has(name) && !EXPANDED_KEYS.has(name) && name !== "gpu_ids");
  }

  /** The config sent to /api/run, with the offsets recorded for the current dataset folder. */
  function runConfig(previewTimesteps: boolean): TrainingConfig {
    const result = readForm(values.value, groups);
    if (offsets.value.dir === datasetDir.value.trim()) {
      const recorded = Object.fromEntries(Object.entries(offsets.value.values).filter(([, value]) => Number(value) !== 0));
      if (Object.keys(recorded).length) result.subset_timestep_sampling_offsets = recorded;
    }
    if (previewTimesteps) {
      // the preview is printed to the training log; "image" would open a matplotlib window on the server
      result.show_timesteps = "console";
    } else {
      // sd-scripts exits after rendering --show_timesteps; a real run must never inherit it
      delete result.show_timesteps;
      delete result.show_timesteps_offset;
      delete result.show_timesteps_resolution;
    }
    return result;
  }

  reset();

  return {
    schema,
    groups,
    values,
    offsets,
    datasetRefresh,
    datasetDir,
    config,
    reset,
    resetField,
    setValue,
    isEdited,
    editedCount,
    applyImported,
    runConfig,
    numberValue: (name: string) => parseInt(String(values.value[name] ?? ""), 10) || 1,
  };
}
