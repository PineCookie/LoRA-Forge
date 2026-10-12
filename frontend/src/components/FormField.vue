<script setup lang="ts">
import { computed } from "vue";
import { controlKind, type FormValue } from "../config/form";
import { t, tr } from "../i18n";
import { schemaOptions, type Field } from "../schema/groups";
import RichText from "./RichText.vue";

const model = defineModel<FormValue>({ required: true });
const props = defineProps<{ field: Field; edited: boolean }>();
const emit = defineEmits<{ pick: [pickerType: string]; reset: [] }>();

const id = computed(() => `field-${props.field.name}`);
const meta = computed(() => props.field.schema.meta);
const kind = computed(() => controlKind(props.field.schema));
const options = computed(() => (schemaOptions(props.field.schema) ?? []).map(String));
const pickerType = computed(() => (meta.value.role?.name === "filepicker" ? String(meta.value.role.options.type || "folder") : null));
const description = computed(() => tr(meta.value.description ?? ""));

// typed views of the model for each kind of control
const text = computed({ get: () => (typeof model.value === "string" ? model.value : ""), set: (value: string) => (model.value = value) });
const checked = computed({ get: () => model.value === true, set: (value: boolean) => (model.value = value) });
const selected = computed({ get: () => (Array.isArray(model.value) ? model.value : []), set: (value: string[]) => (model.value = value) });
</script>

<template>
  <div class="field" :class="{ edited, inline: kind === 'checkbox' }">
    <div class="head">
      <input v-if="kind === 'checkbox'" :id="id" v-model="checked" type="checkbox" :disabled="meta.disabled" />
      <label class="name" :for="id">{{ field.name }}</label>
      <button v-if="edited && !meta.disabled" type="button" class="ghost small reset" :title="t('field.reset')" :aria-label="t('field.reset')" @click="emit('reset')">
        ↺
      </button>
    </div>

    <div v-if="kind !== 'checkbox'" class="control">
      <select v-if="kind === 'select'" :id="id" v-model="text" :disabled="meta.disabled">
        <option v-for="option in options" :key="option" :value="option">{{ option }}</option>
      </select>
      <select v-else-if="kind === 'multiselect'" :id="id" v-model="selected" multiple :disabled="meta.disabled">
        <option v-for="option in options" :key="option" :value="option">{{ option }}</option>
      </select>
      <textarea v-else-if="kind === 'textarea'" :id="id" v-model="text" rows="4" :disabled="meta.disabled" />
      <!-- not v-model: Vue casts type=number values to numbers, the form keeps the typed text -->
      <input
        v-else-if="kind === 'number'"
        :id="id"
        type="number"
        :value="text"
        :min="meta.min"
        :max="meta.max"
        :step="meta.step ?? 'any'"
        :disabled="meta.disabled"
        @input="text = ($event.target as HTMLInputElement).value"
      />
      <input v-else :id="id" v-model="text" type="text" :disabled="meta.disabled" />
      <button v-if="pickerType" type="button" @click="emit('pick', pickerType)">{{ t("field.browse") }}</button>
    </div>

    <RichText v-if="description" class="hint" tag="p" :text="description" />
  </div>
</template>

<style scoped>
.field {
  display: grid;
  align-content: start;
  gap: 6px;
  min-width: 0;
  padding-left: 9px;
  border-left: 3px solid transparent;
}

.field.edited {
  border-left-color: var(--edited);
}

.head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 22px;
}

.name {
  color: var(--text-strong);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.edited .name {
  color: var(--edited-text);
}

.inline .name {
  cursor: pointer;
}

.reset {
  margin-left: auto;
  padding: 0 7px;
  font-size: 14px;
}

.control {
  display: flex;
  gap: 8px;
  align-items: center;
}

.edited .control :is(input, select, textarea) {
  border-color: var(--edited);
}

.hint {
  margin: 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
</style>
