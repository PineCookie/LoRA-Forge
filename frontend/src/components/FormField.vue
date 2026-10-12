<script setup lang="ts">
import { computed } from "vue";
import { controlKind, type FormValue } from "../config/form";
import { schemaOptions, type Field } from "../schema/groups";

const props = defineProps<{
  field: Field;
  modelValue: FormValue | undefined;
  edited: boolean;
}>();

const emit = defineEmits<{
  "update:modelValue": [value: FormValue];
  pick: [pickerType: string];
}>();

const id = computed(() => `field-${props.field.name}`);
const meta = computed(() => props.field.schema.meta);
const kind = computed(() => controlKind(props.field.schema));
const options = computed(() => (schemaOptions(props.field.schema) ?? []).map(String));
const pickerType = computed(() => (meta.value.role?.name === "filepicker" ? String(meta.value.role.options.type || "folder") : null));
const step = computed(() => meta.value.step ?? "any");

// Descriptions come from the bundled schema files and use `code` and **bold** inline markup.
const description = computed(() =>
  (meta.value.description ?? "").replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>"),
);

const text = computed(() => (typeof props.modelValue === "string" ? props.modelValue : ""));
const selected = computed(() => (Array.isArray(props.modelValue) ? props.modelValue : []));

function onSelectMultiple(event: Event) {
  emit("update:modelValue", Array.from((event.target as HTMLSelectElement).selectedOptions, (option) => option.value));
}

function onValue(event: Event) {
  emit("update:modelValue", (event.target as HTMLInputElement).value);
}

function onChecked(event: Event) {
  emit("update:modelValue", (event.target as HTMLInputElement).checked);
}
</script>

<template>
  <label class="field" :class="{ 'is-edited': edited, 'is-checkbox': kind === 'checkbox' }" :for="id">
    <span class="field-name">{{ field.name }}</span>
    <div class="control-row">
      <select v-if="kind === 'select'" :id="id" :value="text" :disabled="meta.disabled" @change="onValue">
        <option v-for="option in options" :key="option" :value="option">{{ option }}</option>
      </select>
      <select v-else-if="kind === 'multiselect'" :id="id" multiple :disabled="meta.disabled" @change="onSelectMultiple">
        <option v-for="option in options" :key="option" :value="option" :selected="selected.includes(option)">{{ option }}</option>
      </select>
      <input v-else-if="kind === 'checkbox'" :id="id" type="checkbox" :checked="modelValue === true" :disabled="meta.disabled" @change="onChecked" />
      <textarea v-else-if="kind === 'textarea'" :id="id" rows="4" :value="text" :disabled="meta.disabled" @input="onValue" />
      <input
        v-else
        :id="id"
        :type="kind === 'number' ? 'number' : 'text'"
        :value="text"
        :min="meta.min"
        :max="meta.max"
        :step="kind === 'number' ? step : undefined"
        :disabled="meta.disabled"
        @input="onValue"
      />
      <button v-if="pickerType" type="button" class="icon-button" @click="emit('pick', pickerType)">选择</button>
    </div>
    <p v-if="description" class="hint" v-html="description" />
  </label>
</template>
