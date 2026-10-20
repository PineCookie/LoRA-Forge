<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { controlKind, isWideField, type FormValue } from "../config/form";
import { t, tr } from "../i18n";
import { schemaOptions, type Field } from "../schema/groups";
import RichText from "./RichText.vue";

// One option as a row: its name and description on the left, the control on the right, and under
// the control what the page knows about the value (a hint, a check, the options an edit changed).

export interface FieldMessage {
  text: string;
  tone?: "ok" | "warn" | "error" | "todo";
}

const model = defineModel<FormValue>({ required: true });
const props = defineProps<{
  field: Field;
  edited: boolean;
  /** the value it starts with, shown while it is changed */
  defaultText: string;
  hint?: FieldMessage | null;
  check?: FieldMessage | null;
  /** "Also changed …" after an edit that changed other options, with an Undo */
  note?: string | null;
  /** shown only as the parent of a search result */
  context?: boolean;
}>();
const emit = defineEmits<{ pick: [pickerType: string]; reset: []; undo: [] }>();

const id = computed(() => `field-${props.field.name}`);
const meta = computed(() => props.field.schema.meta);
const kind = computed(() => controlKind(props.field.schema));
const wide = computed(() => isWideField(props.field.name, props.field.schema));
const options = computed(() => (schemaOptions(props.field.schema) ?? []).map(String));
const pickerType = computed(() => (meta.value.role?.name === "filepicker" ? String(meta.value.role.options.type || "folder") : null));
const description = computed(() => tr(meta.value.description ?? ""));

// a short choice is a row of buttons; a long one, or a value outside the list (an import), a select
const segmented = computed(
  () =>
    kind.value === "select" &&
    options.value.length <= 5 &&
    options.value.every((option) => option.length <= 16) &&
    options.value.join("").length <= 52 &&
    options.value.includes(String(model.value)),
);
// a fraction or rate between two close bounds gets a slider next to its number
const slider = computed(() => kind.value === "number" && meta.value.min !== undefined && meta.value.max !== undefined && meta.value.max - meta.value.min <= 1);

// typed views of the model for each kind of control
const text = computed({ get: () => (typeof model.value === "string" ? model.value : ""), set: (value: string) => (model.value = value) });
const checked = computed({ get: () => model.value === true, set: (value: boolean) => (model.value = value) });
const selected = computed({ get: () => (Array.isArray(model.value) ? model.value : []), set: (value: string[]) => (model.value = value) });
const optionLabel = (option: string) => option.replace(/^networks\./, "").replace(/^prodigyplus\./, "");

// the description shows two lines; "more" opens the rest when there is more
const descriptionEl = ref<HTMLElement | null>(null);
const open = ref(false);
const clamped = ref(false);
let observer: ResizeObserver | undefined;
function measure() {
  const element = descriptionEl.value;
  if (element && !open.value) clamped.value = element.scrollHeight > element.clientHeight + 1;
}
onMounted(() => {
  if (!descriptionEl.value || typeof ResizeObserver === "undefined") return;
  observer = new ResizeObserver(measure);
  observer.observe(descriptionEl.value);
});
onBeforeUnmount(() => observer?.disconnect());
watch(description, () => nextTick(measure));
</script>

<template>
  <div :id="`option-${field.name}`" class="field" :class="{ edited, wide, context, expanded: open }">
    <div class="meta">
      <div class="name-line">
        <label class="name" :for="id">{{ field.name }}</label>
        <span v-if="edited" class="dot" :title="t('field.changed')" />
        <button v-if="edited && !meta.disabled" type="button" class="reset" :title="t('field.reset')" :aria-label="t('field.reset')" @click="emit('reset')">↺</button>
      </div>
      <div v-if="description" ref="descriptionEl" class="description"><RichText tag="span" :text="description" /></div>
      <button v-if="description && (clamped || open)" type="button" class="more" @click="open = !open">{{ t(open ? "field.less" : "field.more") }}</button>
      <div v-if="edited" class="default">
        {{ t("field.default") }} <code>{{ defaultText === "" ? t("field.empty") : defaultText }}</code>
      </div>
    </div>

    <div class="control">
      <button
        v-if="kind === 'checkbox'"
        :id="id"
        type="button"
        role="switch"
        class="switch"
        :aria-checked="checked"
        :disabled="meta.disabled"
        @click="checked = !checked"
      />
      <div v-else-if="segmented" :id="id" class="segmented" role="radiogroup" :aria-label="field.name">
        <button
          v-for="option in options"
          :key="option"
          type="button"
          role="radio"
          :aria-checked="text === option"
          :disabled="meta.disabled"
          @click="text = option"
        >
          {{ optionLabel(option) }}
        </button>
      </div>
      <select v-else-if="kind === 'select'" :id="id" v-model="text" :disabled="meta.disabled">
        <option v-for="option in options" :key="option" :value="option">{{ option }}</option>
      </select>
      <div v-else-if="kind === 'multiselect'" :id="id" class="choices" role="group" :aria-label="field.name">
        <label v-for="option in options" :key="option" class="choice">
          <input v-model="selected" type="checkbox" :value="option" :disabled="meta.disabled" />
          {{ option }}
        </label>
      </div>
      <textarea v-else-if="kind === 'textarea'" :id="id" v-model="text" rows="3" spellcheck="false" :disabled="meta.disabled" />
      <div v-else-if="kind === 'number'" class="number" :class="{ slider }">
        <input
          v-if="slider"
          type="range"
          :min="meta.min"
          :max="meta.max"
          :step="meta.step ?? 0.01"
          :value="text === '' ? meta.min : text"
          :aria-label="field.name"
          :disabled="meta.disabled"
          @input="text = ($event.target as HTMLInputElement).value"
        />
        <!-- not v-model: Vue casts type=number values to numbers, the form keeps the typed text -->
        <input
          :id="id"
          type="number"
          :value="text"
          :min="meta.min"
          :max="meta.max"
          :step="meta.step ?? 'any'"
          :disabled="meta.disabled"
          :class="{ invalid: check?.tone === 'error' }"
          @input="text = ($event.target as HTMLInputElement).value"
        />
      </div>
      <div v-else class="text">
        <input
          :id="id"
          v-model="text"
          type="text"
          spellcheck="false"
          :class="{ path: pickerType, invalid: check?.tone === 'error' }"
          :placeholder="pickerType ? t('field.notSet') : undefined"
          :disabled="meta.disabled"
        />
        <button v-if="pickerType" type="button" @click="emit('pick', pickerType)">{{ t("field.browse") }}</button>
      </div>

      <div v-if="check" class="check" :class="check.tone">{{ check.text }}</div>
      <div v-else-if="hint" class="hint" :class="hint.tone">{{ hint.text }}</div>
      <div v-if="note" class="note">
        <span>{{ note }}</span>
        <button type="button" @click="emit('undo')">{{ t("field.undo") }}</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.field {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 340px);
  gap: 6px 28px;
  padding: 12px 18px;
  border-top: 1px solid var(--border);
}

.field.wide {
  grid-template-columns: minmax(0, 1fr);
}

/* jumped to from the run summary */
.field.flash {
  animation: flash 1.4s ease;
}

@keyframes flash {
  0%,
  40% {
    background: var(--primary-soft);
  }

  100% {
    background: transparent;
  }
}

.field.context {
  opacity: 0.55;
}

.name-line {
  display: flex;
  align-items: center;
  gap: 7px;
  min-height: 22px;
}

.name {
  color: var(--text-strong);
  font-family: var(--font-mono);
  font-size: 13.5px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.dot {
  flex: 0 0 auto;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--primary);
}

.reset {
  padding: 0 6px;
  border: 0;
  background: transparent;
  box-shadow: none;
  color: var(--faint);
  font-size: 14px;
}

.reset:hover {
  background: var(--surface-muted);
  color: var(--text-strong);
}

.description {
  display: -webkit-box;
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.5;
}

.field.expanded > .meta > .description {
  display: block;
}

.more {
  padding: 0;
  border: 0;
  background: none;
  box-shadow: none;
  color: var(--primary-text);
  font-size: 12px;
  font-weight: 600;
}

.more:hover {
  text-decoration: underline;
}

.default {
  margin-top: 2px;
  color: var(--faint);
  font-size: 12px;
}

.control {
  display: grid;
  align-content: start;
  gap: 6px;
  min-width: 0;
}

.text,
.number {
  display: flex;
  gap: 8px;
  align-items: center;
}

.text input,
.number input {
  min-width: 0;
}

.number.slider input[type="number"] {
  flex: 0 0 84px;
}

.number input[type="range"] {
  flex: 1;
  padding: 0;
  border: 0;
  background: none;
  accent-color: var(--primary);
}

.number input[type="range"]:focus {
  box-shadow: none;
}

input.path,
textarea {
  font-family: var(--font-mono);
  font-size: 13px;
}

input.invalid {
  border-color: var(--danger);
}

.switch {
  position: relative;
  width: 40px;
  height: 22px;
  padding: 0;
  border: 0;
  border-radius: 999px;
  background: var(--border-strong);
  box-shadow: none;
}

.switch::after {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #fff;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
  content: "";
  transition: transform 0.15s ease;
}

.switch[aria-checked="true"] {
  background: var(--primary);
}

.switch[aria-checked="true"]::after {
  transform: translateX(18px);
}

.segmented {
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  padding: 2px;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-sm);
  background: var(--surface-muted);
}

.segmented button {
  flex: 1 1 auto;
  padding: 5px 9px;
  border: 0;
  border-radius: 5px;
  background: transparent;
  box-shadow: none;
  color: var(--muted);
  font-size: 13px;
}

.segmented button:hover {
  color: var(--text-strong);
}

.segmented button[aria-checked="true"] {
  background: var(--surface);
  box-shadow: var(--shadow-sm), 0 0 0 1px var(--border);
  color: var(--text-strong);
  font-weight: 600;
}

.choices {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 18px;
  padding: 4px 0;
}

.choice {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}

.hint,
.check {
  font-size: 12.5px;
  line-height: 1.45;
}

.hint {
  color: var(--muted);
}

.hint.ok {
  color: var(--success);
}

.hint.warn {
  color: var(--warning);
}

.check {
  padding: 5px 9px;
  border-radius: var(--radius-sm);
}

.check.error {
  background: var(--danger-soft);
  color: var(--danger);
}

.check.warn {
  background: var(--warning-soft);
  color: var(--warning);
}

.check.todo {
  padding: 0;
  color: var(--primary-text);
  font-weight: 600;
}

.note {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 6px 10px;
  border-radius: var(--radius-sm);
  background: var(--primary-soft);
  color: var(--primary-text);
  font-size: 12.5px;
}

.note button {
  margin-left: auto;
  padding: 0;
  border: 0;
  background: none;
  box-shadow: none;
  color: inherit;
  font-size: 12.5px;
  font-weight: 700;
  text-decoration: underline;
}

@media (max-width: 720px) {
  .field {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
