<script setup lang="ts">
import { computed } from "vue";
import { t } from "../i18n";

// Quick picks under max_train_epochs: a target step count sets the epochs that come closest to it.
const props = defineProps<{ stepsPerEpoch: number; epochs: string }>();
const emit = defineEmits<{ set: [epochs: string] }>();

const TARGETS = [500, 1000, 1500, 2000, 3000];
const targets = computed(() => {
  const seen = new Set<number>();
  return TARGETS.map((steps) => ({ steps, epochs: Math.max(1, Math.round(steps / props.stepsPerEpoch)) })).filter(({ epochs }) => !seen.has(epochs) && seen.add(epochs));
});
</script>

<template>
  <div class="targets">
    <span class="label">{{ t("dataset.aimFor") }}</span>
    <button
      v-for="target in targets"
      :key="target.steps"
      type="button"
      :class="{ current: String(target.epochs) === epochs.trim() }"
      :title="t('dataset.targetTitle', { epochs: target.epochs, steps: (target.epochs * stepsPerEpoch).toLocaleString() })"
      @click="emit('set', String(target.epochs))"
    >
      {{ target.steps.toLocaleString() }}
    </button>
  </div>
</template>

<style scoped>
.targets {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 6px;
  font-size: 12.5px;
}

.label {
  color: var(--muted);
}

button {
  padding: 1px 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface);
  box-shadow: none;
  color: var(--text);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

button:hover {
  border-color: var(--primary);
  color: var(--primary-text);
}

button.current {
  border-color: var(--primary);
  background: var(--primary-soft);
  color: var(--primary-text);
  font-weight: 600;
}
</style>
