<script setup lang="ts">
import { computed } from "vue";
import { parseResolution } from "../config/normalize";
import { t } from "../i18n";

// Quick picks under `resolution`: square sizes the model handles. A square is all buckets need, since
// they only take the pixel count from it.
const props = defineProps<{ sizes: number[]; value: string }>();
const emit = defineEmits<{ set: [value: string] }>();

const current = computed(() => parseResolution(props.value));
const isCurrent = (size: number) => current.value?.[0] === size && current.value?.[1] === size;
</script>

<template>
  <div class="presets">
    <span class="label">{{ t("resolution.sizes") }}</span>
    <button v-for="size in sizes" :key="size" type="button" :class="{ current: isCurrent(size) }" @click="emit('set', `${size}x${size}`)">
      {{ size }}
    </button>
  </div>
</template>

<style scoped>
.presets {
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
