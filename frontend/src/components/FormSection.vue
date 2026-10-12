<script setup lang="ts">
import { computed } from "vue";
import { tr } from "../i18n";

const props = defineProps<{
  /** English section title from the schema; shown translated */
  title: string;
  collapsed: boolean;
  note?: string;
}>();

defineEmits<{ toggle: [] }>();

const TONES: [RegExp, string][] = [
  [/Quick setup/, "quick"],
  [/^Model/, "model"],
  [/Dataset|Caption/, "data"],
  [/Saving/, "save"],
  [/Learning rate|Optimizer|Loss/, "optimize"],
  [/Network|Base weights|Block-wise/, "network"],
  [/Preview|Logging/, "observe"],
  [/Noise|augmentation|Speed|precision|Distributed|GPU/, "performance"],
];

const tone = computed(() => `section-tone-${TONES.find(([pattern]) => pattern.test(props.title))?.[1] ?? "general"}`);
</script>

<template>
  <section class="panel form-section" :class="[tone, { 'is-collapsed': collapsed }]">
    <button type="button" class="section-header" :aria-expanded="!collapsed" @click="$emit('toggle')">
      <span class="section-title">{{ tr(title) }}</span>
      <span class="section-toggle-icon" aria-hidden="true">{{ collapsed ? "+" : "−" }}</span>
    </button>
    <div v-show="!collapsed" class="section-body">
      <p v-if="note" class="section-note">{{ note }}</p>
      <slot />
    </div>
  </section>
</template>
