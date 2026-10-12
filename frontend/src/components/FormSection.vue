<script setup lang="ts">
import { tr, t } from "../i18n";
import type { Tone } from "../schema/groups";

defineProps<{
  /** position in the page's groups; anchors the section for the index */
  index: number;
  /** English section title from the schema; shown translated */
  title: string;
  tone: Tone;
  collapsed: boolean;
  editedCount: number;
  note?: string;
}>();

defineEmits<{ toggle: [] }>();
</script>

<template>
  <section
    :id="`section-${index}`"
    class="form-section"
    :class="[`tone-${tone}`, { collapsed }]"
    :data-section-index="index"
    :style="{ '--accent': `var(--tone-${tone})` }"
  >
    <button type="button" class="header" :aria-expanded="!collapsed" @click="$emit('toggle')">
      <span class="title">{{ tr(title) }}</span>
      <span v-if="editedCount" class="badge" :title="t('trainer.editedTitle', { count: editedCount })">{{ editedCount }}</span>
      <span class="chevron" aria-hidden="true">{{ collapsed ? "▸" : "▾" }}</span>
    </button>
    <div v-show="!collapsed" class="body">
      <p v-if="note" class="note">{{ note }}</p>
      <slot />
    </div>
  </section>
</template>

<style scoped>
.form-section {
  /* below the sticky page header and toolbar when jumped to */
  scroll-margin-top: calc(var(--header-height) + 64px);
  border: 1px solid var(--border);
  border-top: 3px solid var(--accent);
  border-radius: var(--radius);
  background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 7%, var(--surface)) 0, var(--surface) 52px);
  box-shadow: var(--shadow-sm);
}

.tone-quick {
  border-width: 2px;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
}

.header {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 14px 18px;
  border: 0;
  border-radius: var(--radius) var(--radius) 0 0;
  background: transparent;
  box-shadow: none;
  text-align: left;
}

.header:hover {
  border: 0;
}

.title {
  color: var(--text-strong);
  font-size: 16px;
  font-weight: 700;
  white-space: normal;
}

.tone-quick .title {
  color: var(--accent);
}

.badge {
  padding: 0 7px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--edited) 18%, transparent);
  color: var(--edited-text);
  font-size: 12px;
  font-weight: 700;
}

.chevron {
  margin-left: auto;
  color: var(--accent);
  font-size: 14px;
}

.body {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px 20px;
  padding: 0 18px 18px;
}

.note {
  grid-column: 1 / -1;
  margin: -4px 0 0;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}

/* full-width children such as the dataset panel */
.body > :deep(.wide) {
  grid-column: 1 / -1;
}
</style>
