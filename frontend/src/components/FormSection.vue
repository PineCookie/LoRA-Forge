<script setup lang="ts">
import { t } from "../i18n";
import type { Tone } from "../schema/groups";

defineProps<{
  /** anchors the section for the index and the scroll position */
  sectionId: string;
  /** shown as is (already translated) */
  title: string;
  tone: Tone;
  collapsed: boolean;
  editedCount: number;
  notes?: string[];
}>();

defineEmits<{ toggle: [] }>();
</script>

<template>
  <section
    :id="`section-${sectionId}`"
    class="form-section"
    :class="[`tone-${tone}`, { collapsed }]"
    :data-section="sectionId"
    :style="{ '--accent': `var(--tone-${tone})` }"
  >
    <button type="button" class="header" :aria-expanded="!collapsed" @click="$emit('toggle')">
      <span class="title">{{ title }}</span>
      <span v-if="editedCount" class="badge" :title="t('trainer.editedTitle', { count: editedCount })">{{ editedCount }}</span>
      <span class="chevron" aria-hidden="true">{{ collapsed ? "▸" : "▾" }}</span>
    </button>
    <div v-show="!collapsed" class="body">
      <p v-for="note in notes" :key="note" class="note">{{ note }}</p>
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
  /* at most three columns; fewer when a column would get narrower than 260px */
  grid-template-columns: repeat(auto-fill, minmax(max(260px, calc((100% - 40px) / 3)), 1fr));
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
