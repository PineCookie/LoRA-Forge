<script setup lang="ts">
import { t } from "../i18n";
import type { MergedSectionId } from "../schema/layout";
import SectionIcon from "./SectionIcon.vue";

defineProps<{
  /** anchors the section for the index and the scroll position */
  sectionId: MergedSectionId;
  /** shown as is (already translated) */
  title: string;
  /** one line of what the section holds, e.g. "Prodigy · 1e-4 · cosine" */
  summary: string;
  collapsed: boolean;
  editedCount: number;
}>();

defineEmits<{ toggle: [] }>();
</script>

<template>
  <section :id="`section-${sectionId}`" class="form-section" :class="{ collapsed }" :data-section="sectionId">
    <button type="button" class="header" :aria-expanded="!collapsed" @click="$emit('toggle')">
      <span class="icon"><SectionIcon :section="sectionId" /></span>
      <span class="title">{{ title }}</span>
      <span v-if="editedCount" class="badge" :title="t('trainer.editedTitle', { count: editedCount })">{{ editedCount }}</span>
      <span class="summary">{{ summary }}</span>
      <span class="chevron" aria-hidden="true">▾</span>
    </button>
    <div class="body" :class="{ open: !collapsed }" :inert="collapsed">
      <div class="inner">
        <slot />
        <div v-if="$slots.footer" class="footer"><slot name="footer" /></div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.form-section {
  /* option rows stack their control under the name when the section is narrow (FormField) */
  container-type: inline-size;
  /* below the sticky page header and toolbar when jumped to */
  scroll-margin-top: calc(var(--header-height) + 64px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow-sm);
}

.header {
  justify-content: flex-start;
  gap: 10px;
  width: 100%;
  padding: 13px 18px;
  border: 0;
  border-radius: var(--radius);
  background: transparent;
  box-shadow: none;
  text-align: left;
  white-space: nowrap;
}

.header:hover {
  border: 0;
}

.icon {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 7px;
  background: var(--surface-sunken);
  color: var(--muted);
}

.title {
  color: var(--text-strong);
  font-size: 15px;
  font-weight: 700;
}

.badge {
  padding: 0 7px;
  border-radius: 999px;
  background: var(--primary-soft);
  color: var(--primary-text);
  font-size: 11px;
  font-weight: 700;
}

.summary {
  min-width: 0;
  margin-left: auto;
  overflow: hidden;
  color: var(--muted);
  font-size: 13px;
  text-overflow: ellipsis;
}

.chevron {
  color: var(--faint);
  transition: transform 0.2s ease;
}

.collapsed .chevron {
  transform: rotate(-90deg);
}

.body {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 0.22s ease;
}

.body.open {
  grid-template-rows: 1fr;
}

.inner {
  min-height: 0;
  overflow: hidden;
}

.footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 14px;
  padding: 6px 18px 14px;
}

.footer:empty {
  display: none;
}

@media (max-width: 640px) {
  .summary {
    display: none;
  }

  .title {
    white-space: normal;
  }
}

@media (prefers-reduced-motion: reduce) {
  .body,
  .chevron {
    transition: none;
  }
}
</style>
