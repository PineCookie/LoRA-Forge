<script setup lang="ts">
import { t } from "../i18n";
import type { MergedSectionId } from "../schema/layout";
import SectionIcon from "./SectionIcon.vue";

export interface IndexItem {
  id: MergedSectionId;
  /** shown as is (already translated) */
  title: string;
  edited: number;
  /** checks that hold back Start in this section */
  blocking: number;
}

defineProps<{ items: IndexItem[]; active: string | null }>();
defineEmits<{ jump: [id: MergedSectionId]; expandAll: []; collapseAll: [] }>();
</script>

<template>
  <nav class="section-index" :aria-label="t('trainer.sections')">
    <div class="heading">
      <h2>{{ t("trainer.sections") }}</h2>
      <button type="button" class="tool" :title="t('trainer.expandAll')" :aria-label="t('trainer.expandAll')" @click="$emit('expandAll')">⊞</button>
      <button type="button" class="tool" :title="t('trainer.collapseAll')" :aria-label="t('trainer.collapseAll')" @click="$emit('collapseAll')">⊟</button>
    </div>
    <button
      v-for="item in items"
      :key="item.id"
      class="item"
      type="button"
      :class="{ active: item.id === active }"
      :aria-current="item.id === active ? 'location' : undefined"
      @click="$emit('jump', item.id)"
    >
      <SectionIcon :section="item.id" />
      <span class="title">{{ item.title }}</span>
      <span v-if="item.blocking" class="alert" :title="t('trainer.blockingTitle', { count: item.blocking })">!</span>
      <span v-if="item.edited" class="badge" :title="t('trainer.editedTitle', { count: item.edited })">{{ item.edited }}</span>
    </button>
  </nav>
</template>

<style scoped>
.section-index {
  position: sticky;
  top: calc(var(--header-height) + 16px);
  display: grid;
  align-content: start;
  gap: 2px;
  max-height: calc(100vh - var(--header-height) - 32px);
  overflow-y: auto;
}

.heading {
  display: flex;
  align-items: center;
  gap: 2px;
  margin: 0 0 6px 10px;
}

h2 {
  flex: 1;
  margin: 0;
  color: var(--faint);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.tool {
  width: 26px;
  height: 24px;
  padding: 0;
  border: 0;
  background: transparent;
  box-shadow: none;
  color: var(--faint);
  font-size: 15px;
}

.tool:hover {
  color: var(--text);
}

.item {
  justify-content: flex-start;
  gap: 9px;
  width: 100%;
  padding: 7px 10px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  box-shadow: none;
  color: var(--muted);
  font-size: 13px;
  text-align: left;
  white-space: normal;
}

.item:hover {
  background: var(--surface-muted);
  color: var(--text);
}

.item.active {
  background: var(--surface);
  box-shadow: var(--shadow-sm);
  color: var(--text-strong);
  font-weight: 600;
}

.title {
  flex: 1;
  min-width: 0;
}

.badge,
.alert {
  min-width: 18px;
  padding: 0 6px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  text-align: center;
}

.badge {
  background: var(--primary-soft);
  color: var(--primary-text);
}

.alert {
  background: var(--danger-soft);
  color: var(--danger);
}
</style>
