<script setup lang="ts">
import { t, tr } from "../i18n";
import type { Tone } from "../schema/groups";

export interface IndexItem {
  index: number;
  title: string;
  tone: Tone;
  edited: number;
}

defineProps<{ items: IndexItem[]; active: number | null }>();
defineEmits<{ jump: [index: number]; expandAll: []; collapseAll: [] }>();
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
      class="item"
      :key="item.index"
      type="button"
      :class="{ active: item.index === active }"
      :aria-current="item.index === active ? 'location' : undefined"
      :style="{ '--accent': `var(--tone-${item.tone})` }"
      @click="$emit('jump', item.index)"
    >
      <span class="dot" aria-hidden="true" />
      <span class="title">{{ tr(item.title) }}</span>
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
  gap: 1px;
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
  gap: 8px;
  width: 100%;
  padding: 6px 10px;
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
  background: color-mix(in srgb, var(--accent) 12%, transparent);
  color: var(--text-strong);
  font-weight: 600;
}

.dot {
  flex: 0 0 auto;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent);
}

.title {
  flex: 1;
  min-width: 0;
}

.badge {
  padding: 0 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--edited) 18%, transparent);
  color: var(--edited-text);
  font-size: 11px;
  font-weight: 700;
}
</style>
