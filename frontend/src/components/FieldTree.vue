<script setup lang="ts">
import type { FieldNode } from "../schema/fieldTree";
import type { Field } from "../schema/groups";

// An option and, indented under it on a guide line, the options it controls. Every row stays in the
// page and slides open or closed, so switching an option never moves it: only what follows makes
// room. Closed rows are inert (no focus, no screen reader).
defineProps<{ node: FieldNode; shown: (name: string) => boolean }>();
defineSlots<{ field(props: { field: Field }): unknown }>();
</script>

<template>
  <div class="reveal" :class="{ open: shown(node.field.name) }" :inert="!shown(node.field.name)">
    <div class="inner">
      <slot name="field" :field="node.field" />
      <div v-if="node.children.length" class="children">
        <FieldTree v-for="child in node.children" :key="child.field.name" :node="child" :shown="shown">
          <template #field="{ field }"><slot name="field" :field="field" /></template>
        </FieldTree>
      </div>
    </div>
  </div>
</template>

<style scoped>
.reveal {
  display: grid;
  grid-template-rows: 0fr;
  opacity: 0;
  transition:
    grid-template-rows 0.22s ease,
    opacity 0.18s ease;
}

.reveal.open {
  grid-template-rows: 1fr;
  opacity: 1;
}

.inner {
  min-height: 0;
  overflow: hidden;
}

.children {
  margin-left: 26px;
  border-left: 2px solid var(--border);
}

/* sub-option rows: tighter, and no line between an option and its sub-options */
.children :deep(.field) {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 16px;
  border-top: 0;
}

@media (prefers-reduced-motion: reduce) {
  .reveal {
    transition: none;
  }
}
</style>
