<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <header class="page-header">
    <div class="heading">
      <h1>{{ title }}</h1>
      <slot name="title" />
    </div>
    <div v-if="$slots.default" class="actions">
      <slot />
    </div>
  </header>
</template>

<style scoped>
.page-header {
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 16px;
  min-height: var(--header-height);
  padding: 8px 24px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--surface) 92%, transparent);
  backdrop-filter: blur(8px);
}

.heading {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 700;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

/* header buttons a size smaller than the page's */
.actions > :slotted(button) {
  padding: 6px 12px;
  font-size: 13.5px;
}
</style>
