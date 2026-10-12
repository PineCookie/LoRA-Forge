<script setup lang="ts">
defineProps<{ title: string }>();
</script>

<template>
  <header class="page-header">
    <h1>{{ title }}</h1>
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
  gap: 10px 16px;
  min-height: var(--header-height);
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter: blur(8px);
}

h1 {
  margin: 0;
  font-size: 22px;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
</style>
