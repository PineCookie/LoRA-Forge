<script setup lang="ts">
import type { MergedSectionId } from "../schema/layout";

// one line icon per page section (24 x 24, stroked with the text color)
defineProps<{ section: MergedSectionId }>();
</script>

<template>
  <svg class="section-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <template v-if="section === 'model'">
      <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" />
      <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
    </template>
    <template v-else-if="section === 'dataset'">
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="10" r="2" />
      <path d="M3 17l5-4 4 3 3-2 6 4" />
    </template>
    <template v-else-if="section === 'captions'">
      <path d="M4 5h16v11H9l-5 4z" />
      <path d="M8 9h8M8 12h5" />
    </template>
    <template v-else-if="section === 'timesteps'">
      <circle cx="12" cy="13" r="8" />
      <path d="M12 9v4l3 2M9 2h6" />
    </template>
    <template v-else-if="section === 'saving'">
      <path d="M12 3v12M7 10l5 5 5-5" />
      <path d="M4 17v3h16v-3" />
    </template>
    <template v-else-if="section === 'memory'">
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
    </template>
    <template v-else-if="section === 'training'">
      <path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1" />
      <circle cx="15" cy="6" r="2" />
      <circle cx="9" cy="12" r="2" />
      <circle cx="17" cy="18" r="2" />
    </template>
    <template v-else-if="section === 'network'">
      <circle cx="5" cy="6" r="2" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="19" cy="12" r="2" />
      <path d="M7 6.5l10 5M7 17.5l10-5" />
    </template>
    <template v-else-if="section === 'preview'">
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
    </template>
    <template v-else>
      <path d="M13 3L5 14h6l-1 7 8-11h-6z" />
    </template>
  </svg>
</template>

<style scoped>
.section-icon {
  width: 16px;
  height: 16px;
  flex: 0 0 auto;
}
</style>
