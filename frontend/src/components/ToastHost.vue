<script setup lang="ts">
import { t } from "../i18n";
import { dismissToast, toasts } from "../toast";
</script>

<template>
  <TransitionGroup id="toast-root" tag="div" name="toast" aria-live="polite">
    <div v-for="toast in toasts" :key="toast.id" class="toast" :class="`toast-${toast.type}`" :role="toast.type === 'error' ? 'alert' : 'status'">
      <div class="toast-body">{{ toast.message }}</div>
      <button type="button" class="toast-close" :aria-label="t('toast.dismiss')" @click="dismissToast(toast.id)">×</button>
    </div>
  </TransitionGroup>
</template>
