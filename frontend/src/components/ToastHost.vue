<script setup lang="ts">
import { t } from "../i18n";
import { dismissToast, toasts } from "../toast";
</script>

<template>
  <TransitionGroup tag="div" name="toast" class="toast-root" aria-live="polite">
    <div v-for="toast in toasts" :key="toast.id" class="toast" :class="toast.type" :role="toast.type === 'error' ? 'alert' : 'status'">
      <div class="body">{{ toast.message }}</div>
      <button type="button" class="ghost close" :aria-label="t('toast.dismiss')" @click="dismissToast(toast.id)">×</button>
    </div>
  </TransitionGroup>
</template>

<style scoped>
.toast-root {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 1000;
  display: grid;
  gap: 10px;
  width: min(420px, calc(100vw - 32px));
  pointer-events: none;
}

.toast {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 10px;
  align-items: start;
  padding: 12px 10px 12px 14px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--primary);
  border-radius: var(--radius);
  background: var(--surface);
  box-shadow: var(--shadow-md);
  pointer-events: auto;
}

.success {
  border-left-color: var(--success);
}

.warning {
  border-left-color: var(--edited);
}

.error {
  border-left-color: var(--danger);
}

.body {
  font-size: 15px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.close {
  width: 28px;
  height: 28px;
  padding: 0;
  font-size: 16px;
}

.toast-enter-active,
.toast-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(8px);
}

.toast-move {
  transition: transform 0.18s ease;
}
</style>
