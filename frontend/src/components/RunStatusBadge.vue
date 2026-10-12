<script setup lang="ts">
import type { RunStatus } from "../composables/useRuns";
import { t } from "../i18n";

defineProps<{ status: RunStatus; compact?: boolean }>();
</script>

<template>
  <span class="badge" :class="[status, { compact }]">
    <span class="dot" aria-hidden="true" />
    {{ t(`runs.status.${status}`) }}
  </span>
</template>

<style scoped>
.badge {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-muted);
  color: var(--muted);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}

.badge.compact {
  padding: 1px 7px;
  font-size: 11px;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--faint);
}

.running,
.starting {
  border-color: var(--success-border);
  background: var(--success-soft);
  color: var(--success);
}

.running .dot,
.starting .dot {
  background: var(--success);
  animation: pulse 1.6s ease-in-out infinite;
}

.finished {
  border-color: var(--info-border);
  background: var(--info-soft);
  color: var(--info);
}

.finished .dot {
  background: var(--primary);
}

.failed {
  border-color: var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
}

.failed .dot {
  background: var(--danger);
}

.stopped,
.interrupted {
  border-color: var(--warning-border);
  background: var(--warning-soft);
  color: var(--warning);
}

.stopped .dot,
.interrupted .dot {
  background: var(--warning);
}

@keyframes pulse {
  50% {
    opacity: 0.35;
  }
}
</style>
