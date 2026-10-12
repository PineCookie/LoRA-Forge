<script setup lang="ts">
import { computed } from "vue";
import type { Task } from "../composables/useTasks";
import { t } from "../i18n";

const props = defineProps<{ task?: Task }>();

const label = computed(() => (props.task ? t(`tasks.status.${props.task.status}`) : t("trainer.status.idle")));
</script>

<template>
  <span class="task-status" :class="task?.status.toLowerCase() ?? 'idle'" :title="task?.id">
    <span class="dot" aria-hidden="true" />
    {{ label }}
  </span>
</template>

<style scoped>
.task-status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-muted);
  color: var(--muted);
  font-size: 13px;
  white-space: nowrap;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--faint);
}

.running {
  border-color: var(--success-border);
  background: var(--success-soft);
  color: var(--success);
}

.running .dot {
  background: var(--success);
  animation: pulse 1.6s ease-in-out infinite;
}

.created {
  border-color: var(--warning-border);
  background: var(--warning-soft);
  color: var(--warning);
}

.created .dot {
  background: var(--warning);
}

.finished .dot {
  background: var(--primary);
}

.terminated {
  border-color: var(--danger-border);
  background: var(--danger-soft);
  color: var(--danger);
}

.terminated .dot {
  background: var(--danger);
}

@keyframes pulse {
  50% {
    opacity: 0.35;
  }
}
</style>
