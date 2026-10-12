<script setup lang="ts">
import { computed } from "vue";
import { RouterLink } from "vue-router";
import type { Task } from "../composables/useTasks";
import { t } from "../i18n";
import { formatDuration } from "../monitor/metrics";
import { phaseLabel } from "../monitor/labels";

const props = defineProps<{ task?: Task }>();

const label = computed(() => {
  const task = props.task;
  if (!task) return t("trainer.status.idle");
  const progress = task.progress;
  if (task.status !== "RUNNING" || !progress) return t(`tasks.status.${task.status}`);
  const phase = progress.phase;
  if (phase && phase.total) return `${phaseLabel(phase.desc)} · ${Math.floor((phase.n / phase.total) * 100)}%`;
  const step = progress.step;
  if (!step || !step.total) return t("runs.preparing");
  const percent = Math.floor((step.n / step.total) * 100);
  const eta = step.rate ? ` · ${t("runs.etaShort", { time: formatDuration((step.total - step.n) / step.rate) })}` : "";
  return `${t("tasks.status.RUNNING")} · ${percent}%${eta}`;
});

const percent = computed(() => {
  const step = props.task?.progress?.step;
  return props.task?.status === "RUNNING" && step?.total ? (step.n / step.total) * 100 : null;
});
</script>

<template>
  <component
    :is="task?.run_id ? RouterLink : 'span'"
    :to="task?.run_id ? { path: '/runs.html', query: { run: task.run_id } } : undefined"
    class="task-status"
    :class="task?.status.toLowerCase() ?? 'idle'"
    :title="task?.run_id ? t('runs.open') : undefined"
  >
    <span class="dot" aria-hidden="true" />
    {{ label }}
    <span v-if="percent !== null" class="bar" aria-hidden="true"><span :style="{ width: `${percent}%` }" /></span>
  </component>
</template>

<style scoped>
.task-status {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  overflow: hidden;
  padding: 4px 10px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-muted);
  color: var(--muted);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

a.task-status:hover {
  border-color: var(--primary);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--faint);
}

.bar {
  position: absolute;
  inset: auto 0 0 0;
  height: 2px;
}

.bar span {
  display: block;
  height: 100%;
  background: var(--success);
  transition: width 0.4s ease;
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
