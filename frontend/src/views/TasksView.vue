<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "../api";
import { t } from "../i18n";
import { notify } from "../toast";

interface Task {
  id: string;
  status: "CREATED" | "RUNNING" | "FINISHED" | "TERMINATED";
}

const tasks = ref<Task[]>([]);
const loaded = ref(false);
let timer: ReturnType<typeof setInterval> | undefined;

async function refresh() {
  try {
    const data = await api<{ tasks: Task[] }>("/api/tasks");
    tasks.value = [...(data.tasks ?? [])].reverse(); // newest first
  } catch (error) {
    console.warn("Failed to load tasks", error);
  } finally {
    loaded.value = true;
  }
}

async function terminate(task: Task) {
  if (!confirm(t("tasks.confirmTerminate", { id: task.id }))) return;
  try {
    await api(`/api/tasks/terminate/${encodeURIComponent(task.id)}`);
    notify(t("tasks.terminated", { id: task.id }), "success");
  } catch (error) {
    notify(t("tasks.terminateFailed", { error: error instanceof Error ? error.message : String(error) }), "error");
  }
  await refresh();
}

onMounted(() => {
  refresh();
  timer = setInterval(refresh, 3000);
});
onBeforeUnmount(() => clearInterval(timer));
</script>

<template>
  <section class="panel">
    <h2>{{ t("tasks.title") }}</h2>
    <p v-if="loaded && !tasks.length" class="muted">{{ t("tasks.empty") }}</p>
    <article v-for="task in tasks" :key="task.id" class="task-row">
      <div>
        <strong class="task-id">{{ task.id }}</strong>
        <p>
          <span class="status-chip" :class="`status-${task.status.toLowerCase()}`">{{ t(`tasks.status.${task.status}`) }}</span>
        </p>
      </div>
      <button v-if="task.status === 'CREATED' || task.status === 'RUNNING'" type="button" class="danger" @click="terminate(task)">
        {{ t("tasks.terminate") }}
      </button>
    </article>
  </section>
</template>
