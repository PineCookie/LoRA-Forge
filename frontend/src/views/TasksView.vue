<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { api } from "../api";
import { notify } from "../toast";

interface Task {
  id: string;
  status: "CREATED" | "RUNNING" | "FINISHED" | "TERMINATED";
}

const STATUS_LABELS: Record<Task["status"], string> = {
  CREATED: "等待启动",
  RUNNING: "运行中",
  FINISHED: "已结束",
  TERMINATED: "已终止",
};

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
  if (!confirm(`确定要终止任务 ${task.id} 吗？`)) return;
  try {
    await api(`/api/tasks/terminate/${encodeURIComponent(task.id)}`);
    notify(`已终止任务 ${task.id}`, "success");
  } catch (error) {
    notify(`终止任务失败：${error instanceof Error ? error.message : error}`, "error");
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
    <h2>任务列表</h2>
    <p v-if="loaded && !tasks.length" class="muted">没有正在记录的任务。</p>
    <article v-for="task in tasks" :key="task.id" class="task-row">
      <div>
        <strong class="task-id">{{ task.id }}</strong>
        <p><span class="status-chip" :class="`status-${task.status.toLowerCase()}`">{{ STATUS_LABELS[task.status] ?? task.status }}</span></p>
      </div>
      <button v-if="task.status === 'CREATED' || task.status === 'RUNNING'" type="button" class="danger" @click="terminate(task)">终止</button>
    </article>
  </section>
</template>
