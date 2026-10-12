<script setup lang="ts">
import PageHeader from "../components/PageHeader.vue";
import TaskStatus from "../components/TaskStatus.vue";
import VersionInfo from "../components/VersionInfo.vue";
import { useTasks } from "../composables/useTasks";
import { t } from "../i18n";

const { tasks, terminate } = useTasks();
</script>

<template>
  <PageHeader :title="t('tasks.title')" />
  <div class="tasks">
    <section class="panel">
      <p v-if="tasks.loaded && !tasks.tasks.length" class="muted">{{ t("tasks.empty") }}</p>
      <article v-for="task in tasks.tasks" :key="task.id" class="task">
        <div>
          <code class="id">{{ task.id }}</code>
          <TaskStatus :task="task" />
        </div>
        <button v-if="task.status === 'CREATED' || task.status === 'RUNNING'" type="button" class="danger" @click="terminate(task)">
          {{ t("tasks.terminate") }}
        </button>
      </article>
    </section>
    <VersionInfo />
  </div>
</template>

<style scoped>
.tasks {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 18px;
  align-items: start;
  padding: 20px 24px 48px;
}

.task {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 0;
  border-top: 1px solid var(--border);
}

.task:first-child {
  border-top: 0;
  padding-top: 0;
}

.task > div {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.id {
  font-size: 13px;
}

@media (max-width: 900px) {
  .tasks {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
