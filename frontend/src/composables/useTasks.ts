import { computed, onBeforeUnmount, onMounted, reactive, readonly } from "vue";
import { api } from "../api";
import { t } from "../i18n";
import { notify } from "../toast";
import { confirmAction } from "./useConfirm";

export interface Task {
  id: string;
  status: "CREATED" | "RUNNING" | "FINISHED" | "TERMINATED";
}

const POLL_INTERVAL = 3000;

// one poller shared by every component that shows task state
const state = reactive({ tasks: [] as Task[], loaded: false });
let subscribers = 0;
let timer: ReturnType<typeof setInterval> | undefined;

async function refresh() {
  try {
    const data = await api<{ tasks: Task[] }>("/api/tasks");
    state.tasks = [...(data.tasks ?? [])].reverse(); // newest first
  } catch (error) {
    console.warn("Failed to load tasks", error);
  } finally {
    state.loaded = true;
  }
}

/** The task that blocks a new run: starting or running. */
const activeTask = computed(() => state.tasks.find((task) => task.status === "RUNNING" || task.status === "CREATED"));

async function terminate(task: Task) {
  if (!(await confirmAction(t("tasks.confirmTerminate", { id: task.id }), t("tasks.terminate")))) return;
  try {
    await api(`/api/tasks/terminate/${encodeURIComponent(task.id)}`);
    notify(t("tasks.terminated", { id: task.id }), "success");
  } catch (error) {
    notify(t("tasks.terminateFailed", { error: error instanceof Error ? error.message : String(error) }), "error");
  }
  await refresh();
}

/** Task list kept fresh while at least one component uses it. */
export function useTasks() {
  onMounted(() => {
    subscribers += 1;
    if (subscribers === 1) {
      void refresh();
      timer = setInterval(refresh, POLL_INTERVAL);
    }
  });
  onBeforeUnmount(() => {
    subscribers -= 1;
    if (!subscribers) clearInterval(timer);
  });
  return { tasks: readonly(state), activeTask, refresh, terminate };
}
