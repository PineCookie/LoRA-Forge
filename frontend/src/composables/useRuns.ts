import { computed, onBeforeUnmount, onMounted, ref, shallowRef, watch, type Ref } from "vue";
import { api } from "../api";
import { addMetrics, type MetricsEvent, type RunEvent, type SeriesMap } from "../monitor/metrics";

export type RunStatus = "starting" | "running" | "finished" | "failed" | "stopped" | "interrupted";

export interface ProgressBar {
  desc: string;
  n: number;
  total: number;
  rate: number | null;
  elapsed: number | null;
  done: boolean;
}

export interface RunProgress {
  /** the training-steps bar */
  step: ProgressBar | null;
  /** another bar while it runs: caching latents / text encoder outputs, validation */
  phase: ProgressBar | null;
  epoch: { n: number; total: number } | null;
  /** sd-scripts' moving average of the loss */
  loss: number | null;
}

export interface Run {
  id: string;
  task_id: string;
  model_train_type: string;
  output_name: string;
  output_dir: string;
  preview: boolean;
  started: number;
  ended: number | null;
  status: RunStatus;
  returncode: number | null;
  config_file: string;
  /** network_reg_lrs patterns by sd-scripts' group name (reg_lr_0, ...) */
  lr_groups?: Record<string, string>;
  progress?: RunProgress;
}

export interface Sample {
  name: string;
  time: number;
  step?: number;
  epoch?: number;
  prompt?: number;
}

export interface Checkpoint {
  name: string;
  size: number;
  time: number;
}

export const isActive = (status: RunStatus | undefined) => status === "starting" || status === "running";

const POLL_INTERVAL = 2000;

/** setInterval that also runs once right away and stops with the component. */
function usePolling(callback: () => Promise<void> | void, interval = POLL_INTERVAL) {
  let timer: ReturnType<typeof setInterval> | undefined;
  onMounted(() => {
    void callback();
    timer = setInterval(callback, interval);
  });
  onBeforeUnmount(() => clearInterval(timer));
}

/** Every recorded run, newest first. */
export function useRunList() {
  const runs = ref<Run[]>([]);
  const loaded = ref(false);
  const error = ref("");

  async function refresh() {
    try {
      runs.value = (await api<{ runs: Run[] }>("/api/runs")).runs ?? [];
      error.value = "";
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loaded.value = true;
    }
  }

  usePolling(refresh, 3000);
  return { runs, loaded, error, refresh };
}

/**
 * One run: its record, progress, metrics (fetched incrementally), samples, checkpoints and, while
 * `withOutput` is on, its console output. Polls while the run is active.
 */
export function useRun(runId: Ref<string | null>, withOutput: Ref<boolean>) {
  const run = ref<Run | null>(null);
  const progress = ref<RunProgress | null>(null);
  const samples = ref<Sample[]>([]);
  const checkpoints = ref<Checkpoint[]>([]);
  const output = ref<string[]>([]);
  const error = ref("");
  // metrics live outside Vue's deep reactivity (thousands of points); `version` signals changes
  const series = shallowRef<SeriesMap>(new Map());
  const version = ref(0);
  const startedAt = ref<number | null>(null);

  let offset = 0;
  // a response for a run that is no longer selected is dropped
  let generation = 0;
  let busy = false;

  function reset() {
    generation += 1;
    run.value = null;
    progress.value = null;
    samples.value = [];
    checkpoints.value = [];
    output.value = [];
    error.value = "";
    series.value = new Map();
    version.value += 1;
    startedAt.value = null;
    offset = 0;
  }

  async function loadEvents(id: string, current: number) {
    // a long run comes in pages; keep reading until caught up
    for (let page = 0; page < 50; page += 1) {
      const data = await api<{ events: RunEvent[]; offset: number }>(`/api/runs/${encodeURIComponent(id)}/events?offset=${offset}`);
      if (current !== generation) return;
      let changed = false;
      for (const event of data.events ?? []) {
        if (event.type === "metrics") {
          addMetrics(series.value, event as MetricsEvent);
          changed = true;
        } else if (event.type === "start" && startedAt.value === null) {
          startedAt.value = event.t;
        }
      }
      if (changed) version.value += 1;
      const moved = data.offset !== offset;
      offset = data.offset;
      if (!moved || !data.events?.length) return;
    }
  }

  async function loadOutput(id: string, current: number) {
    const data = await api<{ lines: string[] }>(`/api/runs/${encodeURIComponent(id)}/output`);
    if (current === generation) output.value = data.lines ?? [];
  }

  async function refresh() {
    const id = runId.value;
    if (!id || busy) return;
    const current = generation;
    busy = true;
    try {
      const detail = await api<{ run: Run; progress: RunProgress; samples: Sample[]; checkpoints: Checkpoint[] }>(
        `/api/runs/${encodeURIComponent(id)}`,
      );
      if (current !== generation) return;
      run.value = detail.run;
      progress.value = detail.progress;
      samples.value = detail.samples ?? [];
      checkpoints.value = detail.checkpoints ?? [];
      await loadEvents(id, current);
      if (withOutput.value) await loadOutput(id, current);
      error.value = "";
    } catch (e) {
      if (current === generation) error.value = e instanceof Error ? e.message : String(e);
    } finally {
      busy = false;
    }
  }

  watch(runId, () => {
    reset();
    void refresh();
  });
  // opening the console loads it right away, not with the next poll
  watch(withOutput, (show) => {
    if (show && runId.value) void loadOutput(runId.value, generation).catch(() => undefined);
  });

  // a finished run is fetched once; an active one keeps polling
  usePolling(async () => {
    if (!run.value || isActive(run.value.status)) await refresh();
  });

  const now = ref(Date.now() / 1000);
  usePolling(() => {
    now.value = Date.now() / 1000;
  }, 1000);

  const elapsed = computed(() => {
    if (!run.value) return null;
    return (run.value.ended ?? now.value) - run.value.started;
  });

  const remaining = computed(() => {
    const step = progress.value?.step;
    if (!step || !step.rate || !isActive(run.value?.status)) return null;
    return Math.max(0, (step.total - step.n) / step.rate);
  });

  return { run, progress, samples, checkpoints, output, error, series, version, elapsed, remaining, refresh, startedAt };
}
