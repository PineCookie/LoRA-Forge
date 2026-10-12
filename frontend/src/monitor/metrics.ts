// Training metrics recorded by lora_forge/training/run_entry.py, grouped into charts. Pure
// functions; the Runs page keeps the series and renders them with LineChart.

import { t, type MessageKey } from "../i18n";

export interface RunEvent {
  t: number;
  type: "start" | "progress" | "epoch" | "metrics";
  [field: string]: unknown;
}

export interface MetricsEvent extends RunEvent {
  type: "metrics";
  step: number | null;
  values: Record<string, number>;
}

/** One metric over the run: x is the step, or the epoch for per-epoch metrics. */
export interface Series {
  xs: number[];
  ys: number[];
}

export type SeriesMap = Map<string, Series>;

/** sd-scripts logs per-epoch values (loss/epoch_average, loss/validation/epoch_*) with the epoch as their step. */
export const isEpochMetric = (key: string) => /(^|\/)epoch_/.test(key);

export function addMetrics(series: SeriesMap, event: MetricsEvent) {
  if (event.step == null) return;
  for (const [key, value] of Object.entries(event.values)) {
    if (!Number.isFinite(value)) continue;
    let entry = series.get(key);
    if (!entry) series.set(key, (entry = { xs: [], ys: [] }));
    const last = entry.xs.length - 1;
    // a resumed run logs some steps again: the newer value replaces the older one
    if (last >= 0 && event.step <= entry.xs[last]) {
      const at = entry.xs.findIndex((x) => x >= event.step!);
      entry.xs.splice(at);
      entry.ys.splice(at);
    }
    entry.xs.push(event.step);
    entry.ys.push(value);
  }
}

export interface ChartLine {
  key: string;
  label: string;
  /** drawn faint, with an EMA-smoothed copy on top (noisy per-step values) */
  smoothed?: boolean;
}

export interface ChartSpec {
  id: string;
  /** a message key for known charts, else the metric prefix as it is */
  title: { key: MessageKey } | { text: string };
  axis: "step" | "epoch";
  lines: ChartLine[];
}

const KNOWN: { id: string; title: MessageKey; match: (key: string) => boolean; axis?: "epoch" }[] = [
  { id: "loss", title: "runs.chart.loss", match: (key) => key === "loss/current" || key === "loss/average" },
  { id: "validation", title: "runs.chart.validation", match: (key) => key.startsWith("loss/validation/") && !isEpochMetric(key) },
  { id: "epoch", title: "runs.chart.epoch", match: isEpochMetric, axis: "epoch" },
  // Prodigy / DAdaptation: d * lr per parameter group; ProdigyPlusScheduleFree also logs the step
  // size after its schedule-free averaging (d * effective_lr)
  { id: "deff", title: "runs.chart.deff", match: (key) => key.startsWith("lr/d*eff_lr/") },
  { id: "dlr", title: "runs.chart.dlr", match: (key) => key.startsWith("lr/d*lr/") },
  { id: "lr", title: "runs.chart.lr", match: (key) => key.startsWith("lr/") },
];

/**
 * A learning rate group as sd-scripts names it ("unet", "unet reg_lr_0", "textencoder 1 plus"),
 * with network_reg_lrs groups shown by their pattern (`.*self_attn.*` -> self_attn).
 */
export function lrGroupLabel(group: string, lrGroups: Record<string, string>): string {
  const named = group.replace(/reg_lr_\d+/g, (name) => {
    const pattern = lrGroups[name];
    return pattern ? pattern.replace(/^\.\*/, "").replace(/\.\*$/, "") || pattern : name;
  });
  // with regex groups, the plain group holds every module no pattern matched
  if (Object.keys(lrGroups).length && !/reg_lr_\d+/.test(group)) return t("runs.lrOther", { group: named });
  return named;
}

function lineLabel(key: string, lrGroups: Record<string, string>) {
  // per-epoch chart: loss/epoch_average -> average, loss/validation/epoch_average -> validation/average
  if (isEpochMetric(key)) return key.replace(/^loss\//, "").replace("epoch_", "");
  if (key.startsWith("loss/validation/")) return key.slice("loss/validation/".length);
  const lr = /^lr\/(?:d\*(?:eff_)?lr\/)?(.+)$/.exec(key);
  if (lr) return lrGroupLabel(lr[1], lrGroups);
  return key.includes("/") ? key.slice(key.indexOf("/") + 1) : key;
}

/**
 * The charts for the metric keys a run has logged, in a fixed order; unknown keys are grouped by
 * prefix. Every learning rate group goes into one chart per kind, as lines (`lrGroups`: see
 * lrGroupLabel).
 */
export function chartSpecs(keys: Iterable<string>, lrGroups: Record<string, string> = {}): ChartSpec[] {
  const charts = new Map<string, ChartSpec>();
  for (const key of [...keys].sort()) {
    const known = KNOWN.find((chart) => chart.match(key));
    const id = known?.id ?? `other:${key.includes("/") ? key.slice(0, key.indexOf("/")) : key}`;
    let chart = charts.get(id);
    if (!chart) {
      chart = {
        id,
        title: known ? { key: known.title } : { text: id.slice("other:".length) },
        axis: known?.axis ?? "step",
        lines: [],
      };
      charts.set(id, chart);
    }
    chart.lines.push({ key, label: lineLabel(key, lrGroups), smoothed: key === "loss/current" });
  }
  const order = KNOWN.map((chart) => chart.id);
  const rank = (id: string) => (order.includes(id) ? order.indexOf(id) : order.length);
  return [...charts.values()].sort((a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id));
}

/** TensorBoard's smoothing: an exponential moving average, debiased for the first points. */
export function smooth(ys: number[], weight: number): number[] {
  if (weight <= 0) return ys.slice();
  const result = new Array<number>(ys.length);
  let last = 0;
  for (let i = 0; i < ys.length; i += 1) {
    last = last * weight + (1 - weight) * ys[i];
    result[i] = last / (1 - weight ** (i + 1));
  }
  return result;
}

/**
 * At most about `maxPoints` points to draw: the lowest and highest value of each bucket, in order,
 * so spikes stay visible.
 */
export function downsample(xs: number[], ys: number[], maxPoints: number): Series {
  if (xs.length <= maxPoints) return { xs, ys };
  const bucket = Math.ceil(xs.length / Math.max(1, Math.floor(maxPoints / 2)));
  const out: Series = { xs: [], ys: [] };
  for (let start = 0; start < xs.length; start += bucket) {
    const end = Math.min(start + bucket, xs.length);
    let lo = start;
    let hi = start;
    for (let i = start + 1; i < end; i += 1) {
      if (ys[i] < ys[lo]) lo = i;
      if (ys[i] > ys[hi]) hi = i;
    }
    for (const i of lo <= hi ? [lo, hi] : [hi, lo]) {
      if (out.xs.length && out.xs[out.xs.length - 1] === xs[i]) continue;
      out.xs.push(xs[i]);
      out.ys.push(ys[i]);
    }
  }
  return out;
}

/** Index of the point whose x is closest to `x` (xs ascending). */
export function nearestIndex(xs: number[], x: number): number {
  let lo = 0;
  let hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && Math.abs(xs[lo - 1] - x) <= Math.abs(xs[lo] - x) ? lo - 1 : lo;
}

/** About `count` round tick values covering [min, max]. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  // d3's tick step: 1, 2 or 5 times a power of ten, whichever is closest to range / count
  const raw = (max - min) / Math.max(1, count);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const error = raw / magnitude;
  const step = magnitude * (error >= Math.sqrt(50) ? 10 : error >= Math.sqrt(10) ? 5 : error >= Math.sqrt(2) ? 2 : 1);
  const ticks = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-9; value += step) {
    ticks.push(Number(value.toPrecision(12)));
  }
  return ticks;
}

/** A metric value for display: small values (learning rates) in exponent form. */
export function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(2);
  return Number(value.toPrecision(4)).toString();
}

/** h:mm:ss or m:ss */
export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "–";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return `${h ? `${h}:` : ""}${mm}:${String(s).padStart(2, "0")}`;
}

/** Steps per second as it/s, or s/it when a step takes longer than a second (as tqdm does). */
export function formatRate(rate: number | null | undefined): string {
  if (!rate || !Number.isFinite(rate)) return "–";
  return rate >= 1 ? `${rate.toFixed(2)} it/s` : `${(1 / rate).toFixed(2)} s/it`;
}

export function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit ? value.toFixed(1) : value} ${units[unit]}`;
}
