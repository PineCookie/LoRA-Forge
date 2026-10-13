<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { t } from "../i18n";
import { chartSpecs, smooth, type SeriesMap } from "../monitor/metrics";
import LineChart, { type ChartSeries } from "./LineChart.vue";

const props = defineProps<{ series: SeriesMap; version: number; lrGroups?: Record<string, string> }>();

// learning rate charts get a line per parameter group (network_reg_lrs, LoRA+, text encoders)
const COLORS = [
  "var(--tone-model)",
  "var(--tone-network)",
  "var(--tone-save)",
  "var(--tone-optimize)",
  "var(--tone-performance)",
  "var(--tone-data)",
  "var(--tone-observe)",
  "var(--tone-general)",
];
const SMOOTHING_KEY = "lora-forge.chartSmoothing";
const HIDDEN_KEY = "lora-forge.chartHidden";

function savedSmoothing() {
  try {
    const value = Number(localStorage.getItem(SMOOTHING_KEY));
    if (localStorage.getItem(SMOOTHING_KEY) !== null && value >= 0 && value < 1) return value;
  } catch {
    // storage blocked: use the default
  }
  return 0.9;
}

const smoothing = ref(savedSmoothing());
watch(smoothing, (value) => {
  try {
    localStorage.setItem(SMOOTHING_KEY, String(value));
  } catch {
    // only kept for this page load
  }
});

// lines the viewer hid by clicking the legend ("<chart>/<metric>", "/raw" for the values under a
// smoothed line); the y axis fits the lines still shown
function savedHidden() {
  try {
    const value = JSON.parse(localStorage.getItem(HIDDEN_KEY) ?? "[]");
    if (Array.isArray(value)) return new Set(value.map(String));
  } catch {
    // storage blocked or unreadable: show everything
  }
  return new Set<string>();
}

const hidden = ref(savedHidden());

function toggle(id: string) {
  const next = new Set(hidden.value);
  if (!next.delete(id)) next.add(id);
  hidden.value = next;
  try {
    localStorage.setItem(HIDDEN_KEY, JSON.stringify([...next]));
  } catch {
    // only kept for this page load
  }
}

interface LegendEntry {
  id: string;
  label: string;
  color: string;
  faint: boolean;
  hidden: boolean;
}

const charts = computed(() => {
  void props.version; // the series map is updated in place
  return chartSpecs(props.series.keys(), props.lrGroups).map((spec) => {
    // drawn in this order: raw values under their smoothed line
    const lines: ChartSeries[] = [];
    const legend: LegendEntry[] = [];
    const add = (id: string, line: ChartSeries) => {
      const off = hidden.value.has(id);
      legend.push({ id, label: line.faint ? `${line.label} (${t("runs.legend.raw")})` : line.label, color: line.color, faint: !!line.faint, hidden: off });
      if (!off) lines.push(line);
    };
    spec.lines.forEach((line, index) => {
      const data = props.series.get(line.key);
      if (!data) return;
      const color = COLORS[index % COLORS.length];
      const id = `${spec.id}/${line.key}`;
      if (line.smoothed && smoothing.value > 0) {
        add(`${id}/raw`, { label: line.label, color, xs: data.xs, ys: data.ys, faint: true });
        add(id, { label: line.label, color, xs: data.xs, ys: smooth(data.ys, smoothing.value) });
        // the legend lists the smoothed line first
        legend.push(...legend.splice(-2, 1));
      } else {
        add(id, { label: line.label, color, xs: data.xs, ys: data.ys });
      }
    });
    return { spec, lines, legend, title: "key" in spec.title ? t(spec.title.key) : spec.title.text };
  });
});

const hasSmoothed = computed(() => charts.value.some(({ spec }) => spec.lines.some((line) => line.smoothed)));
</script>

<template>
  <section class="panel charts">
    <header>
      <h2>{{ t("runs.metrics") }}</h2>
      <label v-if="hasSmoothed" class="smoothing">
        {{ t("runs.smoothing") }}
        <input v-model.number="smoothing" type="range" min="0" max="0.99" step="0.01" />
        <span class="value">{{ smoothing.toFixed(2) }}</span>
      </label>
    </header>
    <p v-if="!charts.length" class="muted">{{ t("runs.noMetrics") }}</p>
    <div class="grid">
      <figure v-for="chart in charts" :key="chart.spec.id">
        <figcaption>
          {{ chart.title }}
          <span class="legend">
            <button
              v-for="entry in chart.legend"
              :key="entry.id"
              type="button"
              class="legend-item"
              :class="{ off: entry.hidden }"
              :aria-pressed="!entry.hidden"
              :title="t('runs.legend.toggle')"
              @click="toggle(entry.id)"
            >
              <span class="swatch" :class="{ faint: entry.faint }" :style="{ background: entry.color }" />{{ entry.label }}
            </button>
          </span>
        </figcaption>
        <LineChart :series="chart.lines" :x-label="chart.spec.axis === 'epoch' ? t('runs.axis.epoch') : t('runs.axis.step')" />
      </figure>
    </div>
  </section>
</template>

<style scoped>
header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px 16px;
  margin-bottom: 10px;
}

h2 {
  margin: 0;
  font-size: 17px;
}

.smoothing {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--muted);
  font-size: 13px;
}

.smoothing input {
  width: 140px;
  padding: 0;
  border: 0;
  box-shadow: none;
  accent-color: var(--primary);
}

.value {
  min-width: 32px;
  font-variant-numeric: tabular-nums;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 440px), 1fr));
  gap: 18px 24px;
}

figure {
  min-width: 0;
  margin: 0;
}

figcaption {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 4px 12px;
  margin-bottom: 4px;
  color: var(--text-strong);
  font-size: 13px;
  font-weight: 600;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 2px 10px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 400;
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 1px 4px;
  border: 0;
  border-radius: var(--radius-sm);
  background: none;
  box-shadow: none;
  color: inherit;
  font: inherit;
  cursor: pointer;
}

.legend-item:hover {
  background: var(--surface-sunken);
  color: var(--text-strong);
}

.legend-item.off {
  opacity: 0.45;
  text-decoration: line-through;
}

.swatch {
  width: 12px;
  height: 3px;
  border-radius: 2px;
}

.swatch.faint {
  opacity: 0.35;
}
</style>
