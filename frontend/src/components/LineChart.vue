<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { downsample, formatValue, nearestIndex, niceTicks } from "../monitor/metrics";

export interface ChartSeries {
  label: string;
  /** a CSS color, usually a token variable */
  color: string;
  xs: number[];
  ys: number[];
  /** raw values under a smoothed line: thin and translucent, left out of the tooltip */
  faint?: boolean;
}

const props = defineProps<{ series: ChartSeries[]; xLabel: string }>();

const HEIGHT = 220;
const PAD = { left: 58, right: 14, top: 12, bottom: 28 };

const root = ref<HTMLElement>();
const width = ref(600);
let observer: ResizeObserver | undefined;

onMounted(() => {
  observer = new ResizeObserver(([entry]) => (width.value = Math.max(240, Math.round(entry.contentRect.width))));
  if (root.value) observer.observe(root.value);
});
onBeforeUnmount(() => observer?.disconnect());

const plotWidth = computed(() => width.value - PAD.left - PAD.right);
const plotHeight = HEIGHT - PAD.top - PAD.bottom;

const drawn = computed(() =>
  props.series
    .filter((line) => line.xs.length)
    .map((line) => ({ ...line, ...downsample(line.xs, line.ys, Math.max(64, plotWidth.value * 2)) })),
);

const bounds = computed(() => {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const line of drawn.value) {
    xMin = Math.min(xMin, line.xs[0]);
    xMax = Math.max(xMax, line.xs[line.xs.length - 1]);
    for (const y of line.ys) {
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
  }
  if (!Number.isFinite(xMin)) return null;
  if (xMin === xMax) xMax = xMin + 1;
  if (yMin === yMax) {
    const pad = Math.abs(yMin) * 0.1 || 1;
    yMin -= pad;
    yMax += pad;
  } else {
    const pad = (yMax - yMin) * 0.06;
    yMin -= pad;
    yMax += pad;
  }
  return { xMin, xMax, yMin, yMax };
});

const sx = (x: number) => PAD.left + ((x - bounds.value!.xMin) / (bounds.value!.xMax - bounds.value!.xMin)) * plotWidth.value;
const sy = (y: number) => PAD.top + (1 - (y - bounds.value!.yMin) / (bounds.value!.yMax - bounds.value!.yMin)) * plotHeight;

const paths = computed(() =>
  bounds.value
    ? drawn.value.map((line) => ({
        line,
        d: line.xs.map((x, i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(line.ys[i]).toFixed(1)}`).join(""),
      }))
    : [],
);

const xTicks = computed(() => (bounds.value ? niceTicks(bounds.value.xMin, bounds.value.xMax, Math.max(2, Math.floor(plotWidth.value / 90))) : []));
const yTicks = computed(() => (bounds.value ? niceTicks(bounds.value.yMin, bounds.value.yMax, 4) : []));

// --- hover -----------------------------------------------------------------------------------

const hoverX = ref<number | null>(null);

function onMove(event: PointerEvent) {
  if (!bounds.value) return;
  const rect = (event.currentTarget as SVGElement).getBoundingClientRect();
  const px = ((event.clientX - rect.left) / rect.width) * width.value;
  if (px < PAD.left || px > width.value - PAD.right) {
    hoverX.value = null;
    return;
  }
  hoverX.value = bounds.value.xMin + ((px - PAD.left) / plotWidth.value) * (bounds.value.xMax - bounds.value.xMin);
}

// values come from the full series, not the downsampled one
const hover = computed(() => {
  if (hoverX.value === null || !bounds.value) return null;
  const rows = props.series
    .filter((line) => !line.faint && line.xs.length)
    .map((line) => {
      const i = nearestIndex(line.xs, hoverX.value!);
      return { label: line.label, color: line.color, x: line.xs[i], y: line.ys[i] };
    });
  if (!rows.length) return null;
  const x = rows[0].x;
  const left = sx(x);
  return { x, left, rows, flip: left > width.value * 0.62 };
});
</script>

<template>
  <div ref="root" class="chart">
    <svg
      :viewBox="`0 0 ${width} ${HEIGHT}`"
      :width="width"
      :height="HEIGHT"
      role="img"
      @pointermove="onMove"
      @pointerleave="hoverX = null"
    >
      <g class="grid">
        <line v-for="tick in yTicks" :key="`y${tick}`" :x1="PAD.left" :x2="width - PAD.right" :y1="sy(tick)" :y2="sy(tick)" />
      </g>
      <g class="axis">
        <text v-for="tick in yTicks" :key="`yl${tick}`" :x="PAD.left - 8" :y="sy(tick)" text-anchor="end" dominant-baseline="middle">
          {{ formatValue(tick) }}
        </text>
        <text v-for="tick in xTicks" :key="`xl${tick}`" :x="sx(tick)" :y="HEIGHT - 8" text-anchor="middle">{{ tick }}</text>
        <!-- in the free corner below the value labels, so it never covers a tick -->
        <text :x="PAD.left - 8" :y="HEIGHT - 8" text-anchor="end" class="x-label">{{ xLabel }}</text>
      </g>
      <path
        v-for="{ line, d } in paths"
        :key="line.label + (line.faint ? ':raw' : '')"
        :d="d"
        class="line"
        :class="{ faint: line.faint }"
        :style="{ stroke: line.color }"
      />
      <g v-if="hover">
        <line class="cursor" :x1="hover.left" :x2="hover.left" :y1="PAD.top" :y2="HEIGHT - PAD.bottom" />
        <circle v-for="row in hover.rows" :key="row.label" :cx="sx(row.x)" :cy="sy(row.y)" r="3.5" :style="{ fill: row.color }" />
      </g>
    </svg>
    <div v-if="hover" class="tooltip" :style="hover.flip ? { right: `${width - hover.left + 10}px` } : { left: `${hover.left + 10}px` }">
      <div class="tooltip-x">{{ xLabel }} {{ hover.x }}</div>
      <div v-for="row in hover.rows" :key="row.label" class="tooltip-row">
        <span class="swatch" :style="{ background: row.color }" />
        <span>{{ row.label }}</span>
        <strong>{{ formatValue(row.y) }}</strong>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chart {
  position: relative;
  width: 100%;
  min-width: 0;
}

svg {
  display: block;
  width: 100%;
  height: auto;
  touch-action: none;
  user-select: none;
}

.grid line {
  stroke: var(--border);
  stroke-width: 1;
}

.axis text {
  fill: var(--faint);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.x-label {
  fill: var(--muted);
}

.line {
  fill: none;
  stroke-width: 1.8;
  stroke-linejoin: round;
  stroke-linecap: round;
}

.line.faint {
  stroke-width: 1;
  opacity: 0.28;
}

.cursor {
  stroke: var(--border-strong);
  stroke-dasharray: 3 3;
}

circle {
  stroke: var(--surface);
  stroke-width: 1.5;
}

.tooltip {
  position: absolute;
  top: 6px;
  z-index: 2;
  display: grid;
  gap: 3px;
  min-width: 140px;
  padding: 8px 10px;
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  background: var(--surface);
  box-shadow: var(--shadow-md);
  font-size: 12px;
  pointer-events: none;
}

.tooltip-x {
  color: var(--muted);
}

.tooltip-row {
  display: grid;
  grid-template-columns: 10px 1fr auto;
  align-items: center;
  gap: 6px;
}

.tooltip-row strong {
  font-variant-numeric: tabular-nums;
}

.swatch {
  width: 10px;
  height: 3px;
  border-radius: 2px;
}
</style>
