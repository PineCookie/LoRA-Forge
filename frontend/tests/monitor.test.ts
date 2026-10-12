import { beforeEach, describe, expect, it } from "vitest";
import { locale } from "../src/i18n";
import { phaseLabel } from "../src/monitor/labels";
import {
  addMetrics,
  chartSpecs,
  downsample,
  formatDuration,
  formatRate,
  formatValue,
  lrGroupLabel,
  nearestIndex,
  niceTicks,
  smooth,
  type MetricsEvent,
  type SeriesMap,
} from "../src/monitor/metrics";

beforeEach(() => {
  locale.value = "en";
});

const metrics = (step: number, values: Record<string, number>): MetricsEvent => ({ t: 0, type: "metrics", step, values });

describe("addMetrics", () => {
  it("collects each key as a series and replaces steps logged again after a resume", () => {
    const series: SeriesMap = new Map();
    for (let step = 1; step <= 4; step += 1) addMetrics(series, metrics(step, { "loss/current": step }));
    addMetrics(series, metrics(3, { "loss/current": 30 }));
    expect(series.get("loss/current")).toEqual({ xs: [1, 2, 3], ys: [1, 2, 30] });
  });

  it("skips non-finite values", () => {
    const series: SeriesMap = new Map();
    addMetrics(series, metrics(1, { a: Number.NaN, b: 2 }));
    expect([...series.keys()]).toEqual(["b"]);
  });
});

describe("chartSpecs", () => {
  it("groups sd-scripts' metric keys into charts in a fixed order", () => {
    const specs = chartSpecs([
      "lr/unet",
      "loss/current",
      "loss/average",
      "lr/d*lr/unet",
      "loss/epoch_average",
      "loss/validation/step_average",
      "loss/validation/epoch_average",
      "norm/avg_key_norm",
    ]);
    expect(specs.map((spec) => [spec.id, spec.axis, spec.lines.map((line) => line.label)])).toEqual([
      ["loss", "step", ["average", "current"]],
      ["validation", "step", ["step_average"]],
      ["epoch", "epoch", ["average", "validation/average"]],
      ["dlr", "step", ["unet"]],
      ["lr", "step", ["unet"]],
      ["other:norm", "step", ["avg_key_norm"]],
    ]);
    expect(specs[0].lines.find((line) => line.key === "loss/current")?.smoothed).toBe(true);
  });

  it("puts every network_reg_lrs group in one chart per kind, named by its pattern", () => {
    // network_args = ["network_reg_lrs=.*self_attn.*=1.0,.*cross_attn.*=1.0,.*mlp.*=1.0"] with
    // ProdigyPlusScheduleFree: sd-scripts logs lr, d*lr and d*eff_lr for each group
    const groups = ["unet reg_lr_0", "unet reg_lr_1", "unet reg_lr_2", "unet"];
    const keys = groups.flatMap((group) => [`lr/${group}`, `lr/d*lr/${group}`, `lr/d*eff_lr/${group}`]);
    const lrGroups = { reg_lr_0: ".*self_attn.*", reg_lr_1: ".*cross_attn.*", reg_lr_2: ".*mlp.*" };
    const specs = chartSpecs(keys, lrGroups);
    expect(specs.map((spec) => spec.id)).toEqual(["deff", "dlr", "lr"]);
    for (const spec of specs) {
      expect(spec.lines.map((line) => line.label).sort()).toEqual(["cross_attn", "mlp", "self_attn", "unet (other modules)"].map((name) => (name.startsWith("unet") ? name : `unet ${name}`)).sort());
    }
  });

  it("names learning rate groups", () => {
    expect(lrGroupLabel("unet", {})).toBe("unet");
    expect(lrGroupLabel("unet reg_lr_1 plus", { reg_lr_1: ".*attn.*" })).toBe("unet attn plus");
    expect(lrGroupLabel("unet reg_lr_5", { reg_lr_1: "x" })).toBe("unet reg_lr_5");
    expect(lrGroupLabel("textencoder 1", { reg_lr_0: "x" })).toBe("textencoder 1 (other modules)");
  });
});

describe("chart helpers", () => {
  it("smooths like TensorBoard (debiased EMA)", () => {
    expect(smooth([1, 1, 1], 0.9)).toEqual([1, 1, 1].map((v) => expect.closeTo(v, 10)));
    expect(smooth([0, 10], 0)).toEqual([0, 10]);
    const [first] = smooth([5, 0], 0.6);
    expect(first).toBeCloseTo(5);
  });

  it("downsamples to bucket minima and maxima, keeping spikes", () => {
    const xs = Array.from({ length: 1000 }, (_, i) => i);
    const ys = xs.map((x) => (x === 500 ? 100 : 1));
    const out = downsample(xs, ys, 100);
    expect(out.xs.length).toBeLessThanOrEqual(100);
    expect(Math.max(...out.ys)).toBe(100);
    expect(out.xs).toEqual([...out.xs].sort((a, b) => a - b));
  });

  it("finds the nearest point", () => {
    expect(nearestIndex([1, 5, 10], 6)).toBe(1);
    expect(nearestIndex([1, 5, 10], 8)).toBe(2);
    expect(nearestIndex([1, 5, 10], -3)).toBe(0);
  });

  it("makes round ticks", () => {
    expect(niceTicks(0, 1000, 5)).toEqual([0, 200, 400, 600, 800, 1000]);
    expect(niceTicks(0.0001, 0.00052, 4)).toEqual([0.0001, 0.0002, 0.0003, 0.0004, 0.0005]);
  });

  it("formats values, durations and speeds", () => {
    expect(formatValue(0.123456)).toBe("0.1235");
    expect(formatValue(0.0001)).toBe("1.00e-4");
    expect(formatDuration(59)).toBe("0:59");
    expect(formatDuration(3725)).toBe("1:02:05");
    expect(formatRate(1.5)).toBe("1.50 it/s");
    expect(formatRate(0.5)).toBe("2.00 s/it");
    expect(formatRate(null)).toBe("–");
  });

  it("names sd-scripts' progress bars", () => {
    expect(phaseLabel("steps")).toBe("Training");
    expect(phaseLabel("")).toBe("Preparing data");
    expect(phaseLabel("Loading model.safetensors")).toBe("Loading model.safetensors");
  });
});
