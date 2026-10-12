<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { api } from "../api";
import type { SubsetOffsets } from "../config/offsets";

interface Subdir {
  name: string;
  concept: string;
  repeat: number;
  image_count: number;
}

interface DatasetInfo {
  subdirs: Subdir[];
  total_images: number;
  total_images_with_repeats: number;
}

const props = defineProps<{
  dir: string;
  batchSize: number;
  gradAcc: number;
  epochs: number;
  supportsOffsets: boolean;
  timestepSampling: string;
  offsets: SubsetOffsets;
  /** bump to force a reload, e.g. after importing a config */
  refreshKey: number;
}>();

const info = ref<DatasetInfo | null>(null);
const loading = ref(false);
const error = ref("");

let timer: ReturnType<typeof setTimeout> | undefined;
let requestId = 0;
let loadedDir: string | null = null;
let loadedKey = -1;

function clearOffsets() {
  props.offsets.values = {};
  props.offsets.dir = null;
}

// Drop offsets for another dataset or for subfolders that no longer exist: /api/run rejects
// unknown folders, and the input to clear them would be gone from the UI.
function pruneOffsets(dir: string, result: DatasetInfo) {
  if (dir !== props.offsets.dir) {
    props.offsets.values = {};
    props.offsets.dir = dir;
  }
  const names = new Set(result.subdirs.map((subdir) => subdir.name));
  for (const name of Object.keys(props.offsets.values)) {
    if (!names.has(name)) delete props.offsets.values[name];
  }
}

watch(
  () => [props.dir.trim(), props.refreshKey] as const,
  ([dir, key]) => {
    clearTimeout(timer);
    if (!dir) {
      requestId += 1;
      info.value = null;
      error.value = "";
      loading.value = false;
      loadedDir = null;
      clearOffsets();
      return;
    }
    if (dir === loadedDir && key === loadedKey) return;

    loading.value = true;
    error.value = "";
    const id = (requestId += 1);
    timer = setTimeout(async () => {
      try {
        const result = await api<DatasetInfo>(`/api/dataset_info?train_data_dir=${encodeURIComponent(dir)}`);
        if (id !== requestId) return;
        info.value = result;
        loadedDir = dir;
        loadedKey = key;
        pruneOffsets(dir, result);
      } catch (err) {
        if (id !== requestId) return;
        info.value = null;
        loadedDir = null;
        error.value = err instanceof Error ? err.message : String(err);
      } finally {
        if (id === requestId) loading.value = false;
      }
    }, 400);
  },
  { immediate: true },
);

onBeforeUnmount(() => clearTimeout(timer));

const effectiveBatch = computed(() => props.batchSize * props.gradAcc);
// Lower bound: all images in one bucket -> ceil(total / batch). ARB buckets batch separately,
// so the real count is a little higher.
const stepsPerEpoch = computed(() => (info.value && effectiveBatch.value > 0 ? Math.ceil(info.value.total_images_with_repeats / effectiveBatch.value) : 0));
const totalSteps = computed(() => stepsPerEpoch.value * props.epochs);
const suggestions = computed(() =>
  [500, 1000, 1500, 2000, 3000].map((steps) => ({ steps, epochs: Math.max(1, Math.round(steps / stepsPerEpoch.value)) })),
);

const recommendation = computed(() => {
  if (["sigma", "uniform"].includes(props.timestepSampling)) return "sigma-uniform";
  if (["shift", "flux_shift"].includes(props.timestepSampling)) return "shift";
  return "default";
});

function setOffset(name: string, event: Event) {
  props.offsets.values[name] = (event.target as HTMLInputElement).value;
}
</script>

<template>
  <div class="dataset-stats">
    <span v-if="loading" class="dataset-stats-loading">正在计算数据集信息...</span>
    <span v-else-if="error" class="dataset-stats-error">无法读取数据集: {{ error }}</span>
    <div v-else-if="info" class="dataset-stats-panel">
      <div class="dataset-stats-title">📊 步数预测</div>

      <template v-if="info.subdirs.length">
        <table class="dataset-stats-table">
          <thead>
            <tr>
              <th>子文件夹</th>
              <th>重复次数</th>
              <th>图片数</th>
              <th>有效图片数</th>
              <th v-if="supportsOffsets">时间步偏移</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="subdir in info.subdirs" :key="subdir.name">
              <td>{{ subdir.concept || subdir.name }}</td>
              <td>×{{ subdir.repeat }}</td>
              <td>{{ subdir.image_count }}</td>
              <td>{{ subdir.image_count * subdir.repeat }}</td>
              <td v-if="supportsOffsets">
                <input
                  class="subset-timestep-offset"
                  type="number"
                  step="0.01"
                  :value="offsets.values[subdir.name] ?? 0"
                  title="负值：低噪声/细节；正值：高噪声/结构。为 0 时不改变原有行为。"
                  @input="setOffset(subdir.name, $event)"
                />
              </td>
            </tr>
          </tbody>
        </table>
        <template v-if="supportsOffsets">
          <p class="dataset-stats-note">
            Timestep offset：负值偏向低噪声（细节），正值偏向高噪声（结构）。记录仅适用于 FLUX / Anima 训练；0 保持旧行为。⚠ 需 sd-scripts main（&gt; v0.11.1，旧版本会静默忽略）。
          </p>
          <div class="timestep-offset-recommendation">
            <template v-if="recommendation === 'sigma-uniform'">
              <b>建议：</b>偏移在 <code>sigma</code> / <code>uniform</code> 模式下无效。如需使用该功能，请选择 <code>sigmoid</code> 作为起点。
            </template>
            <template v-else-if="recommendation === 'shift'">
              <b>建议：</b><code>shift</code> / <code>flux_shift</code> 已经偏向高 timestep。从 <b>±0.25</b> 开始，谨慎使用正偏移；用
              <code>show_timesteps=console</code> 先预览分布。
            </template>
            <template v-else>
              <b>推荐起点：</b><code>timestep_sampling=sigmoid</code>、<code>sigmoid_scale=1.0</code>。先保持全部为 0；细节/近景子集可试 <b>-0.25</b>，全身/架构子集可试
              <b>+0.25</b>。只在对比预览后再扩大到 <b>±0.5</b>。
            </template>
          </div>
        </template>
      </template>
      <p v-else class="dataset-stats-note">未检测到符合 <code>数字_概念名</code> 格式的子文件夹</p>

      <div class="dataset-stats-summary">
        <div class="stat-row"><span>原始图片总数</span><span>{{ info.total_images }}</span></div>
        <div class="stat-row"><span>乘以重复后的有效图片数</span><span>{{ info.total_images_with_repeats }}</span></div>
        <div class="stat-row"><span>有效 batch size (batch × grad_accum)</span><span>{{ effectiveBatch }}</span></div>
        <div class="stat-row"><span>每 epoch 最少步数</span><span>{{ stepsPerEpoch.toLocaleString() }}</span></div>
        <div class="stat-row stat-row-highlight">
          <span>{{ epochs }} epoch → 最少总步数</span><span>{{ totalSteps.toLocaleString() }}</span>
        </div>
      </div>

      <template v-if="info.total_images_with_repeats > 0 && stepsPerEpoch > 0">
        <div class="dataset-stats-note-bucket">
          ⚠ 以上为<b>下限估算</b>（假设所有图片落入同一分辨率桶）。ARB 分桶按宽高比将图片分入不同桶，每桶独立组 batch，碎片化会导致实际步数略高（通常 +1%~10%）。
        </div>
        <div class="dataset-stats-suggestions">
          <div class="dataset-stats-suggest-title">💡 参考 epoch 数（基于下限估算）</div>
          <span>
            <template v-for="(item, index) in suggestions" :key="item.steps">
              {{ index ? "； " : "" }}达到 ~{{ item.steps.toLocaleString() }} 步需要 {{ item.epochs }} epoch
            </template>
          </span>
        </div>
      </template>
    </div>
  </div>
</template>
