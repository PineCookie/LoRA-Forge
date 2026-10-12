<script setup lang="ts">
import { computed } from "vue";

const props = defineProps<{
  title: string;
  collapsed: boolean;
  note?: string;
}>();

defineEmits<{ toggle: [] }>();

const TONES: [RegExp, string][] = [
  [/快速配置/i, "quick"],
  [/模型|Model/i, "model"],
  [/数据集|caption|Tag/i, "data"],
  [/保存|输出/i, "save"],
  [/学习率|优化器|损失/i, "optimize"],
  [/网络|基础权重|分层/i, "network"],
  [/预览图|日志|Tensorboard|wandb/i, "observe"],
  [/噪声|增强|速度|精度|分布式|显卡/i, "performance"],
];

const tone = computed(() => `section-tone-${TONES.find(([pattern]) => pattern.test(props.title))?.[1] ?? "general"}`);
</script>

<template>
  <section class="panel form-section" :class="[tone, { 'is-collapsed': collapsed }]">
    <button type="button" class="section-header" :aria-expanded="!collapsed" @click="$emit('toggle')">
      <span class="section-title">{{ title }}</span>
      <span class="section-toggle-icon" aria-hidden="true">{{ collapsed ? "+" : "−" }}</span>
    </button>
    <div v-show="!collapsed" class="section-body">
      <p v-if="note" class="section-note">{{ note }}</p>
      <slot />
    </div>
  </section>
</template>
