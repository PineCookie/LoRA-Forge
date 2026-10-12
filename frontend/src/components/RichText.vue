<script setup lang="ts">
import { computed } from "vue";
import { parseRichText } from "../richText";

const props = withDefaults(defineProps<{ text: string; tag?: string }>(), { tag: "span" });
const parts = computed(() => parseRichText(props.text));
</script>

<template>
  <component :is="tag">
    <template v-for="(part, index) in parts" :key="index">
      <br v-if="part.kind === 'br'" />
      <code v-else-if="part.kind === 'code'">{{ part.text }}</code>
      <strong v-else-if="part.kind === 'strong'">{{ part.text }}</strong>
      <template v-else>{{ part.text }}</template>
    </template>
  </component>
</template>
