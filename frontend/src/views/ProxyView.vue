<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import { store } from "../store";

const props = defineProps<{ proxy: string; service: string }>();
const route = useRoute();
const available = computed(() => store.runtime?.services?.[props.service] !== false);
</script>

<template>
  <iframe v-if="available" class="proxy-frame" :src="proxy" :title="route.meta.title" />
  <section v-else class="panel service-unavailable">
    <h2>{{ route.meta.title }} 未启动</h2>
    <p>当前 UI 启动时没有启用这个服务，因此不会连接 {{ proxy }}。</p>
    <p>需要使用时，请用 <code>--enable-{{ service }}</code> 重新启动 GUI，例如 <code>run_gui.ps1 --enable-{{ service }}</code>。</p>
  </section>
</template>
