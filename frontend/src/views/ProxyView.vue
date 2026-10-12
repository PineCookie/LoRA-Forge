<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import { t } from "../i18n";
import { labelText } from "../router";
import { store } from "../store";

const props = defineProps<{ proxy: string; service: string }>();
const route = useRoute();
const available = computed(() => store.runtime?.services?.[props.service] !== false);
const name = computed(() => labelText(route.meta.title));
</script>

<template>
  <iframe v-if="available" class="proxy-frame" :src="proxy" :title="name" />
  <section v-else class="panel service-unavailable">
    <h2>{{ t("proxy.notRunning", { name }) }}</h2>
    <p>{{ t("proxy.notStarted", { proxy }) }}</p>
    <p v-html="t('proxy.howToStart', { service })" />
  </section>
</template>
