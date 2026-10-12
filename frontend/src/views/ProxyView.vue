<script setup lang="ts">
import { computed } from "vue";
import { useRoute } from "vue-router";
import PageHeader from "../components/PageHeader.vue";
import RichText from "../components/RichText.vue";
import { t } from "../i18n";
import { labelText } from "../router";
import { store } from "../store";

const props = defineProps<{ proxy: string; service: string }>();
const route = useRoute();
const available = computed(() => store.runtime?.services?.[props.service] !== false);
const name = computed(() => labelText(route.meta.title));
</script>

<template>
  <PageHeader :title="name" />
  <div class="proxy">
    <iframe v-if="available" :src="proxy" :title="name" />
    <section v-else class="panel unavailable">
      <h2>{{ t("proxy.notRunning", { name }) }}</h2>
      <p>{{ t("proxy.notStarted", { proxy }) }}</p>
      <RichText tag="p" :text="t('proxy.howToStart', { service })" />
    </section>
  </div>
</template>

<style scoped>
.proxy {
  padding: 16px 24px 24px;
}

iframe {
  width: 100%;
  height: calc(100vh - var(--header-height) - 40px);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.unavailable {
  max-width: 720px;
}

.unavailable p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.5;
}
</style>
