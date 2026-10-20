<script setup lang="ts">
import { onMounted, watch } from "vue";
import { RouterView, useRoute } from "vue-router";
import AppSidebar from "./components/AppSidebar.vue";
import ConfirmDialog from "./components/ConfirmDialog.vue";
import ToastHost from "./components/ToastHost.vue";
import { locale, t } from "./i18n";
import { loadApp, loadRuntime, store } from "./store";

const route = useRoute();

onMounted(loadApp);

watch(
  locale,
  (value, previous) => {
    document.documentElement.lang = value;
    // the backend words the sd-scripts status in the request's language
    if (previous && store.ready) void loadRuntime();
  },
  { immediate: true },
);
</script>

<template>
  <AppSidebar />

  <main class="workspace">
    <section v-if="store.fatal" class="panel state fatal">
      <h2>{{ t("app.loadFailed") }}</h2>
      <pre>{{ store.fatal }}</pre>
    </section>
    <p v-else-if="!store.ready" class="state muted">{{ t("app.loading") }}</p>
    <RouterView v-else :key="route.path" />
  </main>

  <ToastHost />
  <ConfirmDialog />
</template>

<style scoped>
.workspace {
  min-width: 0;
  min-height: var(--viewport-height);
}

.state {
  margin: 24px;
}

.fatal {
  border-color: var(--danger-border);
  background: var(--danger-soft);
}

.fatal pre {
  margin: 0;
  font-size: 12px;
  white-space: pre-wrap;
}
</style>
