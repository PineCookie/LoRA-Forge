<script setup lang="ts">
import { onMounted, watch } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";
import logoUrl from "../../assets/logo.svg";
import ToastHost from "./components/ToastHost.vue";
import { LOCALES, locale, setLocale, t, type Locale } from "./i18n";
import { labelText, navigation } from "./router";
import { loadApp, loadRuntime, store } from "./store";
import { statusMessage } from "./toast";

const route = useRoute();

onMounted(loadApp);

watch(
  locale,
  (value, previous) => {
    document.documentElement.lang = value;
    // the backend words the sd-scripts status in the request's language
    if (previous && store.ready) loadRuntime();
  },
  { immediate: true },
);

const onLocaleChange = (event: Event) => setLocale((event.target as HTMLSelectElement).value as Locale);
</script>

<template>
  <aside class="sidebar">
    <RouterLink class="brand" to="/">
      <img class="brand-logo" :src="logoUrl" alt="" />
      <span>LoRA-Forge</span>
    </RouterLink>
    <nav>
      <RouterLink v-for="item in navigation" :key="item.path" :to="item.path" exact-active-class="active">{{ labelText(item.label) }}</RouterLink>
    </nav>
    <div class="sidebar-footer">
      <select class="locale-select" :value="locale" :aria-label="t('app.language')" @change="onLocaleChange">
        <option v-for="(name, code) in LOCALES" :key="code" :value="code">{{ name }}</option>
      </select>
      <!-- AGPL-3.0 section 13: users of the UI over a network must be offered the source -->
      <p>
        <a href="https://github.com/PineCookie/lora-forge" target="_blank" rel="noopener">{{ t("app.sourceCode") }}</a>
        <span>AGPL-3.0</span>
      </p>
    </div>
  </aside>

  <main class="workspace">
    <header class="topbar">
      <div>
        <p class="eyebrow">{{ t("app.eyebrow") }}</p>
        <h1>{{ labelText(route.meta.title) }}</h1>
      </div>
      <span id="status">{{ statusMessage }}</span>
    </header>
    <section id="content">
      <section v-if="store.fatal" class="panel fatal">
        <h2>{{ t("app.loadFailed") }}</h2>
        <pre>{{ store.fatal }}</pre>
      </section>
      <section v-else-if="!store.ready" class="panel loading">{{ t("app.loading") }}</section>
      <RouterView v-else :key="route.path" />
    </section>
  </main>

  <ToastHost />
</template>
