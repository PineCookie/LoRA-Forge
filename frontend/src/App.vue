<script setup lang="ts">
import { onMounted } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";
import logoUrl from "../../assets/logo.svg";
import ToastHost from "./components/ToastHost.vue";
import { navigation } from "./router";
import { loadApp, store } from "./store";
import { statusMessage } from "./toast";

const route = useRoute();

onMounted(loadApp);
</script>

<template>
  <aside class="sidebar">
    <RouterLink class="brand" to="/">
      <img class="brand-logo" :src="logoUrl" alt="" />
      <span>LoRA-Forge</span>
    </RouterLink>
    <nav>
      <RouterLink v-for="item in navigation" :key="item.path" :to="item.path" exact-active-class="active">{{ item.label }}</RouterLink>
    </nav>
    <!-- AGPL-3.0 section 13: users of the UI over a network must be offered the source -->
    <p class="sidebar-footer">
      <a href="https://github.com/PineCookie/lora-forge" target="_blank" rel="noopener">源代码 · Source code</a>
      <span>AGPL-3.0</span>
    </p>
  </aside>

  <main class="workspace">
    <header class="topbar">
      <div>
        <p class="eyebrow">Training UI</p>
        <h1>{{ route.meta.title }}</h1>
      </div>
      <span id="status">{{ statusMessage }}</span>
    </header>
    <section id="content">
      <section v-if="store.fatal" class="panel fatal">
        <h2>前端加载失败</h2>
        <pre>{{ store.fatal }}</pre>
      </section>
      <section v-else-if="!store.ready" class="panel loading">正在加载训练配置…</section>
      <RouterView v-else :key="route.path" />
    </section>
  </main>

  <ToastHost />
</template>
