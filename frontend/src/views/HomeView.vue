<script setup lang="ts">
import { RouterLink } from "vue-router";
import VersionInfo from "../components/VersionInfo.vue";
import { t, type MessageKey } from "../i18n";

const modes: { path: string; title: string; text: MessageKey }[] = [
  { path: "/lora/anima.html", title: "Anima", text: "home.mode.anima" },
  { path: "/lora/flux.html", title: "Flux", text: "home.mode.flux" },
  { path: "/lora/master.html", title: "Stable Diffusion", text: "home.mode.master" },
  { path: "/lora/sd3.html", title: "SD3.5", text: "home.mode.sd3" },
  { path: "/lora/basic.html", title: "SD1.5", text: "home.mode.basic" },
  { path: "/dreambooth/index.html", title: "Dreambooth", text: "home.mode.dreambooth" },
];
</script>

<template>
  <div class="home">
    <section class="panel">
      <h2>{{ t("home.title") }}</h2>
      <p class="muted">{{ t("home.intro") }}</p>
    </section>
    <div class="mode-grid">
      <RouterLink v-for="mode in modes" :key="mode.path" :to="mode.path" class="panel mode-card">
        <strong>{{ mode.title }}</strong>
        <span>{{ t(mode.text) }}</span>
      </RouterLink>
    </div>
    <VersionInfo />
  </div>
</template>
