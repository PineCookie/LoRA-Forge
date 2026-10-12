<script setup lang="ts">
import { RouterLink } from "vue-router";
import PageHeader from "../components/PageHeader.vue";
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
  <PageHeader :title="t('home.title')" />
  <div class="home">
    <p class="muted">{{ t("home.intro") }}</p>
    <div class="modes">
      <RouterLink v-for="mode in modes" :key="mode.path" :to="mode.path" class="panel mode">
        <strong>{{ mode.title }}</strong>
        <span>{{ t(mode.text) }}</span>
      </RouterLink>
    </div>
    <VersionInfo class="system" />
  </div>
</template>

<style scoped>
.home {
  display: grid;
  gap: 18px;
  max-width: 980px;
  padding: 20px 24px 48px;
}

.modes {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 12px;
}

.mode {
  display: grid;
  align-content: start;
  gap: 6px;
  border-top: 3px solid var(--primary);
  transition: box-shadow 0.16s ease, transform 0.16s ease;
}

.mode:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}

.mode strong {
  color: var(--text-strong);
}

.mode span {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}

.system {
  max-width: 480px;
}
</style>
