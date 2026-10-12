<script setup lang="ts">
import { RouterLink } from "vue-router";
import PageHeader from "../components/PageHeader.vue";
import VersionInfo from "../components/VersionInfo.vue";
import { t, type MessageKey } from "../i18n";
import { TRAINER_PAGES } from "../pages";
import { labelText } from "../router";

const DESCRIPTIONS: Record<string, MessageKey> = {
  "anima-lora": "home.mode.anima",
  "sdxl-lora": "home.mode.sdxl",
  "flux-lora": "home.mode.flux",
  "sd3-lora": "home.mode.sd3",
  "sdxl-finetune": "home.mode.sdxlFinetune",
  "sd-lora": "home.mode.sd15",
  "sd-dreambooth": "home.mode.sdDreambooth",
};

const modes = TRAINER_PAGES.map((page) => ({ path: page.path, title: page.title, text: DESCRIPTIONS[page.schemaName] }));
</script>

<template>
  <PageHeader :title="t('home.title')" />
  <div class="home">
    <p class="muted">{{ t("home.intro") }}</p>
    <div class="modes">
      <RouterLink v-for="mode in modes" :key="mode.path" :to="mode.path" class="panel mode">
        <strong>{{ labelText(mode.title) }}</strong>
        <span v-if="mode.text">{{ t(mode.text) }}</span>
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
