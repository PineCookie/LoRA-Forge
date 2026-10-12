<script setup lang="ts">
import { ref, watch } from "vue";
import { t } from "../i18n";

const open = defineModel<boolean>("open", { required: true });
defineProps<{ toml: string; generatedArgs: { name: string; args: string[] }[] }>();
const emit = defineEmits<{ copy: [] }>();

const dialog = ref<HTMLDialogElement>();

watch(open, (value) => {
  if (value && !dialog.value?.open) dialog.value?.showModal();
  if (!value && dialog.value?.open) dialog.value.close();
});

// a click on the backdrop lands on the <dialog> element itself
function onDialogClick(event: MouseEvent) {
  if (event.target === dialog.value) open.value = false;
}
</script>

<template>
  <dialog ref="dialog" class="drawer" :aria-label="t('trainer.tomlPreview')" @close="open = false" @click="onDialogClick">
    <div class="content">
      <header>
        <h2>{{ t("trainer.tomlPreview") }}</h2>
        <button type="button" class="small" :title="t('trainer.copyTitle')" @click="emit('copy')">{{ t("trainer.copy") }}</button>
        <button type="button" class="ghost" :aria-label="t('drawer.close')" @click="open = false">×</button>
      </header>

      <pre class="code-block toml">{{ toml }}</pre>

      <section>
        <h3>{{ t("trainer.generatedArgs") }}</h3>
        <div class="code-block args">
          <div v-for="section in generatedArgs" :key="section.name">
            <strong>{{ section.name }}:</strong>
            <div v-for="arg in section.args" :key="arg" class="arg">{{ arg }}</div>
          </div>
          <span v-if="!generatedArgs.length" class="empty">{{ t("trainer.noGeneratedArgs") }}</span>
        </div>
      </section>
    </div>
  </dialog>
</template>

<style scoped>
.drawer {
  position: fixed;
  inset: 0 0 0 auto;
  width: min(560px, 100vw);
  max-width: none;
  height: 100vh;
  max-height: none;
  margin: 0;
  padding: 0;
  border: 0;
  border-left: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-lg);
}

.drawer[open] {
  animation: slide-in 0.18s ease;
}

.drawer::backdrop {
  background: var(--backdrop);
}

.content {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  padding: 18px 20px;
}

header {
  display: flex;
  align-items: center;
  gap: 10px;
}

h2 {
  flex: 1;
  margin: 0;
  font-size: 18px;
}

h3 {
  margin: 0 0 8px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.toml {
  flex: 1;
  min-height: 0;
}

.args {
  max-height: 22vh;
}

.arg {
  padding-left: 18px;
}

.empty {
  color: var(--faint);
}

@keyframes slide-in {
  from {
    transform: translateX(40px);
    opacity: 0;
  }
}
</style>
