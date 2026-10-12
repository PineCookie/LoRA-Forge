<script setup lang="ts">
import { ref, watch } from "vue";
import { confirmState, settleConfirm } from "../composables/useConfirm";
import { t } from "../i18n";

const dialog = ref<HTMLDialogElement>();

watch(
  () => confirmState.request,
  (request) => {
    if (request && !dialog.value?.open) dialog.value?.showModal();
    if (!request && dialog.value?.open) dialog.value.close();
  },
);
</script>

<template>
  <!-- closing by Esc counts as cancel -->
  <dialog ref="dialog" class="confirm" @close="confirmState.request && settleConfirm(false)">
    <p>{{ confirmState.request?.message }}</p>
    <div class="buttons">
      <button type="button" @click="settleConfirm(false)">{{ t("confirm.cancel") }}</button>
      <button type="button" class="danger" autofocus @click="settleConfirm(true)">{{ confirmState.request?.confirmLabel }}</button>
    </div>
  </dialog>
</template>

<style scoped>
.confirm {
  width: min(420px, calc(100vw - 32px));
  padding: 20px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow-lg);
}

.confirm::backdrop {
  background: var(--backdrop);
}

p {
  margin: 0 0 18px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.buttons {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
</style>
