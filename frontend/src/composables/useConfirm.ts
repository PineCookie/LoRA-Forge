import { shallowReactive } from "vue";

interface ConfirmRequest {
  message: string;
  confirmLabel: string;
  resolve: (confirmed: boolean) => void;
}

/** The open confirmation, rendered by ConfirmDialog in App.vue. */
export const confirmState = shallowReactive<{ request: ConfirmRequest | null }>({ request: null });

/** In-app replacement for window.confirm(); resolves true when the user confirms. */
export function confirmAction(message: string, confirmLabel: string): Promise<boolean> {
  confirmState.request?.resolve(false);
  return new Promise((resolve) => {
    confirmState.request = { message, confirmLabel, resolve };
  });
}

export function settleConfirm(confirmed: boolean) {
  const request = confirmState.request;
  confirmState.request = null;
  request?.resolve(confirmed);
}
