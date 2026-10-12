import { reactive, ref } from "vue";

export type ToastType = "info" | "success" | "warning" | "error";

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

export const toasts = reactive<Toast[]>([]);
/** last message, also shown in the top bar */
export const statusMessage = ref("");

let nextId = 0;

export function dismissToast(id: number) {
  const index = toasts.findIndex((toast) => toast.id === id);
  if (index >= 0) toasts.splice(index, 1);
}

export function notify(message: string, type: ToastType = "info") {
  statusMessage.value = message;
  const id = (nextId += 1);
  toasts.push({ id, message, type });
  setTimeout(() => dismissToast(id), type === "error" ? 11000 : 7200);
}
