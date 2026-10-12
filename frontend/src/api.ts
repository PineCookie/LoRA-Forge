import { locale, t } from "./i18n";

interface ApiEnvelope {
  status?: "success" | "fail" | "pending";
  message?: string;
  data?: Record<string, unknown>;
}

export type ApiResult<T> = T & { message?: string };

/** Call a backend endpoint; throws with the backend's message when it answers `status: fail`. */
export async function api<T = Record<string, unknown>>(path: string, options: RequestInit = {}): Promise<ApiResult<T>> {
  // the backend words its messages in the UI language
  const headers = new Headers(options.headers);
  headers.set("Accept-Language", locale.value);
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let json: ApiEnvelope;
  try {
    json = JSON.parse(text);
  } catch {
    const hint = text.trim().startsWith("<") ? t("api.html") : t("api.unparsable");
    throw new Error(`${hint} (${path})`);
  }
  if (json.status === "fail") throw new Error(json.message || t("api.failed"));
  return { message: json.message, ...json.data } as ApiResult<T>;
}

export function postJson<T = Record<string, unknown>>(path: string, body: unknown) {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** GPU list; the backend answers `pending` until torch has enumerated the devices. */
export async function loadGraphicCards(): Promise<string[]> {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const response = await fetch("/api/graphic_cards");
      const result: ApiEnvelope = await response.json();
      if (result.status === "success") return (result.data?.cards as string[] | undefined) ?? [];
      if (result.status !== "pending") return [];
    } catch (error) {
      console.warn("Failed to load GPU list", error);
      return [];
    }
    if (attempt < 2) await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  return [];
}
