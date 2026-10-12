interface ApiEnvelope {
  status?: "success" | "fail" | "pending";
  message?: string;
  data?: Record<string, unknown>;
}

export type ApiResult<T> = T & { message?: string };

/** Call a backend endpoint; throws with the backend's message when it answers `status: fail`. */
export async function api<T = Record<string, unknown>>(path: string, options?: RequestInit): Promise<ApiResult<T>> {
  const response = await fetch(path, options);
  const text = await response.text();
  let json: ApiEnvelope;
  try {
    json = JSON.parse(text);
  } catch {
    const hint = text.trim().startsWith("<")
      ? "API 返回了 HTML。请通过 Python GUI 后端访问页面，而不是只用静态文件服务器。"
      : "API 返回了无法解析的内容。";
    throw new Error(`${hint} (${path})`);
  }
  if (json.status === "fail") throw new Error(json.message || "请求失败");
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
