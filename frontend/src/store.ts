import { markRaw, reactive } from "vue";
import { api, loadGraphicCards } from "./api";
import { evaluateSchema, type SchemaNode } from "./schema/dsl";

export interface Runtime {
  services: Record<string, boolean>;
  versions: Record<string, string>;
}

interface RawSchema {
  name: string;
  schema: string;
}

export const store = reactive({
  ready: false,
  fatal: "",
  runtime: null as Runtime | null,
  cards: [] as string[],
  /** collapsed form sections, kept while navigating between pages */
  collapsedSections: new Set<string>(),
});

// evaluated schemas never change; keep them out of Vue's reactivity
const schemas = new Map<string, SchemaNode>();

export function getSchema(name: string) {
  return schemas.get(name);
}

async function loadRuntime() {
  try {
    store.runtime = await api<Runtime>("/api/runtime");
  } catch (error) {
    console.warn("Failed to load runtime information", error);
    store.runtime = { services: {}, versions: {} };
  }
}

async function loadSchemas() {
  const data = await api<{ schemas: RawSchema[] }>("/api/schemas/all");
  const rawSchemas = data.schemas ?? [];
  const shared = rawSchemas.find((schema) => schema.name === "shared");
  const sharedSchemas = shared ? evaluateSchema(shared.schema, null) : {};

  for (const entry of rawSchemas) {
    if (entry.name === "shared") continue;
    try {
      schemas.set(entry.name, markRaw(evaluateSchema(entry.schema, sharedSchemas) as SchemaNode));
    } catch (error) {
      console.warn(`Failed to evaluate schema ${entry.name}`, error);
    }
  }
}

export async function loadApp() {
  await loadRuntime();
  try {
    await loadSchemas();
    store.cards = await loadGraphicCards();
    store.ready = true;
  } catch (error) {
    store.fatal = error instanceof Error ? error.stack || error.message : String(error);
  }
}
