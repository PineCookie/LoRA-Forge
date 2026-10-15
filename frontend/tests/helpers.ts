import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { evaluateSchema, SchemaNode } from "../src/schema/dsl";
import { buildGroups } from "../src/schema/groups";

export const schemaDir = fileURLToPath(new URL("../../lora_forge/schema/", import.meta.url));

/** Every trainer schema the backend serves. */
export const SCHEMA_NAMES = ["anima-lora", "flux-lora", "krea2-lora", "sd-dreambooth", "sd-lora", "sd3-lora", "sdxl-finetune", "sdxl-lora"];

export const loadShared = () => evaluateSchema(readFileSync(`${schemaDir}shared.ts`, "utf8"), null);

export function loadSchema(name: string) {
  return evaluateSchema(readFileSync(`${schemaDir}${name}.ts`, "utf8"), loadShared()) as SchemaNode;
}

/** Groups for one of the real backend schema files, as the trainer page builds them. */
export function loadGroups(name: string, cards: string[] = []) {
  return buildGroups(loadSchema(name), name, cards);
}
