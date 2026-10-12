import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { evaluateSchema, SchemaNode } from "../src/schema/dsl";
import { buildGroups } from "../src/schema/groups";

const schemaDir = fileURLToPath(new URL("../../mikazuki/schema/", import.meta.url));

/** Groups for one of the real backend schema files, as the trainer page builds them. */
export function loadGroups(name: string, cards: string[] = []) {
  const shared = evaluateSchema(readFileSync(`${schemaDir}shared.ts`, "utf8"), null);
  const schema = evaluateSchema(readFileSync(`${schemaDir}${name}.ts`, "utf8"), shared) as SchemaNode;
  return buildGroups(schema, name, cards);
}
