import { Schema, SchemaNode, type Primitive } from "./dsl";
import { FALLBACK_SECTION, SECTION_LAYOUT, sectionByTitle, type MergedSectionId, type Tone } from "./layout";

export type { MergedSectionId, Tone } from "./layout";

export interface Field {
  name: string;
  schema: SchemaNode;
}

export interface Group {
  /** English section title from the layout; also identifies the section (translated when shown) */
  title: string;
  fields: Field[];
}

/** The sections of a trainer page, in page order; each holds some of layout.ts's sections. */
export const MERGED_SECTIONS: { id: MergedSectionId; tone: Tone }[] = [
  { id: "model", tone: "model" },
  { id: "dataset", tone: "data" },
  { id: "captions", tone: "data" },
  { id: "training", tone: "optimize" },
  { id: "timesteps", tone: "optimize" },
  { id: "network", tone: "network" },
  { id: "saving", tone: "save" },
  { id: "preview", tone: "observe" },
  { id: "memory", tone: "performance" },
  { id: "speed", tone: "performance" },
];

/** The trainer page section a layout section belongs to. */
export function mergedSectionOf(title: string): MergedSectionId {
  return sectionByTitle(title).merged;
}

/** Accent colour of a section. */
export function groupTone(title: string): Tone {
  return sectionByTitle(title).tone;
}

/** A schema's options (plus the ones the UI adds itself), arranged in the sections of layout.ts. */
export function buildGroups(schema: SchemaNode, schemaName: string, cards: string[]): Group[] {
  const fields = withGpuSelector(withFrontendAnimaNetworkArgsFields(withFrontendOptimizerFields(schemaFields(schema)), schemaName), cards);
  return arrange(fields);
}

/** Every option of a schema, in definition order; the first definition of a name wins. */
export function schemaFields(schema: SchemaNode): Field[] {
  const fields = new Map<string, Field>();
  function visit(node: SchemaNode | undefined) {
    if (!node) return;
    if (node.type === "intersect" || node.type === "union") {
      for (const item of node.items ?? []) visit(item);
    } else if (node.type === "object") {
      for (const [name, field] of Object.entries(node.fields ?? {})) if (!fields.has(name)) fields.set(name, { name, schema: field });
    }
  }
  visit(schema);
  return [...fields.values()];
}

function arrange(fields: Field[]): Group[] {
  const byName = new Map(fields.map((field) => [field.name, field]));
  const groups: Group[] = [];
  for (const section of SECTION_LAYOUT) {
    const members = section.fields.flatMap((name) => {
      const field = byName.get(name);
      byName.delete(name);
      return field ? [field] : [];
    });
    if (members.length) groups.push({ title: section.title, fields: members });
  }
  if (byName.size) groups.push({ title: FALLBACK_SECTION.title, fields: [...byName.values()] });
  return groups;
}

function withGpuSelector(fields: Field[], cards: string[]): Field[] {
  if (cards.length < 2) return fields;
  const schema = Schema.array(Schema.union(cards)).default([cards[0]]).description("GPUs to use. When none is selected, the first GPU is used").role("select");
  return [...fields, { name: "gpu_ids", schema }];
}

function withFrontendOptimizerFields(fields: Field[]): Field[] {
  const names = new Set(fields.map((field) => field.name));
  const optimizerField = fields.find((field) => field.name === "optimizer_type");
  const optimizerOptions = optimizerField ? schemaOptions(optimizerField.schema) ?? [] : [];
  if (!optimizerOptions.includes("prodigyplus.ProdigyPlusScheduleFree")) return fields;

  const extra: [string, SchemaNode][] = [
    ["prodigy_d0", Schema.number().min(0).description("Prodigy d0: the initial LR estimate; try 1e-7, 1e-6 or 1e-5")],
    ["prodigy_d_coef", Schema.number().min(0).step(0.1).default("2.0").description("Prodigy d_coef. Default 2.0")],
    ["prodigyplus_d_coef", Schema.number().min(0).step(0.1).default("1.0").description("Prodigy Plus d_coef. Official default 1.0; try 2 or higher to help the LR grow")],
    ["prodigyplus_betas", Schema.string().default("(0.9, 0.99)").description("Prodigy Plus betas as a Python tuple, e.g. (0.95, 0.99)")],
    ["prodigyplus_schedulefree_c", Schema.number().min(0).default("0").description("Prodigy Plus schedulefree_c. Official default 0")],
    ["prodigyplus_prodigy_steps", Schema.number().min(0).step(1).default("0").description("Prodigy Plus prodigy_steps. Official default 0; on older versions, 5-25% of the total steps")],
  ];
  return [...fields, ...extra.filter(([name]) => !names.has(name)).map(([name, schema]) => ({ name, schema }))];
}

function withFrontendAnimaNetworkArgsFields(fields: Field[], schemaName: string): Field[] {
  if (schemaName !== "anima-lora") return fields;
  const names = new Set(fields.map((field) => field.name));
  const extra: [string, SchemaNode][] = [
    [
      "network_reg_dims",
      Schema.string().default("").description("Rank per module, matched by regex (Anima). Format: pattern=rank,pattern=rank. Example: .*self_attn.*=8,.*cross_attn.*=4,.*mlp.*=8"),
    ],
    [
      "network_reg_lrs",
      Schema.string().default("").description("Learning rate per module, matched by regex (Anima). Format: pattern=lr,pattern=lr. Example: .*self_attn.*=1e-4,.*cross_attn.*=5e-5"),
    ],
  ];
  return [...fields, ...extra.filter(([name]) => !names.has(name)).map(([name, schema]) => ({ name, schema }))];
}

export function schemaOptions(node: SchemaNode): Primitive[] | null {
  if (node.type === "enum") return node.options ?? [];
  if (node.type === "const") return node.value === undefined ? [] : [node.value];
  if (node.type === "array" && node.item instanceof SchemaNode && node.item.type === "enum") return node.item.options ?? [];
  return null;
}

export function defaultFor(node: SchemaNode): unknown {
  if (Object.hasOwn(node.meta, "default")) return node.meta.default;
  if (node.type === "boolean") return false;
  if (node.type === "array") return [];
  if (node.type === "const") return node.value;
  return "";
}
