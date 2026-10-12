// Browser-side implementation of the schema DSL used by mikazuki/schema/*.ts. The backend serves
// those files as source text; they are evaluated here with `Schema`, `SHARED_SCHEMAS`,
// `UpdateSchema` and a `window` stand-in in scope.

export type Primitive = string | number | boolean;

export type SchemaType = "string" | "number" | "boolean" | "const" | "array" | "object" | "intersect" | "union" | "enum";

export interface SchemaMeta {
  default?: unknown;
  description?: string;
  role?: { name: string; options: Record<string, unknown> };
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  disabled?: boolean;
}

export class SchemaNode {
  type: SchemaType;
  /** const */
  value?: Primitive;
  /** array; schema files also pass the `String` constructor here */
  item?: unknown;
  /** object */
  fields?: Record<string, SchemaNode>;
  /** intersect / union */
  items?: SchemaNode[];
  /** enum (a union of literals) */
  options?: Primitive[];
  meta: SchemaMeta = {};

  constructor(type: SchemaType, data: Partial<Pick<SchemaNode, "value" | "item" | "fields" | "items" | "options">> = {}) {
    this.type = type;
    Object.assign(this, data);
  }

  default(value: unknown) {
    this.meta.default = value;
    return this;
  }

  description(value: string) {
    this.meta.description = value;
    return this;
  }

  role(name: string, options: Record<string, unknown> = {}) {
    this.meta.role = { name, options };
    return this;
  }

  min(value: number) {
    this.meta.min = value;
    return this;
  }

  max(value: number) {
    this.meta.max = value;
    return this;
  }

  step(value: number) {
    this.meta.step = value;
    return this;
  }

  required() {
    this.meta.required = true;
    return this;
  }

  disabled() {
    this.meta.disabled = true;
    return this;
  }
}

function isLiteral(item: unknown): item is Primitive {
  return typeof item === "string" || typeof item === "number" || typeof item === "boolean";
}

export const Schema = {
  string: () => new SchemaNode("string"),
  number: () => new SchemaNode("number"),
  boolean: () => new SchemaNode("boolean"),
  const: (value: Primitive) => new SchemaNode("const", { value }).default(value),
  array: (item: unknown) => new SchemaNode("array", { item }),
  object: (fields: Record<string, SchemaNode>) => new SchemaNode("object", { fields }),
  intersect: (items: SchemaNode[]) => new SchemaNode("intersect", { items }),
  union: (items: (SchemaNode | Primitive)[]) =>
    items.every(isLiteral)
      ? new SchemaNode("enum", { options: items as Primitive[] })
      : new SchemaNode("union", { items: items as SchemaNode[] }),
};

export function UpdateSchema<T extends Record<string, unknown>>(base: T, updates: Record<string, unknown> = {}, removes: string[] = []) {
  const result: Record<string, unknown> = { ...base };
  for (const key of removes) delete result[key];
  return { ...result, ...updates };
}

export const SAMPLE_PROMPTS = {
  SAMPLE_PROMPTS_DEFAULT:
    "(masterpiece, best quality:1.2), 1girl, solo, --n lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts,signature, watermark, username, blurry,  --w 512  --h 768  --l 7  --s 24  --d 1337",
  SAMPLE_PROMPTS_DESCRIPTION:
    "预览图生成参数。可填写直接填写参数，或单独写入txt文件填写路径<br>`--n` 后方为反向提示词<br>`--w`宽，`--h`高<br>`--l`: CFG Scale<br>`--s`: 迭代步数<br>`--d`: 种子",
};

/** Evaluate one schema file's source. Schema files are trusted: they ship with the backend. */
export function evaluateSchema(source: string, sharedSchemas: unknown): unknown {
  const trimmed = source.trim().replace(/;\s*$/, "");
  const evaluate = new Function("Schema", "SHARED_SCHEMAS", "UpdateSchema", "window", `"use strict"; return (${trimmed});`);
  return evaluate(Schema, sharedSchemas, UpdateSchema, { __MIKAZUKI__: SAMPLE_PROMPTS });
}
