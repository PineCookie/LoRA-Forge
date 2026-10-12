import type { SchemaNode } from "../schema/dsl";
import { defaultFor, schemaOptions, type Group } from "../schema/groups";
import { cleanConfig, formatResolutionForInput, isResolutionField, normalizeResolutionForConfig, type TrainingConfig } from "./normalize";

// What a form control holds: text inputs, number inputs, selects and textareas hold strings,
// checkboxes booleans, multi-selects string arrays. cleanConfig() turns these into typed config.
export type FormValue = string | boolean | string[];
export type FormValues = Record<string, FormValue>;

export type ControlKind = "select" | "multiselect" | "checkbox" | "textarea" | "number" | "text";

export function controlKind(node: SchemaNode): ControlKind {
  if (schemaOptions(node)) return node.type === "array" ? "multiselect" : "select";
  if (node.type === "boolean") return "checkbox";
  const role = node.meta.role?.name;
  if (role === "textarea" || role === "table" || node.type === "array") return "textarea";
  if (node.type === "number") return "number";
  return "text";
}

function optionStrings(node: SchemaNode) {
  return (schemaOptions(node) ?? []).map(String);
}

/** A schema default or imported value as the control holds it. */
export function toFormValue(name: string, node: SchemaNode, value: unknown, { imported = false } = {}): FormValue {
  switch (controlKind(node)) {
    case "checkbox":
      return Boolean(value);
    case "multiselect": {
      const options = optionStrings(node);
      const list = Array.isArray(value) ? value : value === undefined || value === "" ? [] : [value];
      return list.map(String).filter((item) => options.includes(item));
    }
    case "select": {
      const options = optionStrings(node);
      const text = String(value ?? "");
      if (options.includes(text) || imported) return text;
      // a select without a matching default shows its first option, and that is what gets submitted
      return options[0] ?? "";
    }
    case "textarea":
      return Array.isArray(value) ? value.join("\n") : String(value ?? "");
    default:
      return isResolutionField(name) ? formatResolutionForInput(value) : String(value ?? "");
  }
}

export function initialFormValues(groups: Group[]): FormValues {
  const values: FormValues = {};
  for (const group of groups) {
    for (const field of group.fields) values[field.name] = toFormValue(field.name, field.schema, defaultFor(field.schema));
  }
  return values;
}

/** Stable string form of a control value, for "edited" comparisons. */
export function formValueKey(value: FormValue | undefined): string {
  if (Array.isArray(value)) return value.join("\n");
  return String(value ?? "");
}

/** The config the current form state stands for (what is previewed, exported and submitted). */
export function readForm(values: FormValues, groups: Group[]): TrainingConfig {
  const raw: TrainingConfig = {};
  for (const group of groups) {
    for (const field of group.fields) {
      const value = values[field.name];
      if (value === undefined) continue;
      raw[field.name] = isResolutionField(field.name) ? normalizeResolutionForConfig(value) : value;
    }
  }
  return cleanConfig(raw, groups);
}
