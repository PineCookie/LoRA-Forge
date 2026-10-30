import { t } from "../i18n";
import { SECTION_LAYOUT } from "../schema/layout";
import type { TrainingConfig } from "./normalize";

// Minimal TOML for the flat training configs this UI exports and imports (no tables). Section
// comments are English in every UI language, so an exported file does not depend on it.

// sd-scripts keys the config has without a form field of that name (the form fields they come from
// are dropped on export), by the section they belong to
const EXTRA_KEYS: Record<string, string[]> = {
  Network: ["network_args"],
  "Learning rate and optimizer": ["optimizer_args", "lr_restart_cycles"],
  "Timesteps and loss": ["logit_mean", "logit_std", "mode_scale"],
  Noise: ["ip_noise_gamma"],
  Saving: ["save_last_n_epochs", "save_n_epoch_ratio"],
  Precision: ["full_fp16", "full_bf16"],
  "torch.compile": ["compile_dynamic"],
};

// the page's sections and order (schema/layout.ts)
const TOML_KEY_ORDER = SECTION_LAYOUT.map(({ title, fields }) => ({ section: title, keys: [...fields, ...(EXTRA_KEYS[title] ?? [])] }));

const KEY_ORDER = new Map(
  TOML_KEY_ORDER.flatMap(({ section, keys }) => keys.map((key) => [key, section] as const)).map(([key, section], order) => [key, { section, order }]),
);

function tomlValue(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(tomlValue).join(", ")}]`;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return JSON.stringify(String(value));
}

/** Serialize a config, grouped into commented sections in a stable key order. */
export function toToml(config: TrainingConfig): string {
  const entries = Object.entries(config);
  const known = entries.filter(([key]) => KEY_ORDER.has(key)).sort(([a], [b]) => KEY_ORDER.get(a)!.order - KEY_ORDER.get(b)!.order);
  const unknown = entries.filter(([key]) => !KEY_ORDER.has(key));

  const lines: string[] = [];
  let currentSection: string | null = null;
  for (const [key, value] of known) {
    const { section } = KEY_ORDER.get(key)!;
    if (section !== currentSection) {
      currentSection = section;
      lines.push(`# --- ${section} ---`);
    }
    lines.push(`${key} = ${tomlValue(value)}`);
  }
  if (unknown.length) {
    lines.push("# --- Other ---");
    for (const [key, value] of unknown) lines.push(`${key} = ${tomlValue(value)}`);
  }
  return `${lines.join("\n")}\n`;
}

function indexOutsideQuotes(line: string, target: string) {
  let quote: string | null = null;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if ((char === '"' || char === "'") && line[index - 1] !== "\\") {
      quote = quote === char ? null : quote || char;
    }
    if (char === target && !quote) return index;
  }
  return -1;
}

function parseTomlString(value: string): string {
  if (value.startsWith("'")) return value.slice(1, -1);
  try {
    return JSON.parse(value);
  } catch {
    return value.slice(1, -1);
  }
}

function parseTomlArray(source: string): unknown[] {
  const values: unknown[] = [];
  let quote: string | null = null;
  let current = "";
  let depth = 0;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if ((char === '"' || char === "'") && source[index - 1] !== "\\") quote = quote === char ? null : quote || char;
    if (!quote && char === "[") depth += 1;
    if (!quote && char === "]") depth -= 1;
    if (char === "," && !quote && depth === 0) {
      if (current.trim()) values.push(parseTomlValue(current.trim()));
      current = "";
    } else {
      current += char;
    }
  }
  if (current.trim()) values.push(parseTomlValue(current.trim()));
  return values;
}

function parseTomlValue(value: string): unknown {
  if (value === "true") return true;
  if (value === "false") return false;
  if (/^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) return Number(value);
  if (value.startsWith("[") && value.endsWith("]")) return parseTomlArray(value.slice(1, -1));
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) return parseTomlString(value);
  return value;
}

/** Parse a flat training-config TOML; tables are rejected. */
export function fromToml(text: string): TrainingConfig {
  const result: TrainingConfig = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const commentAt = indexOutsideQuotes(rawLine, "#");
    const line = (commentAt >= 0 ? rawLine.slice(0, commentAt) : rawLine).trim();
    if (!line || line.startsWith("[[")) continue;
    if (line.startsWith("[")) throw new Error(t("trainer.tomlTables"));
    const separator = indexOutsideQuotes(line, "=");
    if (separator < 1) continue;
    result[line.slice(0, separator).trim()] = parseTomlValue(line.slice(separator + 1).trim());
  }
  return result;
}
