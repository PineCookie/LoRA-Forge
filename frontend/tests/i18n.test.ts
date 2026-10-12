import { readFileSync } from "node:fs";
import { beforeEach, describe, expect, it } from "vitest";
import { locale, setSchemaCatalogs, t, tr } from "../src/i18n";
import { SchemaNode } from "../src/schema/dsl";
import { loadGroups, loadSchema, loadShared, SCHEMA_NAMES, schemaDir } from "./helpers";

const catalogs = Object.fromEntries(
  ["zh", "ja"].map((code) => [code, JSON.parse(readFileSync(`${schemaDir}locales/${code}.json`, "utf8")) as Record<string, string>]),
);

/** Every description in a schema tree (field help texts and section titles). */
function collectText(value: unknown, out: Set<string>) {
  if (value instanceof SchemaNode) {
    if (value.meta.description) out.add(value.meta.description);
    for (const child of [...Object.values(value.fields ?? {}), ...(value.items ?? []), value.item]) collectText(child, out);
  } else if (value && typeof value === "object") {
    for (const child of Object.values(value)) collectText(child, out);
  }
}

function schemaText() {
  const text = new Set<string>();
  collectText(loadShared(), text);
  for (const name of SCHEMA_NAMES) {
    collectText(loadSchema(name), text);
    // titles inferred by the UI and the fields it adds itself
    for (const group of loadGroups(name, ["GPU 0: A", "GPU 1: B"])) {
      text.add(group.title);
      for (const field of group.fields) collectText(field.schema, text);
    }
  }
  return text;
}

const count = (text: string, token: string) => text.split(token).length - 1;

describe("schema translations", () => {
  const text = schemaText();

  it.each(["zh", "ja"])("%s covers every schema text", (code) => {
    expect([...text].filter((source) => !catalogs[code][source])).toEqual([]);
  });

  it.each(["zh", "ja"])("%s has no entries for text the schemas no longer use", (code) => {
    expect(Object.keys(catalogs[code]).filter((source) => !text.has(source))).toEqual([]);
  });

  it.each(["zh", "ja"])("%s keeps the inline markup", (code) => {
    const broken = Object.entries(catalogs[code]).filter(
      ([source, translated]) => count(source, "`") !== count(translated, "`") || count(source, "<br>") !== count(translated, "<br>"),
    );
    expect(broken.map(([source]) => source)).toEqual([]);
  });
});

describe("t and tr", () => {
  // the starting language follows the machine's locale
  beforeEach(() => {
    locale.value = "en";
    setSchemaCatalogs({});
  });

  it("fills in parameters in the current language", () => {
    expect(t("runs.stepOf", { n: 12, total: 400 })).toBe("Step 12 / 400");
    locale.value = "zh";
    expect(t("runs.stepOf", { n: 12, total: 400 })).toBe("第 12 / 400 步");
  });

  it("translates schema text and falls back to English", () => {
    setSchemaCatalogs({ ja: { Dataset: "データセット" } });
    locale.value = "ja";
    expect(tr("Dataset")).toBe("データセット");
    expect(tr("Not translated")).toBe("Not translated");
  });
});
