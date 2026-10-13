import { describe, expect, it } from "vitest";
import en from "../src/locales/en";
import ja from "../src/locales/ja";
import zh from "../src/locales/zh";
import { isEssential } from "../src/config/essentials";
import { initialFormValues } from "../src/config/form";
import { isFieldVisible } from "../src/config/visibility";
import { parseRichText } from "../src/richText";
import { groupTone, mergedSectionOf } from "../src/schema/groups";
import { FALLBACK_SECTION, SECTION_LAYOUT } from "../src/schema/layout";
import { loadGroups, SCHEMA_NAMES } from "./helpers";

describe("parseRichText", () => {
  it("splits code, bold and line breaks", () => {
    expect(parseRichText("Use `bf16` on **RTX 30**<br>or newer")).toEqual([
      { kind: "text", text: "Use " },
      { kind: "code", text: "bf16" },
      { kind: "text", text: " on " },
      { kind: "strong", text: "RTX 30" },
      { kind: "br" },
      { kind: "text", text: "or newer" },
    ]);
  });

  it("keeps lone markers and other tags as text", () => {
    expect(parseRichText("a ` b ** c <script>")).toEqual([{ kind: "text", text: "a ` b ** c <script>" }]);
  });
});

describe("page layout", () => {
  const pages = SCHEMA_NAMES.map((name) => ({ name, groups: loadGroups(name, ["GPU 0: A", "GPU 1: B"]) }));
  const listed = SECTION_LAYOUT.flatMap((section) => section.fields);

  it("places every option of every page in a section", () => {
    for (const { name, groups } of pages) {
      const unplaced = groups.filter((group) => group.title === FALLBACK_SECTION.title).flatMap((group) => group.fields.map((field) => field.name));
      expect(unplaced, name).toEqual([]);
    }
  });

  it("lists each option once, and only options some page has", () => {
    expect(listed.filter((name, index) => listed.indexOf(name) !== index)).toEqual([]);
    const used = new Set(pages.flatMap(({ groups }) => groups.flatMap((group) => group.fields.map((field) => field.name))));
    expect(listed.filter((name) => !used.has(name))).toEqual([]);
  });

  it("uses the same section order on every page", () => {
    const order = SECTION_LAYOUT.map((section) => section.title);
    for (const { name, groups } of pages) {
      const titles = groups.map((group) => group.title);
      expect(titles, name).toEqual(order.filter((title) => titles.includes(title)));
    }
  });

  it("gives every section an accent", () => {
    expect(SECTION_LAYOUT.filter((section) => groupTone(section.title) === "general").map((section) => section.title)).toEqual([]);
  });
});

describe("essentials", () => {
  it.each(SCHEMA_NAMES)("%s shows a short, complete set of options", (name) => {
    const groups = loadGroups(name, ["GPU 0: A", "GPU 1: B"]);
    const values = initialFormValues(groups);
    const shown = groups.flatMap((group) => group.fields).filter((field) => isEssential(field.name) && isFieldVisible(field.name, values));
    const names = shown.map((field) => field.name);
    expect(names).toEqual(expect.arrayContaining(["pretrained_model_name_or_path", "train_data_dir", "output_name", "max_train_epochs", "network_dim"].filter((n) => n !== "network_dim" || !["sd-dreambooth", "sdxl-finetune"].includes(name))));
    expect(shown.length, names.join(", ")).toBeGreaterThanOrEqual(12);
    expect(shown.length, names.join(", ")).toBeLessThanOrEqual(30);
  });
});

describe("merged Essentials sections", () => {
  it("puts the dataset folder and outputs together", () => {
    expect(["Dataset", "Saving", "Logging", "Captions"].map(mergedSectionOf)).toEqual(["data", "data", "data", "data"]);
  });
});

describe("UI messages", () => {
  it("use RichText markup instead of HTML", () => {
    for (const messages of [en, zh, ja]) {
      expect(Object.values(messages).filter((text) => /<(?!br>)[a-z]/i.test(text) || /&[a-z]+;/.test(text))).toEqual([]);
    }
  });
});
