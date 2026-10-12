import { describe, expect, it } from "vitest";
import en from "../src/locales/en";
import ja from "../src/locales/ja";
import zh from "../src/locales/zh";
import { parseRichText } from "../src/richText";
import { groupTone } from "../src/schema/groups";
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

describe("section tones", () => {
  // sections that are deliberately neutral
  const GENERAL = new Set(["Training", "Other", "Advanced", "Options"]);

  it("gives every schema section an accent", () => {
    const untoned = new Set<string>();
    for (const name of SCHEMA_NAMES) {
      for (const group of loadGroups(name, ["GPU 0: A", "GPU 1: B"])) {
        if (groupTone(group.title) === "general" && !GENERAL.has(group.title)) untoned.add(group.title);
      }
    }
    expect([...untoned]).toEqual([]);
  });
});

describe("UI messages", () => {
  it("use RichText markup instead of HTML", () => {
    for (const messages of [en, zh, ja]) {
      expect(Object.values(messages).filter((text) => /<(?!br>)[a-z]/i.test(text) || /&[a-z]+;/.test(text))).toEqual([]);
    }
  });
});
