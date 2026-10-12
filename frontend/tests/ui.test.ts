import { describe, expect, it } from "vitest";
import en from "../src/locales/en";
import ja from "../src/locales/ja";
import zh from "../src/locales/zh";
import { isEssential } from "../src/config/essentials";
import { initialFormValues } from "../src/config/form";
import { isFieldVisible } from "../src/config/visibility";
import { parseRichText } from "../src/richText";
import { groupTone, mergedSectionOf } from "../src/schema/groups";
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

describe("essentials", () => {
  it.each(SCHEMA_NAMES)("%s shows a short, complete set of options", (name) => {
    const groups = loadGroups(name, ["GPU 0: A", "GPU 1: B"]);
    const values = initialFormValues(groups);
    const shown = groups.flatMap((group) => group.fields).filter((field) => isEssential(field.name) && isFieldVisible(field.name, values));
    const names = shown.map((field) => field.name);
    expect(names).toEqual(expect.arrayContaining(["pretrained_model_name_or_path", "train_data_dir", "output_name", "max_train_epochs", "network_dim"].filter((n) => n !== "network_dim" || name !== "dreambooth")));
    expect(shown.length, names.join(", ")).toBeGreaterThanOrEqual(12);
    expect(shown.length, names.join(", ")).toBeLessThanOrEqual(30);
  });
});

describe("merged Essentials sections", () => {
  // sections that belong under "Training"; any other title must be mapped explicitly
  const TRAINING = new Set(["Training", "Learning rate and optimizer", "Optimizer options", "Loss", "Noise", "Anima options", "FLUX options", "SD3 options", "Options", "Other", "Advanced"]);

  it("maps every schema section on purpose", () => {
    const unmapped = new Set<string>();
    for (const name of SCHEMA_NAMES) {
      for (const group of loadGroups(name, ["GPU 0: A", "GPU 1: B"])) {
        if (mergedSectionOf(group.title) === "training" && !TRAINING.has(group.title)) unmapped.add(group.title);
      }
    }
    expect([...unmapped]).toEqual([]);
  });

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
