// The small inline markup used by schema descriptions and some UI messages: `code`, **bold** and
// <br> line breaks. Parsed into parts and rendered by RichText.vue, so no text goes through v-html.

export type RichPart = { kind: "text" | "code" | "strong"; text: string } | { kind: "br" };

const TOKEN = /(`[^`]+`|\*\*[^*]+\*\*|<br\s*\/?>)/;

export function parseRichText(source: string): RichPart[] {
  return source
    .split(TOKEN)
    .filter(Boolean)
    .map((part): RichPart => {
      if (/^<br\s*\/?>$/.test(part)) return { kind: "br" };
      if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) return { kind: "code", text: part.slice(1, -1) };
      if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) return { kind: "strong", text: part.slice(2, -2) };
      return { kind: "text", text: part };
    });
}
