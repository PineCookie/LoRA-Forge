import { computed } from "vue";
import { fromToml, toToml } from "../config/toml";
import { t } from "../i18n";
import { notify } from "../toast";
import type { TrainerForm } from "./useTrainerForm";

export type ConfigFormat = "json" | "toml";

function download(text: string, type: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** Export, import and preview of a trainer form's config as JSON / TOML. */
export function useConfigFile(form: TrainerForm) {
  const toml = computed(() => toToml(form.config.value));
  const generatedArgs = computed(() =>
    (["optimizer_args", "network_args"] as const)
      .map((name) => ({ name, args: (form.config.value[name] as string[] | undefined) ?? [] }))
      .filter((section) => section.args.length),
  );

  function exportAs(format: ConfigFormat) {
    const name = `${form.config.value.output_name || "training-config"}.${format}`;
    if (format === "json") download(JSON.stringify(form.config.value, null, 2), "application/json", name);
    else download(toml.value, "application/toml;charset=utf-8", name);
  }

  /** Import a .json or .toml config; anything not named .json is read as TOML. */
  async function importFile(file: File) {
    const format: ConfigFormat = file.name.toLowerCase().endsWith(".json") ? "json" : "toml";
    try {
      const text = await file.text();
      form.applyImported(format === "json" ? JSON.parse(text) : fromToml(text));
      notify(t("trainer.imported", { format: format.toUpperCase() }), "success");
    } catch (error) {
      notify(t("trainer.importFailed", { format: format.toUpperCase(), error: errorText(error) }), "error");
    }
  }

  async function copyToml() {
    try {
      await navigator.clipboard.writeText(toml.value);
      notify(t("trainer.copied"), "success");
    } catch {
      notify(t("trainer.copyFailed"), "error");
    }
  }

  return { toml, generatedArgs, exportAs, importFile, copyToml };
}
