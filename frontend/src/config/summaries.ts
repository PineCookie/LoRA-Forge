import { t } from "../i18n";
import type { MergedSectionId } from "../schema/layout";
import type { FormValues } from "./form";
import { isFieldVisible } from "./visibility";

// One line per page section for its header ("Prodigy · 1e-4 · cosine"), so a collapsed section
// still says what it holds.

const text = (value: unknown) => String(value ?? "").trim();
const fileName = (path: unknown) => text(path).split(/[\\/]/).filter(Boolean).pop() ?? "";
const SHORT_NAMES: Record<string, string> = {
  "prodigyplus.ProdigyPlusScheduleFree": "ProdigyPlus",
  "pytorch_optimizer.CAME": "CAME",
};
const short = (value: unknown) => SHORT_NAMES[text(value)] ?? text(value).replace(/^networks\./, "").replace(/\.kohya$/, "");

export function sectionSummary(section: MergedSectionId, values: FormValues): string {
  const parts: string[] = [];
  const add = (part: unknown) => text(part) && parts.push(text(part));
  switch (section) {
    case "model":
      add(fileName(values.pretrained_model_name_or_path) || t("summary.noModel"));
      break;
    case "data":
      add(fileName(values.train_data_dir));
      add(text(values.resolution).replace(",", "x"));
      add(fileName(values.output_name));
      break;
    case "training": {
      if (text(values.max_train_epochs)) add(t("summary.epochs", { count: text(values.max_train_epochs) }));
      if (text(values.train_batch_size)) add(t("summary.batch", { count: text(values.train_batch_size) }));
      add(short(values.optimizer_type));
      add(values.learning_rate ?? values.unet_lr);
      add(isFieldVisible("prodigyplus_d_coef", values) ? t("summary.scheduleFree") : values.lr_scheduler);
      break;
    }
    case "network":
      add(short(values.network_module));
      if (text(values.network_dim)) add(t("summary.dimAlpha", { dim: text(values.network_dim), alpha: text(values.network_alpha) || "-" }));
      break;
    case "preview":
      add(values.enable_preview ? t("summary.previewEvery", { count: text(values.sample_every_n_epochs) || "1" }) : t("summary.off"));
      break;
    case "speed":
      add(values.training_precision ?? values.mixed_precision);
      if (values.compile === true) add(t("summary.compiled"));
      if (values.gradient_checkpointing === true) add(t("summary.checkpointing"));
      if (Number(values.blocks_to_swap) > 0) add(t("summary.swap", { count: text(values.blocks_to_swap) }));
      break;
  }
  return parts.join(" · ");
}
