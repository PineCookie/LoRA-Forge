import { compileShapesOf, optimizerRuleFor, trainingPrecisionOf, type TrainingConfig } from "./normalize";

// Inverse of normalizeTrainingConfig() for import. Anything it generated must go back into its
// form field: *_args_custom is appended after the generated args and sd-scripts keeps the last
// duplicate key, so a leftover imported copy would silently override later edits in the form.

const OPTIMIZER_ARG_TO_FIELD: Record<string, { prodigy?: string; prodigyplus?: string }> = {
  d_coef: { prodigy: "prodigy_d_coef", prodigyplus: "prodigyplus_d_coef" },
  d0: { prodigy: "prodigy_d0" },
  betas: { prodigyplus: "prodigyplus_betas" },
  schedulefree_c: { prodigyplus: "prodigyplus_schedulefree_c" },
  prodigy_steps: { prodigyplus: "prodigyplus_prodigy_steps" },
};

const NETWORK_ARG_TO_FIELD: Record<string, Record<string, string>> = {
  "lycoris.kohya": { conv_dim: "conv_dim", conv_alpha: "conv_alpha", dropout: "dropout", algo: "lycoris_algo", factor: "lokr_factor", train_norm: "train_norm" },
  "networks.dylora": { unit: "dylora_unit" },
};

const BLOCK_WEIGHT_ARGS = ["down_lr_weight", "mid_lr_weight", "up_lr_weight", "block_lr_zero_threshold"];

function splitArg(arg: unknown): [text: string, name: string, value: string] {
  const text = String(arg);
  const separator = text.indexOf("=");
  return separator >= 0 ? [text, text.slice(0, separator), text.slice(separator + 1)] : [text, text, ""];
}

/** Keys an import uses without a field of the same name: expanded into other fields by
 * expandImportedArgs(), or set by the page itself (show_timesteps, by "Preview timesteps"). */
export const EXPANDED_KEYS = new Set([
  "network_args", "optimizer_args", "full_fp16", "full_bf16", "show_timesteps", "compile_dynamic", "compile_dynamic_seq",
  // training_precision (FLUX / SD3 / Krea 2 have mixed_precision itself)
  "mixed_precision",
]);

export function expandImportedArgs(config: TrainingConfig): TrainingConfig {
  const result: TrainingConfig = { ...config };

  if (Array.isArray(config.network_args)) {
    const moduleFields = NETWORK_ARG_TO_FIELD[config.network_module] ?? {};
    const customArgs: string[] = [];
    for (const [text, name, value] of config.network_args.map(splitArg)) {
      if (["network_reg_dims", "network_reg_lrs"].includes(name)) {
        result[name] = value;
      } else if (moduleFields[name] === "train_norm") {
        result.train_norm = value.toLowerCase() === "true";
      } else if (moduleFields[name]) {
        result[moduleFields[name]] = value;
      } else if (BLOCK_WEIGHT_ARGS.includes(name)) {
        result[name] = value;
        result.enable_block_weights = true;
      } else {
        customArgs.push(text);
      }
    }
    result.network_args_custom = customArgs;
  }

  if (Array.isArray(config.optimizer_args)) {
    const optimType = String(config.optimizer_type ?? "");
    // only map an arg to a field the current optimizer actually uses; e.g. d_coef is also a
    // DAdaptation arg and must stay a custom arg there
    const family = optimType === "Prodigy" ? "prodigy" : optimType.includes("ProdigyPlusScheduleFree") ? "prodigyplus" : null;
    let customArgs: string[] = [];
    for (const [text, name, value] of config.optimizer_args.map(splitArg)) {
      const fieldName = family ? OPTIMIZER_ARG_TO_FIELD[name]?.[family] : undefined;
      if (fieldName) result[fieldName] = value;
      else customArgs.push(text);
    }
    // fixed args the optimizer rule adds again on submit (decouple=True, weight_decay=0.01, ...)
    const rule = optimizerRuleFor(optimType);
    if (rule) {
      const regenerated = new Set(rule.args(result, optimType).filter(Boolean));
      customArgs = customArgs.filter((arg) => !regenerated.has(arg));
    }
    result.optimizer_args_custom = customArgs;
  }

  // full_precision is a form-only field that normalizeTrainingConfig() turns into these flags
  if ("full_fp16" in config || "full_bf16" in config) {
    result.full_precision = config.full_fp16 ? "full_fp16" : config.full_bf16 ? "full_bf16" : "none";
  }

  // training_precision is a form-only field that normalizeTrainingConfig() turns into these keys
  const precision = trainingPrecisionOf(config);
  if (precision) result.training_precision = precision;

  // compile_shapes is a form-only field that normalizeTrainingConfig() turns into compile_dynamic(_seq)
  const shapes = compileShapesOf(config);
  if (shapes) result.compile_shapes = shapes;

  return result;
}
