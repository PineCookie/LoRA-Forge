import { Schema, SchemaNode, type Primitive } from "./dsl";

export interface Field {
  name: string;
  schema: SchemaNode;
}

export interface Group {
  title: string;
  fields: Field[];
}

/** Flatten an evaluated schema into titled field groups, as the form renders them. */
export function buildGroups(schema: SchemaNode, schemaName: string, cards: string[]): Group[] {
  return withGpuSelector(withFrontendAnimaNetworkArgsFields(withFrontendOptimizerFields(flattenSchema(schema)), schemaName), cards);
}

export function flattenSchema(schema: SchemaNode): Group[] {
  const groups: Group[] = [];
  const seen = new Set<string>();

  function visit(node: SchemaNode | undefined, fallbackTitle = "Options") {
    if (!node) return;
    if (node.type === "intersect" || node.type === "union") {
      for (const item of node.items ?? []) visit(item, fallbackTitle);
      return;
    }
    if (node.type !== "object") return;

    const fields: Field[] = [];
    for (const [name, field] of Object.entries(node.fields ?? {})) {
      if (seen.has(name)) continue;
      seen.add(name);
      fields.push({ name, schema: field });
    }
    if (fields.length) {
      const title = node.meta.description || inferGroupTitle(fields, fallbackTitle);
      const group = groups.find((item) => item.title === title);
      if (group) group.fields.push(...fields);
      else groups.push({ title, fields });
    }
  }

  visit(schema);
  return normalizeGroups(groups);
}

function normalizeGroups(groups: Group[]): Group[] {
  const normalized = groups.filter((group) => group.fields.length);

  moveFieldToGroup(normalized, "seed", "Training");
  return normalized;
}

function moveFieldToGroup(groups: Group[], fieldName: string, targetTitle: string) {
  let movedField: Field | undefined;
  for (const group of groups) {
    const index = group.fields.findIndex((field) => field.name === fieldName);
    if (index >= 0) {
      [movedField] = group.fields.splice(index, 1);
      break;
    }
  }
  if (!movedField) return;

  let target = groups.find((group) => group.title === targetTitle);
  if (!target) {
    target = { title: targetTitle, fields: [] };
    groups.push(target);
  }
  target.fields.push(movedField);

  for (let index = groups.length - 1; index >= 0; index -= 1) {
    if (!groups[index].fields.length) groups.splice(index, 1);
  }
}

const GROUP_TITLE_HINTS: [string, string[]][] = [
  ["Model", ["model_train_type", "pretrained_model_name_or_path", "qwen3", "vae", "clip_l", "clip_g", "t5xxl", "v2", "v_parameterization", "scale_v_pred_loss_like_noise_pred"]],
  ["Dataset", ["train_data_dir", "resolution", "enable_bucket", "bucket_no_upscale"]],
  ["Saving", ["output_name", "output_dir", "save_model_as", "save_state", "save_last_n_epochs_state"]],
  ["Training", ["max_train_epochs", "train_batch_size", "gradient_checkpointing", "seed"]],
  ["Learning rate and optimizer", ["optimizer_type", "learning_rate", "lr_scheduler", "lr_scheduler_num_cycles", "unet_lr", "text_encoder_lr", "optimizer_args_custom"]],
  ["Loss", ["loss_type", "huber_schedule", "huber_c", "min_snr_gamma", "debiased_estimation_loss"]],
  ["Optimizer options", ["prodigy_d0", "prodigyplus_d_coef"]],
  ["Network", ["network_module", "network_dim", "network_alpha", "network_weights"]],
  ["Network options", ["lycoris_algo", "conv_dim", "conv_alpha", "lokr_factor", "dylora_unit"]],
  ["Base weights", ["enable_base_weight", "base_weights", "base_weights_multiplier"]],
  ["Block-wise learning rates", ["enable_block_weights", "down_lr_weight", "mid_lr_weight", "up_lr_weight"]],
  ["Preview images", ["enable_preview", "sample_prompts", "positive_prompts", "sample_width"]],
  ["Logging", ["log_with", "logging_dir", "wandb_api_key"]],
  ["Captions", ["caption_extension", "shuffle_caption", "keep_tokens", "max_token_length"]],
  ["Noise", ["noise_offset", "multires_noise_iterations"]],
  ["Data augmentation", ["color_aug", "flip_aug", "random_crop"]],
  ["Speed and memory", ["mixed_precision", "xformers", "sdpa", "cache_latents", "attn_mode"]],
  ["Distributed training", ["ddp_timeout", "ddp_gradient_as_bucket_view"]],
  ["Other", ["clip_skip"]],
];

function inferGroupTitle(fields: Field[], fallbackTitle: string) {
  const names = new Set(fields.map((field) => field.name));
  return GROUP_TITLE_HINTS.find(([, candidates]) => candidates.some((name) => names.has(name)))?.[0] ?? fallbackTitle;
}

function fieldNames(groups: Group[]) {
  return new Set(groups.flatMap((group) => group.fields.map((field) => field.name)));
}

function withGpuSelector(groups: Group[], cards: string[]): Group[] {
  if (cards.length < 2) return groups;
  return [
    ...groups,
    {
      title: "GPU",
      fields: [
        {
          name: "gpu_ids",
          schema: Schema.array(Schema.union(cards)).default([cards[0]]).description("GPUs to use. When none is selected, the first GPU is used").role("select"),
        },
      ],
    },
  ];
}

function withFrontendOptimizerFields(groups: Group[]): Group[] {
  const allNames = fieldNames(groups);
  const optimizerField = groups.flatMap((group) => group.fields).find((field) => field.name === "optimizer_type");
  const optimizerOptions = optimizerField ? schemaOptions(optimizerField.schema) ?? [] : [];
  if (!optimizerOptions.includes("prodigyplus.ProdigyPlusScheduleFree")) return groups;

  const extra: [string, SchemaNode][] = [
    ["prodigy_d0", Schema.string().description("Prodigy d0: the initial LR estimate; try 1e-7, 1e-6 or 1e-5")],
    ["prodigy_d_coef", Schema.string().default("2.0").description("Prodigy d_coef. Default 2.0")],
    ["prodigyplus_d_coef", Schema.string().default("1.0").description("Prodigy Plus d_coef. Official default 1.0; try 2 or higher to help the LR grow")],
    ["prodigyplus_betas", Schema.string().default("(0.9, 0.99)").description("Prodigy Plus betas as a Python tuple, e.g. (0.95, 0.99)")],
    ["prodigyplus_schedulefree_c", Schema.string().default("0").description("Prodigy Plus schedulefree_c. Official default 0")],
    ["prodigyplus_prodigy_steps", Schema.string().default("0").description("Prodigy Plus prodigy_steps. Official default 0; on older versions, 5-25% of the total steps")],
  ];
  const fields = extra.filter(([name]) => !allNames.has(name)).map(([name, schema]) => ({ name, schema }));
  if (!fields.length) return groups;
  return [...groups, { title: "Optimizer options", fields }];
}

function withFrontendAnimaNetworkArgsFields(groups: Group[], schemaName: string): Group[] {
  if (schemaName !== "anima-lora") return groups;

  const allNames = fieldNames(groups);
  const extra: [string, SchemaNode][] = [
    [
      "network_reg_dims",
      Schema.string().default("").description("Rank per module, matched by regex (Anima). Format: pattern=rank,pattern=rank. Example: .*self_attn.*=8,.*cross_attn.*=4,.*mlp.*=8"),
    ],
    [
      "network_reg_lrs",
      Schema.string().default("").description("Learning rate per module, matched by regex (Anima). Format: pattern=lr,pattern=lr. Example: .*self_attn.*=1e-4,.*cross_attn.*=5e-5"),
    ],
  ];
  const fields = extra.filter(([name]) => !allNames.has(name)).map(([name, schema]) => ({ name, schema }));
  if (!fields.length) return groups;

  const networkGroup = groups.find((group) => group.fields.some((field) => field.name === "network_module"));
  if (networkGroup) {
    networkGroup.fields.push(...fields);
    return groups;
  }
  return [...groups, { title: "Network", fields }];
}

export function schemaOptions(node: SchemaNode): Primitive[] | null {
  if (node.type === "enum") return node.options ?? [];
  if (node.type === "const") return node.value === undefined ? [] : [node.value];
  if (node.type === "array" && node.item instanceof SchemaNode && node.item.type === "enum") return node.item.options ?? [];
  return null;
}

export function defaultFor(node: SchemaNode): unknown {
  if (Object.hasOwn(node.meta, "default")) return node.meta.default;
  if (node.type === "boolean") return false;
  if (node.type === "array") return [];
  if (node.type === "const") return node.value;
  return "";
}
