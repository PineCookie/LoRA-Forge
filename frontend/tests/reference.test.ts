import { describe, expect, it } from "vitest";
import { formValueKey, initialFormValues, readForm, toFormValue } from "../src/config/form";
import { isEssential } from "../src/config/essentials";
import { EXPANDED_KEYS, expandImportedArgs } from "../src/config/import";
import { applyDependentValues, applyExclusiveRules } from "../src/config/rules";
import { fromToml } from "../src/config/toml";
import { loadGroups } from "./helpers";

// A typical Anima LoRA config as people export it (ProdigyPlus, flash attention, compile, cached
// encoders), with placeholder paths.
const ANIMA_TOML = `
model_train_type = "anima-lora"
pretrained_model_name_or_path = "./models/anima-base-v1.0.safetensors"
qwen3 = "./models/qwen_3_06b_base.safetensors"
vae = "./models/qwen_image_vae.safetensors"
timestep_sampling = "sigmoid"
weighting_scheme = "uniform"
sigmoid_scale = 1
discrete_flow_shift = 1
llm_adapter_lr = 0
train_data_dir = "./train/287_data"
resolution = "1024,1024"
enable_bucket = true
min_bucket_reso = 256
max_bucket_reso = 2048
bucket_reso_steps = 64
bucket_no_upscale = true
output_name = "287_anima"
output_dir = "./output"
save_every_n_epochs = 1
max_train_epochs = 13
train_batch_size = 2
gradient_checkpointing = true
network_train_unet_only = true
seed = 42
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1
unet_lr = 1
text_encoder_lr = 1
lr_scheduler = "constant"
max_grad_norm = 0
network_module = "networks.lora_anima"
network_dim = 32
network_alpha = 16
logging_dir = "./logs/287"
weighted_captions = false
keep_tokens = 4
mixed_precision = "bf16"
xformers = false
sdpa = true
cache_text_encoder_outputs = true
attn_mode = "flash"
qwen_image_vae_2d = true
compile = true
compile_dynamic = "false"
show_timesteps = "console"
gpu_ids = ["GPU 0"]
network_args = ["network_reg_lrs=.*self_attn.*=1.0,.*cross_attn.*=1.0,.*mlp.*=1.0"]
optimizer_args = ["d_coef=1.0", "betas=(0.95, 0.99)", "schedulefree_c=20", "prodigy_steps=1073"]
full_fp16 = false
full_bf16 = true
`;

describe("typical Anima config", () => {
  const source = fromToml(ANIMA_TOML);
  const groups = loadGroups("anima-lora", ["GPU 0: A (24 GB)", "GPU 1: B (24 GB)"]);
  const fields = new Map(groups.flatMap((group) => group.fields.map((field) => [field.name, field] as const)));
  const defaults = initialFormValues(groups);
  applyDependentValues(defaults);
  const values = structuredClone(defaults);
  for (const [name, value] of Object.entries(expandImportedArgs(source))) {
    const field = fields.get(name);
    if (field) values[name] = toFormValue(name, field.schema, value, { imported: true });
  }
  applyExclusiveRules(values, null);
  applyDependentValues(values);
  const config = readForm(values, groups);

  it("imports into the form", () => {
    expect(values.gpu_ids).toEqual(["GPU 0: A (24 GB)"]);
    expect(values.full_precision).toBe("full_bf16");
    expect(values.prodigyplus_prodigy_steps).toBe("1073");
    expect(values.network_reg_lrs).toBe(".*self_attn.*=1.0,.*cross_attn.*=1.0,.*mlp.*=1.0");
  });

  it("reports only the keys the Anima page has no option for", () => {
    const unused = Object.keys(source).filter((name) => !fields.has(name) && !EXPANDED_KEYS.has(name));
    expect(unused.sort()).toEqual(["sdpa", "weighted_captions", "xformers"]);
  });

  it("exports the same training arguments", () => {
    expect(config.optimizer_args).toEqual(expect.arrayContaining(["d_coef=1.0", "betas=(0.95, 0.99)", "schedulefree_c=20", "prodigy_steps=1073"]));
    expect(config.network_args).toEqual(["network_reg_lrs=.*self_attn.*=1.0,.*cross_attn.*=1.0,.*mlp.*=1.0"]);
    for (const key of ["attn_mode", "compile", "qwen_image_vae_2d", "full_bf16", "max_grad_norm", "keep_tokens", "bucket_reso_steps"]) {
      expect(config[key], key).toEqual(source[key]);
    }
  });

  it("keeps most of what such a run changes in Essentials", () => {
    const changed = [...fields.keys()].filter((name) => formValueKey(values[name]) !== formValueKey(defaults[name]));
    // set automatically by the optimizer rule, or rarely changed on purpose
    const outside = changed.filter((name) => !isEssential(name)).sort();
    expect(outside).toEqual(["bucket_reso_steps", "logging_dir", "max_grad_norm", "min_bucket_reso", "network_reg_lrs"]);
  });
});
