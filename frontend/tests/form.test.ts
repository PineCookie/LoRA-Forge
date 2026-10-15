import { describe, expect, it } from "vitest";
import { initialFormValues, readForm, toFormValue } from "../src/config/form";
import { compileShapesOf } from "../src/config/normalize";
import { applyDependentValues, applyExclusiveRules } from "../src/config/rules";
import { isFieldVisible } from "../src/config/visibility";
import { loadGroups, SCHEMA_NAMES } from "./helpers";

describe("schema groups", () => {
  it("builds every backend schema", () => {
    for (const name of SCHEMA_NAMES) {
      const groups = loadGroups(name);
      expect(groups.length, name).toBeGreaterThan(3);
      // the base model has no placeholder path: the user picks it
      const model = groups.flatMap((group) => group.fields).find((field) => field.name === "pretrained_model_name_or_path");
      expect(model?.schema.meta.default, name).toBeUndefined();
    }
  });

  it("adds the GPU picker only with two or more cards, defaulting to the first", () => {
    expect(loadGroups("anima-lora", ["GPU 0: A"]).some((group) => group.fields.some((field) => field.name === "gpu_ids"))).toBe(false);
    const groups = loadGroups("anima-lora", ["GPU 0: A", "GPU 1: B"]);
    expect(initialFormValues(groups).gpu_ids).toEqual(["GPU 0: A"]);
  });
});

describe("anima defaults", () => {
  const groups = loadGroups("anima-lora");
  const values = initialFormValues(groups);
  applyDependentValues(values);
  const config = readForm(values, groups);

  it("submits the expected core options", () => {
    expect(config.model_train_type).toBe("anima-lora");
    expect(config.network_module).toBe("networks.lora_anima");
    expect(config.resolution).toBe("1024,1024");
    expect(config.full_bf16).toBe(true);
    expect(config.mixed_precision).toBe("bf16");
    expect(config.network_train_unet_only).toBe(true);
    expect(config.qwen_image_vae_2d).toBe(true);
  });

  it("drops preview-only and form-only keys", () => {
    for (const key of ["enable_preview", "positive_prompts", "full_precision", "network_reg_dims", "optimizer_args_custom"]) {
      expect(config, key).not.toHaveProperty(key);
    }
  });

  it("shows the resolution as WxH in the form", () => {
    expect(values.resolution).toBe("1024x1024");
  });

  it("compiles with bounded shapes, sent as sd-scripts' compile_dynamic plus compile_dynamic_seq", () => {
    expect(config).toMatchObject({ compile: true, compile_dynamic: "false", compile_dynamic_seq: true });
    expect(config).not.toHaveProperty("compile_shapes");
    const shapes = (value: string, compile = true) => readForm({ ...values, compile_shapes: value, compile }, groups);
    expect(shapes("static")).toMatchObject({ compile_dynamic: "false" });
    expect(shapes("static")).not.toHaveProperty("compile_dynamic_seq");
    expect(shapes("auto").compile_dynamic).toBe("auto");
    expect(shapes("dynamic").compile_dynamic).toBe("true");
    expect(shapes("bounded", false)).not.toHaveProperty("compile_dynamic");
  });

  it("submits learning rates as numbers", () => {
    expect([config.learning_rate, config.unet_lr]).toEqual([1e-4, 1e-4]);
  });
});

describe("compile_shapes import", () => {
  it.each([
    [{ compile_dynamic: "false", compile_dynamic_seq: true }, "bounded"],
    [{ compile_dynamic: "false", compile_dynamic_seq: false }, "static"],
    // from before compile_dynamic_seq: false was the only working choice
    [{ compile_dynamic: "false" }, "bounded"],
    [{ compile_dynamic: "auto" }, "auto"],
    [{ compile_dynamic: "true", compile_dynamic_seq: false }, "dynamic"],
    [{}, undefined],
  ])("%o -> %s", (config, shapes) => {
    expect(compileShapesOf(config)).toBe(shapes);
  });
});

describe("max_token_length", () => {
  it("is 75, 150 or 225 and submitted as a number; 75 is sd-scripts' default", () => {
    const groups = loadGroups("sdxl-lora");
    const values = initialFormValues(groups);
    expect(readForm(values, groups).max_token_length).toBe(225);
    expect(readForm({ ...values, max_token_length: "75" }, groups)).not.toHaveProperty("max_token_length");
  });

  it("turns an older config's 255 into 225", () => {
    const values = initialFormValues(loadGroups("sdxl-lora"));
    values.max_token_length = "255";
    applyExclusiveRules(values, null);
    expect(values.max_token_length).toBe("225");
  });
});

describe("rules", () => {
  it("forces Prodigy values and the max_grad_norm reset", () => {
    const groups = loadGroups("anima-lora");
    const values = initialFormValues(groups);
    values.optimizer_type = "Prodigy";
    applyDependentValues(values);
    expect(values.max_grad_norm).toBe("0");
    const config = readForm(values, groups);
    expect(config.optimizer_args).toEqual(expect.arrayContaining(["decouple=True", "use_bias_correction=True", "d_coef=2.0"]));
  });

  it("keeps offload options exclusive, block swap winning", () => {
    const values = initialFormValues(loadGroups("anima-lora"));
    values.cpu_offload_checkpointing = true;
    applyExclusiveRules(values, "cpu_offload_checkpointing");
    expect(values.blocks_to_swap).toBe("");
    values.blocks_to_swap = "8";
    applyExclusiveRules(values, "blocks_to_swap");
    expect(values.cpu_offload_checkpointing).toBe(false);
  });

  it("keeps activation_memory_budget and the VRAM-saving options apart, the last edit winning", () => {
    const groups = loadGroups("anima-lora");
    const values = initialFormValues(groups);
    values.blocks_to_swap = "8";
    values.activation_memory_budget = "auto";
    applyExclusiveRules(values, "activation_memory_budget");
    expect([values.blocks_to_swap, values.activation_memory_budget]).toEqual(["", "auto"]);
    values.unsloth_offload_checkpointing = true;
    applyExclusiveRules(values, "unsloth_offload_checkpointing");
    expect(values.activation_memory_budget).toBe("off");
  });

  it("keeps Krea 2's base quantization, Turbo source and H2D-only block swap consistent", () => {
    const groups = loadGroups("krea2-lora");
    const values = initialFormValues(groups);
    expect([values.convrot_int8, values.fp8_base, values.block_swap_h2d_only, values.gradient_checkpointing]).toEqual([true, false, true, true]);
    values.fp8_base = true;
    applyExclusiveRules(values, "fp8_base");
    expect(values.convrot_int8).toBe(false);
    values.turbo_lora = "./sd-models/turbo_lora.safetensors";
    applyExclusiveRules(values, "turbo_lora");
    values.turbo_dit = "./sd-models/turbo.safetensors";
    applyExclusiveRules(values, "turbo_dit");
    expect([values.turbo_dit, values.turbo_lora]).toEqual(["./sd-models/turbo.safetensors", ""]);
    values.gradient_checkpointing = false;
    applyExclusiveRules(values, "gradient_checkpointing");
    expect(values.block_swap_h2d_only).toBe(false);
    values.block_swap_h2d_only = true;
    applyExclusiveRules(values, "block_swap_h2d_only");
    expect(values.gradient_checkpointing).toBe(true);
    // an imported combination, which skips the live rules
    expect(readForm({ ...values, gradient_checkpointing: false }, groups)).not.toHaveProperty("block_swap_h2d_only");
  });

  it("sends activation_memory_budget only when on, with compile, and not with an offload option", () => {
    const groups = loadGroups("anima-lora");
    const values = initialFormValues(groups);
    expect(values.activation_memory_budget).toBe("off");
    expect(readForm(values, groups)).not.toHaveProperty("activation_memory_budget");
    for (const budget of ["auto", "0.3"]) expect(readForm({ ...values, activation_memory_budget: budget }, groups).activation_memory_budget).toBe(budget);
    const on = { ...values, activation_memory_budget: "auto" };
    expect(readForm({ ...on, compile: false }, groups)).not.toHaveProperty("activation_memory_budget");
    // an imported combination, which skips the live rules
    expect(readForm({ ...on, cpu_offload_checkpointing: true }, groups)).not.toHaveProperty("activation_memory_budget");
  });

  it("locks network_train_unet_only while caching text encoder outputs on Anima", () => {
    const values = initialFormValues(loadGroups("anima-lora"));
    values.network_train_text_encoder_only = true;
    applyExclusiveRules(values, "network_train_text_encoder_only");
    expect(values.cache_text_encoder_outputs).toBe(false);
    expect(values.network_train_unet_only).toBe(false);
  });

  it("defaults bucket_reso_steps to the smallest step each model accepts", () => {
    // sd-scripts' verify_bucket_reso_steps: Anima 16, FLUX / SD3.5 / SDXL 32, SD1.5 64
    const step = (schema: string) => initialFormValues(loadGroups(schema)).bucket_reso_steps;
    const pages = ["anima-lora", "flux-lora", "sd3-lora", "sdxl-lora", "sdxl-finetune", "sd-lora", "sd-dreambooth"];
    expect(pages.map(step)).toEqual(["16", "32", "32", "32", "32", "64", "64"]);
  });

  it("corrects an SD1.5 bucket step sd-scripts would reject (e.g. from an imported SDXL config)", () => {
    for (const schema of ["sd-lora", "sd-dreambooth"]) {
      const values = initialFormValues(loadGroups(schema));
      values.bucket_reso_steps = "32";
      applyExclusiveRules(values, null);
      expect(values.bucket_reso_steps, schema).toBe("64");
    }
  });

  it("gives every model page its own training type and SDXL its own defaults", () => {
    const initial = (schema: string) => initialFormValues(loadGroups(schema));
    expect(["sdxl-lora", "sdxl-finetune", "sd-lora", "sd-dreambooth"].map((schema) => initial(schema).model_train_type)).toEqual([
      "sdxl-lora", "sdxl-finetune", "sd-lora", "sd-dreambooth",
    ]);
    const sdxl = initial("sdxl-lora");
    expect([sdxl.resolution, sdxl.max_bucket_reso, sdxl.network_train_unet_only, sdxl.cache_text_encoder_outputs]).toEqual(["1024x1024", "2048", true, true]);
    // SDXL's U-Net: 9 down / 3 mid / 9 up blocks; SD1.5: 12 / 1 / 12
    expect([sdxl.down_lr_weight, sdxl.mid_lr_weight].map((v) => String(v).split(",").length)).toEqual([9, 3]);
    expect(String(initial("sd-lora").down_lr_weight).split(",").length).toBe(12);
    // options that mean nothing for a model are not on its page
    expect(["v2", "clip_skip"].filter((name) => name in sdxl)).toEqual([]);
    // clip_skip only exists for SD1.5 (FLUX ignores it, SD3.5 just warns)
    expect(["anima-lora", "flux-lora", "sd3-lora", "sdxl-lora", "sdxl-finetune"].filter((schema) => "clip_skip" in initial(schema))).toEqual([]);
    expect(["sd-lora", "sd-dreambooth"].filter((schema) => "clip_skip" in initial(schema))).toEqual(["sd-lora", "sd-dreambooth"]);
    expect(["cache_text_encoder_outputs"].filter((name) => name in initial("sd-lora"))).toEqual([]);
  });

  it("shows SDXL fine-tune's text encoder learning rates only when they are used", () => {
    const groups = loadGroups("sdxl-finetune");
    const values = initialFormValues(groups);
    expect(isFieldVisible("learning_rate_te1", values)).toBe(false);
    expect(readForm(values, groups).learning_rate_te1).toBeUndefined();
    values.train_text_encoder = true;
    expect(isFieldVisible("learning_rate_te1", values)).toBe(true);
    expect(readForm(values, groups)).toMatchObject({ train_text_encoder: true, learning_rate_te1: 5e-7, learning_rate_te2: 5e-7 });
  });

  it("syncs mixed_precision with full_precision", () => {
    const values = initialFormValues(loadGroups("anima-lora"));
    values.full_precision = "full_fp16";
    applyExclusiveRules(values, "full_precision");
    expect(values.mixed_precision).toBe("fp16");
  });
});

describe("visibility", () => {
  it("shows dependent fields only for their controlling value", () => {
    const values = initialFormValues(loadGroups("anima-lora"));
    expect(isFieldVisible("huber_c", values)).toBe(false);
    values.loss_type = "huber";
    expect(isFieldVisible("huber_c", values)).toBe(true);
    expect(isFieldVisible("wandb_api_key", values)).toBe(false);
    expect(isFieldVisible("network_dim", values)).toBe(true);
  });

  it("shows the bucket side limits only without bucket_no_upscale, which makes sd-scripts ignore them", () => {
    for (const schema of ["anima-lora", "flux-lora", "sd3-lora", "sdxl-lora", "sd-lora", "sdxl-finetune", "sd-dreambooth"]) {
      const values = initialFormValues(loadGroups(schema));
      // the two full-training pages keep sd-scripts' default (off), so their limits stay in use
      const upscaleOff = ["sdxl-finetune", "sd-dreambooth"].includes(schema);
      expect(values.bucket_no_upscale, schema).toBe(!upscaleOff);
      expect(isFieldVisible("min_bucket_reso", values), schema).toBe(upscaleOff);
      values.bucket_no_upscale = upscaleOff;
      expect(isFieldVisible("max_bucket_reso", values), schema).toBe(!upscaleOff);
    }
  });
});

describe("imported values", () => {
  it("selects imported GPUs in the multi-select", () => {
    const groups = loadGroups("anima-lora", ["GPU 0: A", "GPU 1: B"]);
    const field = groups.flatMap((group) => group.fields).find((item) => item.name === "gpu_ids")!;
    expect(toFormValue("gpu_ids", field.schema, ["GPU 1: B", "GPU 9: gone"], { imported: true })).toEqual(["GPU 1: B"]);
  });

  it("selects a GPU saved without its name", () => {
    const groups = loadGroups("anima-lora", ["GPU 0: A (24 GB)", "GPU 1: B (24 GB)"]);
    const field = groups.flatMap((group) => group.fields).find((item) => item.name === "gpu_ids")!;
    expect(toFormValue("gpu_ids", field.schema, ["GPU 1"], { imported: true })).toEqual(["GPU 1: B (24 GB)"]);
    expect(toFormValue("gpu_ids", field.schema, ["GPU 10"], { imported: true })).toEqual([]);
  });
});
