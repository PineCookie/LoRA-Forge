import { describe, expect, it } from "vitest";
import { initialFormValues, readForm, toFormValue } from "../src/config/form";
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
    expect(loadGroups("anima-lora", ["GPU 0: A"]).some((group) => group.title === "GPU")).toBe(false);
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
