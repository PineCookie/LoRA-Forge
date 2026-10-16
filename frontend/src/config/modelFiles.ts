import type { MessageKey } from "../i18n";

// The model files a trainer page needs, with where to download them. The page checks each path
// the user enters with /api/model_file (lora_forge/training/model_files.py).

export interface ModelFileSpec {
  /** the form field that holds the file's path */
  field: string;
  role: MessageKey;
  repo: string;
  /** the file inside the repository */
  path: string;
  size: string;
  note?: MessageKey;
}

export const MODEL_FILES: Record<string, ModelFileSpec[]> = {
  "anima-lora": [
    {
      field: "pretrained_model_name_or_path",
      role: "modelFiles.role.dit",
      repo: "circlestone-labs/Anima",
      path: "split_files/diffusion_models/anima-base-v1.0.safetensors",
      size: "4.18 GB",
      note: "modelFiles.note.animaDit",
    },
    {
      field: "qwen3",
      role: "modelFiles.role.textEncoder",
      repo: "circlestone-labs/Anima",
      path: "split_files/text_encoders/qwen_3_06b_base.safetensors",
      size: "1.19 GB",
      note: "modelFiles.note.animaQwen3",
    },
    {
      field: "vae",
      role: "modelFiles.role.vae",
      repo: "circlestone-labs/Anima",
      path: "split_files/vae/qwen_image_vae.safetensors",
      size: "254 MB",
      note: "modelFiles.note.animaVae",
    },
  ],
  "moss-image-lora": [
    {
      field: "pretrained_model_name_or_path",
      role: "modelFiles.role.dit",
      repo: "Comfy-Org/MOSS-Image",
      path: "diffusion_models/moss_image_base_bf16.safetensors",
      size: "10.2 GB",
    },
    {
      field: "text_encoder",
      role: "modelFiles.role.textEncoder",
      repo: "Comfy-Org/Qwen3.5",
      path: "text_encoders/qwen3.5_9b_bf16.safetensors",
      size: "19.3 GB",
      note: "modelFiles.note.mossTextEncoder",
    },
    {
      field: "vae",
      role: "modelFiles.role.vae",
      repo: "OpenMOSS-Team/MOSS-Image-Base",
      path: "vae/diffusion_pytorch_model.safetensors",
      size: "250 MB",
      note: "modelFiles.note.mossVae",
    },
  ],
};

export type ModelFileState = "empty" | "missing" | "repo" | "quantized" | "wrong" | "ok" | "unknown";
