import type { MessageKey } from "../i18n";

// The model files a trainer page needs, with where to download them. The page checks each path
// the user enters with /api/model_file (lora_forge/training/moss_image.py check_model_file).

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

export type ModelFileState = "empty" | "missing" | "quantized" | "wrong" | "ok" | "unknown";
