// Per model: the square sizes offered as quick picks under `resolution` (what the model was trained
// around, smaller ones for faster runs), and the step a fixed size (buckets off) must be a multiple of:
// U-Net models downscale the latent three times (8 x 8 = 64 px), the DiTs patch 2x2 latents (16 px).

export interface ResolutionInfo {
  sizes: number[];
  step: number;
}

const UNET_SD15: ResolutionInfo = { sizes: [512, 640, 768], step: 64 };
const UNET_SDXL: ResolutionInfo = { sizes: [768, 896, 1024, 1280], step: 64 };

export const RESOLUTIONS: Record<string, ResolutionInfo> = {
  "sd-lora": UNET_SD15,
  "sd-dreambooth": UNET_SD15,
  "sdxl-lora": UNET_SDXL,
  "sdxl-finetune": UNET_SDXL,
  "flux-lora": { sizes: [512, 768, 1024], step: 16 },
  "sd3-lora": { sizes: [512, 768, 1024], step: 16 },
  // Anima supports 512x512 to 1536x1536
  "anima-lora": { sizes: [512, 768, 1024, 1280, 1536], step: 16 },
  // krea2_shift covers 256-1280 px
  "krea2-lora": { sizes: [768, 1024, 1280], step: 16 },
  // MOSS-Image trained at about 1024x1024
  "moss-image-lora": { sizes: [768, 1024, 1280], step: 16 },
};
