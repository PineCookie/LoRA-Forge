import type { Label } from "./router";

// One trainer page per model. The .html paths are the URLs the GUI has always used; oldPaths
// are URLs of pages a page replaced, so bookmarks keep working. guide names the model's user guide
// in docs/ (linked in the page header).
export const TRAINER_PAGES: { path: string; title: Label; schemaName: string; oldPaths?: string[]; guide?: string }[] = [
  { path: "/lora/anima.html", title: { text: "Anima" }, schemaName: "anima-lora", guide: "anima" },
  { path: "/lora/krea2.html", title: { text: "Krea 2" }, schemaName: "krea2-lora", guide: "krea2" },
  { path: "/lora/moss-image.html", title: { text: "MOSS-Image" }, schemaName: "moss-image-lora", guide: "moss-image" },
  // the combined SD1.5 / SDXL page was /lora/master.html
  { path: "/lora/sdxl.html", title: { text: "SDXL" }, schemaName: "sdxl-lora", oldPaths: ["/lora/master.html"] },
  { path: "/lora/flux.html", title: { text: "Flux" }, schemaName: "flux-lora" },
  { path: "/lora/sd3.html", title: { text: "SD3.5" }, schemaName: "sd3-lora" },
  { path: "/finetune/sdxl.html", title: { key: "nav.sdxlFinetune" }, schemaName: "sdxl-finetune" },
  // the reduced SD1.5 page was /lora/basic.html
  { path: "/lora/sd15.html", title: { text: "SD1.5" }, schemaName: "sd-lora", oldPaths: ["/lora/basic.html"] },
  { path: "/dreambooth/index.html", title: { key: "nav.sdDreambooth" }, schemaName: "sd-dreambooth" },
];
