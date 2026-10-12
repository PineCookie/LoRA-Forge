import { fileURLToPath } from "node:url";
import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vitest/config";

// `npm run dev` serves the UI on Vite's port and forwards API calls to a running GUI backend.
const backend = process.env.LORA_FORGE_BACKEND ?? "http://127.0.0.1:28000";

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      "/api": backend,
      "/proxy": backend,
      "/favicon.ico": backend,
    },
    // the logo is imported from the repository's assets/ folder
    fs: { allow: [fileURLToPath(new URL("..", import.meta.url))] },
  },
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
