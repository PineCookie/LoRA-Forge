import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";
import HomeView from "./views/HomeView.vue";
import ProxyView from "./views/ProxyView.vue";
import TasksView from "./views/TasksView.vue";
import TrainerView from "./views/TrainerView.vue";

declare module "vue-router" {
  interface RouteMeta {
    title: string;
  }
}

function trainer(path: string, title: string, schemaName: string): RouteRecordRaw {
  return { path, alias: path.replace(/\.html$/, ".md"), component: TrainerView, props: { schemaName }, meta: { title } };
}

// The .html paths are the URLs the GUI has always used, so bookmarks keep working.
const routes: RouteRecordRaw[] = [
  { path: "/", component: HomeView, meta: { title: "LoRA-Forge" } },
  { path: "/lora/index.html", alias: "/lora/index.md", component: HomeView, meta: { title: "LoRA 训练" } },
  trainer("/lora/master.html", "Stable Diffusion", "lora-master"),
  trainer("/lora/flux.html", "Flux", "flux-lora"),
  trainer("/lora/anima.html", "Anima", "anima-lora"),
  trainer("/lora/sd3.html", "SD3.5", "sd3-lora"),
  trainer("/lora/basic.html", "SD1.5", "lora-basic"),
  trainer("/dreambooth/index.html", "Dreambooth", "dreambooth"),
  { path: "/task.html", alias: "/task.md", component: TasksView, meta: { title: "任务" } },
  {
    path: "/tensorboard.html",
    alias: "/tensorboard.md",
    component: ProxyView,
    props: { proxy: "/proxy/tensorboard/", service: "tensorboard" },
    meta: { title: "Tensorboard" },
  },
  { path: "/other/about.html", alias: "/other/about.md", component: HomeView, meta: { title: "关于" } },
  { path: "/:unknown(.*)*", component: HomeView, meta: { title: "LoRA-Forge" } },
];

export const navigation = [
  { path: "/lora/index.html", label: "LoRA 训练" },
  { path: "/lora/master.html", label: "Stable Diffusion" },
  { path: "/lora/flux.html", label: "Flux" },
  { path: "/lora/anima.html", label: "Anima" },
  { path: "/lora/sd3.html", label: "SD3.5" },
  { path: "/lora/basic.html", label: "SD1.5" },
  { path: "/dreambooth/index.html", label: "Dreambooth" },
  { path: "/task.html", label: "任务" },
  { path: "/tensorboard.html", label: "Tensorboard" },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.afterEach((to) => {
  document.title = to.meta.title === "LoRA-Forge" ? "LoRA-Forge" : `${to.meta.title} · LoRA-Forge`;
});
