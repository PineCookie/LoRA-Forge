import { watch } from "vue";
import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";
import { locale, t, type MessageKey } from "./i18n";
import HomeView from "./views/HomeView.vue";
import ProxyView from "./views/ProxyView.vue";
import TasksView from "./views/TasksView.vue";
import TrainerView from "./views/TrainerView.vue";

/** A name shown as it is (model names) or a UI message key. */
export type Label = { text: string } | { key: MessageKey };

// meta is empty until the first navigation has resolved
export const labelText = (label: Label | undefined) => (!label ? "" : "key" in label ? t(label.key) : label.text);

declare module "vue-router" {
  interface RouteMeta {
    title: Label;
  }
}

const BRAND = "LoRA-Forge";

function trainer(path: string, title: string, schemaName: string): RouteRecordRaw {
  return { path, alias: path.replace(/\.html$/, ".md"), component: TrainerView, props: { schemaName }, meta: { title: { text: title } } };
}

// The .html paths are the URLs the GUI has always used, so bookmarks keep working.
const routes: RouteRecordRaw[] = [
  { path: "/", component: HomeView, meta: { title: { text: BRAND } } },
  { path: "/lora/index.html", alias: "/lora/index.md", component: HomeView, meta: { title: { key: "nav.lora" } } },
  trainer("/lora/master.html", "Stable Diffusion", "lora-master"),
  trainer("/lora/flux.html", "Flux", "flux-lora"),
  trainer("/lora/anima.html", "Anima", "anima-lora"),
  trainer("/lora/sd3.html", "SD3.5", "sd3-lora"),
  trainer("/lora/basic.html", "SD1.5", "lora-basic"),
  trainer("/dreambooth/index.html", "Dreambooth", "dreambooth"),
  { path: "/task.html", alias: "/task.md", component: TasksView, meta: { title: { key: "nav.tasks" } } },
  {
    path: "/tensorboard.html",
    alias: "/tensorboard.md",
    component: ProxyView,
    props: { proxy: "/proxy/tensorboard/", service: "tensorboard" },
    meta: { title: { text: "Tensorboard" } },
  },
  { path: "/other/about.html", alias: "/other/about.md", component: HomeView, meta: { title: { key: "nav.about" } } },
  { path: "/:unknown(.*)*", component: HomeView, meta: { title: { text: BRAND } } },
];

export const navigation: { path: string; label: Label }[] = [
  { path: "/lora/index.html", label: { key: "nav.lora" } },
  { path: "/lora/master.html", label: { text: "Stable Diffusion" } },
  { path: "/lora/flux.html", label: { text: "Flux" } },
  { path: "/lora/anima.html", label: { text: "Anima" } },
  { path: "/lora/sd3.html", label: { text: "SD3.5" } },
  { path: "/lora/basic.html", label: { text: "SD1.5" } },
  { path: "/dreambooth/index.html", label: { text: "Dreambooth" } },
  { path: "/task.html", label: { key: "nav.tasks" } },
  { path: "/tensorboard.html", label: { text: "Tensorboard" } },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});

function updateDocumentTitle() {
  const title = labelText(router.currentRoute.value.meta.title) || BRAND;
  document.title = title === BRAND ? BRAND : `${title} · ${BRAND}`;
}

router.afterEach(updateDocumentTitle);
watch(locale, updateDocumentTitle);
