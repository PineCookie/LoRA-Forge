import { watch } from "vue";
import { createRouter, createWebHistory, type RouteLocationNormalized, type RouteRecordRaw } from "vue-router";
import { locale, t, type MessageKey } from "./i18n";
import { TRAINER_PAGES } from "./pages";
import HomeView from "./views/HomeView.vue";
import ProxyView from "./views/ProxyView.vue";
import RunsView from "./views/RunsView.vue";
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

/** A trainer page; `oldPaths` are URLs of pages it replaced, so bookmarks keep working. */
function trainer(path: string, title: Label, schemaName: string, oldPaths: string[] = []): RouteRecordRaw {
  const alias = [path, ...oldPaths].flatMap((url) => (url === path ? [url.replace(/\.html$/, ".md")] : [url, url.replace(/\.html$/, ".md")]));
  return { path, alias, component: TrainerView, props: { schemaName }, meta: { title } };
}

const routes: RouteRecordRaw[] = [
  { path: "/", component: HomeView, meta: { title: { text: BRAND } } },
  { path: "/lora/index.html", alias: "/lora/index.md", component: HomeView, meta: { title: { key: "nav.lora" } } },
  ...TRAINER_PAGES.map((page) => trainer(page.path, page.title, page.schemaName, page.oldPaths)),
  // /task.html was the task list this page replaced
  { path: "/runs.html", alias: ["/runs.md", "/task.html", "/task.md"], component: RunsView, meta: { title: { key: "nav.runs" } } },
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

// /lora/index.html is the home page under its old URL, so it is not listed
export const navigation: { heading: { key: MessageKey }; items: { path: string; label: Label }[] }[] = [
  {
    heading: { key: "nav.train" },
    items: TRAINER_PAGES.map((page) => ({ path: page.path, label: page.title })),
  },
  {
    heading: { key: "nav.monitor" },
    items: [
      { path: "/runs.html", label: { key: "nav.runs" } },
      { path: "/tensorboard.html", label: { text: "Tensorboard" } },
    ],
  },
];

// a trainer page keeps its form while the app is open (useTrainerForm), so going back to it also
// returns to where it was scrolled
const trainerPaths = new Set(TRAINER_PAGES.map((page) => page.path));
const scrollPositions = new Map<string, number>();
// the page's own path, also when it was opened under an old URL (an alias)
const pagePath = (route: RouteLocationNormalized) => {
  const record = route.matched[0];
  return record ? (record.aliasOf ?? record).path : route.path;
};

export const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) return savedPosition;
    if (to.path === from.path) return false;
    return { top: scrollPositions.get(pagePath(to)) ?? 0 };
  },
});

router.beforeEach((_to, from) => {
  const path = pagePath(from);
  if (trainerPaths.has(path)) scrollPositions.set(path, window.scrollY);
});

function updateDocumentTitle() {
  const title = labelText(router.currentRoute.value.meta.title) || BRAND;
  document.title = title === BRAND ? BRAND : `${title} · ${BRAND}`;
}

router.afterEach(updateDocumentTitle);
watch(locale, updateDocumentTitle);
