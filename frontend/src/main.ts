import { createApp } from "vue";
import App from "./App.vue";
import { applyTheme } from "./composables/useTheme";
import { applyUiScale } from "./composables/useUiScale";
import { router } from "./router";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";

applyTheme();
applyUiScale();
createApp(App).use(router).mount("#app");
