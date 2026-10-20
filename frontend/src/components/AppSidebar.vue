<script setup lang="ts">
import { RouterLink } from "vue-router";
import logoUrl from "../../../assets/logo.svg";
import { useTasks } from "../composables/useTasks";
import { setTheme, THEME_CHOICES, themeChoice } from "../composables/useTheme";
import { setUiScale, UI_SCALE_CHOICES, uiScaleChoice } from "../composables/useUiScale";
import { LOCALES, locale, setLocale, t, type Locale } from "../i18n";
import { REPOSITORY } from "../links";
import { labelText, navigation } from "../router";

const { activeTask } = useTasks();
const locales = Object.entries(LOCALES) as [Locale, string][];
</script>

<template>
  <aside class="sidebar">
    <RouterLink class="brand" to="/">
      <img class="logo" :src="logoUrl" alt="" />
      <span>LoRA-Forge</span>
    </RouterLink>

    <nav>
      <section v-for="group in navigation" :key="group.heading.key">
        <h2>{{ t(group.heading.key) }}</h2>
        <RouterLink v-for="item in group.items" :key="item.path" :to="item.path" exact-active-class="active">
          <span class="tile" aria-hidden="true">
            <template v-if="item.mark">{{ item.mark }}</template>
            <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path v-if="item.icon === 'runs'" d="M3 12h4l3-8 4 16 3-8h4" />
              <path v-else d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
            </svg>
          </span>
          <span class="label">{{ labelText(item.label) }}</span>
          <span v-if="item.path === '/runs.html' && activeTask" class="running-dot" :title="t(`tasks.status.${activeTask.status}`)" />
        </RouterLink>
      </section>
    </nav>

    <div class="footer">
      <div class="setting">
        <span>{{ t("app.language") }}</span>
        <div class="segmented" role="radiogroup" :aria-label="t('app.language')">
          <button v-for="[code, name] in locales" :key="code" type="button" role="radio" :aria-checked="locale === code" @click="setLocale(code)">
            {{ name }}
          </button>
        </div>
      </div>
      <div class="setting">
        <span>{{ t("app.theme") }}</span>
        <div class="segmented" role="radiogroup" :aria-label="t('app.theme')">
          <button v-for="choice in THEME_CHOICES" :key="choice" type="button" role="radio" :aria-checked="themeChoice === choice" @click="setTheme(choice)">
            {{ t(`theme.${choice}`) }}
          </button>
        </div>
      </div>
      <div class="setting">
        <span>{{ t("app.size") }}</span>
        <div class="segmented" role="radiogroup" :aria-label="t('app.size')">
          <button v-for="choice in UI_SCALE_CHOICES" :key="choice" type="button" role="radio" :aria-checked="uiScaleChoice === choice" @click="setUiScale(choice)">
            {{ choice === "auto" ? t("size.auto") : `${Math.round(Number(choice) * 100)}%` }}
          </button>
        </div>
      </div>
      <!-- AGPL-3.0 section 13: users of the UI over a network must be offered the source -->
      <p class="source">
        <a :href="REPOSITORY" target="_blank" rel="noopener">{{ t("app.sourceCode") }}</a>
        <span>AGPL-3.0</span>
      </p>
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  position: sticky;
  top: 0;
  display: flex;
  flex-direction: column;
  height: var(--viewport-height);
  padding: 0 12px 16px;
  overflow-y: auto;
  border-right: 1px solid var(--border);
  background: var(--surface);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  /* as tall as the page header, so the two line up */
  min-height: var(--header-height);
  margin: 0 -12px 14px;
  padding: 0 20px;
  border-bottom: 1px solid var(--border);
  color: var(--text-strong);
  font-size: 18px;
  font-weight: 700;
}

.logo {
  width: 32px;
  height: 32px;
  border-radius: 8px;
}

nav {
  display: grid;
  gap: 18px;
}

nav section {
  display: grid;
  gap: 2px;
}

h2 {
  margin: 0 10px 6px;
  color: var(--muted);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

nav a {
  display: flex;
  align-items: center;
  gap: 11px;
  padding: 6px 8px;
  border-radius: var(--radius-sm);
  color: var(--text);
  font-size: 14.5px;
  font-weight: 500;
  transition: background 0.16s ease, color 0.16s ease;
}

nav a:hover {
  background: var(--surface-muted);
  color: var(--text-strong);
}

nav a.active {
  background: var(--surface-sunken);
  color: var(--text-strong);
  font-weight: 650;
}

.tile {
  display: grid;
  flex: 0 0 auto;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 7px;
  background: var(--surface-sunken);
  color: var(--text);
  font-size: 11px;
  font-weight: 800;
  letter-spacing: -0.02em;
}

nav a:hover .tile {
  background: var(--border);
}

.active .tile,
nav a.active:hover .tile {
  background: var(--primary);
  color: var(--on-primary);
}

.tile svg {
  width: 16px;
  height: 16px;
}

.label {
  flex: 1;
  min-width: 0;
}

.running-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 0 3px var(--success-soft);
}

.footer {
  display: grid;
  gap: 12px;
  margin: auto 4px 0;
  padding-top: 18px;
  color: var(--muted);
  font-size: 12px;
}

.setting {
  display: grid;
  gap: 5px;
}

.setting .segmented button {
  padding: 4px 6px;
  font-size: 12px;
}

.source {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin: 2px 0 0;
  color: var(--faint);
}

.source a {
  color: var(--muted);
  text-decoration: underline;
}

.source a:hover {
  color: var(--primary-text);
}

@media (max-width: 860px) {
  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }

  nav section {
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  }

  h2 {
    grid-column: 1 / -1;
  }

  .footer {
    grid-template-columns: 1fr 1fr;
    margin-top: 16px;
  }

  .source {
    grid-column: 1 / -1;
  }
}
</style>
