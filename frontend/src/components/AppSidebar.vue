<script setup lang="ts">
import { RouterLink } from "vue-router";
import logoUrl from "../../../assets/logo.svg";
import { useTasks } from "../composables/useTasks";
import { setTheme, THEME_CHOICES, themeChoice, type ThemeChoice } from "../composables/useTheme";
import { LOCALES, locale, setLocale, t, type Locale } from "../i18n";
import { REPOSITORY } from "../links";
import { labelText, navigation } from "../router";

const { activeTask } = useTasks();
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
          {{ labelText(item.label) }}
          <span v-if="item.path === '/runs.html' && activeTask" class="running-dot" :title="t(`tasks.status.${activeTask.status}`)" />
        </RouterLink>
      </section>
    </nav>

    <div class="footer">
      <label>
        <span>{{ t("app.language") }}</span>
        <select :value="locale" @change="setLocale(($event.target as HTMLSelectElement).value as Locale)">
          <option v-for="(name, code) in LOCALES" :key="code" :value="code">{{ name }}</option>
        </select>
      </label>
      <label>
        <span>{{ t("app.theme") }}</span>
        <select :value="themeChoice" @change="setTheme(($event.target as HTMLSelectElement).value as ThemeChoice)">
          <option v-for="choice in THEME_CHOICES" :key="choice" :value="choice">{{ t(`theme.${choice}`) }}</option>
        </select>
      </label>
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
  height: 100vh;
  padding: 18px 14px;
  overflow-y: auto;
  border-right: 1px solid var(--border);
  background: var(--surface);
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 8px 20px;
  font-size: 20px;
  font-weight: 700;
}

.brand span {
  background: linear-gradient(90deg, #2563eb 0%, #0891b2 42%, #16a34a 76%, #ea580c 100%);
  background-clip: text;
  color: transparent;
}

.logo {
  width: 34px;
  height: 34px;
  border-radius: 8px;
  box-shadow: var(--shadow-sm);
}

nav {
  display: grid;
  gap: 16px;
}

nav section {
  display: grid;
  gap: 2px;
}

h2 {
  margin: 0 10px 4px;
  color: var(--faint);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

nav a {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  color: var(--muted);
  font-weight: 500;
  transition: background 0.16s ease, color 0.16s ease;
}

nav a:hover {
  background: var(--surface-muted);
  color: var(--text);
}

nav a.active {
  background: var(--primary-soft);
  color: var(--primary-text);
  font-weight: 700;
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
  gap: 10px;
  margin: auto 8px 0;
  padding-top: 18px;
  color: var(--muted);
  font-size: 12px;
}

.footer label {
  display: grid;
  gap: 4px;
}

.footer select {
  padding: 5px 8px;
  font-size: 12px;
}

.source {
  display: grid;
  gap: 2px;
  margin: 4px 0 0;
}

.source a {
  text-decoration: underline;
}

.source a:hover {
  color: var(--primary);
}

@media (max-width: 860px) {
  .sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }

  nav section {
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
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
