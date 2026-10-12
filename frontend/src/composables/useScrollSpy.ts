import { onBeforeUnmount, onMounted, ref } from "vue";

const USER_SCROLL_EVENTS = ["wheel", "touchstart", "keydown", "mousedown"] as const;

/**
 * Index of the section currently at the top of the page. Sections are the elements matching
 * `selector`, each with its index in `data-section-index`; `offset` is the sticky header height.
 */
export function useScrollSpy(selector: string, offset = 160) {
  const active = ref<number | null>(null);
  let frame = 0;
  // after a jump the target stays active until the user scrolls themselves: a section near the
  // end of the page cannot reach the top, so position alone would highlight a later one
  let pinned = false;

  function update() {
    frame = 0;
    if (pinned) return;
    const sections = Array.from(document.querySelectorAll<HTMLElement>(selector));
    if (!sections.length) {
      active.value = null;
      return;
    }
    let current = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top - offset <= 0) current = section;
    }
    const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
    if (atBottom) current = sections[sections.length - 1];
    active.value = Number(current.dataset.sectionIndex);
  }

  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(update);
  };

  const unpin = () => {
    pinned = false;
  };

  /** Mark `index` active (after a jump) until the user scrolls. */
  function pin(index: number) {
    pinned = true;
    active.value = index;
  }

  onMounted(() => {
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    for (const name of USER_SCROLL_EVENTS) window.addEventListener(name, unpin, { passive: true, capture: true });
    schedule();
  });
  onBeforeUnmount(() => {
    window.removeEventListener("scroll", schedule);
    window.removeEventListener("resize", schedule);
    for (const name of USER_SCROLL_EVENTS) window.removeEventListener(name, unpin, { capture: true });
    cancelAnimationFrame(frame);
  });

  return { active, refresh: schedule, pin };
}
