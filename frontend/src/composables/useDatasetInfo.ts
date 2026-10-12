import { onWatcherCleanup, ref, watch, type Ref } from "vue";
import { api } from "../api";

export interface Subdir {
  name: string;
  concept: string;
  repeat: number;
  image_count: number;
}

export interface DatasetInfo {
  subdirs: Subdir[];
  total_images: number;
  total_images_with_repeats: number;
}

const DEBOUNCE_MS = 400;

/** Image counts of the dataset folder, refetched (debounced) when the folder or `refreshKey` changes. */
export function useDatasetInfo(dir: Ref<string>, refreshKey: Ref<number>) {
  const info = ref<DatasetInfo | null>(null);
  const loading = ref(false);
  const error = ref("");

  watch(
    [() => dir.value.trim(), refreshKey],
    ([folder]) => {
      error.value = "";
      if (!folder) {
        info.value = null;
        loading.value = false;
        return;
      }
      loading.value = true;
      let stale = false;
      const timer = setTimeout(async () => {
        try {
          const result = await api<DatasetInfo>(`/api/dataset_info?train_data_dir=${encodeURIComponent(folder)}`);
          if (!stale) info.value = result;
        } catch (err) {
          if (stale) return;
          info.value = null;
          error.value = err instanceof Error ? err.message : String(err);
        } finally {
          if (!stale) loading.value = false;
        }
      }, DEBOUNCE_MS);
      // a newer folder (or unmount) wins over a request still in flight
      onWatcherCleanup(() => {
        stale = true;
        clearTimeout(timer);
      });
    },
    { immediate: true },
  );

  return { info, loading, error };
}
