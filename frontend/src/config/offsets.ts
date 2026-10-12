/** Per-subfolder timestep offsets and the dataset folder they were recorded for. */
export interface SubsetOffsets {
  dir: string | null;
  values: Record<string, string>;
}
