/** sd-scripts' name for the offset, set per subset in its dataset config (custom_attributes.timestep_sampling.offset). */
export const OFFSET_NAME = "timestep_sampling.offset";

/** Per-subfolder timestep offsets and the dataset folder they were recorded for. */
export interface SubsetOffsets {
  dir: string | null;
  values: Record<string, string>;
}
