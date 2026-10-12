// UI text, English (the source language). zh.ts and ja.ts must have the same keys; the type
// checker enforces it. Values marked "HTML" are rendered with v-html.

const en = {
  "app.eyebrow": "Training UI",
  "app.loading": "Loading training options…",
  "app.loadFailed": "The web UI failed to load",
  "app.sourceCode": "Source code",
  "app.language": "Language",

  "nav.lora": "LoRA training",
  "nav.tasks": "Tasks",
  "nav.about": "About",

  "home.title": "LoRA-Forge training UI",
  "home.intro": "Pick a training mode to configure it. Configs can be exported as JSON / TOML and imported again later.",
  "home.mode.anima": "Anima DiT LoRA, with per-block torch.compile and per-subfolder timestep offsets.",
  "home.mode.flux": "FLUX.1 LoRA training.",
  "home.mode.master": "SD1.5 / SDXL LoRA with every option.",
  "home.mode.sd3": "Stable Diffusion 3.5 LoRA training.",
  "home.mode.basic": "SD1.5 LoRA with a reduced set of options.",
  "home.mode.dreambooth": "Dreambooth / full fine-tuning.",

  "tasks.title": "Tasks",
  "tasks.empty": "No tasks recorded.",
  "tasks.status.CREATED": "Waiting to start",
  "tasks.status.RUNNING": "Running",
  "tasks.status.FINISHED": "Finished",
  "tasks.status.TERMINATED": "Terminated",
  "tasks.terminate": "Terminate",
  "tasks.confirmTerminate": "Terminate task {id}?",
  "tasks.terminated": "Task {id} terminated",
  "tasks.terminateFailed": "Could not terminate the task: {error}",

  "trainer.schemaMissing": "Schema not found: {name}",
  "trainer.search": "Search options by name or description…",
  "trainer.searchLabel": "Search options",
  "trainer.matchCount": "{count} found",
  "trainer.expandAll": "Expand all",
  "trainer.collapseAll": "Collapse all",
  "trainer.noMatch": "No options match “{query}”.",
  "trainer.note.optimizer": "Switching the optimizer resets some learning rates, the scheduler and optimizer_args to the values recommended for it.",
  "trainer.note.lossAnima":
    "Anima uses the FlowMatch scheduler: the snr Huber schedule is not supported, and exponential may run on the network training path, but its timestep scaling makes it close to constant. constant is usually the better choice.",
  "trainer.note.lossMinSnr": "Min-SNR and Debiased Estimation loss have similar effects; enabling both is usually not recommended.",
  "trainer.training": "Training",
  "trainer.start": "Start training",
  "trainer.submitting": "Submitting…",
  "trainer.previewTimesteps": "Preview timesteps",
  "trainer.stop": "Stop training",
  "trainer.config": "Config",
  "trainer.reset": "Reset all",
  "trainer.downloadJson": "Download JSON",
  "trainer.importJson": "Import JSON",
  "trainer.downloadToml": "Download TOML",
  "trainer.importToml": "Import TOML",
  "trainer.generatedArgs": "Generated arguments",
  "trainer.noGeneratedArgs": "This config adds no extra arguments",
  "trainer.tomlPreview": "TOML preview",
  "trainer.copy": "Copy",
  "trainer.copyTitle": "Copy to the clipboard",
  "trainer.previewing": "Generating the timestep distribution preview…",
  "trainer.starting": "Starting training…",
  "trainer.previewSubmitted": "Timestep preview submitted",
  "trainer.submitted": "Training submitted",
  "trainer.noRunningTask": "No training task is running",
  "trainer.confirmStop": "Stop task {id}?",
  "trainer.stopped": "Task {id} stopped",
  "trainer.stopFailed": "Could not stop training: {error}",
  "trainer.copied": "TOML copied to the clipboard",
  "trainer.copyFailed": "Copy failed: the browser denied clipboard access",
  "trainer.wrongTrainType": "This page supports {allowed}, but the file is for {type}",
  "trainer.imported": "{format} imported",
  "trainer.importFailed": "{format} import failed: {error}",
  "trainer.tomlTables": "Only the flat TOML of training configs can be imported (no tables)",

  "field.browse": "Browse",

  "dataset.loading": "Reading the dataset…",
  "dataset.error": "Cannot read the dataset: {error}",
  "dataset.title": "📊 Step estimate",
  "dataset.subfolder": "Subfolder",
  "dataset.repeats": "Repeats",
  "dataset.images": "Images",
  "dataset.effectiveImages": "With repeats",
  "dataset.offset": "Timestep offset",
  "dataset.offsetTitle": "Negative: low noise / detail; positive: high noise / structure. 0 keeps the default behavior.",
  // HTML
  "dataset.offsetNote":
    "Timestep offset: negative values favor low noise (detail), positive values high noise (structure). Only used for FLUX / Anima training; 0 keeps the default behavior. ⚠ Needs sd-scripts main (&gt; v0.11.1); older versions silently ignore it.",
  // HTML
  "dataset.adviceSigmaUniform":
    "<b>Advice:</b> offsets have no effect with <code>sigma</code> / <code>uniform</code>. To use them, start from <code>sigmoid</code>.",
  // HTML
  "dataset.adviceShift":
    "<b>Advice:</b> <code>shift</code> / <code>flux_shift</code> already favor high timesteps. Start at <b>±0.25</b>, be careful with positive offsets, and preview the distribution with <code>show_timesteps=console</code> first.",
  // HTML
  "dataset.adviceDefault":
    "<b>Suggested start:</b> <code>timestep_sampling=sigmoid</code>, <code>sigmoid_scale=1.0</code>. Keep every offset at 0 first; try <b>-0.25</b> for detail / close-up subsets and <b>+0.25</b> for full-body / composition subsets. Only go up to <b>±0.5</b> after comparing previews.",
  // HTML
  "dataset.noSubsets": "No subfolders in the <code>repeats_name</code> format found",
  "dataset.totalImages": "Images",
  "dataset.totalWithRepeats": "Images × repeats",
  "dataset.effectiveBatch": "Effective batch size (batch × grad_accum)",
  "dataset.stepsPerEpoch": "Minimum steps per epoch",
  "dataset.totalSteps": "Minimum total steps for {epochs} epochs",
  // HTML
  "dataset.bucketNote":
    "⚠ These are <b>lower bounds</b>: they assume every image lands in one resolution bucket. ARB sorts images into buckets by aspect ratio and batches each bucket separately, so the real step count is a little higher (usually +1% to 10%).",
  "dataset.suggestTitle": "💡 Epochs for a target step count (from the lower bound)",
  "dataset.suggestion": "~{steps} steps: {epochs} epochs",

  "system.title": "System info",
  "system.sdScriptsStatus": "sd-scripts compatibility",
  "system.loading": "Loading…",

  "proxy.notRunning": "{name} is not running",
  "proxy.notStarted": "The UI was started without this service, so {proxy} is not connected.",
  // HTML
  "proxy.howToStart":
    "To use it, restart the GUI with <code>--enable-{service}</code>, for example <code>run_gui.ps1 --enable-{service}</code>.",

  "toast.dismiss": "Dismiss",

  "api.html": "The API returned HTML. Open the page through the Python GUI backend, not a static file server.",
  "api.unparsable": "The API returned a response that could not be parsed.",
  "api.failed": "Request failed",
};

export default en;
