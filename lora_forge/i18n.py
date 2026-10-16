"""Messages the web UI shows, in English, Chinese and Japanese.

The UI sends its language in Accept-Language; `LocaleMiddleware` makes it the language of
everything `t()` returns while that request is handled. Terminal logs stay in English.
Schema text (field descriptions, section titles) is translated in the browser from
lora_forge/schema/locales/.
"""

from contextvars import ContextVar

LOCALES = ("en", "zh", "ja")

_locale: ContextVar[str] = ContextVar("locale", default="en")


def pick_locale(accept_language: str) -> str:
    """The first supported language of an Accept-Language header, else English."""
    for part in accept_language.split(","):
        language = part.split(";")[0].strip().lower().split("-")[0]
        if language in LOCALES:
            return language
    return "en"


def t(key: str, **params) -> str:
    """The message `key` in the current request's language, with `{name}` fields filled in."""
    texts = MESSAGES[key]
    text = texts.get(_locale.get(), texts["en"])
    return text.format(**params) if params else text


def join_list(items) -> str:
    return ("、" if _locale.get() in ("zh", "ja") else ", ").join(items)


class LocaleMiddleware:
    """Pure ASGI middleware, so the context variable is set in the task that runs the endpoint."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        header = dict(scope["headers"]).get(b"accept-language", b"").decode("latin-1")
        token = _locale.set(pick_locale(header))
        try:
            await self.app(scope, receive, send)
        finally:
            _locale.reset(token)


MESSAGES: dict[str, dict[str, str]] = {
    # --- /api/run ----------------------------------------------------------------------------
    "run.started": {
        "en": "Training started. Follow it on the Runs page.",
        "zh": "训练已开始，可在「运行记录」页面查看进度。",
        "ja": "学習を開始しました。「実行履歴」ページで進捗を確認できます。",
    },
    "runs.notFound": {
        "en": "This run record does not exist (any more).",
        "zh": "该运行记录不存在（或已删除）。",
        "ja": "この実行記録は存在しません（削除済みの可能性があります）。",
    },
    "runs.stillRunning": {
        "en": "This run is still running. Stop it before deleting its record.",
        "zh": "该运行仍在进行中，请先停止再删除记录。",
        "ja": "この実行はまだ進行中です。記録を削除する前に停止してください。",
    },
    "run.serverError": {
        "en": "Server error: {error}",
        "zh": "服务器错误：{error}",
        "ja": "サーバーエラー：{error}",
    },
    "run.alreadyRunning": {
        "en": "A training task is already running, so no new task was started.",
        "zh": "已有训练任务正在运行，无法创建新任务。",
        "ja": "学習タスクがすでに実行中のため、新しいタスクを開始できません。",
    },
    "run.unknownType": {
        "en": "Unknown training type: {type}",
        "zh": "未知的训练类型：{type}",
        "ja": "不明な学習タイプ：{type}",
    },
    "run.noBaseModel": {
        "en": "Select a base model: pretrained_model_name_or_path is empty.",
        "zh": "请选择底模：pretrained_model_name_or_path 未填写。",
        "ja": "ベースモデルを選択してください：pretrained_model_name_or_path が空欄です。",
    },
    "run.memoryBudgetRange": {
        "en": "activation_memory_budget must be off, auto, or a number above 0 and at most 1.",
        "zh": "activation_memory_budget 必须是 off、auto，或大于 0 且不超过 1 的数。",
        "ja": "activation_memory_budget は off、auto、または 0 より大きく 1 以下の数値にしてください。",
    },
    "run.fullFp16Optimizer": {
        "en": "full_fp16 does not work with {optimizer}: with fp16 weights it stops learning or turns the weights "
              "into NaN. Use {works}, or turn full_fp16 off (with mixed precision fp16 the weights then train in "
              "fp32; on the Anima page: training_precision fp16).",
        "zh": "full_fp16 与 {optimizer} 不兼容：权重为 fp16 时它会停止学习，或让权重变成 NaN。请改用 {works}，"
              "或关闭 full_fp16（混合精度为 fp16 时权重会以 fp32 训练；Anima 页面请把 training_precision 选为 fp16）。",
        "ja": "full_fp16 は {optimizer} と組み合わせられません。重みが fp16 だと学習が止まるか、重みが NaN になります。"
              "{works} を使うか、full_fp16 をオフにしてください（混合精度が fp16 なら重みは fp32 で学習します。"
              "Anima のページでは training_precision を fp16 に）。",
    },
    "run.weightedCaptions": {
        "en": "This training type does not support weighted_captions: only SD / SDXL implement weighted "
              "tokenization, and Anima / FLUX / SD3 ignore it or fail. Turn the option off and try again.",
        "zh": "当前训练类型不支持 weighted_captions：只有 SD / SDXL 实现了加权分词，"
              "Anima / FLUX / SD3 会静默忽略或直接报错。请关闭该选项后重试。",
        "ja": "この学習タイプは weighted_captions に対応していません。重み付きトークン化は SD / SDXL のみで、"
              "Anima / FLUX / SD3 では無視されるかエラーになります。オプションをオフにして再試行してください。",
    },
    "run.promptFileMissing": {
        "en": "The prompt file {path} does not exist. Check the path.",
        "zh": "Prompt 文件 {path} 不存在，请检查路径。",
        "ja": "プロンプトファイル {path} が存在しません。パスを確認してください。",
    },
    "run.randomPromptSubsets": {
        "en": "A random preview prompt needs exactly one subfolder in the training dataset.",
        "zh": "训练数据集下有多个子文件夹，无法启用随机选取 Prompt 功能。",
        "ja": "ランダムなプレビュー用プロンプトを使うには、学習データセットのサブフォルダが 1 つだけである必要があります。",
    },
    "run.randomPromptNoTxt": {
        "en": "The training dataset subfolder has no txt caption files.",
        "zh": "训练数据集路径没有 txt 文件。",
        "ja": "学習データセットのサブフォルダに txt キャプションファイルがありません。",
    },
    "run.offsetsNoSubsets": {
        "en": "No training subfolders found; name them in the `repeats_name` format.",
        "zh": "未检测到可用的训练子文件夹；请使用 `重复次数_名称` 格式。",
        "ja": "学習用のサブフォルダが見つかりません。`繰り返し回数_名前` の形式で作成してください。",
    },
    "run.offsetsUnknown": {
        "en": "These timestep offsets belong to subfolders that do not exist: {names}",
        "zh": "下列时间步偏移对应的子文件夹不存在：{names}",
        "ja": "次のタイムステップオフセットに対応するサブフォルダが存在しません：{names}",
    },
    # --- dataset -----------------------------------------------------------------------------
    "dataset.missing": {
        "en": "The training dataset folder does not exist: {path}",
        "zh": "训练数据集路径不存在，请检查目录：{path}",
        "ja": "学習データセットのフォルダが存在しません：{path}",
    },
    "dataset.noSubsets": {
        "en": "The training dataset has no subfolders in the `repeats_name` format (for example `10_mychar`). "
              "{found}Create such a subfolder and put the images and caption files in it; the number is how "
              "often each image repeats per epoch.",
        "zh": "训练数据集路径下没有 `重复次数_名称` 格式的子文件夹（例如 `10_mychar`）。{found}"
              "请新建这样的子文件夹，把图片和标注文件放进去；数字为每个 epoch 的重复次数。",
        "ja": "学習データセットに `繰り返し回数_名前` 形式のサブフォルダ（例：`10_mychar`）がありません。{found}"
              "このようなサブフォルダを作成し、画像とキャプションファイルを入れてください。数字は 1 エポックあたりの繰り返し回数です。",
    },
    "dataset.looseImages": {
        "en": "It has {count} images directly inside. ",
        "zh": "目录下直接放了 {count} 张图片。",
        "ja": "フォルダ直下に画像が {count} 枚あります。",
    },
    "dataset.noImages": {
        "en": "The subfolders of the training dataset contain no images (supported: jpg / jpeg / png / webp / bmp).",
        "zh": "训练数据集的子文件夹中没有图片（支持 jpg / jpeg / png / webp / bmp）。",
        "ja": "学習データセットのサブフォルダに画像がありません（対応形式：jpg / jpeg / png / webp / bmp）。",
    },
    "dataset.invalid": {
        "en": "The training dataset folder does not exist or is not a folder.",
        "zh": "训练数据集路径不存在或无效。",
        "ja": "学習データセットのフォルダが存在しないか、フォルダではありません。",
    },
    "dataset.unreadable": {
        "en": "The training dataset folder cannot be read.",
        "zh": "无法读取训练数据集目录。",
        "ja": "学習データセットのフォルダを読み込めません。",
    },
    # --- models ------------------------------------------------------------------------------
    "model.notFound": {
        "en": "Base model not found: {path}",
        "zh": "找不到底模：{path}",
        "ja": "ベースモデルが見つかりません：{path}",
    },
    "model.notBaseModel": {
        "en": "The base model is not a Stable Diffusion, FLUX or Anima checkpoint.",
        "zh": "校验失败：底模不是 Stable Diffusion、FLUX 或 Anima 模型。",
        "ja": "ベースモデルが Stable Diffusion、FLUX、Anima のいずれのモデルでもありません。",
    },
    "model.animaWrongMode": {
        "en": "The base model is Anima, but this is not the Anima LoRA page.",
        "zh": "校验失败：预训练模型是 Anima，但当前训练种类不是 Anima LoRA。",
        "ja": "ベースモデルは Anima ですが、学習タイプが Anima LoRA ではありません。",
    },
    "model.notAnima": {
        "en": "Anima LoRA training needs an Anima base model, but this one is not.",
        "zh": "校验失败：你选择的是 Anima LoRA 训练，但预训练模型不是 Anima。",
        "ja": "Anima LoRA の学習を選んでいますが、ベースモデルが Anima ではありません。",
    },
    "model.sdxlOnSd15": {
        "en": "The base model is SDXL, but this page trains SD1.5 LoRAs. Use the Stable Diffusion page instead.",
        "zh": "校验失败：你选择的是 SD1.5 LoRA 训练，但预训练模型是 SDXL。请前往 Stable Diffusion 模式选择正确的模型种类。",
        "ja": "SD1.5 LoRA の学習を選んでいますが、ベースモデルは SDXL です。Stable Diffusion のページを使ってください。",
    },
    "model.animaPathMissing": {
        "en": "{label} is empty",
        "zh": "{label} 未填写",
        "ja": "{label} が空欄です",
    },
    "model.animaPathNotFound": {
        "en": "{label} does not exist: {path}",
        "zh": "{label} 路径不存在：{path}",
        "ja": "{label} が存在しません：{path}",
    },
    "model.qwen3": {
        "en": "Qwen3-0.6B text encoder (qwen3)",
        "zh": "Qwen3-0.6B 文本编码器（qwen3）",
        "ja": "Qwen3-0.6B テキストエンコーダ（qwen3）",
    },
    "model.vae": {
        "en": "Qwen-Image VAE (vae)",
        "zh": "Qwen-Image VAE（vae）",
        "ja": "Qwen-Image VAE（vae）",
    },
    "model.dit": {
        "en": "Krea 2 DiT (base model)",
        "zh": "Krea 2 DiT（底模）",
        "ja": "Krea 2 DiT（ベースモデル）",
    },
    "model.textEncoder": {
        "en": "Qwen3-VL-4B text encoder (text_encoder)",
        "zh": "Qwen3-VL-4B 文本编码器（text_encoder）",
        "ja": "Qwen3-VL-4B テキストエンコーダー（text_encoder）",
    },
    "run.krea2SingleGpu": {
        "en": "Krea 2 training runs on one GPU: pick a single GPU.",
        "zh": "Krea 2 训练只支持单卡：请只选择一个 GPU。",
        "ja": "Krea 2 の学習は 1 GPU のみ対応です。GPU を 1 つだけ選んでください。",
    },
    "run.krea2PinnedRam": {
        "en": "Swapping {blocks} blocks in pinned RAM needs about {need} GB of free RAM, but {free} GB is free. "
              "Pinned RAM cannot be paged out, so this could freeze the PC. Lower blocks_to_swap, turn off "
              "use_pinned_memory_for_block_swap, or close other programs.",
        "zh": "在锁页内存中换出 {blocks} 个 block 需要约 {need} GB 可用内存，但目前只有 {free} GB。锁页内存不能被换页，"
              "可能导致电脑卡死。请调低 blocks_to_swap、关闭 use_pinned_memory_for_block_swap，或关闭其他程序。",
        "ja": "ピン留めメモリで {blocks} ブロックをスワップするには約 {need} GB の空き RAM が必要ですが、空きは {free} GB です。"
              "ピン留めメモリはページアウトできないため、PC が固まるおそれがあります。blocks_to_swap を下げるか、"
              "use_pinned_memory_for_block_swap をオフにするか、ほかのプログラムを閉じてください。",
    },
    "run.krea2TurboBoth": {
        "en": "Set only one of turbo_dit and turbo_lora: both are sources for the Turbo previews.",
        "zh": "turbo_dit 和 turbo_lora 只能设置一个：两者都是 Turbo 预览图的来源。",
        "ja": "turbo_dit と turbo_lora はどちらか一方だけ設定してください。どちらも Turbo プレビューの元になります。",
    },
    "run.krea2TurboDit": {
        "en": "turbo_dit does not work with blocks_to_swap or convrot_int8. Use turbo_lora for Turbo previews "
              "instead, or clear turbo_dit to preview with RAW.",
        "zh": "turbo_dit 不能与 blocks_to_swap 或 convrot_int8 同时使用。请改用 turbo_lora 生成 Turbo 预览图，"
              "或清空 turbo_dit 用 RAW 预览。",
        "ja": "turbo_dit は blocks_to_swap や convrot_int8 と併用できません。Turbo プレビューには turbo_lora を使うか、"
              "turbo_dit を空にして RAW でプレビューしてください。",
    },
    "model.mossDit": {
        "en": "MOSS-Image DiT (base model)",
        "zh": "MOSS-Image DiT（底模）",
        "ja": "MOSS-Image DiT（ベースモデル）",
    },
    "model.mossTextEncoder": {
        "en": "Qwen3.5-9B text encoder (text_encoder)",
        "zh": "Qwen3.5-9B 文本编码器（text_encoder）",
        "ja": "Qwen3.5-9B テキストエンコーダー（text_encoder）",
    },
    "model.mossVae": {
        "en": "FLUX.2 VAE (vae)",
        "zh": "FLUX.2 VAE（vae）",
        "ja": "FLUX.2 VAE（vae）",
    },
    "run.mossSingleGpu": {
        "en": "MOSS-Image training runs on one GPU: pick a single GPU.",
        "zh": "MOSS-Image 训练只支持单卡：请只选择一个 GPU。",
        "ja": "MOSS-Image の学習は 1 GPU のみ対応です。GPU を 1 つだけ選んでください。",
    },
    "run.mossQuantized": {
        "en": "{label} is a quantized file (int8 / fp8): {path}. MOSS-Image training needs the bf16 file.",
        "zh": "{label} 是量化文件（int8 / fp8）：{path}。MOSS-Image 训练需要 bf16 文件。",
        "ja": "{label} は量子化されたファイル（int8 / fp8）です：{path}。MOSS-Image の学習には bf16 のファイルが必要です。",
    },
    # --- sd-scripts compatibility ------------------------------------------------------------
    "sdScripts.missing": {
        "en": "The installed sd-scripts ({ref}) does not support: {features}. sd-scripts would silently ignore "
              "these options; update it to the main branch and try again: run_gui.ps1 update-sd-scripts "
              "(Linux: bash run_gui.sh update-sd-scripts)",
        "zh": "已安装的 sd-scripts（{ref}）不支持：{features}。这些选项会被 sd-scripts 静默忽略，请先更新到 main 分支后重试："
              "run_gui.ps1 update-sd-scripts（Linux: bash run_gui.sh update-sd-scripts）",
        "ja": "インストール済みの sd-scripts（{ref}）は次に対応していません：{features}。これらのオプションは sd-scripts に"
              "無視されるため、main ブランチに更新してから再試行してください：run_gui.ps1 update-sd-scripts"
              "（Linux：bash run_gui.sh update-sd-scripts）",
    },
    "sdScripts.outdated": {
        "en": "Outdated: missing {features}",
        "zh": "过旧：缺少 {features}",
        "ja": "古いバージョン：{features} がありません",
    },
    "sdScripts.tested": {
        "en": "Compatible (tested version)",
        "zh": "兼容（已测试版本）",
        "ja": "対応（検証済みバージョン）",
    },
    "sdScripts.noGit": {
        "en": "Compatible (git version unknown)",
        "zh": "兼容（未能读取 git 版本信息）",
        "ja": "対応（git のバージョン情報を取得できません）",
    },
    "sdScripts.compatible": {
        "en": "Compatible",
        "zh": "兼容",
        "ja": "対応",
    },
    "feature.compile": {
        "en": "per-block torch.compile / CUDA performance options",
        "zh": "逐 block torch.compile / CUDA 性能开关",
        "ja": "ブロック単位の torch.compile / CUDA 性能オプション",
    },
    "feature.qwenImageVae2d": {
        "en": "2D Qwen-Image VAE",
        "zh": "2D Qwen-Image VAE",
        "ja": "2D Qwen-Image VAE",
    },
    "feature.showTimesteps": {
        "en": "timestep distribution preview",
        "zh": "时间步分布预览",
        "ja": "タイムステップ分布のプレビュー",
    },
    "feature.timestepOffset": {
        "en": "per-subfolder timestep offsets / offset preview",
        "zh": "子文件夹时间步偏移 / 偏移预览",
        "ja": "サブフォルダ別タイムステップオフセット / オフセットのプレビュー",
    },
    # --- file picker, services, pages --------------------------------------------------------
    "picker.unsupported": {
        "en": "Unsupported picker_type: {type}",
        "zh": "不支持的 picker_type：{type}",
        "ja": "未対応の picker_type：{type}",
    },
    "picker.cancelled": {
        "en": "Selection cancelled",
        "zh": "已取消选择",
        "ja": "選択がキャンセルされました",
    },
    "proxy.unavailable": {
        "en": "The service has not started yet or failed to start. On the first start this can take a while; "
              "reload the page later.",
        "zh": "请求的服务尚未启动或启动失败。若是第一次启动，可能需要等待一段时间后再刷新网页。",
        "ja": "サービスがまだ起動していないか、起動に失敗しました。初回起動時は時間がかかることがあるので、しばらくしてからページを再読み込みしてください。",
    },
    "page.frontendMissing.title": {
        "en": "The web UI is not built",
        "zh": "网页界面尚未构建",
        "ja": "Web UI がまだビルドされていません",
    },
    "page.frontendMissing.body": {
        "en": "<code>frontend/dist</code> is missing. Either install <a href=\"https://nodejs.org/\">Node.js</a> 20 "
              "or newer and restart the GUI (it builds the UI automatically), or build it yourself with "
              "<code>npm ci &amp;&amp; npm run build</code> in the <code>frontend</code> folder, or use the "
              "ready-to-run zip from the GitHub Releases page.",
        "zh": "缺少 <code>frontend/dist</code>。请安装 <a href=\"https://nodejs.org/\">Node.js</a> 20 或更新版本后重新启动 GUI"
              "（会自动构建），或在 <code>frontend</code> 目录运行 <code>npm ci &amp;&amp; npm run build</code>，"
              "或下载 GitHub Releases 中可直接运行的压缩包。",
        "ja": "<code>frontend/dist</code> がありません。<a href=\"https://nodejs.org/\">Node.js</a> 20 以降をインストールして GUI を"
              "再起動する（UI は自動でビルドされます）か、<code>frontend</code> フォルダで "
              "<code>npm ci &amp;&amp; npm run build</code> を実行するか、GitHub Releases からそのまま使える zip を"
              "ダウンロードしてください。",
    },
}
