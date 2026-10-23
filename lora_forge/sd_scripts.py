"""Guard rails for the sd-scripts checkout the GUI drives.

The GUI hands its options to sd-scripts as TOML config keys, and sd-scripts ignores unknown
keys silently: an outdated checkout accepts an option and simply does nothing. These helpers
map GUI options to the sd-scripts feature they need and report which ones the installed
checkout cannot provide.

Each feature records the upstream commit that introduced it (used for a precise
``git merge-base --is-ancestor`` check) plus a cheap file marker, used as a fallback when git
cannot answer -- for example a checkout whose ``.git`` directory was removed.
"""

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Set

from lora_forge.i18n import join_list, t
from lora_forge.launcher import base_dir_path
from lora_forge.log import log


@dataclass(frozen=True)
class Feature:
    label: str  # i18n message key
    min_commit: str
    marker_path: str
    marker_text: Optional[str] = None


# upstream commits that introduced the options the GUI exposes
V0_11_1 = "6721028c79ee85a78b3a06dfd8954dae310a1cce"
TIMESTEP_OFFSET = "5bf1dd510e49efc9304f140773a8465d5b0fd299"
BLOCK_SWAP_H2D = "143786ab5171803a3605ced14e82f8354f707947"  # v0.12.2

FEATURES: Dict[str, Feature] = {
    "compile": Feature("feature.compile", V0_11_1, "library/compile_utils.py"),
    "qwen_image_vae_2d": Feature("feature.qwenImageVae2d", V0_11_1, "library/qwen_image_autoencoder_kl_2d.py"),
    "show_timesteps": Feature("feature.showTimesteps", V0_11_1, "library/args.py", '"--show_timesteps"'),
    "timestep_offset": Feature(
        "feature.timestepOffset", TIMESTEP_OFFSET, "library/flux_train_utils.py", "timestep_sampling_offset"
    ),
    "block_swap_h2d": Feature("feature.blockSwapH2d", BLOCK_SWAP_H2D, "library/custom_offloading_utils.py", "h2d_only"),
}

# last checkout this GUI was verified against, and its tag: a fresh install clones this tag, so a
# release keeps working when sd-scripts' main moves on
SD_SCRIPTS_TESTED_REF = "11d8cedd0c67b93a10f8b2d75f43554f6f12c794"
SD_SCRIPTS_TESTED_TAG = "v0.12.2"
SD_SCRIPTS_SUBPATH = ("scripts", "sd-scripts")

# config key -> feature the key needs. Keys that are always sent with their default value
# (compile_backend, compile_mode, compile_cache_size_limit, ...) are deliberately absent:
# they only take effect once `compile` itself is enabled.
CONFIG_KEY_FEATURES = {
    "compile": "compile",
    "cuda_allow_tf32": "compile",
    "cuda_cudnn_benchmark": "compile",
    "qwen_image_vae_2d": "qwen_image_vae_2d",
    "show_timesteps": "show_timesteps",
    "show_timesteps_offset": "timestep_offset",
    "block_swap_h2d_only": "block_swap_h2d",
}

_SUPPORT_CACHE: Dict[tuple, bool] = {}
_HEAD_CACHE: Dict[tuple, Optional[str]] = {}


def sd_scripts_dir(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else base_dir_path().joinpath(*SD_SCRIPTS_SUBPATH)


def _checkout_stamp(directory: Path) -> float:
    """Cheap change marker for a checkout, so the HEAD lookup can be cached.

    ``.git/HEAD`` is rewritten on every checkout/branch switch; fall back to the ``.git``
    entry (or the directory itself) for worktrees where ``.git`` is a file.
    """
    for candidate in (directory / ".git" / "HEAD", directory / ".git", directory):
        try:
            return candidate.stat().st_mtime
        except OSError:
            continue
    return 0.0


def requested_features(config: Mapping, has_offsets: bool = False) -> Set[str]:
    """The sd-scripts features a submitted config actually relies on."""
    features = {feature for key, feature in CONFIG_KEY_FEATURES.items() if config.get(key)}
    if has_offsets:
        features.add("timestep_offset")
    return features


def head_commit(root: Optional[Path] = None) -> Optional[str]:
    """HEAD of the checkout, or None when git cannot tell. Cached until the checkout changes."""
    directory = sd_scripts_dir(root)
    cache_key = (str(directory), _checkout_stamp(directory))
    if cache_key in _HEAD_CACHE:
        return _HEAD_CACHE[cache_key]

    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug(f"cannot read sd-scripts HEAD: {e}")
        return None
    if result.returncode != 0:
        head = None
    else:
        head = result.stdout.strip() or None
    _HEAD_CACHE[cache_key] = head
    return head


def _introduced_before(feature: Feature, root: Optional[Path], head: Optional[str]) -> Optional[bool]:
    if head is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(sd_scripts_dir(root)), "merge-base", "--is-ancestor", feature.min_commit, head],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    return None  # 128: unknown commit or not a repository


def _marker_present(feature: Feature, root: Optional[Path]) -> Optional[bool]:
    path = sd_scripts_dir(root) / feature.marker_path
    try:
        if feature.marker_text is None:
            return path.is_file()
        return feature.marker_text in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def feature_supported(key: str, root: Optional[Path] = None, head: Optional[str] = None) -> bool:
    """Whether the checkout can apply `key`. Undeterminable situations count as supported."""
    feature = FEATURES[key]
    if head is None:
        head = head_commit(root)

    cache_key = (str(sd_scripts_dir(root)), head or "", key)
    if cache_key in _SUPPORT_CACHE:
        return _SUPPORT_CACHE[cache_key]

    supported = _introduced_before(feature, root, head)
    if supported is None:
        supported = _marker_present(feature, root)
    result = True if supported is None else supported
    _SUPPORT_CACHE[cache_key] = result
    return result


def missing_features(features: Iterable[str], root: Optional[Path] = None) -> List[str]:
    """Requested features the installed checkout cannot provide (sorted, deduplicated)."""
    head = head_commit(root)
    return [key for key in sorted(set(features)) if not feature_supported(key, root, head)]


def _feature_labels(keys: Iterable[str]) -> str:
    return join_list(t(FEATURES[key].label) for key in keys)


def missing_features_message(missing: List[str], root: Optional[Path] = None) -> str:
    ref = (head_commit(root) or "unknown")[:12]
    return t("sdScripts.missing", ref=ref, features=_feature_labels(missing))


def compatibility(root: Optional[Path] = None) -> tuple[str, str]:
    """State of the checkout for /api/runtime: `tested`, `compatible`, `unknown` (no git information)
    or `outdated`, with the message to show for `outdated`."""
    missing = missing_features(FEATURES, root)
    if missing:
        return "outdated", missing_features_message(missing, root)
    head = head_commit(root)
    if head == SD_SCRIPTS_TESTED_REF:
        return "tested", ""
    if head is None:
        return "unknown", ""
    return "compatible", ""
