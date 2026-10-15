"""The kohya-ss/musubi-tuner checkout the GUI drives for Krea 2 (see sd_scripts.py for sd-scripts).

musubi-tuner is run from its source tree (`src` on sys.path), not installed as a package: its
requirements pin exact versions that differ from this project's, and the Krea 2 scripts only need
what pyproject.toml already declares.
"""

from pathlib import Path
from typing import Optional

from lora_forge.launcher import base_dir_path

# last checkout this GUI was verified against, and its tag: a fresh install clones this tag
MUSUBI_TUNER_TESTED_REF = "f8a1b03794a49239a3539015075f5123d6c07d66"
MUSUBI_TUNER_TESTED_TAG = "v0.3.6"
MUSUBI_TUNER_SUBPATH = ("scripts", "musubi-tuner")


def musubi_tuner_dir(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else base_dir_path().joinpath(*MUSUBI_TUNER_SUBPATH)


def src_dir(root: Optional[Path] = None) -> Path:
    """The folder to put on sys.path (PYTHONPATH) so `musubi_tuner.*` imports."""
    return musubi_tuner_dir(root) / "src"


def script(name: str, root: Optional[Path] = None) -> Path:
    """Path of a musubi-tuner entry script, e.g. `krea2_train_network.py`."""
    return src_dir(root) / "musubi_tuner" / name
