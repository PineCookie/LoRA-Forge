"""Dataset folder layout shared by validation, the dataset stats endpoint and the generated
dataset configs, so they all agree on which folders and images count."""

import os

# image types sd-scripts loads (library/train_util.py IMAGE_EXTENSIONS, common ones)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_repeat_folder(name: str):
    """Split a `repeats_name` folder into `(num_repeats, class_tokens)`.

    Mirrors sd-scripts' `extract_dreambooth_params()`. Returns None for folders without a valid
    positive leading repeat count.
    """
    tokens = name.split("_")
    try:
        num_repeats = int(tokens[0])
    except ValueError:
        return None
    if num_repeats < 1:
        return None
    return num_repeats, "_".join(tokens[1:])


def subset_folders(root: str):
    """`(folder name, num_repeats, class_tokens)` for every `repeats_name` subfolder, sorted."""
    subsets = []
    for entry in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, entry)):
            continue
        parsed = parse_repeat_folder(entry)
        if parsed is not None:
            subsets.append((entry, *parsed))
    return subsets


def count_images(path: str, recursive: bool = True) -> int:
    if not recursive:
        return sum(1 for entry in os.scandir(path) if entry.is_file() and os.path.splitext(entry.name)[1].lower() in IMAGE_EXTENSIONS)
    return sum(
        1
        for _root, _dirs, files in os.walk(path)
        for name in files
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )
