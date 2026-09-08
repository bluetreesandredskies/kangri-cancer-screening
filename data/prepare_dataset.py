"""
Splits the raw downloads into a torchvision ImageFolder-style layout.

Assumes images have already been placed into:
    data/raw/<class_name>/*.jpg   (+ optional .json metadata sidecars)

Class names are NOT hardcoded — this script just lists whatever
subdirectories exist under data/raw/ and sorts them alphabetically. That
alphabetical order becomes the label order for the whole project, and gets
written out to data/processed/class_names.json so training/eval/backend code
can all agree on it without re-deriving it. Add or remove a class folder
under data/raw/ and this script picks it up automatically — no edits needed.

Run from the repo root, e.g.:
    python data/prepare_dataset.py
    python data/prepare_dataset.py --max-per-class 50   # quick smoke test
"""

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMAGE_SIZE = (224, 224)
SPLIT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def discover_class_names(raw_dir: Path):
    """Any subdirectory of data/raw/ is treated as a class. Sorted
    alphabetically so the label order is deterministic and reproducible."""
    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    return sorted(p.name for p in class_dirs)


def find_images_for_class(class_dir: Path):
    """Grab every jpg/jpeg/png in a class folder, once each.

    Uses a manual case-insensitive suffix check + dedupe-by-resolved-path,
    instead of combining separate lowercase and uppercase glob patterns
    (e.g. "*.jpg" AND "*.JPG"). On case-insensitive filesystems — which
    includes Windows, and default macOS — those two patterns match the exact
    same files, so combining them would silently return every image twice.
    That double-counting doesn't just waste disk/CPU: since the train/val/test
    split below is randomized, the two duplicate entries for the same image
    could land in *different* splits, meaning the same photo could end up in
    both the training set and the test set — invisibly leaking training data
    into evaluation and inflating the reported accuracy/F1 numbers. This
    version is safe on both Windows and Colab's Linux, so no platform-specific
    handling is needed.

    Non-image files that sit alongside the images in a class folder (e.g. an
    isic-cli "attribution" text file, or a "licenses" subfolder) are ignored
    automatically, since they don't have a .jpg/.jpeg/.png suffix and/or
    aren't files at all.
    """
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    seen = set()
    files = []
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in valid_suffixes:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(p)
    return sorted(files)


def resize_and_save(src_path: Path, dst_path: Path):
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, quality=95)


def split_paths(paths, seed=SPLIT_SEED):
    """70/15/15 split. sklearn's train_test_split only splits two ways at a
    time, so we do it in two steps: first carve off train, then split the
    remainder evenly into val/test."""
    train_paths, remainder = train_test_split(
        paths, train_size=TRAIN_FRAC, random_state=seed, shuffle=True
    )
    # remainder is 30% of the data, we want it split 15/15 -> 50/50 of what's left
    val_paths, test_paths = train_test_split(
        remainder, train_size=0.5, random_state=seed, shuffle=True
    )
    return train_paths, val_paths, test_paths


def main():
    parser = argparse.ArgumentParser(description="Prepare raw images into train/val/test folders.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap the number of images used per class (useful for a quick test run).",
    )
    args = parser.parse_args()

    if not RAW_DIR.exists():
        sys.exit(f"Could not find {RAW_DIR}/ — did you run the download step first?")

    class_names = discover_class_names(RAW_DIR)
    if not class_names:
        sys.exit(f"No class subfolders found under {RAW_DIR}/ — nothing to do.")

    print(f"Discovered {len(class_names)} class(es) under {RAW_DIR}: {class_names}")

    summary_rows = []

    for class_name in class_names:
        class_dir = RAW_DIR / class_name
        image_paths = find_images_for_class(class_dir)

        if not image_paths:
            print(f"[warn] no images found in {class_dir}, skipping.")
            continue

        if args.max_per_class is not None:
            # shuffle before truncating so we don't just take an alphabetically-biased subset
            rng = random.Random(SPLIT_SEED)
            rng.shuffle(image_paths)
            image_paths = image_paths[: args.max_per_class]

        train_paths, val_paths, test_paths = split_paths(image_paths)

        splits = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        }

        for split_name, paths in splits.items():
            desc = f"{class_name} -> {split_name}"
            for src in tqdm(paths, desc=desc, unit="img"):
                dst = PROCESSED_DIR / split_name / class_name / src.name
                try:
                    resize_and_save(src, dst)
                except Exception as e:
                    print(f"[error] failed on {src}: {e}")

        summary_rows.append(
            {
                "class": class_name,
                "train": len(train_paths),
                "val": len(val_paths),
                "test": len(test_paths),
                "total": len(image_paths),
            }
        )

    write_class_names_file(summary_rows)
    print_summary(summary_rows)


def write_class_names_file(rows):
    """Writes the alphabetical class order out so later scripts (training,
    eval, the backend) can all agree on label indices without having to
    re-derive them from folder listings themselves."""
    if not rows:
        return

    ordered_names = [row["class"] for row in rows]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "class_names.json"
    with open(out_path, "w") as f:
        json.dump(ordered_names, f, indent=2)

    print(f"\nWrote class order to {out_path}")


def print_summary(rows):
    if not rows:
        print("Nothing was processed — check that data/raw/<class_name>/ folders have images in them.")
        return

    print("\nClass balance summary")
    print("-" * 60)
    header = f"{'class':30} {'train':>8} {'val':>8} {'test':>8} {'total':>8}"
    print(header)
    print("-" * 60)

    totals = {"train": 0, "val": 0, "test": 0, "total": 0}
    for row in rows:
        print(f"{row['class']:30} {row['train']:>8} {row['val']:>8} {row['test']:>8} {row['total']:>8}")
        for key in totals:
            totals[key] += row[key]

    print("-" * 60)
    print(f"{'TOTAL':30} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8} {totals['total']:>8}")


if __name__ == "__main__":
    main()
