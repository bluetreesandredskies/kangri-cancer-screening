"""
Splits the raw ISIC downloads into a torchvision ImageFolder-style layout.

Assumes isic-cli has already dumped images + json sidecars into:
    data/raw/<class_name>/*.jpg

Run from the repo root, e.g.:
    python data/prepare_dataset.py
    python data/prepare_dataset.py --max-per-class 50   # quick smoke test
"""

import argparse
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

CLASS_NAMES = [
    "squamous_cell_carcinoma",
    "actinic_keratosis",
    "nevus",
    "seborrheic_keratosis",
]

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def find_images_for_class(class_dir: Path):
    """Grab every jpg/jpeg/png in a class folder. We only care about the
    image files here — the matching .json metadata from isic-cli is ignored
    for this stage, it's not needed to build the ImageFolder split."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for ext in exts:
        files.extend(class_dir.glob(ext))
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
    parser = argparse.ArgumentParser(description="Prepare ISIC images into train/val/test folders.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap the number of images used per class (useful for a quick test run).",
    )
    args = parser.parse_args()

    if not RAW_DIR.exists():
        sys.exit(f"Could not find {RAW_DIR}/ — did you run the isic-cli download step first?")

    summary_rows = []

    for class_name in CLASS_NAMES:
        class_dir = RAW_DIR / class_name
        if not class_dir.exists():
            print(f"[warn] {class_dir} does not exist, skipping this class.")
            continue

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

    print_summary(summary_rows)


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
