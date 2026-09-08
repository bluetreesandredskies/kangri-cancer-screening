"""
Loads the best checkpoint from training and evaluates it on the held-out
test set. Writes a classification report + confusion matrix both to stdout
and to model/checkpoints/eval_report.txt.

Class names come from the checkpoint itself (saved by train.py as
checkpoint["class_names"]), not hardcoded or re-derived here — this avoids
label-order bugs if the dataset's class list ever changes.
"""

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.augment import eval_transforms  # noqa: E402
from model.model_def import build_model  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
REPORT_PATH = Path(__file__).resolve().parent / "checkpoints" / "eval_report.txt"
TEST_DIR = REPO_ROOT / "data" / "processed" / "test"
CLASS_NAMES_JSON = REPO_ROOT / "data" / "processed" / "class_names.json"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        sys.exit(f"No checkpoint found at {CHECKPOINT_PATH} — run model/train.py first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if "class_names" not in checkpoint or "model_state_dict" not in checkpoint:
        sys.exit(
            "Checkpoint is missing 'class_names' or 'model_state_dict' — "
            "this checkpoint looks like it was saved by an older version of train.py. "
            "Retrain to get a checkpoint in the expected format."
        )

    class_names = checkpoint["class_names"]

    # cross-check against class_names.json if it's around, same reasoning as in train.py:
    # fail loudly on a mismatch rather than silently mislabeling the report
    if CLASS_NAMES_JSON.exists():
        with open(CLASS_NAMES_JSON, "r") as f:
            expected = json.load(f)
        if expected != class_names:
            sys.exit(
                f"Checkpoint class_names disagree with {CLASS_NAMES_JSON}!\n"
                f"  checkpoint:       {class_names}\n"
                f"  class_names.json: {expected}\n"
                f"This checkpoint may be from a different dataset version — retrain before evaluating."
            )

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_ds = ImageFolder(TEST_DIR, transform=eval_transforms)
    if test_ds.classes != class_names:
        sys.exit(
            f"Test set classes {test_ds.classes} don't match checkpoint classes {class_names}. "
            f"Did the data folders change since this checkpoint was trained?"
        )

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    cm_lines = ["Confusion matrix (rows = true label, cols = predicted label):"]
    header = "                    " + " ".join(f"{c[:10]:>10}" for c in class_names)
    cm_lines.append(header)
    for class_name, row in zip(class_names, cm):
        cm_lines.append(f"{class_name[:18]:>18}  " + " ".join(f"{v:>10}" for v in row))
    cm_text = "\n".join(cm_lines)

    full_output = (
        f"Evaluation on test set ({len(test_ds)} images)\n"
        f"Checkpoint: {CHECKPOINT_PATH} (from epoch {checkpoint.get('epoch', '?')}, "
        f"val macro-F1 {checkpoint.get('val_macro_f1', float('nan')):.4f})\n"
        f"Class order: {class_names}\n\n"
        f"Classification report:\n{report}\n\n"
        f"{cm_text}\n"
    )

    print(full_output)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(full_output)

    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()