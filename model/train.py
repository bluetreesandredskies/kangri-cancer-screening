# ---------------------------------------------------------------------------
# If you get a CUDA out of memory error:
#   1) lower batch_size in config.yaml to 8
#   2) lower image_size to 192
#   3) as a last resort, run model/train_colab.ipynb on Google Colab instead.
# ---------------------------------------------------------------------------

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# make sure "data" package (repo root) is importable regardless of where
# this script is invoked from
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.augment import eval_transforms, train_transforms  # noqa: E402
from model.model_def import build_model  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def check_against_class_names_file(data_dir: Path, discovered_classes):
    """data/processed/class_names.json is the source of truth for label
    order (written by prepare_dataset.py). ImageFolder derives the same
    alphabetical order on its own, but we cross-check here so a stale or
    hand-edited class_names.json fails loudly instead of causing a silent
    label mismatch down the line."""
    class_names_path = data_dir / "class_names.json"
    if not class_names_path.exists():
        print(f"[warn] {class_names_path} not found — proceeding with ImageFolder's own class order.")
        return

    with open(class_names_path, "r") as f:
        expected = json.load(f)

    if expected != discovered_classes:
        sys.exit(
            f"class_names.json disagrees with the folders on disk!\n"
            f"  class_names.json: {expected}\n"
            f"  ImageFolder sees: {discovered_classes}\n"
            f"Re-run data/prepare_dataset.py to regenerate class_names.json."
        )


def build_dataloaders(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_ds = ImageFolder(train_dir, transform=train_transforms)
    val_ds = ImageFolder(val_dir, transform=eval_transforms)

    # sanity check: class ordering must match between splits or metrics
    # will silently be wrong
    if train_ds.classes != val_ds.classes:
        sys.exit(
            f"Class mismatch between train and val folders!\n"
            f"train: {train_ds.classes}\nval: {val_ds.classes}"
        )

    check_against_class_names_file(data_dir, train_ds.classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, train_ds, val_ds


def compute_class_weights(train_ds: ImageFolder, device):
    """Inverse-frequency class weights. This matters a lot with the current
    dataset since 'healthy' only has ~70-100 images versus several hundred
    for the ISIC lesion classes — without this weighting the model would
    have little incentive to learn the minority class well."""
    counts = np.bincount([label for _, label in train_ds.samples], minlength=len(train_ds.classes))
    counts = counts.astype(np.float32)
    counts[counts == 0] = 1  # avoid divide-by-zero if a class is somehow empty

    inv_freq = 1.0 / counts
    weights = inv_freq * (len(counts) / inv_freq.sum())  # normalize so weights average to ~1
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch_train(model, loader, optimizer, criterion, scaler, device, use_amp, accum_steps):
    model.train()
    running_loss = 0.0
    n_batches = 0

    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accum_steps  # normalize so accumulated grad matches a full batch

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps  # undo the normalization for logging
        n_batches += 1

    # flush any leftover accumulated gradient at the end of the epoch
    if (n_batches % accum_steps) != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def run_epoch_eval(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        n_batches += 1

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(n_batches, 1)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1, all_labels, all_preds


def print_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    print("\nConfusion matrix (rows = true label, cols = predicted label):")
    header = "                    " + " ".join(f"{c[:10]:>10}" for c in class_names)
    print(header)
    for class_name, row in zip(class_names, cm):
        print(f"{class_name[:18]:>18}  " + " ".join(f"{v:>10}" for v in row))


def main():
    parser = argparse.ArgumentParser(description="Train the lesion classifier.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Root folder containing train/ and val/ subfolders.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available, falling back to cpu.")
        device = torch.device("cpu")

    config = load_config()
    data_dir = Path(args.data_dir)

    print(f"Using device: {device}")
    print(f"Config: {config}")

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(data_dir, config["batch_size"])
    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(num_classes=len(class_names)).to(device)

    class_weights = compute_class_weights(train_ds, device)
    print(f"Class weights: {dict(zip(class_names, class_weights.cpu().numpy().round(3)))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    use_amp = bool(config["mixed_precision"]) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_val_labels, best_val_preds = None, None

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        train_loss = run_epoch_train(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            use_amp,
            config["gradient_accumulation_steps"],
        )

        val_loss, val_acc, val_macro_f1, val_labels, val_preds = run_epoch_eval(
            model, val_loader, criterion, device, use_amp
        )

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:2d}/{config['epochs']} "
            f"| train_loss {train_loss:.4f} "
            f"| val_loss {val_loss:.4f} "
            f"| val_acc {val_acc:.4f} "
            f"| val_macro_f1 {val_macro_f1:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
            best_val_labels, best_val_preds = val_labels, val_preds
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,  # same order as train_ds.classes / class_names.json
                    "val_macro_f1": best_macro_f1,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
            print(f"  -> new best model saved (macro_f1 {best_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"No improvement for {config['patience']} epochs, stopping early.")
                break

    print(f"\nBest val macro-F1: {best_macro_f1:.4f} (saved to {best_ckpt_path})")

    if best_val_labels is not None:
        print_confusion_matrix(best_val_labels, best_val_preds, class_names)


if __name__ == "__main__":
    main()
