"""
Inference wrapper around the trained lesion classifier.

Loads the checkpoint once at import time and exposes predict(image) for
main.py to call. Class names are read from the checkpoint itself, never
hardcoded here, since their order depends on alphabetical folder sorting
done at data-prep time and must always match what the model was actually
trained with.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# repo root is the parent of backend/, so this works when uvicorn is run
# from the repo root (as backend/README.md instructs)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.model_def import build_model  # noqa: E402

CHECKPOINT_PATH = REPO_ROOT / "model" / "checkpoints" / "best_model.pt"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Tunable safety-net threshold — see the comment above apply_safety_net()
# for why this exists.
SAFETY_THRESHOLD = 0.15

DISCLAIMER = (
    "This is a screening aid, not a medical diagnosis. Please consult a "
    "qualified dermatologist for confirmation."
)

BASE_RISK_BY_CLASS = {
    "healthy": "none",
    "nevus": "low",
    "seborrheic_keratosis": "low",
    "actinic_keratosis": "moderate",
    "squamous_cell_carcinoma": "high",
}

RECOMMENDATION_BY_RISK = {
    "none": "No lesion detected — continue routine skin self-checks.",
    "low": "No immediate concern detected, continue routine skin checks",
    "moderate": "Please see a dermatologist for confirmation",
    "high": "Please see a dermatologist for confirmation",
}

_inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def _load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH} — train the model first "
            f"(model/train.py) before starting the backend."
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if "class_names" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint is missing 'class_names' or 'model_state_dict' — "
            "this doesn't look like a checkpoint produced by model/train.py."
        )

    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


# loaded once at import time, reused across requests
_model, _class_names, _device = _load_model()


def apply_safety_net(class_names, probs, base_risk_level, predicted_class):
    """
    During test-set evaluation we found cases where squamous_cell_carcinoma
    or actinic_keratosis had meaningful predicted probability but weren't
    the top-1 class, so the base risk mapping (which only looks at top-1)
    was reporting a reassuring "low"/"none" risk on lesions that actually
    warranted more caution. This safety net looks at the full probability
    distribution — not just the top prediction — and escalates the risk
    level when those two higher-severity classes cross a threshold, even if
    something else won top-1.
    """
    class_to_prob = dict(zip(class_names, probs))
    p_scc = class_to_prob.get("squamous_cell_carcinoma", 0.0)
    p_ak = class_to_prob.get("actinic_keratosis", 0.0)

    risk_level = base_risk_level
    safety_escalated = False
    escalation_note = None

    if p_scc >= SAFETY_THRESHOLD and base_risk_level != "high":
        risk_level = "high"
        safety_escalated = True
        escalation_note = (
            f"Note: while the top prediction was {predicted_class}, elevated "
            f"squamous cell carcinoma indicators ({p_scc:.0%}) were also "
            f"detected — professional evaluation is recommended out of caution."
        )
    elif p_ak >= SAFETY_THRESHOLD and base_risk_level in ("low", "none"):
        risk_level = "moderate"
        safety_escalated = True
        escalation_note = (
            f"Note: while the top prediction was {predicted_class}, elevated "
            f"actinic keratosis indicators ({p_ak:.0%}) were also detected — "
            f"professional evaluation is recommended out of caution."
        )

    return risk_level, safety_escalated, escalation_note


def predict(image: Image.Image) -> dict:
    image = image.convert("RGB")
    tensor = _inference_transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    predicted_class = _class_names[top_idx]
    confidence = round(float(probs[top_idx]) * 100, 1)

    base_risk_level = BASE_RISK_BY_CLASS.get(predicted_class, "moderate")

    risk_level, safety_escalated, escalation_note = apply_safety_net(
        _class_names, probs, base_risk_level, predicted_class
    )

    # recommendation always reflects the FINAL risk level, post-escalation
    recommendation = RECOMMENDATION_BY_RISK[risk_level]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "disclaimer": DISCLAIMER,
        "safety_escalated": safety_escalated,
        "escalation_note": escalation_note,
    }