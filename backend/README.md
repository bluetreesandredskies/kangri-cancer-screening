# Backend — Lesion Screening API

## Running locally

Run this from the **repo root**, not from inside `backend/` — the app uses
`backend.main:app` as the module path and does an internal import of
`model/model_def.py`, both of which assume the repo root is the working
directory.

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The model checkpoint is loaded once at startup from
`model/checkpoints/best_model.pt`. If that file doesn't exist yet, run
`model/train.py` first.

## Testing the /predict endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/sample_lesion_photo.jpg"
```

You should get back JSON like:

```json
{
  "predicted_class": "actinic_keratosis",
  "confidence": 62.3,
  "risk_level": "moderate",
  "recommendation": "Please see a dermatologist for confirmation",
  "disclaimer": "This is a screening aid, not a medical diagnosis. Please consult a qualified dermatologist for confirmation.",
  "safety_escalated": false,
  "escalation_note": null
}
```

## About the safety-net escalation rule

The model's risk mapping is normally based on whichever class it ranks
highest (top-1). During test-set evaluation, though, we found cases where
`squamous_cell_carcinoma` or `actinic_keratosis` — the two classes that
matter most to catch — had a real, meaningful predicted probability without
actually winning top-1. Those cases were slipping through with a falsely
reassuring "low" or "none" risk level, which is exactly the kind of mistake
a screening tool shouldn't make.

To guard against that, `inference.py` looks at the model's *full* probability
distribution, not just the top prediction. If the probability assigned to
`squamous_cell_carcinoma` or `actinic_keratosis` crosses a threshold
(`SAFETY_THRESHOLD = 0.15`, easy to tune in `inference.py`), the risk level
gets bumped up (to "high" or "moderate" respectively) even though a
different, lower-risk class won top-1. When this happens, the response sets
`"safety_escalated": true` and fills in `"escalation_note"` with a
human-readable explanation, so the escalation is visible and explainable
rather than a silent override. This is a deliberate design choice — biasing
the tool toward over-caution rather than a missed high-risk case — not a bug.