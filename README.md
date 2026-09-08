# Paakzir

**Team Paakzir** · `bluetreesandredskies/kangri-cancer-screening`

Paakzir is an AI-assisted skin lesion screening web app. A user uploads or
photographs a skin lesion, and a PyTorch image classifier returns an instant,
risk-tiered assessment — a starting point for deciding whether to seek
professional dermatological evaluation, not a replacement for one.

## Problem statement

Kangri cancer is a form of skin cancer (typically squamous cell carcinoma)
linked to chronic thermal injury from the *kangri*, a traditional firepot used
for warmth during Kashmir's harsh winters. Early lesions often resemble
ordinary burns, calluses, or benign skin changes, and dermatology access in
the affected region is limited — so cases are frequently caught only once the
disease has progressed. Paakzir is a step toward closing that detection gap
with an accessible, low-cost screening tool that anyone with a phone camera
can use.

## Architecture

```
ISIC public dataset  →  EfficientNet-B0 classifier (PyTorch / timm)
                              ↓
                     FastAPI backend  (/predict, /health)
                              ↓
                     React (Vite) frontend
```

- **Data** — Lesion images pulled from the [ISIC Archive](https://www.isic-archive.com/)
  across 5 classes (`actinic_keratosis`, `healthy`, `nevus`,
  `seborrheic_keratosis`, `squamous_cell_carcinoma`), cleaned and split into
  train/val/test sets.
- **Model** — An EfficientNet-B0 (via `timm`), fine-tuned on the processed
  dataset, saved as a checkpoint that bundles both weights and the class name
  order.
- **Backend** — A FastAPI service that loads the checkpoint once at startup,
  runs inference on uploaded images, and applies a confidence-margin safety
  net that escalates the reported risk level when malignant/precancerous
  classes show meaningful probability even outside the top-1 prediction.
- **Frontend** — A Vite + React app that uploads a photo to the backend and
  displays the risk badge, confidence, recommendation, and any safety
  escalation notice.

## Setup

Run these in order from the repo root.

### 1. `data/` — dataset prep

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python data/prepare_dataset.py
python data/augment.py
```

Populates `data/processed/` with train/val/test folders for all 5 classes.

### 2. `model/` — training

```bash
python model/train.py
python model/evaluate.py
```

Produces `model/checkpoints/best_model.pt` and `model/checkpoints/eval_report.txt`.

### 3. `backend/` — FastAPI service

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Run from the repo root (not from inside `backend/`), so the relative import
to `model/model_def.py` resolves correctly. Verify with:

```bash
curl http://localhost:8000/health
```

### 4. `frontend/` — React web app

```bash
cd frontend
npm install
cp .env.example .env      # then edit VITE_API_URL if the backend isn't on localhost:8000
npm run dev
```

Open the printed local URL (typically `http://localhost:5173`).

## Tech stack

- **ML**: Python, PyTorch, `timm` (EfficientNet-B0), scikit-learn, Pillow
- **Backend**: FastAPI, Uvicorn, python-multipart
- **Frontend**: React, Vite
- **DevOps**: Docker, Docker Compose, GitHub Actions
- **Deployment**: Render (backend), Vercel (frontend)

## Current Scope & Limitations

Paakzir's classifier is trained entirely on **public ISIC data** and
recognizes general precancerous/malignant lesion patterns (actinic
keratosis, squamous cell carcinoma) alongside benign classes (nevus,
seborrheic keratosis) and a "no lesion" class. **It does not currently
target the kangri-specific erythema ab igne precursor pattern** — no public
dataset labels that pattern, since it's tied to a regional exposure not
represented in ISIC's contributor base.

This project is a **pilot**: a working, validated screening pipeline built on
the best publicly available data, intended as the foundation for a
kangri-cancer-specific model once clinical partnership data (e.g. from a
regional hospital such as SKIMS) becomes available. Results should be read as
a general skin-lesion risk screen, not a kangri-cancer-specific diagnosis, and
never as a substitute for a qualified dermatologist.
