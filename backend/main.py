"""
FastAPI app serving the lesion classifier.

Run from the repo root:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io

from backend import inference

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB

app = FastAPI(title="Kangri Cancer Screening API")

# hackathon demo only — do not ship this CORS config to a real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image upload, got content type '{file.content_type}'.",
        )

    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image is too large ({len(raw_bytes) / 1_000_000:.1f}MB) — max allowed is 10MB.",
        )

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force-decode now so a truncated/corrupt file fails here, not mid-inference
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not read this file as an image. Please upload a valid JPG or PNG.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")

    try:
        result = inference.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return result