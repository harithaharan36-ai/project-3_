"""
FastAPI entry point for the CIFAR-10 image-classification service.

Endpoints
---------
GET  /             -> service info
GET  /health       -> readiness probe
GET  /classes      -> list of class names
POST /predict      -> multipart file upload (image) -> top-k predictions
POST /predict_url  -> JSON {"url": "..."} -> top-k predictions
GET  /docs         -> Swagger UI (auto)
"""

from __future__ import annotations

import logging
from typing import List

import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.model import get_classifier
from app.schemas import HealthResponse, Prediction, PredictResponse, URLRequest

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(
    title="CIFAR-10 Image Classifier API",
    description=(
        "REST API wrapping a ResNet18 model trained on CIFAR-10. "
        "Send an image and receive the top-k class predictions."
    ),
    version="1.0.0",
)


# --------------------------- startup ---------------------------
@app.on_event("startup")
def _warm_up() -> None:
    clf = get_classifier()  # forces model load
    logger.info("Service ready (device=%s, checkpoint_loaded=%s)",
                clf.device, clf.checkpoint_loaded)


# --------------------------- routes ---------------------------
@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "CIFAR-10 Image Classifier",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/health", "/classes", "/predict", "/predict_url"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    clf = get_classifier()
    return HealthResponse(
        status="ok",
        model_loaded=clf.checkpoint_loaded,
        device=str(clf.device),
        num_classes=len(clf.classes),
    )


@app.get("/classes", tags=["meta"])
def classes() -> dict:
    return {"classes": get_classifier().classes}


_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/bmp"}


def _to_predictions(pairs) -> List[Prediction]:
    return [Prediction(**{"class": c, "confidence": round(p, 4)}) for c, p in pairs]


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Image file (jpeg/png/webp/bmp)"),
    topk: int = Query(3, ge=1, le=10),
) -> PredictResponse:
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Allowed: {sorted(_ALLOWED_TYPES)}",
        )
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        preds = get_classifier().predict(image_bytes, topk=topk)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictResponse(filename=file.filename, predictions=_to_predictions(preds))


@app.post("/predict_url", response_model=PredictResponse, tags=["inference"])
def predict_url(req: URLRequest) -> PredictResponse:
    try:
        resp = requests.get(str(req.url), timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}")

    try:
        preds = get_classifier().predict(resp.content, topk=req.topk)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictResponse(filename=str(req.url), predictions=_to_predictions(preds))


# --------------------------- error handler ---------------------------
@app.exception_handler(Exception)
async def _unhandled(_, exc):  # pragma: no cover
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
