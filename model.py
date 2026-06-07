"""
Model loading, preprocessing, and inference for the CIFAR-10 ResNet18 classifier.

Designed to *always* start cleanly:
- If a trained checkpoint exists at MODEL_PATH, load it.
- Otherwise fall back to a freshly-initialized ResNet18 head so the API still runs
  (useful for CI / Docker smoke tests). A warning is emitted.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

CIFAR_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/resnet18_cifar10.pth"))

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class CIFARClassifier:
    """Singleton-style wrapper around the ResNet18 CIFAR-10 model."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classes = CIFAR_CLASSES
        self.model = self._build_model()
        self.checkpoint_loaded = self._maybe_load_weights()
        self.model.to(self.device).eval()
        logger.info("Model ready on %s (checkpoint_loaded=%s)",
                    self.device, self.checkpoint_loaded)

    # --------------------------- internals ---------------------------
    def _build_model(self) -> nn.Module:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        return model

    def _maybe_load_weights(self) -> bool:
        if not self.model_path.exists():
            logger.warning(
                "No checkpoint found at %s — serving with an UNTRAINED head. "
                "Predictions will be random until you provide a trained .pth file.",
                self.model_path,
            )
            return False
        try:
            ckpt = torch.load(self.model_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state)
            if isinstance(ckpt, dict) and "classes" in ckpt:
                self.classes = list(ckpt["classes"])
            logger.info("Loaded checkpoint from %s", self.model_path)
            return True
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to load checkpoint: %s", exc)
            return False

    # --------------------------- public API ---------------------------
    def preprocess(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return _transform(img).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_bytes: bytes, topk: int = 3) -> List[Tuple[str, float]]:
        topk = max(1, min(topk, len(self.classes)))
        x = self.preprocess(image_bytes).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = torch.topk(probs, k=topk)
        return [
            (self.classes[int(idx)], float(prob))
            for prob, idx in zip(top.values.cpu(), top.indices.cpu())
        ]


# Singleton accessor -------------------------------------------------
_classifier: CIFARClassifier | None = None


def get_classifier() -> CIFARClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CIFARClassifier()
    return _classifier
