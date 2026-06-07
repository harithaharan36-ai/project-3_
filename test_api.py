"""
Smoke tests for the CIFAR-10 API.

Run:
    pip install pytest httpx
    pytest tests/ -v
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _png_bytes(color=(255, 0, 0), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["num_classes"] == 10


def test_classes():
    r = client.get("/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 10


def test_predict_file():
    img = _png_bytes()
    r = client.post(
        "/predict?topk=3",
        files={"file": ("test.png", img, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0.0 <= p["confidence"] <= 1.0


def test_predict_rejects_bad_type():
    r = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415
