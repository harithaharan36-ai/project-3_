# Model Deployment — FastAPI + Docker

A production-style REST API that wraps the **CIFAR-10 ResNet18 image classifier** (from the deep-learning-tasks repo) using **FastAPI**, containerized with **Docker**.

> ✅ Works even without a real trained model — it ships with a fallback that auto-builds a fresh ResNet18 if no checkpoint is found, so the API and Docker image always start cleanly.

---

## 🎯 What's inside

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Liveness check (model loaded? device?) |
| `/classes` | GET | List of 10 CIFAR-10 class names |
| `/predict` | POST | Upload an image file → top-k predictions |
| `/predict_url` | POST | Send a JSON `{"url": "..."}` → top-k predictions |
| `/docs` | GET | Auto-generated Swagger UI |

---

## 📁 Structure

```
model-deployment-api/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app + endpoints
│   ├── model.py           # Model loading + preprocessing + inference
│   └── schemas.py         # Pydantic request/response models
├── tests/
│   └── test_api.py        # pytest smoke tests
├── examples/
│   ├── sample_request.sh  # curl examples
│   └── sample_request.py  # Python requests client
├── docs/
│   └── DemoScreenshot.md  # how to capture screenshots
└── models/
    └── README.md          # where to put your .pth checkpoint
```

---

## ⚙️ Run Locally (without Docker)

```bash
# 1. Create venv & install
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) Drop your trained checkpoint at:
#    models/resnet18_cifar10.pth
#    -> if missing, the API still starts with an untrained ResNet18 head.

# 3. Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

---

## 🐳 Run with Docker

```bash
# Build
docker build -t cifar10-api:latest .

# Run
docker run -p 8000:8000 cifar10-api:latest

# With your own model file mounted in
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    cifar10-api:latest
```

Then visit **http://localhost:8000/docs**.

---

## 📡 Example Requests

### 1. Health check
```bash
curl http://localhost:8000/health
```
Response:
```json
{"status": "ok", "model_loaded": true, "device": "cpu", "num_classes": 10}
```

### 2. Predict from file upload
```bash
curl -X POST "http://localhost:8000/predict?topk=3" \
     -F "file=@examples/cat.jpg"
```
Response:
```json
{
  "filename": "cat.jpg",
  "predictions": [
    {"class": "cat", "confidence": 0.8742},
    {"class": "dog", "confidence": 0.0613},
    {"class": "deer", "confidence": 0.0287}
  ]
}
```

### 3. Predict from URL
```bash
curl -X POST "http://localhost:8000/predict_url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/cat.jpg", "topk": 3}'
```

### 4. Python client
```bash
python examples/sample_request.py --image examples/cat.jpg
```

---

## 🧪 Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 📸 Demo Screenshot

Once running, capture screenshots of:
1. `http://localhost:8000/docs` (Swagger UI)
2. A successful curl `/predict` response in your terminal
3. `docker ps` showing the running container

Save them in `docs/` and reference them in your submission.

---

## 🔧 Reproducibility

- Python 3.10
- Versions pinned in `requirements.txt`
- Dockerfile uses `python:3.10-slim` for a small, deterministic image
