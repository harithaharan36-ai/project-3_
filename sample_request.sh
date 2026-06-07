#!/usr/bin/env bash
# -------------------------------------------------------------
# Example curl requests for the CIFAR-10 API.
# Run the server first:  uvicorn app.main:app --reload
# -------------------------------------------------------------

BASE=${BASE:-http://localhost:8000}

echo "1) Service info"
curl -s "$BASE/" | python -m json.tool
echo

echo "2) Health"
curl -s "$BASE/health" | python -m json.tool
echo

echo "3) Class list"
curl -s "$BASE/classes" | python -m json.tool
echo

echo "4) Predict from file (replace cat.jpg with your image)"
curl -s -X POST "$BASE/predict?topk=3" \
     -F "file=@cat.jpg" | python -m json.tool
echo

echo "5) Predict from URL"
curl -s -X POST "$BASE/predict_url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg", "topk": 3}' \
     | python -m json.tool
