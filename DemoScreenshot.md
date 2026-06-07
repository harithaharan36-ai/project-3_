# Demo Screenshots — How to capture

After starting the API (`uvicorn app.main:app --reload` **or** `docker run -p 8000:8000 cifar10-api`), capture the following and save them in `docs/`:

| # | What | Suggested filename |
|---|------|--------------------|
| 1 | Swagger UI at `http://localhost:8000/docs` | `swagger_ui.png` |
| 2 | `/health` response in browser or curl | `health_check.png` |
| 3 | A successful `/predict` curl call in your terminal | `predict_curl.png` |
| 4 | `docker ps` showing the running container | `docker_ps.png` |
| 5 | Container logs from `docker logs <id>` | `docker_logs.png` |

## Reproducible curl demo

```bash
# Start container
docker run -d --name cifar10-api -p 8000:8000 cifar10-api:latest

# Wait for health to flip to "ok"
sleep 5 && curl -s http://localhost:8000/health

# Predict
curl -X POST "http://localhost:8000/predict?topk=3" \
     -F "file=@examples/cat.jpg" | python -m json.tool

# Tear down
docker stop cifar10-api && docker rm cifar10-api
```

Embed the screenshots in your task submission document (DOC/PDF).
