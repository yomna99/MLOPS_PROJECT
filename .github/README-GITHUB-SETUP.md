# GitHub Setup

## 1. Initialize Git Locally

```powershell
cd C:\Users\HP\Desktop\Mlops_yomna
git init
git branch -M main
git remote add origin https://github.com/yomna99/MLOPS_PROJECT.git
```

## 2. Commit And Push

```powershell
git add .
git commit -m "Add fraud detection API, Docker deployment, and CI/CD workflows"
git push -u origin main
```

## 3. What Happens On GitHub

Two workflows will run:

- `.github/workflows/ci.yml`
  Runs Python tests, starts the API and Streamlit app with Docker Compose, and smoke-tests both services.

- `.github/workflows/cd-ghcr.yml`
  Publishes both Docker images to GitHub Container Registry on pushes to `main` or `master`.

## 4. Published Docker Image

The images will be published to:

```text
ghcr.io/yomna99/fraud-api
ghcr.io/yomna99/fraud-web
```

Possible tags include:

- `latest`
- branch names
- git tags like `v1.0.0`
- commit SHA tags

## 5. Pull The Published Image

```powershell
docker pull ghcr.io/yomna99/fraud-api:latest
docker run --rm -p 8000:8000 ghcr.io/yomna99/fraud-api:latest

docker pull ghcr.io/yomna99/fraud-web:latest
docker run --rm -p 8501:8501 -e FRAUD_API_URL=http://host.docker.internal:8000 ghcr.io/yomna99/fraud-web:latest
```

## 6. Test The Running Image

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"

$body = Get-Content "artifacts/sample_prediction_input.json" -Raw
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```
