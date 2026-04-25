# Docker Deployment

## Build

```powershell
docker build -f deployment/docker/Dockerfile -t fraud-api:latest .
```

## Run

```powershell
docker run --rm -p 8000:8000 fraud-api:latest
```

## Build Web App

```powershell
docker build -f deployment/docker/Dockerfile.web -t fraud-web:latest .
docker run --rm -p 8501:8501 fraud-web:latest
```

## Run With Compose

```powershell
docker compose -f deployment/docker/compose.yaml up --build
```

This starts:

- the FastAPI service on `http://127.0.0.1:8000`
- the Streamlit web app on `http://127.0.0.1:8501`

## Test

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"

$body = @'
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.0,
  "oldbalanceOrg": 181.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0,
  "isFlaggedFraud": 0
}
'@

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

Then open the Streamlit interface in a browser:

```text
http://127.0.0.1:8501
```
