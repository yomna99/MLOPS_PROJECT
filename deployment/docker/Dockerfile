FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV FRAUD_MODEL_ARTIFACT_PATH=/app/artifacts/best_fraud_model_f1.joblib

WORKDIR /app

COPY deployment/docker/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-api.txt

COPY src /app/src
COPY artifacts/best_fraud_model_f1.joblib /app/artifacts/best_fraud_model_f1.joblib
COPY artifacts/best_fraud_model_f1.pkl /app/artifacts/best_fraud_model_f1.pkl

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
