from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.inference.service import DEFAULT_ARTIFACT_PATH, FraudPredictionService


class FraudPredictionRequest(BaseModel):
    step: int = Field(..., ge=0)
    type: str
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = Field(default=0, ge=0, le=1)


class FraudPredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    predicted_label: Literal["fraud", "not_fraud"]
    fraud_probability: float
    threshold: float
    model_name: str


class FraudBatchPredictionRequest(BaseModel):
    transactions: list[FraudPredictionRequest] = Field(..., min_length=1)


class FraudBatchSummary(BaseModel):
    total_transactions: int
    fraud_predictions: int
    non_fraud_predictions: int
    average_fraud_probability: float


class FraudBatchPredictionResponse(BaseModel):
    summary: FraudBatchSummary
    predictions: list[FraudPredictionResponse]


@lru_cache(maxsize=1)
def get_prediction_service() -> FraudPredictionService:
    return FraudPredictionService.from_environment()


app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.get("/health")
def healthcheck(service: FraudPredictionService = Depends(get_prediction_service)) -> dict:
    metadata = service.metadata()
    return {
        "status": "ok",
        "artifact_path": metadata["artifact_path"] or str(Path(DEFAULT_ARTIFACT_PATH)),
        "model_name": metadata["model_name"],
        "threshold": metadata["threshold"],
    }


@app.post("/predict", response_model=FraudPredictionResponse)
def predict(
    request: FraudPredictionRequest,
    service: FraudPredictionService = Depends(get_prediction_service),
) -> FraudPredictionResponse:
    result = service.predict(request.model_dump())
    return FraudPredictionResponse(**result.to_dict())


@app.post("/predict-batch", response_model=FraudBatchPredictionResponse)
def predict_batch(
    request: FraudBatchPredictionRequest,
    service: FraudPredictionService = Depends(get_prediction_service),
) -> FraudBatchPredictionResponse:
    results = service.predict_batch([transaction.model_dump() for transaction in request.transactions])
    prediction_rows = [
        FraudPredictionResponse(
            prediction=int(row["prediction"]),
            predicted_label=row["predicted_label"],
            fraud_probability=float(row["fraud_probability"]),
            threshold=float(row["threshold"]),
            model_name=row["model_name"],
        )
        for row in results.to_dict(orient="records")
    ]
    fraud_predictions = int(results["prediction"].sum())
    summary = FraudBatchSummary(
        total_transactions=int(len(results)),
        fraud_predictions=fraud_predictions,
        non_fraud_predictions=int(len(results) - fraud_predictions),
        average_fraud_probability=float(results["fraud_probability"].mean()),
    )
    return FraudBatchPredictionResponse(summary=summary, predictions=prediction_rows)
