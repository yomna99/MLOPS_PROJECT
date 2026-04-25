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
