from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd

from src.features.preprocess import FraudFeatureSpec
from src.models.artifact import FraudModelArtifact


DEFAULT_ARTIFACT_PATH = Path("artifacts/best_fraud_model_f1.joblib")
ARTIFACT_PATH_ENV_VAR = "FRAUD_MODEL_ARTIFACT_PATH"


@dataclass(frozen=True)
class FraudPredictionResult:
    """Prediction result returned by the inference service."""

    prediction_id: str
    prediction: int
    predicted_label: str
    fraud_probability: float
    threshold: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "prediction": self.prediction,
            "predicted_label": self.predicted_label,
            "fraud_probability": self.fraud_probability,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }


class FraudPredictionService:
    """Load the trained fraud artifact and serve single-transaction predictions."""

    def __init__(self, artifact: FraudModelArtifact, artifact_path: Path | None = None) -> None:
        self.artifact = artifact
        self.feature_spec = artifact.feature_spec
        self.artifact_path = artifact_path

    @classmethod
    def from_path(cls, artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> "FraudPredictionService":
        artifact_path = Path(artifact_path)
        loaded_artifact = joblib.load(artifact_path)
        if not isinstance(loaded_artifact, FraudModelArtifact):
            raise TypeError(
                f"Expected a FraudModelArtifact in {artifact_path}, got {type(loaded_artifact).__name__}."
            )
        return cls(loaded_artifact, artifact_path=artifact_path)

    @classmethod
    def from_environment(cls) -> "FraudPredictionService":
        artifact_path = Path(os.getenv(ARTIFACT_PATH_ENV_VAR, str(DEFAULT_ARTIFACT_PATH)))
        return cls.from_path(artifact_path)

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        feature_spec = self.feature_spec
        feature_values: dict[str, Any] = {}

        for feature_name in feature_spec.numeric_features:
            if feature_name not in payload:
                raise ValueError(f"Missing numeric feature: {feature_name}")
            feature_values[feature_name] = payload[feature_name]

        for feature_name in feature_spec.categorical_features:
            if feature_name not in payload:
                raise ValueError(f"Missing categorical feature: {feature_name}")
            feature_values[feature_name] = payload[feature_name]

        return feature_values

    def _build_feature_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        feature_spec = self.feature_spec
        feature_values = self._normalize_payload(payload)
        ordered_columns = [*feature_spec.numeric_features, *feature_spec.categorical_features]
        return pd.DataFrame([[feature_values[column] for column in ordered_columns]], columns=ordered_columns)

    def predict(self, payload: dict[str, Any]) -> FraudPredictionResult:
        features = self._build_feature_frame(payload)
        score = float(self.artifact.predict_scores(features).iloc[0])
        prediction = int(self.artifact.predict(features).iloc[0])
        predicted_label = "fraud" if prediction == 1 else "not_fraud"
        return FraudPredictionResult(
            prediction_id=str(uuid4()),
            prediction=prediction,
            predicted_label=predicted_label,
            fraud_probability=score,
            threshold=float(self.artifact.threshold),
            model_name=self.artifact.model_name,
        )

    def predict_batch(self, payloads: list[dict[str, Any]]) -> pd.DataFrame:
        if not payloads:
            raise ValueError("Batch payload must contain at least one transaction.")

        ordered_columns = [
            *self.feature_spec.numeric_features,
            *self.feature_spec.categorical_features,
        ]
        normalized_rows = [self._normalize_payload(payload) for payload in payloads]
        features = pd.DataFrame(normalized_rows, columns=ordered_columns)

        scores = self.artifact.predict_scores(features).astype(float)
        predictions = self.artifact.predict(features).astype(int)

        results = features.copy()
        results["prediction_id"] = [str(uuid4()) for _ in range(len(results))]
        results["fraud_probability"] = scores.values
        results["prediction"] = predictions.values
        results["predicted_label"] = results["prediction"].map({0: "not_fraud", 1: "fraud"})
        results["threshold"] = float(self.artifact.threshold)
        results["model_name"] = self.artifact.model_name
        return results

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.artifact.model_name,
            "threshold": float(self.artifact.threshold),
            "artifact_path": str(self.artifact_path) if self.artifact_path is not None else None,
            "feature_spec": {
                "numeric_features": list(self.feature_spec.numeric_features),
                "categorical_features": list(self.feature_spec.categorical_features),
            },
            "metadata": self.artifact.metadata,
        }
