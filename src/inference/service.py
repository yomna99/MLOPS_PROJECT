from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features.preprocess import FraudFeatureSpec
from src.models.artifact import FraudModelArtifact


DEFAULT_ARTIFACT_PATH = Path("artifacts/best_fraud_model_f1.joblib")
ARTIFACT_PATH_ENV_VAR = "FRAUD_MODEL_ARTIFACT_PATH"


@dataclass(frozen=True)
class FraudPredictionResult:
    """Prediction result returned by the inference service."""

    prediction: int
    predicted_label: str
    fraud_probability: float
    threshold: float
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
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

    def _build_feature_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
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

        ordered_columns = [*feature_spec.numeric_features, *feature_spec.categorical_features]
        return pd.DataFrame([[feature_values[column] for column in ordered_columns]], columns=ordered_columns)

    def predict(self, payload: dict[str, Any]) -> FraudPredictionResult:
        features = self._build_feature_frame(payload)
        score = float(self.artifact.predict_scores(features).iloc[0])
        prediction = int(self.artifact.predict(features).iloc[0])
        predicted_label = "fraud" if prediction == 1 else "not_fraud"
        return FraudPredictionResult(
            prediction=prediction,
            predicted_label=predicted_label,
            fraud_probability=score,
            threshold=float(self.artifact.threshold),
            model_name=self.artifact.model_name,
        )

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
