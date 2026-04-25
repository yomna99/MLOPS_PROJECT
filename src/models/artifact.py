from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features.preprocess import FraudFeatureSpec
from src.models.train import get_positive_class_scores, scores_to_predictions


@dataclass(frozen=True)
class FraudModelArtifact:
    """Serializable fraud model bundle containing the estimator and tuned threshold."""

    estimator: Pipeline
    threshold: float
    model_name: str
    feature_spec: FraudFeatureSpec = field(default_factory=FraudFeatureSpec)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        return get_positive_class_scores(self.estimator, features)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        scores = self.predict_scores(features)
        return pd.Series(scores_to_predictions(scores, self.threshold), index=features.index)

    def predict_proba(self, features: pd.DataFrame):
        return self.estimator.predict_proba(features)
