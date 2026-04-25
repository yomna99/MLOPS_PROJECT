from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "isFraud"
DROP_COLUMNS = ["nameOrig", "nameDest"]
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]
CATEGORICAL_FEATURES = ["type"]


@dataclass(frozen=True)
class FraudFeatureSpec:
    target_column: str = TARGET_COLUMN
    numeric_features: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    drop_columns: tuple[str, ...] = tuple(DROP_COLUMNS)


def prepare_training_frame(
    dataframe: pd.DataFrame,
    spec: FraudFeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into model features and target."""
    spec = spec or FraudFeatureSpec()

    missing_columns = {
        spec.target_column,
        *spec.numeric_features,
        *spec.categorical_features,
    } - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")

    features = dataframe.drop(columns=list(spec.drop_columns), errors="ignore").copy()
    target = features.pop(spec.target_column).astype(int)
    return features, target
