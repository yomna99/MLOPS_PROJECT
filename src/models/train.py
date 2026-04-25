from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data.dataset import FraudDatasetConfig, load_fraud_dataset
from src.features.preprocess import FraudFeatureSpec, prepare_training_frame


@dataclass(frozen=True)
class TrainingConfig:
    dataset: FraudDatasetConfig = FraudDatasetConfig()
    features: FraudFeatureSpec = FraudFeatureSpec()
    test_size: float = 0.2
    random_state: int = 42
    max_majority_to_minority_ratio: int = 10
    n_estimators: int = 160
    max_depth: int = 12
    n_jobs: int = 1
    model_output_path: Path = Path("artifacts/fraud_model.joblib")
    metrics_output_path: Path = Path("reports/fraud_metrics.json")


def build_preprocessor(spec: FraudFeatureSpec) -> ColumnTransformer:
    """Build the common preprocessing block shared by all candidate models."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                list(spec.numeric_features),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(spec.categorical_features),
            ),
        ]
    )


def build_model_pipeline(config: TrainingConfig) -> Pipeline:
    """Create a baseline fraud-detection pipeline."""
    preprocessor = build_preprocessor(config.features)
    classifier = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        class_weight="balanced_subsample",
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def downsample_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    max_majority_to_minority_ratio: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reduce the majority class to keep training tractable on a large dataset."""
    combined = features.copy()
    combined["__target__"] = target

    minority = combined[combined["__target__"] == 1]
    majority = combined[combined["__target__"] == 0]
    if minority.empty or majority.empty:
        raise ValueError("Training split must contain both classes.")

    max_majority = len(minority) * max_majority_to_minority_ratio
    if len(majority) > max_majority:
        majority = majority.sample(n=max_majority, random_state=random_state)

    balanced = pd.concat([minority, majority], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    sampled_target = balanced.pop("__target__").astype(int)
    return balanced, sampled_target


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def split_train_validation_test(
    features: pd.DataFrame,
    target: pd.Series,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/validation/test splits."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1.")

    features_train_val, features_test, target_train_val, target_test = split_dataset(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    validation_relative_size = validation_size / (1 - test_size)
    features_train, features_validation, target_train, target_validation = split_dataset(
        features_train_val,
        target_train_val,
        test_size=validation_relative_size,
        random_state=random_state,
    )
    return (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
    )


def get_default_threshold(model: Pipeline) -> float:
    """Return the default decision threshold for the model output."""
    if hasattr(model, "predict_proba"):
        return 0.5
    if hasattr(model, "decision_function"):
        return 0.0
    raise ValueError("Model must support predict_proba or decision_function.")


def get_positive_class_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return scores associated with the positive class."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        if hasattr(model, "predict_proba"):
            return pd.Series(model.predict_proba(features)[:, 1])
        if hasattr(model, "decision_function"):
            return pd.Series(model.decision_function(features))
    raise ValueError("Model must support predict_proba or decision_function for evaluation.")


def scores_to_predictions(scores: pd.Series | np.ndarray, threshold: float) -> np.ndarray:
    """Convert model scores to binary predictions with a configurable threshold."""
    scores_array = np.asarray(scores)
    return (scores_array >= threshold).astype(int)


def evaluate_scores(
    scores: pd.Series | np.ndarray,
    target_true: pd.Series,
    threshold: float,
) -> dict:
    """Compute thresholded and ranking metrics from positive-class scores."""
    probabilities = np.asarray(scores)
    predictions = scores_to_predictions(probabilities, threshold)

    report = classification_report(
        target_true,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(target_true, probabilities),
        "average_precision": average_precision_score(target_true, probabilities),
        "fraud_precision": precision_score(target_true, predictions, zero_division=0),
        "fraud_recall": recall_score(target_true, predictions, zero_division=0),
        "fraud_f1": f1_score(target_true, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(target_true, predictions).tolist(),
        "classification_report": report,
    }
    return metrics


def evaluate_model(
    model: Pipeline,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    threshold: float | None = None,
) -> dict:
    scores = get_positive_class_scores(model, features_test)
    decision_threshold = threshold if threshold is not None else get_default_threshold(model)
    return evaluate_scores(scores, target_test, decision_threshold)


def train_and_save(config: TrainingConfig | None = None) -> dict:
    config = config or TrainingConfig()

    dataframe = load_fraud_dataset(config.dataset)
    features, target = prepare_training_frame(dataframe, config.features)

    features_train, features_test, target_train, target_test = split_dataset(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    sampled_train_features, sampled_train_target = downsample_training_data(
        features_train,
        target_train,
        max_majority_to_minority_ratio=config.max_majority_to_minority_ratio,
        random_state=config.random_state,
    )

    model = build_model_pipeline(config)
    model.fit(sampled_train_features, sampled_train_target)

    metrics = evaluate_model(model, features_test, target_test)
    metrics["dataset"] = {
        "raw_rows": int(len(dataframe)),
        "train_rows": int(len(features_train)),
        "train_rows_after_downsampling": int(len(sampled_train_features)),
        "test_rows": int(len(features_test)),
        "fraud_rate_full_sample": float(target.mean()),
    }
    metrics["config"] = {
        "dataset": {
            "csv_path": str(config.dataset.csv_path),
            "sample_size": config.dataset.sample_size,
            "random_state": config.dataset.random_state,
        },
        "test_size": config.test_size,
        "random_state": config.random_state,
        "max_majority_to_minority_ratio": config.max_majority_to_minority_ratio,
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "n_jobs": config.n_jobs,
    }

    config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.model_output_path)
    config.metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
