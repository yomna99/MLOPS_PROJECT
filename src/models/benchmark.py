from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import ClassifierTags, InputTags, Tags, TargetTags

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.data.dataset import FraudDatasetConfig, load_fraud_dataset
from src.features.preprocess import FraudFeatureSpec, prepare_training_frame
from src.models.artifact import FraudModelArtifact
from src.models.train import (
    build_preprocessor,
    downsample_training_data,
    evaluate_model,
    evaluate_scores,
    get_default_threshold,
    get_positive_class_scores,
    split_train_validation_test,
)


DEFAULT_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "fraud_f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    "fraud_precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "fraud_recall": make_scorer(recall_score, pos_label=1, zero_division=0),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: FraudDatasetConfig = FraudDatasetConfig()
    features: FraudFeatureSpec = FraudFeatureSpec()
    validation_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42
    max_majority_to_minority_ratio: int = 10
    cv_folds: int = 3
    search_iterations: int = 6
    selection_metric: str = "average_precision"
    threshold_metric: str = "fraud_f1"
    candidate_models: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "bagging_tree",
        "adaboost",
        "catboost",
        "lightgbm",
        "xgboost",
    )
    model_output_path: Path = Path("artifacts/best_fraud_model.joblib")
    pickle_output_path: Path = Path("artifacts/best_fraud_model.pkl")
    report_output_path: Path = Path("reports/model_benchmark.json")
    leaderboard_output_path: Path = Path("reports/model_leaderboard.csv")
    n_jobs: int = 1


@dataclass(frozen=True)
class CandidateModelSpec:
    name: str
    estimator: object
    param_distributions: dict[str, list]
    notes: str


class SklearnCompatibleCatBoostClassifier(CatBoostClassifier if CatBoostClassifier is not None else object):
    """Patch CatBoost to expose sklearn tags expected by sklearn>=1.6."""

    def __sklearn_tags__(self) -> Tags:
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True, one_d_labels=True),
            classifier_tags=ClassifierTags(multi_class=True, multi_label=False),
            input_tags=InputTags(two_d_array=True, sparse=True, allow_nan=True),
        )


def get_available_optional_models() -> dict[str, bool]:
    """Return the availability of optional third-party boosters."""
    return {
        "catboost": CatBoostClassifier is not None,
        "lightgbm": LGBMClassifier is not None,
        "xgboost": XGBClassifier is not None,
    }


def get_candidate_model_specs(config: BenchmarkConfig) -> list[CandidateModelSpec]:
    """Return the benchmark candidates available for the fraud task."""
    all_specs = {
        "logistic_regression": CandidateModelSpec(
            name="logistic_regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1_500,
                solver="liblinear",
                random_state=config.random_state,
            ),
            param_distributions={
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
            notes="Linear baseline with class balancing.",
        ),
        "random_forest": CandidateModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=config.n_jobs,
                random_state=config.random_state,
            ),
            param_distributions={
                "classifier__n_estimators": [120, 180, 240],
                "classifier__max_depth": [8, 12, 18, None],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
            notes="Bagging ensemble of decision trees.",
        ),
        "extra_trees": CandidateModelSpec(
            name="extra_trees",
            estimator=ExtraTreesClassifier(
                class_weight="balanced_subsample",
                n_jobs=config.n_jobs,
                random_state=config.random_state,
            ),
            param_distributions={
                "classifier__n_estimators": [160, 220, 300],
                "classifier__max_depth": [8, 12, 18, None],
                "classifier__min_samples_leaf": [1, 2, 4],
            },
            notes="Bagging-style ensemble with stronger randomization.",
        ),
        "gradient_boosting": CandidateModelSpec(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(
                random_state=config.random_state,
            ),
            param_distributions={
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [None, 6, 10],
                "classifier__n_estimators": [120, 180, 260],
                "classifier__subsample": [0.7, 0.85, 1.0],
            },
            notes="Gradient boosting baseline for tabular fraud data.",
        ),
        "bagging_tree": CandidateModelSpec(
            name="bagging_tree",
            estimator=BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
                random_state=config.random_state,
                n_jobs=config.n_jobs,
            ),
            param_distributions={
                "classifier__n_estimators": [60, 100, 140],
                "classifier__estimator__max_depth": [4, 6, 10, None],
                "classifier__estimator__min_samples_leaf": [1, 2, 4],
            },
            notes="Explicit bagging with balanced decision trees.",
        ),
        "adaboost": CandidateModelSpec(
            name="adaboost",
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=2,
                    random_state=config.random_state,
                ),
                random_state=config.random_state,
            ),
            param_distributions={
                "classifier__n_estimators": [80, 120, 180],
                "classifier__learning_rate": [0.03, 0.05, 0.1, 0.2],
                "classifier__estimator__max_depth": [1, 2, 3],
            },
            notes="Classic boosting baseline over shallow trees.",
        ),
    }

    if CatBoostClassifier is not None:
        all_specs["catboost"] = CandidateModelSpec(
            name="catboost",
            estimator=SklearnCompatibleCatBoostClassifier(
                auto_class_weights="Balanced",
                random_seed=config.random_state,
                allow_writing_files=False,
                verbose=False,
                thread_count=config.n_jobs,
            ),
            param_distributions={
                "classifier__depth": [4, 6, 8],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__iterations": [120, 200, 320],
                "classifier__l2_leaf_reg": [1, 3, 5, 7],
            },
            notes="Gradient boosting with ordered boosting and balanced class handling.",
        )

    if LGBMClassifier is not None:
        all_specs["lightgbm"] = CandidateModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                class_weight="balanced",
                random_state=config.random_state,
                verbosity=-1,
                n_jobs=config.n_jobs,
            ),
            param_distributions={
                "classifier__n_estimators": [120, 200, 320],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__num_leaves": [15, 31, 63],
                "classifier__max_depth": [-1, 6, 10],
                "classifier__min_child_samples": [10, 20, 40],
            },
            notes="Leaf-wise gradient boosting for tabular classification.",
        )

    if XGBClassifier is not None:
        all_specs["xgboost"] = CandidateModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                eval_metric="logloss",
                tree_method="hist",
            ),
            param_distributions={
                "classifier__n_estimators": [120, 200, 320],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_depth": [4, 6, 8],
                "classifier__subsample": [0.7, 0.85, 1.0],
                "classifier__colsample_bytree": [0.7, 0.85, 1.0],
            },
            notes="Optimized gradient boosting often strong on fraud tabular datasets.",
        )

    unknown_models = set(config.candidate_models) - set(all_specs)
    if unknown_models:
        availability = get_available_optional_models()
        missing_optional = sorted(
            model_name
            for model_name in unknown_models
            if model_name in availability and not availability[model_name]
        )
        truly_unknown = sorted(set(unknown_models) - set(missing_optional))
        details = []
        if missing_optional:
            details.append(
                "optional packages not installed for: " + ", ".join(missing_optional)
            )
        if truly_unknown:
            details.append("unknown models: " + ", ".join(truly_unknown))
        raise ValueError("; ".join(details))

    return [all_specs[name] for name in config.candidate_models]


def build_candidate_pipeline(feature_spec: FraudFeatureSpec, estimator: object) -> Pipeline:
    """Wrap a candidate estimator with the shared preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_spec)),
            ("classifier", estimator),
        ]
    )


def summarize_search(search: RandomizedSearchCV, spec: CandidateModelSpec) -> dict:
    """Extract the most useful tuning details from a fitted search object."""
    best_index = search.best_index_
    cv_results = search.cv_results_
    summary = {
        "model_name": spec.name,
        "notes": spec.notes,
        "best_params": search.best_params_,
        "cv_scores": {
            "selection_metric": search.refit,
            "best_selection_score": float(search.best_score_),
            "roc_auc": float(cv_results["mean_test_roc_auc"][best_index]),
            "average_precision": float(cv_results["mean_test_average_precision"][best_index]),
            "fraud_f1": float(cv_results["mean_test_fraud_f1"][best_index]),
            "fraud_precision": float(cv_results["mean_test_fraud_precision"][best_index]),
            "fraud_recall": float(cv_results["mean_test_fraud_recall"][best_index]),
        },
    }
    return summary


def select_best_threshold(
    target_true: pd.Series,
    scores: pd.Series,
    metric_name: str,
    default_threshold: float,
) -> dict:
    """Pick the decision threshold that maximizes a validation metric."""
    supported_metrics = {"fraud_f1", "fraud_precision", "fraud_recall"}
    if metric_name not in supported_metrics:
        raise ValueError(f"Unsupported threshold metric: {metric_name}")

    score_values = np.asarray(scores, dtype=float)
    quantile_grid = np.quantile(score_values, np.linspace(0.0, 1.0, 201))
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                quantile_grid,
                np.array([default_threshold]),
            ]
        )
    )

    target_values = np.asarray(target_true, dtype=int)
    metric_functions = {
        "fraud_f1": f1_score,
        "fraud_precision": precision_score,
        "fraud_recall": recall_score,
    }
    metric_function = metric_functions[metric_name]

    best_threshold = float(default_threshold)
    best_score = float("-inf")

    for threshold in candidate_thresholds:
        predictions = (score_values >= threshold).astype(int)
        current_score = float(metric_function(target_values, predictions, zero_division=0))
        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)

    best_metrics = evaluate_scores(score_values, target_true, best_threshold)
    default_metrics = evaluate_scores(scores, target_true, float(default_threshold))
    return {
        "metric": metric_name,
        "selected_threshold": best_threshold,
        "selected_metric_value": best_score,
        "default_threshold": float(default_threshold),
        "default_metric_value": float(default_metrics[metric_name]),
        "candidate_threshold_count": int(len(candidate_thresholds)),
        "tuned_validation_metrics": best_metrics,
        "default_validation_metrics": default_metrics,
    }


def benchmark_and_select_best(config: BenchmarkConfig | None = None) -> dict:
    """Tune several candidate models and save the best-performing one."""
    config = config or BenchmarkConfig()

    dataframe = load_fraud_dataset(config.dataset)
    features, target = prepare_training_frame(dataframe, config.features)
    (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
    ) = split_train_validation_test(
        features,
        target,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    sampled_train_features, sampled_train_target = downsample_training_data(
        features_train,
        target_train,
        max_majority_to_minority_ratio=config.max_majority_to_minority_ratio,
        random_state=config.random_state,
    )

    if config.selection_metric not in DEFAULT_SCORING:
        raise ValueError(f"Unsupported selection metric: {config.selection_metric}")
    if config.threshold_metric not in {"fraud_f1", "fraud_precision", "fraud_recall"}:
        raise ValueError(f"Unsupported threshold metric: {config.threshold_metric}")

    candidate_results: list[dict] = []
    best_result: dict | None = None
    best_estimator: Pipeline | None = None

    for spec in get_candidate_model_specs(config):
        pipeline = build_candidate_pipeline(config.features, clone(spec.estimator))
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=spec.param_distributions,
            n_iter=min(config.search_iterations, count_search_space(spec.param_distributions)),
            scoring=DEFAULT_SCORING,
            refit=config.selection_metric,
            cv=config.cv_folds,
            n_jobs=1,
            random_state=config.random_state,
            verbose=0,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            )
            search.fit(sampled_train_features, sampled_train_target)

        result = summarize_search(search, spec)
        validation_scores = get_positive_class_scores(search.best_estimator_, features_validation)
        default_threshold = get_default_threshold(search.best_estimator_)
        threshold_selection = select_best_threshold(
            target_validation,
            validation_scores,
            metric_name=config.threshold_metric,
            default_threshold=default_threshold,
        )
        validation_metrics = threshold_selection["tuned_validation_metrics"]
        holdout_metrics = evaluate_model(
            search.best_estimator_,
            features_test,
            target_test,
            threshold=threshold_selection["selected_threshold"],
        )
        result["threshold_selection"] = {
            "metric": threshold_selection["metric"],
            "selected_threshold": float(threshold_selection["selected_threshold"]),
            "selected_metric_value": float(threshold_selection["selected_metric_value"]),
            "default_threshold": float(threshold_selection["default_threshold"]),
            "default_metric_value": float(threshold_selection["default_metric_value"]),
            "candidate_threshold_count": int(threshold_selection["candidate_threshold_count"]),
        }
        result["validation_metrics"] = {
            "threshold": float(validation_metrics["threshold"]),
            "roc_auc": float(validation_metrics["roc_auc"]),
            "average_precision": float(validation_metrics["average_precision"]),
            "fraud_f1": float(validation_metrics["fraud_f1"]),
            "fraud_precision": float(validation_metrics["fraud_precision"]),
            "fraud_recall": float(validation_metrics["fraud_recall"]),
            "confusion_matrix": validation_metrics["confusion_matrix"],
        }
        result["validation_metrics_default_threshold"] = {
            "threshold": float(threshold_selection["default_validation_metrics"]["threshold"]),
            "roc_auc": float(threshold_selection["default_validation_metrics"]["roc_auc"]),
            "average_precision": float(threshold_selection["default_validation_metrics"]["average_precision"]),
            "fraud_f1": float(threshold_selection["default_validation_metrics"]["fraud_f1"]),
            "fraud_precision": float(threshold_selection["default_validation_metrics"]["fraud_precision"]),
            "fraud_recall": float(threshold_selection["default_validation_metrics"]["fraud_recall"]),
            "confusion_matrix": threshold_selection["default_validation_metrics"]["confusion_matrix"],
        }
        result["holdout_metrics"] = {
            "threshold": float(holdout_metrics["threshold"]),
            "roc_auc": float(holdout_metrics["roc_auc"]),
            "average_precision": float(holdout_metrics["average_precision"]),
            "fraud_f1": float(holdout_metrics["fraud_f1"]),
            "fraud_precision": float(holdout_metrics["fraud_precision"]),
            "fraud_recall": float(holdout_metrics["fraud_recall"]),
            "confusion_matrix": holdout_metrics["confusion_matrix"],
        }
        candidate_results.append(result)

        current_score = result["validation_metrics"][config.threshold_metric]
        if best_result is None or current_score > best_result["validation_metrics"][config.threshold_metric]:
            best_result = result
            best_estimator = search.best_estimator_

    assert best_result is not None
    assert best_estimator is not None

    leaderboard = build_leaderboard(candidate_results)
    report = {
        "selected_model": best_result["model_name"],
        "selection_strategy": {
            "hyperparameter_selection_metric_cv": config.selection_metric,
            "threshold_selection_metric_validation": config.threshold_metric,
            "final_model_selection_stage": "validation",
        },
        "artifacts": {
            "joblib_model_path": str(config.model_output_path),
            "pickle_model_path": str(config.pickle_output_path),
            "report_output_path": str(config.report_output_path),
            "leaderboard_output_path": str(config.leaderboard_output_path),
        },
        "dataset": {
            "csv_path": str(config.dataset.csv_path),
            "raw_rows": int(len(dataframe)),
            "train_rows": int(len(features_train)),
            "validation_rows": int(len(features_validation)),
            "train_rows_after_downsampling": int(len(sampled_train_features)),
            "test_rows": int(len(features_test)),
            "fraud_rate_full_sample": float(target.mean()),
        },
        "benchmark_config": {
            "validation_size": config.validation_size,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "max_majority_to_minority_ratio": config.max_majority_to_minority_ratio,
            "cv_folds": config.cv_folds,
            "search_iterations": config.search_iterations,
            "selection_metric": config.selection_metric,
            "threshold_metric": config.threshold_metric,
            "candidate_models": list(config.candidate_models),
            "n_jobs": config.n_jobs,
        },
        "best_model": best_result,
        "candidates": sorted(
            candidate_results,
            key=lambda item: item["validation_metrics"][config.threshold_metric],
            reverse=True,
        ),
    }

    config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.pickle_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_output_path.parent.mkdir(parents=True, exist_ok=True)
    config.leaderboard_output_path.parent.mkdir(parents=True, exist_ok=True)

    best_artifact = FraudModelArtifact(
        estimator=best_estimator,
        threshold=float(best_result["threshold_selection"]["selected_threshold"]),
        model_name=best_result["model_name"],
        feature_spec=config.features,
        metadata={
            "best_params": best_result["best_params"],
            "selection_strategy": report["selection_strategy"],
        },
    )
    joblib.dump(best_artifact, config.model_output_path)
    with config.pickle_output_path.open("wb") as pickle_file:
        pickle.dump(best_artifact, pickle_file)
    config.report_output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    leaderboard.to_csv(config.leaderboard_output_path, index=False)
    return report


def count_search_space(param_distributions: dict[str, list]) -> int:
    """Compute the total number of combinations in a discrete search space."""
    total = 1
    for values in param_distributions.values():
        total *= len(values)
    return total


def build_leaderboard(candidate_results: list[dict]) -> pd.DataFrame:
    """Create a flat leaderboard that is easy to inspect or plot."""
    rows = []
    for item in candidate_results:
        rows.append(
            {
                "model_name": item["model_name"],
                "cv_selection_score": item["cv_scores"]["best_selection_score"],
                "cv_roc_auc": item["cv_scores"]["roc_auc"],
                "cv_average_precision": item["cv_scores"]["average_precision"],
                "cv_fraud_f1": item["cv_scores"]["fraud_f1"],
                "validation_selection_metric": item["threshold_selection"]["metric"],
                "validation_selection_score": item["threshold_selection"]["selected_metric_value"],
                "validation_threshold": item["threshold_selection"]["selected_threshold"],
                "validation_fraud_f1": item["validation_metrics"]["fraud_f1"],
                "validation_fraud_precision": item["validation_metrics"]["fraud_precision"],
                "validation_fraud_recall": item["validation_metrics"]["fraud_recall"],
                "validation_average_precision": item["validation_metrics"]["average_precision"],
                "test_roc_auc": item["holdout_metrics"]["roc_auc"],
                "test_average_precision": item["holdout_metrics"]["average_precision"],
                "test_fraud_f1": item["holdout_metrics"]["fraud_f1"],
                "test_fraud_precision": item["holdout_metrics"]["fraud_precision"],
                "test_fraud_recall": item["holdout_metrics"]["fraud_recall"],
                "test_threshold": item["holdout_metrics"]["threshold"],
            }
        )

    leaderboard = pd.DataFrame(rows)
    return leaderboard.sort_values(by="validation_selection_score", ascending=False).reset_index(drop=True)
