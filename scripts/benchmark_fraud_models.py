from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FraudDatasetConfig
from src.models.benchmark import (
    BenchmarkConfig,
    benchmark_and_select_best,
    get_available_optional_models,
)


def parse_args() -> argparse.Namespace:
    optional_models = get_available_optional_models()
    default_models = [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "bagging_tree",
        "adaboost",
    ]
    for model_name in ("catboost", "lightgbm", "xgboost"):
        if optional_models.get(model_name):
            default_models.append(model_name)

    parser = argparse.ArgumentParser(
        description="Benchmark several fraud detection models and keep the best one."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/AIML Dataset.csv"),
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=120_000,
        help="Maximum number of rows loaded for the benchmark.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=6,
        help="Number of hyperparameter combinations tried per model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds used during tuning.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Share of the full dataset reserved for validation.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="average_precision",
        choices=["average_precision", "roc_auc", "fraud_f1", "fraud_precision", "fraud_recall"],
        help="Metric used inside CV to tune hyperparameters.",
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="fraud_f1",
        choices=["fraud_f1", "fraud_precision", "fraud_recall"],
        help="Validation metric used to choose the decision threshold and final model.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help="Subset of candidate models to benchmark.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/best_fraud_model.joblib"),
        help="Path where the best model is saved.",
    )
    parser.add_argument(
        "--pickle-output",
        type=Path,
        default=Path("artifacts/best_fraud_model.pkl"),
        help="Path where the best model is also saved as a pickle file.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/model_benchmark.json"),
        help="Path where the benchmark report is saved.",
    )
    parser.add_argument(
        "--leaderboard-output",
        type=Path,
        default=Path("reports/model_leaderboard.csv"),
        help="Path where the leaderboard CSV is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = FraudDatasetConfig(
        csv_path=args.data_path,
        sample_size=args.sample_size,
    )
    benchmark_config = BenchmarkConfig(
        dataset=dataset_config,
        validation_size=args.validation_size,
        search_iterations=args.search_iterations,
        cv_folds=args.cv_folds,
        selection_metric=args.selection_metric,
        threshold_metric=args.threshold_metric,
        candidate_models=tuple(args.models),
        model_output_path=args.model_output,
        pickle_output_path=args.pickle_output,
        report_output_path=args.report_output,
        leaderboard_output_path=args.leaderboard_output,
    )
    report = benchmark_and_select_best(benchmark_config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
