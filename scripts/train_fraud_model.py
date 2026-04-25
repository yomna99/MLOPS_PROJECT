from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FraudDatasetConfig
from src.models.train import TrainingConfig, train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fraud detection baseline model.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/AIML Dataset.csv"),
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=250_000,
        help="Maximum number of rows loaded for a training run.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("artifacts/fraud_model.joblib"),
        help="Path where the trained model is saved.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/fraud_metrics.json"),
        help="Path where evaluation metrics are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = FraudDatasetConfig(
        csv_path=args.data_path,
        sample_size=args.sample_size,
    )
    training_config = TrainingConfig(
        dataset=dataset_config,
        model_output_path=args.model_output,
        metrics_output_path=args.metrics_output,
    )
    metrics = train_and_save(training_config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
