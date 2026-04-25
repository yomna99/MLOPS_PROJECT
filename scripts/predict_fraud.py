from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.service import FraudPredictionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud prediction for a single transaction.")
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to a JSON file containing one transaction payload.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/best_fraud_model_f1.joblib"),
        help="Path to the trained fraud artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    service = FraudPredictionService.from_path(args.artifact_path)
    result = service.predict(payload)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
