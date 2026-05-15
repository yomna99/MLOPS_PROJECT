from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.service import FraudPredictionService
from src.monitoring.evidently_service import EvidentlyMonitoringService


DEFAULT_REFERENCE_PATH = Path("data/processed/ref_data_sample.csv")
DEFAULT_PRODUCTION_PATH = Path("data/production/prod_data.csv")
DEFAULT_OUTPUT_DIR = Path("reports/evidently")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Evidently monitoring reports.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH, help="Reference dataset CSV.")
    parser.add_argument("--production", type=Path, default=DEFAULT_PRODUCTION_PATH, help="Production dataset CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for HTML reports.")
    parser.add_argument(
        "--base-url",
        default="https://github-actions.local",
        help="Base URL used when building report links in the summary.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if report files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_service = FraudPredictionService.from_environment()
    monitoring_service = EvidentlyMonitoringService(
        prediction_service=prediction_service,
        reference_data_path=args.reference,
        production_data_path=args.production,
        output_dir=args.output,
        public_base_url=args.base_url,
    )
    bundle = monitoring_service.generate_reports(force=args.force)
    print(json.dumps(bundle.to_dict(), indent=2))


if __name__ == "__main__":
    main()
