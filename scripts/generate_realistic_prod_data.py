from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.service import FraudPredictionService
from src.production.store import PREFERRED_COLUMN_ORDER


DEFAULT_REFERENCE_DATA_PATH = Path("data/raw/AIML Dataset.csv")
DEFAULT_OUTPUT_PATH = Path("data/production/prod_data.csv")
DEFAULT_ARTIFACT_PATH = Path("artifacts/best_fraud_model_f1.joblib")
MODEL_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]
RAW_COLUMNS = [*MODEL_FEATURE_COLUMNS, "isFraud"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a realistic production feedback dataset from the raw fraud dataset."
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_DATA_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--rows", type=int, default=180)
    parser.add_argument("--fraud-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def sample_raw_data(reference_path: Path, total_rows: int, fraud_ratio: float, seed: int) -> pd.DataFrame:
    raw = pd.read_csv(reference_path, usecols=RAW_COLUMNS)
    fraud_ratio = min(max(fraud_ratio, 0.02), 0.45)
    fraud_rows = max(1, round(total_rows * fraud_ratio))
    legit_rows = max(1, total_rows - fraud_rows)

    fraud_pool = raw[raw["isFraud"] == 1]
    legit_pool = raw[raw["isFraud"] == 0]

    sampled_fraud = fraud_pool.sample(n=fraud_rows, replace=len(fraud_pool) < fraud_rows, random_state=seed)
    sampled_legit = legit_pool.sample(n=legit_rows, replace=len(legit_pool) < legit_rows, random_state=seed + 1)

    combined = pd.concat([sampled_fraud, sampled_legit], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return combined


def generate_feedback_notes(prediction: int, ground_truth: int) -> str:
    if prediction == 1 and ground_truth == 1:
        return "Customer confirmed this transaction as fraud after investigation."
    if prediction == 0 and ground_truth == 0:
        return "Customer confirmed this transaction as legitimate."
    if prediction == 1 and ground_truth == 0:
        return "False positive alert: customer validated the transaction as legitimate."
    return "Missed fraud scenario recovered by customer feedback."


def build_customer_email(index: int, transaction_type: str) -> str:
    prefix = transaction_type.lower().replace("_", "")[:8]
    return f"{prefix}.client{index:04d}@bankmail.com"


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)

    sampled = sample_raw_data(args.reference, args.rows, args.fraud_ratio, args.seed)
    prediction_service = FraudPredictionService.from_path(args.artifact)
    prediction_frame = prediction_service.predict_batch(sampled[MODEL_FEATURE_COLUMNS].to_dict(orient="records"))

    start_time = datetime.now(timezone.utc) - timedelta(days=45)
    records: list[dict[str, object]] = []
    for index, row in sampled.reset_index(drop=True).iterrows():
        predicted = prediction_frame.iloc[index]
        ground_truth = int(row["isFraud"])
        prediction_value = int(predicted["prediction"])
        user_feedback = "reported_fraud" if ground_truth == 1 else "confirmed_legit"
        feedback_time = start_time + timedelta(hours=index * 6)
        record = {
            "prediction_id": f"realistic-prod-{index + 1:04d}",
            "feedback_timestamp": feedback_time.isoformat(),
            "step": int(row["step"]),
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrig": float(row["newbalanceOrig"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
            "isFlaggedFraud": int(row["isFlaggedFraud"]),
            "customer_email": build_customer_email(index + 1, str(row["type"])),
            "prediction": prediction_value,
            "predicted_label": str(predicted["predicted_label"]),
            "fraud_probability": float(predicted["fraud_probability"]),
            "threshold": float(predicted["threshold"]),
            "model_name": str(predicted["model_name"]),
            "user_feedback": user_feedback,
            "ground_truth_label": ground_truth,
            "feedback_notes": generate_feedback_notes(prediction_value, ground_truth),
        }
        records.append(record)

    output_frame = pd.DataFrame(records)
    output_frame = output_frame[PREFERRED_COLUMN_ORDER]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output, index=False)

    fraud_count = int(output_frame["ground_truth_label"].sum())
    model_alerts = int(output_frame["prediction"].sum())
    print(
        f"Generated {len(output_frame)} realistic production rows at {args.output} "
        f"(ground-truth frauds={fraud_count}, model alerts={model_alerts})."
    )


if __name__ == "__main__":
    main()
