from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas.errors import ParserError


DEFAULT_PRODUCTION_DATA_PATH = Path("data/production/prod_data.csv")
PRODUCTION_DATA_PATH_ENV_VAR = "FRAUD_PRODUCTION_DATA_PATH"

USER_FEEDBACK_TO_LABEL = {
    "confirmed_legit": 0,
    "reported_fraud": 1,
}

PREFERRED_COLUMN_ORDER = [
    "prediction_id",
    "feedback_timestamp",
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "customer_email",
    "prediction",
    "predicted_label",
    "fraud_probability",
    "threshold",
    "model_name",
    "user_feedback",
    "ground_truth_label",
    "feedback_notes",
]


class ProductionFeedbackStore:
    """Append reviewed production predictions to a flat CSV dataset."""

    def __init__(self, output_path: Path | str = DEFAULT_PRODUCTION_DATA_PATH) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls) -> "ProductionFeedbackStore":
        return cls(os.getenv(PRODUCTION_DATA_PATH_ENV_VAR, str(DEFAULT_PRODUCTION_DATA_PATH)))

    def append_feedback(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        user_feedback: str,
        feedback_notes: str | None = None,
    ) -> dict[str, Any]:
        if user_feedback not in USER_FEEDBACK_TO_LABEL:
            raise ValueError(f"Unsupported user feedback: {user_feedback}")

        prediction_id = prediction.get("prediction_id") or str(uuid4())
        feedback_timestamp = datetime.now(timezone.utc).isoformat()
        normalized_notes = (feedback_notes or "").strip()

        record = {
            "prediction_id": prediction_id,
            "feedback_timestamp": feedback_timestamp,
            **transaction,
            "prediction": int(prediction["prediction"]),
            "predicted_label": prediction["predicted_label"],
            "fraud_probability": float(prediction["fraud_probability"]),
            "threshold": float(prediction["threshold"]),
            "model_name": prediction["model_name"],
            "user_feedback": user_feedback,
            "ground_truth_label": USER_FEEDBACK_TO_LABEL[user_feedback],
            "feedback_notes": normalized_notes,
        }

        existing_frame = self._load_feedback_frame()
        row = pd.DataFrame([record])
        combined_frame = self._merge_frames(existing_frame, row)
        combined_frame.to_csv(self.output_path, index=False)
        return record

    def has_feedback(self, prediction_id: str) -> bool:
        feedback_frame = self._load_feedback_frame()
        if feedback_frame.empty or "prediction_id" not in feedback_frame.columns:
            return False
        return bool((feedback_frame["prediction_id"].astype(str) == str(prediction_id)).any())

    def summary(self) -> dict[str, Any]:
        feedback_frame = self._load_feedback_frame()
        if feedback_frame.empty:
            return {
                "production_data_path": str(self.output_path),
                "total_feedback": 0,
                "confirmed_legit": 0,
                "reported_fraud": 0,
                "last_feedback_timestamp": None,
            }

        legit_count = int((feedback_frame["user_feedback"] == "confirmed_legit").sum())
        fraud_count = int((feedback_frame["user_feedback"] == "reported_fraud").sum())
        last_feedback_timestamp = str(feedback_frame["feedback_timestamp"].iloc[-1])

        return {
            "production_data_path": str(self.output_path),
            "total_feedback": int(len(feedback_frame)),
            "confirmed_legit": legit_count,
            "reported_fraud": fraud_count,
            "last_feedback_timestamp": last_feedback_timestamp,
        }

    def records(self, limit: int | None = None) -> list[dict[str, Any]]:
        feedback_frame = self._load_feedback_frame()
        if feedback_frame.empty:
            return []

        ordered = feedback_frame.iloc[::-1].copy()
        if limit is not None:
            ordered = ordered.head(max(int(limit), 0))
        records = ordered.to_dict(orient="records")
        sanitized_records = []
        for record in records:
            sanitized_records.append(
                {
                    key: (None if pd.isna(value) else value)
                    for key, value in record.items()
                }
            )
        return sanitized_records

    def _load_feedback_frame(self) -> pd.DataFrame:
        if not self.output_path.exists():
            return pd.DataFrame(columns=PREFERRED_COLUMN_ORDER)

        try:
            feedback_frame = pd.read_csv(self.output_path)
        except ParserError:
            feedback_frame = self._repair_misaligned_csv()

        return self._normalize_columns(feedback_frame)

    def _repair_misaligned_csv(self) -> pd.DataFrame:
        with self.output_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))

        if not rows:
            return pd.DataFrame(columns=PREFERRED_COLUMN_ORDER)

        header = rows[0]
        data_rows = rows[1:]
        if "customer_email" not in header:
            insert_at = header.index("isFlaggedFraud") + 1 if "isFlaggedFraud" in header else len(header)
            repaired_header = header[:insert_at] + ["customer_email"] + header[insert_at:]
            repaired_rows = []
            for row in data_rows:
                if len(row) == len(header):
                    repaired_rows.append(row[:insert_at] + [""] + row[insert_at:])
                elif len(row) == len(repaired_header):
                    repaired_rows.append(row)
                else:
                    raise ParserError(
                        f"Could not repair production feedback CSV at {self.output_path}: inconsistent row length {len(row)}."
                    )
            repaired_frame = pd.DataFrame(repaired_rows, columns=repaired_header)
            repaired_frame = self._normalize_columns(repaired_frame)
            repaired_frame.to_csv(self.output_path, index=False)
            return repaired_frame

        raise ParserError(
            f"Could not repair production feedback CSV at {self.output_path}: unsupported header shape."
        )

    def _merge_frames(self, existing_frame: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
        normalized_existing = self._normalize_columns(existing_frame.copy())
        normalized_new = self._normalize_columns(new_rows.copy())
        if normalized_existing.empty:
            return normalized_new
        return pd.concat([normalized_existing, normalized_new], ignore_index=True)

    def _normalize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        for column in PREFERRED_COLUMN_ORDER:
            if column not in normalized.columns:
                normalized[column] = ""

        ordered_columns = [column for column in PREFERRED_COLUMN_ORDER if column in normalized.columns]
        remaining_columns = [column for column in normalized.columns if column not in ordered_columns]
        return normalized[ordered_columns + remaining_columns]
