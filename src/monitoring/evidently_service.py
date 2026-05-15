from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.features.preprocess import FraudFeatureSpec, prepare_training_frame
from src.inference.service import FraudPredictionService


REFERENCE_DATA_PATH_ENV_VAR = "FRAUD_REFERENCE_DATA_PATH"
MONITORING_OUTPUT_DIR_ENV_VAR = "FRAUD_MONITORING_OUTPUT_DIR"
DEFAULT_REFERENCE_DATA_PATH = Path("data/raw/AIML Dataset.csv")
DEFAULT_MONITORING_OUTPUT_DIR = Path("reports/evidently")
DEFAULT_REFERENCE_SAMPLE_SIZE = 2000
DEFAULT_REFERENCE_CHUNK_SIZE = 10000
DEFAULT_REFERENCE_MAX_CHUNKS = 12

REPORT_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]
REPORT_COLUMNS = [
    *REPORT_FEATURE_COLUMNS,
    "ground_truth_label",
    "prediction",
    "fraud_probability",
]
SUPPORTED_REPORT_NAMES = {
    "data-drift": "data_drift_report.html",
    "classification": "classification_report.html",
}
SUMMARY_FILE_NAME = "monitoring_summary.json"


@dataclass(frozen=True)
class MonitoringReportBundle:
    generated_at: str
    reference_rows: int
    production_rows: int
    data_drift_report_path: str
    classification_report_path: str
    data_drift_report_url: str
    classification_report_url: str
    drift_detected: bool | None = None
    drift_threshold: float | None = None
    total_columns: int | None = None
    drifted_columns: int | None = None
    share_drifted_columns: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "reference_rows": self.reference_rows,
            "production_rows": self.production_rows,
            "data_drift_report_path": self.data_drift_report_path,
            "classification_report_path": self.classification_report_path,
            "data_drift_report_url": self.data_drift_report_url,
            "classification_report_url": self.classification_report_url,
            "drift_detected": self.drift_detected,
            "drift_threshold": self.drift_threshold,
            "total_columns": self.total_columns,
            "drifted_columns": self.drifted_columns,
            "share_drifted_columns": self.share_drifted_columns,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


class EvidentlyMonitoringService:
    """Generate Evidently HTML reports for drift and classification quality."""

    def __init__(
        self,
        prediction_service: FraudPredictionService,
        reference_data_path: Path | str = DEFAULT_REFERENCE_DATA_PATH,
        production_data_path: Path | str = Path("data/production/prod_data.csv"),
        output_dir: Path | str = DEFAULT_MONITORING_OUTPUT_DIR,
        public_base_url: str = "http://127.0.0.1:8000",
        reference_sample_size: int = DEFAULT_REFERENCE_SAMPLE_SIZE,
    ) -> None:
        self.prediction_service = prediction_service
        self.reference_data_path = Path(reference_data_path)
        self.production_data_path = Path(production_data_path)
        self.output_dir = Path(output_dir)
        self.public_base_url = public_base_url.rstrip("/")
        self.reference_sample_size = int(reference_sample_size)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(
        cls,
        prediction_service: FraudPredictionService,
        production_data_path: Path | str,
        public_base_url: str,
    ) -> "EvidentlyMonitoringService":
        reference_path = Path(os.getenv(REFERENCE_DATA_PATH_ENV_VAR, str(DEFAULT_REFERENCE_DATA_PATH)))
        output_dir = Path(os.getenv(MONITORING_OUTPUT_DIR_ENV_VAR, str(DEFAULT_MONITORING_OUTPUT_DIR)))
        return cls(
            prediction_service=prediction_service,
            reference_data_path=reference_path,
            production_data_path=production_data_path,
            output_dir=output_dir,
            public_base_url=public_base_url,
        )

    def generate_reports(self, force: bool = False) -> MonitoringReportBundle:
        data_drift_path = self.output_dir / SUPPORTED_REPORT_NAMES["data-drift"]
        classification_path = self.output_dir / SUPPORTED_REPORT_NAMES["classification"]
        summary_path = self.output_dir / SUMMARY_FILE_NAME
        current_frame = self._load_current_frame()
        reference_frame = self._build_reference_frame()

        if force or not data_drift_path.exists() or not classification_path.exists():
            self._generate_evidently_html(
                current_frame=current_frame,
                reference_frame=reference_frame,
                data_drift_path=data_drift_path,
                classification_path=classification_path,
            )

        drift_summary = self._extract_drift_summary(data_drift_path)
        performance_summary = self._compute_classification_summary(current_frame)
        generated_at = datetime.now(timezone.utc).isoformat()
        bundle = MonitoringReportBundle(
            generated_at=generated_at,
            reference_rows=int(len(reference_frame)),
            production_rows=int(len(current_frame)),
            data_drift_report_path=str(data_drift_path),
            classification_report_path=str(classification_path),
            data_drift_report_url=f"{self.public_base_url}/monitoring/reports/data-drift",
            classification_report_url=f"{self.public_base_url}/monitoring/reports/classification",
            **drift_summary,
            **performance_summary,
        )
        summary_path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")
        return bundle

    def read_report_html(self, report_name: str) -> str:
        output_name = SUPPORTED_REPORT_NAMES.get(report_name)
        if output_name is None:
            raise ValueError(f"Unsupported report name: {report_name}")

        path = self.output_dir / output_name
        if not path.exists():
            self.generate_reports(force=True)
        if not path.exists():
            raise FileNotFoundError(f"Monitoring report was not generated: {path}")
        return path.read_text(encoding="utf-8")

    def read_summary(self) -> MonitoringReportBundle | None:
        summary_path = self.output_dir / SUMMARY_FILE_NAME
        if not summary_path.exists():
            return None
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return MonitoringReportBundle(**payload)

    def _load_current_frame(self) -> pd.DataFrame:
        if not self.production_data_path.exists():
            raise FileNotFoundError(
                f"Production dataset not found at {self.production_data_path}. Collect feedback before running monitoring."
            )

        frame = pd.read_csv(self.production_data_path)
        if frame.empty:
            raise ValueError("Production dataset is empty. Collect feedback before running monitoring.")

        missing = [column for column in REPORT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Production dataset is missing required columns: {missing}")

        current = frame[REPORT_COLUMNS].copy()
        return self._normalize_report_frame(current)

    def _build_reference_frame(self) -> pd.DataFrame:
        if not self.reference_data_path.exists():
            raise FileNotFoundError(
                f"Reference dataset not found at {self.reference_data_path}. Mount or provide the raw training data."
            )

        raw_frame = self._load_reference_raw_frame()
        feature_spec = self.prediction_service.artifact.feature_spec
        features, target = prepare_training_frame(raw_frame, feature_spec)

        reference_eval = features[list(REPORT_FEATURE_COLUMNS)].copy()
        reference_eval["ground_truth_label"] = target.astype(int).values
        reference_eval = self._sample_reference_frame(reference_eval)

        reference_features = reference_eval[list(REPORT_FEATURE_COLUMNS)].copy()
        reference_eval["prediction"] = (
            self.prediction_service.artifact.predict(reference_features).astype(int).values
        )
        reference_eval["fraud_probability"] = (
            self.prediction_service.artifact.predict_scores(reference_features).astype(float).values
        )
        return self._normalize_report_frame(reference_eval[REPORT_COLUMNS].copy())

    def _sample_reference_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= self.reference_sample_size:
            return frame.reset_index(drop=True)

        sampled, _ = train_test_split(
            frame,
            train_size=self.reference_sample_size,
            stratify=frame["ground_truth_label"],
            random_state=42,
        )
        return sampled.reset_index(drop=True)

    def _load_reference_raw_frame(self) -> pd.DataFrame:
        use_columns = [
            "isFraud",
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFlaggedFraud",
            "nameOrig",
            "nameDest",
        ]
        sampled_chunks: list[pd.DataFrame] = []
        per_chunk_sample = max(self.reference_sample_size // 4, 250)

        for chunk_index, chunk in enumerate(
            pd.read_csv(self.reference_data_path, usecols=use_columns, chunksize=DEFAULT_REFERENCE_CHUNK_SIZE)
        ):
            if chunk.empty:
                continue

            sample_size = min(len(chunk), per_chunk_sample)
            sampled_chunks.append(chunk.sample(n=sample_size, random_state=42 + chunk_index))

            if len(sampled_chunks) >= DEFAULT_REFERENCE_MAX_CHUNKS:
                break

        if not sampled_chunks:
            raise ValueError(f"Reference dataset at {self.reference_data_path} is empty.")

        return pd.concat(sampled_chunks, ignore_index=True)

    def _normalize_report_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        numeric_columns = [
            "step",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "isFlaggedFraud",
            "ground_truth_label",
            "prediction",
            "fraud_probability",
        ]
        for column in numeric_columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized["type"] = normalized["type"].astype(str)
        normalized = normalized.dropna(subset=["ground_truth_label", "prediction", "fraud_probability"])
        normalized["ground_truth_label"] = normalized["ground_truth_label"].astype(int)
        normalized["prediction"] = normalized["prediction"].astype(int)
        normalized["isFlaggedFraud"] = normalized["isFlaggedFraud"].astype(int)
        return normalized.reset_index(drop=True)

    def _generate_evidently_html(
        self,
        *,
        current_frame: pd.DataFrame,
        reference_frame: pd.DataFrame,
        data_drift_path: Path,
        classification_path: Path,
    ) -> None:
        try:
            from evidently import BinaryClassification, DataDefinition, Dataset, Report
            from evidently.presets import ClassificationPreset, DataDriftPreset
        except ImportError as exc:
            raise RuntimeError(
                "Evidently is not installed. Add the 'evidently' package to the API environment before generating reports."
            ) from exc

        classification_definition = BinaryClassification(
            target="ground_truth_label",
            prediction_labels="prediction",
            prediction_probas="fraud_probability",
            pos_label=1,
        )
        data_definition = DataDefinition(
            numerical_columns=[
                "step",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
                "isFlaggedFraud",
            ],
            categorical_columns=["type"],
            classification=[classification_definition],
        )

        current_dataset = Dataset.from_pandas(current_frame, data_definition=data_definition)
        reference_dataset = Dataset.from_pandas(reference_frame, data_definition=data_definition)

        try:
            data_drift_report = Report(metrics=[DataDriftPreset()])
        except TypeError:
            data_drift_report = Report([DataDriftPreset()])
        try:
            data_drift_result = data_drift_report.run(
                current_data=current_dataset,
                reference_data=reference_dataset,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Data Drift report generation failed: {type(exc).__name__}: {exc!r}"
            ) from exc
        self._save_report_html(data_drift_result, data_drift_path)

        try:
            classification_report = Report(metrics=[ClassificationPreset()])
        except TypeError:
            classification_report = Report([ClassificationPreset()])
        try:
            classification_result = classification_report.run(
                current_data=current_dataset,
                reference_data=None,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Classification report generation failed: {type(exc).__name__}: {exc!r}"
            ) from exc
        self._save_report_html(classification_result, classification_path)

    def _save_report_html(self, report_result: Any, output_path: Path) -> None:
        if hasattr(report_result, "save_html"):
            report_result.save_html(str(output_path))
            return
        if hasattr(report_result, "html") and callable(report_result.html):
            output_path.write_text(report_result.html(), encoding="utf-8")
            return
        render_html = getattr(report_result, "_repr_html_", None)
        if callable(render_html):
            output_path.write_text(render_html(), encoding="utf-8")
            return
        raise RuntimeError("Unsupported Evidently report object: unable to export HTML.")

    def _extract_drift_summary(self, report_path: Path) -> dict[str, Any]:
        html = report_path.read_text(encoding="utf-8")
        compact = re.sub(r"<[^>]+>", " ", html)
        compact = re.sub(r"\s+", " ", compact)

        detected_match = re.search(r"Dataset Drift is (detected|not detected)", compact, re.IGNORECASE)
        threshold_match = re.search(r"threshold is ([0-9]+(?:\.[0-9]+)?)", compact, re.IGNORECASE)
        total_match = re.search(r"([0-9]+)\s+Columns", compact)
        drifted_match = re.search(r"([0-9]+)\s+Drifted Columns", compact)
        share_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s+Share of Drifted Columns", compact)

        return {
            "drift_detected": None if not detected_match else detected_match.group(1).lower() == "detected",
            "drift_threshold": None if not threshold_match else float(threshold_match.group(1)),
            "total_columns": None if not total_match else int(total_match.group(1)),
            "drifted_columns": None if not drifted_match else int(drifted_match.group(1)),
            "share_drifted_columns": None if not share_match else float(share_match.group(1)),
        }

    def _compute_classification_summary(self, current_frame: pd.DataFrame) -> dict[str, float]:
        y_true = current_frame["ground_truth_label"].astype(int)
        y_pred = current_frame["prediction"].astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }
