from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
API_URL_ENV_VAR = "FRAUD_API_URL"


@dataclass(frozen=True)
class FraudApiClient:
    """Small HTTP client used by the Streamlit frontend."""

    base_url: str = DEFAULT_API_URL
    timeout_seconds: int = 15

    @classmethod
    def from_environment(cls) -> "FraudApiClient":
        return cls(base_url=os.getenv(API_URL_ENV_VAR, DEFAULT_API_URL))

    def healthcheck(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/predict-batch",
            json={"transactions": payloads},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def submit_feedback(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        user_feedback: str,
        feedback_notes: str | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/feedback",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "user_feedback": user_feedback,
                "feedback_notes": feedback_notes,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def feedback_summary(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/feedback-summary", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def feedback_records(self, limit: int = 50) -> list[dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/feedback-records",
            params={"limit": limit},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def generate_monitoring_reports(self, force_refresh: bool = False) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/monitoring/reports/generate",
            params={"force_refresh": force_refresh},
            timeout=max(self.timeout_seconds, 180),
        )
        response.raise_for_status()
        return response.json()

    def monitoring_reports_summary(self) -> dict[str, Any] | None:
        response = requests.get(
            f"{self.base_url}/monitoring/reports/summary",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def explain_prediction(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        language: str = "fr",
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/explain-prediction",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "language": language,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def notify_customer(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        explanation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "transaction": transaction,
            "prediction": prediction,
        }
        if explanation is not None:
            payload["explanation"] = explanation
        response = requests.post(
            f"{self.base_url}/notify-customer",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def notify_user(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        explanation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "transaction": transaction,
            "prediction": prediction,
        }
        if explanation is not None:
            payload["explanation"] = explanation
        response = requests.post(
            f"{self.base_url}/notify_user",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
