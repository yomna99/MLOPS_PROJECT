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
