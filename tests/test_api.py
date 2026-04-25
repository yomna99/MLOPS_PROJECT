from __future__ import annotations

import unittest

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import app, get_prediction_service
from src.models.artifact import FraudModelArtifact
from src.inference.service import FraudPredictionService


class DummyEstimator:
    def predict_proba(self, features):
        score = 1.0 if float(features["amount"].iloc[0]) > 500.0 else 0.0
        return np.array([[1.0 - score, score]], dtype=float)


class FraudApiTest(unittest.TestCase):
    def setUp(self) -> None:
        artifact = FraudModelArtifact(
            estimator=DummyEstimator(),
            threshold=0.7,
            model_name="api_dummy_model",
        )
        self.service = FraudPredictionService(artifact)
        app.dependency_overrides = {}
        get_prediction_service.cache_clear()
        app.dependency_overrides[get_prediction_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides = {}
        get_prediction_service.cache_clear()

    def test_healthcheck_returns_model_metadata(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model_name"], "api_dummy_model")
        self.assertEqual(body["threshold"], 0.7)

    def test_predict_returns_fraud_response(self) -> None:
        response = self.client.post(
            "/predict",
            json={
                "step": 1,
                "type": "TRANSFER",
                "amount": 1000.0,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 1000.0,
                "isFlaggedFraud": 0,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["prediction"], 1)
        self.assertEqual(body["predicted_label"], "fraud")
        self.assertEqual(body["model_name"], "api_dummy_model")


if __name__ == "__main__":
    unittest.main()
