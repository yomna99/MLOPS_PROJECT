from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.artifact import FraudModelArtifact
from src.inference.service import FraudPredictionService


class DummyEstimator:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        scores = (features["amount"].to_numpy(dtype=float) > 500.0).astype(float)
        probabilities = np.column_stack([1.0 - scores, scores])
        return probabilities


class FraudPredictionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        artifact = FraudModelArtifact(
            estimator=DummyEstimator(),
            threshold=0.7,
            model_name="dummy_model",
        )
        self.service = FraudPredictionService(artifact)

    def test_predict_returns_not_fraud_for_low_amount(self) -> None:
        result = self.service.predict(
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 100.0,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 100.0,
                "isFlaggedFraud": 0,
            }
        )

        self.assertEqual(result.prediction, 0)
        self.assertEqual(result.predicted_label, "not_fraud")
        self.assertEqual(result.threshold, 0.7)
        self.assertTrue(result.prediction_id)

    def test_predict_returns_fraud_for_high_amount(self) -> None:
        result = self.service.predict(
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 900.0,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 100.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 900.0,
                "isFlaggedFraud": 0,
            }
        )

        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.predicted_label, "fraud")
        self.assertEqual(result.model_name, "dummy_model")

    def test_predict_raises_when_feature_is_missing(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing numeric feature: amount"):
            self.service.predict(
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 100.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 900.0,
                    "isFlaggedFraud": 0,
                }
            )

    def test_predict_batch_returns_dataframe_with_predictions(self) -> None:
        result = self.service.predict_batch(
            [
                {
                    "step": 1,
                    "type": "PAYMENT",
                    "amount": 100.0,
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 900.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 100.0,
                    "isFlaggedFraud": 0,
                },
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 900.0,
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 100.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 900.0,
                    "isFlaggedFraud": 0,
                },
            ]
        )

        self.assertEqual(result["prediction"].tolist(), [0, 1])
        self.assertEqual(result["predicted_label"].tolist(), ["not_fraud", "fraud"])
        self.assertEqual(len(result["prediction_id"]), 2)


if __name__ == "__main__":
    unittest.main()
