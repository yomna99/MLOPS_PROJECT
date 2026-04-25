from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.client import FraudApiClient


class FraudApiClientTest(unittest.TestCase):
    @patch("app.client.requests.get")
    def test_healthcheck_calls_expected_endpoint(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"status": "ok"}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        result = client.healthcheck()

        self.assertEqual(result, {"status": "ok"})
        mock_get.assert_called_once_with("http://fraud-api:8000/health", timeout=5)

    @patch("app.client.requests.post")
    def test_predict_calls_expected_endpoint(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"prediction": 0}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        payload = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
        }
        result = client.predict(payload)

        self.assertEqual(result, {"prediction": 0})
        mock_post.assert_called_once_with(
            "http://fraud-api:8000/predict",
            json=payload,
            timeout=5,
        )

    @patch("app.client.requests.post")
    def test_predict_batch_calls_expected_endpoint(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"summary": {"total_transactions": 2}}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        payloads = [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 2,
                "type": "PAYMENT",
                "amount": 50.0,
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 50.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 50.0,
                "isFlaggedFraud": 0,
            },
        ]
        result = client.predict_batch(payloads)

        self.assertEqual(result, {"summary": {"total_transactions": 2}})
        mock_post.assert_called_once_with(
            "http://fraud-api:8000/predict-batch",
            json={"transactions": payloads},
            timeout=5,
        )


if __name__ == "__main__":
    unittest.main()
