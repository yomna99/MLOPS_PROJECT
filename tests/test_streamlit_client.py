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

    @patch("app.client.requests.post")
    def test_submit_feedback_calls_expected_endpoint(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"status": "saved"}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
        }
        prediction = {
            "prediction_id": "pred-1",
            "prediction": 1,
            "predicted_label": "fraud",
            "fraud_probability": 0.91,
            "threshold": 0.5,
            "model_name": "random_forest",
        }

        result = client.submit_feedback(
            transaction=transaction,
            prediction=prediction,
            user_feedback="reported_fraud",
            feedback_notes="Customer denied it.",
        )

        self.assertEqual(result, {"status": "saved"})
        mock_post.assert_called_once_with(
            "http://fraud-api:8000/feedback",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "user_feedback": "reported_fraud",
                "feedback_notes": "Customer denied it.",
            },
            timeout=5,
        )

    @patch("app.client.requests.get")
    def test_feedback_summary_calls_expected_endpoint(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"total_feedback": 1}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        result = client.feedback_summary()

        self.assertEqual(result, {"total_feedback": 1})
        mock_get.assert_called_once_with("http://fraud-api:8000/feedback-summary", timeout=5)

    @patch("app.client.requests.get")
    def test_feedback_records_calls_expected_endpoint(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = [{"prediction_id": "pred-1"}]
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        result = client.feedback_records(limit=25)

        self.assertEqual(result, [{"prediction_id": "pred-1"}])
        mock_get.assert_called_once_with(
            "http://fraud-api:8000/feedback-records",
            params={"limit": 25},
            timeout=5,
        )

    @patch("app.client.requests.get")
    def test_generate_monitoring_reports_calls_expected_endpoint(self, mock_get: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"data_drift_report_url": "http://fraud-api:8000/monitoring/reports/data-drift"}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        result = client.generate_monitoring_reports(force_refresh=True)

        self.assertEqual(result["data_drift_report_url"], "http://fraud-api:8000/monitoring/reports/data-drift")
        mock_get.assert_called_once_with(
            "http://fraud-api:8000/monitoring/reports/generate",
            params={"force_refresh": True},
            timeout=180,
        )

    @patch("app.client.requests.post")
    def test_explain_prediction_calls_expected_endpoint(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"provider": "template"}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
        }
        prediction = {
            "prediction_id": "pred-1",
            "prediction": 1,
            "predicted_label": "fraud",
            "fraud_probability": 0.91,
            "threshold": 0.5,
            "model_name": "random_forest",
        }

        result = client.explain_prediction(transaction=transaction, prediction=prediction, language="fr")

        self.assertEqual(result, {"provider": "template"})
        mock_post.assert_called_once_with(
            "http://fraud-api:8000/explain-prediction",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "language": "fr",
            },
            timeout=5,
        )

    @patch("app.client.requests.post")
    def test_notify_customer_calls_expected_endpoint(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"status": "accepted", "workflow_provider": "n8n"}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
            "customer_email": "client@example.com",
        }
        prediction = {
            "prediction_id": "pred-1",
            "prediction": 1,
            "predicted_label": "fraud",
            "fraud_probability": 0.91,
            "threshold": 0.5,
            "model_name": "random_forest",
        }
        explanation = {
            "provider": "template",
            "model": None,
            "language": "fr",
            "risk_level": "medium",
            "reasons": ["probabilite fraud 53%"],
            "analyst_summary": "Alerte",
            "customer_message": "Veuillez confirmer cette operation.",
            "email_subject": "Verification de votre transaction",
            "recommended_action": "request_customer_confirmation",
        }

        result = client.notify_customer(transaction=transaction, prediction=prediction, explanation=explanation)

        self.assertEqual(result, {"status": "accepted", "workflow_provider": "n8n"})
        mock_post.assert_called_once_with(
            "http://fraud-api:8000/notify-customer",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "explanation": explanation,
            },
            timeout=5,
        )

    @patch("app.client.requests.post")
    def test_notify_user_calls_expected_endpoint(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.json.return_value = {"status": "accepted", "workflow_provider": "n8n"}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = FraudApiClient(base_url="http://fraud-api:8000", timeout_seconds=5)
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
            "customer_email": "client@bankmail.com",
        }
        prediction = {
            "prediction_id": "pred-1",
            "prediction": 1,
            "predicted_label": "fraud",
            "fraud_probability": 0.91,
            "threshold": 0.5,
            "model_name": "random_forest",
        }
        explanation = {
            "provider": "template",
            "model": None,
            "language": "fr",
            "risk_level": "medium",
            "reasons": ["probabilite fraud 53%"],
            "analyst_summary": "Alerte",
            "customer_message": "Veuillez confirmer cette operation.",
            "email_subject": "Verification de votre transaction",
            "recommended_action": "request_customer_confirmation",
        }

        result = client.notify_user(transaction=transaction, prediction=prediction, explanation=explanation)

        self.assertEqual(result, {"status": "accepted", "workflow_provider": "n8n"})
        mock_post.assert_called_once_with(
            "http://fraud-api:8000/notify_user",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "explanation": explanation,
            },
            timeout=5,
        )


if __name__ == "__main__":
    unittest.main()
