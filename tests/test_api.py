from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.agent.explainer import AgentExplanationService
from src.agent.n8n_client import N8nNotificationClient
from src.api.main import (
    app,
    get_explanation_service,
    get_feedback_store,
    get_feedback_token_service,
    get_monitoring_service,
    get_n8n_notification_client,
    get_prediction_service,
)
from src.inference.service import FraudPredictionService
from src.models.artifact import FraudModelArtifact
from src.monitoring.evidently_service import MonitoringReportBundle
from src.notifications.feedback_tokens import FeedbackTokenService
from src.production.store import ProductionFeedbackStore


class DummyEstimator:
    def predict_proba(self, features):
        scores = (features["amount"].astype(float).to_numpy() > 500.0).astype(float)
        return np.column_stack([1.0 - scores, scores]).astype(float)


class FakeN8nNotificationClient(N8nNotificationClient):
    def __init__(self) -> None:
        super().__init__(webhook_url="http://n8n.local/webhook/fraud-notify")
        self.sent_payloads: list[dict] = []

    def send_notification(self, payload: dict) -> dict:
        self.sent_payloads.append(payload)
        return {
            "status": "accepted",
            "workflow_status": "email_sent",
            "email_subject": payload["email_subject"],
            "provider": "n8n",
        }


class FakeMonitoringService:
    reference_data_path = Path("data/raw/test_reference.csv")
    output_dir = Path("reports/evidently")

    def generate_reports(self, force: bool = False) -> MonitoringReportBundle:
        return MonitoringReportBundle(
            generated_at="2026-05-02T18:00:00+00:00",
            reference_rows=500,
            production_rows=50,
            data_drift_report_path="reports/evidently/data_drift_report.html",
            classification_report_path="reports/evidently/classification_report.html",
            data_drift_report_url="http://127.0.0.1:8000/monitoring/reports/data-drift",
            classification_report_url="http://127.0.0.1:8000/monitoring/reports/classification",
        )

    def read_report_html(self, report_name: str) -> str:
        return f"<html><body><h1>{report_name}</h1></body></html>"


class FraudApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prod_data_path = Path("data/production/test_api_prod_data.csv")
        if self.prod_data_path.exists():
            self.prod_data_path.unlink()

        artifact = FraudModelArtifact(
            estimator=DummyEstimator(),
            threshold=0.7,
            model_name="api_dummy_model",
        )
        self.service = FraudPredictionService(artifact)
        self.feedback_store = ProductionFeedbackStore(self.prod_data_path)
        self.explanation_service = AgentExplanationService(default_language="fr")
        self.token_service = FeedbackTokenService(secret="test-secret", ttl_seconds=3600)
        self.n8n_client = FakeN8nNotificationClient()
        self.monitoring_service = FakeMonitoringService()

        os.environ["FRAUD_PUBLIC_BASE_URL"] = "http://127.0.0.1:8000"

        app.dependency_overrides = {}
        get_prediction_service.cache_clear()
        get_feedback_store.cache_clear()
        get_explanation_service.cache_clear()
        get_feedback_token_service.cache_clear()
        get_monitoring_service.cache_clear()
        get_n8n_notification_client.cache_clear()
        app.dependency_overrides[get_prediction_service] = lambda: self.service
        app.dependency_overrides[get_feedback_store] = lambda: self.feedback_store
        app.dependency_overrides[get_explanation_service] = lambda: self.explanation_service
        app.dependency_overrides[get_feedback_token_service] = lambda: self.token_service
        app.dependency_overrides[get_monitoring_service] = lambda: self.monitoring_service
        app.dependency_overrides[get_n8n_notification_client] = lambda: self.n8n_client
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides = {}
        get_prediction_service.cache_clear()
        get_feedback_store.cache_clear()
        get_explanation_service.cache_clear()
        get_feedback_token_service.cache_clear()
        get_monitoring_service.cache_clear()
        get_n8n_notification_client.cache_clear()
        if self.prod_data_path.exists():
            os.remove(self.prod_data_path)

    def test_healthcheck_returns_agent_metadata(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["model_name"], "api_dummy_model")
        self.assertEqual(body["threshold"], 0.7)
        self.assertEqual(body["explanation_provider"], "template")
        self.assertEqual(body["agent_provider"], "n8n")
        self.assertTrue(body["agent_ready"])
        self.assertFalse(body["groq_in_fastapi_enabled"])
        self.assertIn("monitoring_reference_data_path", body)
        self.assertIn("monitoring_reports_dir", body)

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
        self.assertIn("prediction_id", body)
        self.assertEqual(body["prediction"], 1)
        self.assertEqual(body["predicted_label"], "fraud")
        self.assertEqual(body["model_name"], "api_dummy_model")

    def test_predict_batch_returns_summary(self) -> None:
        response = self.client.post(
            "/predict-batch",
            json={
                "transactions": [
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
                        "amount": 1000.0,
                        "oldbalanceOrg": 1000.0,
                        "newbalanceOrig": 0.0,
                        "oldbalanceDest": 0.0,
                        "newbalanceDest": 1000.0,
                        "isFlaggedFraud": 0,
                    },
                ]
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["summary"]["total_transactions"], 2)
        self.assertEqual(body["summary"]["fraud_predictions"], 1)
        self.assertEqual(len(body["predictions"]), 2)

    def test_feedback_persists_review_to_prod_data(self) -> None:
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 1000.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "isFlaggedFraud": 0,
        }
        prediction = self.client.post("/predict", json=transaction).json()

        response = self.client.post(
            "/feedback",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "user_feedback": "reported_fraud",
                "feedback_notes": "Customer denied this transfer.",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "saved")
        self.assertEqual(body["prediction_id"], prediction["prediction_id"])
        self.assertEqual(body["ground_truth_label"], 1)

    def test_explain_prediction_returns_template_payload(self) -> None:
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 1000.0,
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "isFlaggedFraud": 0,
        }
        prediction = self.client.post("/predict", json=transaction).json()

        response = self.client.post(
            "/explain-prediction",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "language": "fr",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["provider"], "template")
        self.assertIsNone(body["model"])
        self.assertGreaterEqual(len(body["reasons"]), 1)

    def test_notify_user_calls_n8n_and_returns_feedback_links(self) -> None:
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
        prediction = self.client.post("/predict", json=transaction).json()

        response = self.client.post(
            "/notify_user",
            json={
                "transaction": transaction,
                "prediction": prediction,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "accepted")
        self.assertEqual(body["workflow_provider"], "n8n")
        self.assertEqual(body["workflow_status"], "email_sent")
        self.assertEqual(body["recipient_email"], "client@bankmail.com")
        self.assertIn("/feedback-action?", body["confirm_legit_url"])
        self.assertEqual(len(self.n8n_client.sent_payloads), 1)

        sent_payload = self.n8n_client.sent_payloads[0]
        self.assertEqual(sent_payload["prediction_id"], prediction["prediction_id"])
        self.assertEqual(sent_payload["customer_email"], "client@bankmail.com")
        self.assertIn("confirm_url", sent_payload)
        self.assertIn("reject_url", sent_payload)
        self.assertIn("explanatory_factors", sent_payload)

    def test_notify_customer_alias_calls_same_n8n_workflow(self) -> None:
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
        prediction = self.client.post("/predict", json=transaction).json()

        response = self.client.post(
            "/notify-customer",
            json={
                "transaction": transaction,
                "prediction": prediction,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["workflow_provider"], "n8n")

    def test_notify_user_rejects_placeholder_email_domains(self) -> None:
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
        prediction = self.client.post("/predict", json=transaction).json()

        response = self.client.post(
            "/notify_user",
            json={
                "transaction": transaction,
                "prediction": prediction,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("example.com", response.json()["detail"])

    def test_feedback_action_saves_feedback_from_email_link(self) -> None:
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
        prediction = self.client.post("/predict", json=transaction).json()
        token = self.token_service.create_token({"transaction": transaction, "prediction": prediction})

        response = self.client.get(f"/feedback-action?token={token}&action=reported_fraud")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Fraude signalee avec succes", response.text)
        self.assertTrue(self.feedback_store.has_feedback(prediction["prediction_id"]))

    def test_feedback_records_returns_saved_rows(self) -> None:
        transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 400.0,
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 100.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 400.0,
            "isFlaggedFraud": 0,
        }
        prediction = self.client.post("/predict", json=transaction).json()
        self.client.post(
            "/feedback",
            json={
                "transaction": transaction,
                "prediction": prediction,
                "user_feedback": "confirmed_legit",
            },
        )

        response = self.client.get("/feedback-records?limit=10")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["prediction_id"], prediction["prediction_id"])

    def test_generate_monitoring_reports_returns_urls(self) -> None:
        response = self.client.get("/monitoring/reports/generate")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["reference_rows"], 500)
        self.assertEqual(body["production_rows"], 50)
        self.assertIn("/monitoring/reports/data-drift", body["data_drift_report_url"])
        self.assertIn("/monitoring/reports/classification", body["classification_report_url"])

    def test_serve_monitoring_report_returns_html(self) -> None:
        response = self.client.get("/monitoring/reports/data-drift")

        self.assertEqual(response.status_code, 200)
        self.assertIn("<h1>data-drift</h1>", response.text)


if __name__ == "__main__":
    unittest.main()
