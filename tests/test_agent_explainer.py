from __future__ import annotations

import unittest

from src.agent.explainer import AgentExplanationService


class AgentExplanationServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = AgentExplanationService(default_language="fr")
        self.transaction = {
            "step": 1,
            "type": "TRANSFER",
            "amount": 1200.0,
            "oldbalanceOrg": 1200.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
        }
        self.prediction = {
            "prediction_id": "pred-1",
            "prediction": 1,
            "predicted_label": "fraud",
            "fraud_probability": 0.91,
            "threshold": 0.5,
            "model_name": "random_forest",
        }

    def test_metadata_exposes_template_provider(self) -> None:
        metadata = self.service.metadata()

        self.assertEqual(metadata["active_provider"], "template")
        self.assertTrue(metadata["ready"])
        self.assertIsNone(metadata["model"])

    def test_explain_prediction_returns_french_payload(self) -> None:
        result = self.service.explain_prediction(self.transaction, self.prediction, language="fr")

        self.assertEqual(result.provider, "template")
        self.assertEqual(result.language, "fr")
        self.assertEqual(result.risk_level, "high")
        self.assertGreaterEqual(len(result.reasons), 1)
        self.assertEqual(result.recommended_action, "request_customer_confirmation")

    def test_explain_prediction_supports_english(self) -> None:
        result = self.service.explain_prediction(self.transaction, self.prediction, language="en")

        self.assertEqual(result.language, "en")
        self.assertIn("Fraud probability estimated", result.reasons[0])
        self.assertIn("Please confirm", result.customer_message)


if __name__ == "__main__":
    unittest.main()
