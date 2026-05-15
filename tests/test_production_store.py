from __future__ import annotations

import unittest
from pathlib import Path
import os

import pandas as pd

from src.production.store import ProductionFeedbackStore


class ProductionFeedbackStoreTest(unittest.TestCase):
    def test_append_feedback_creates_prod_data_csv_and_updates_summary(self) -> None:
        output_path = Path("data/production/test_store_prod_data.csv")
        if output_path.exists():
            output_path.unlink()
        try:
            store = ProductionFeedbackStore(output_path)

            record = store.append_feedback(
                transaction={
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 1000.0,
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 1000.0,
                    "isFlaggedFraud": 0,
                },
                prediction={
                    "prediction_id": "pred-123",
                    "prediction": 1,
                    "predicted_label": "fraud",
                    "fraud_probability": 0.93,
                    "threshold": 0.5,
                    "model_name": "random_forest",
                },
                user_feedback="reported_fraud",
                feedback_notes="Customer denied this transaction.",
            )

            self.assertEqual(record["ground_truth_label"], 1)
            self.assertTrue(store.output_path.exists())

            saved_frame = pd.read_csv(store.output_path)
            self.assertEqual(len(saved_frame), 1)
            self.assertEqual(saved_frame.loc[0, "prediction_id"], "pred-123")
            self.assertEqual(saved_frame.loc[0, "user_feedback"], "reported_fraud")

            summary = store.summary()
            self.assertEqual(summary["total_feedback"], 1)
            self.assertEqual(summary["reported_fraud"], 1)
        finally:
            if output_path.exists():
                os.remove(output_path)

    def test_append_feedback_handles_schema_evolution_for_customer_email(self) -> None:
        output_path = Path("data/production/test_store_schema_prod_data.csv")
        if output_path.exists():
            output_path.unlink()
        try:
            store = ProductionFeedbackStore(output_path)

            store.append_feedback(
                transaction={
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 1000.0,
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 1000.0,
                    "isFlaggedFraud": 0,
                },
                prediction={
                    "prediction_id": "pred-old",
                    "prediction": 1,
                    "predicted_label": "fraud",
                    "fraud_probability": 0.93,
                    "threshold": 0.5,
                    "model_name": "random_forest",
                },
                user_feedback="reported_fraud",
            )

            store.append_feedback(
                transaction={
                    "step": 2,
                    "type": "PAYMENT",
                    "amount": 50.0,
                    "oldbalanceOrg": 100.0,
                    "newbalanceOrig": 50.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 50.0,
                    "isFlaggedFraud": 0,
                    "customer_email": "person@example.com",
                },
                prediction={
                    "prediction_id": "pred-new",
                    "prediction": 0,
                    "predicted_label": "not_fraud",
                    "fraud_probability": 0.2,
                    "threshold": 0.5,
                    "model_name": "random_forest",
                },
                user_feedback="confirmed_legit",
            )

            saved_frame = pd.read_csv(output_path)
            self.assertEqual(len(saved_frame), 2)
            self.assertIn("customer_email", saved_frame.columns)
            self.assertEqual(saved_frame.loc[0, "prediction_id"], "pred-old")
            self.assertTrue(pd.isna(saved_frame.loc[0, "customer_email"]) or saved_frame.loc[0, "customer_email"] == "")
            self.assertEqual(saved_frame.loc[1, "customer_email"], "person@example.com")
        finally:
            if output_path.exists():
                os.remove(output_path)


if __name__ == "__main__":
    unittest.main()
