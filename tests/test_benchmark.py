from __future__ import annotations

from pathlib import Path
import unittest

from sklearn.base import is_classifier

from src.models.benchmark import (
    BenchmarkConfig,
    CatBoostClassifier,
    SklearnCompatibleCatBoostClassifier,
    build_leaderboard,
    get_available_optional_models,
    get_candidate_model_specs,
)


class BenchmarkHelpersTest(unittest.TestCase):
    def test_optional_models_are_reported(self) -> None:
        availability = get_available_optional_models()

        self.assertIn("catboost", availability)
        self.assertIn("lightgbm", availability)
        self.assertIn("xgboost", availability)

    def test_catboost_wrapper_exposes_classifier_tags(self) -> None:
        if CatBoostClassifier is None:
            self.skipTest("CatBoost is not installed.")

        model = SklearnCompatibleCatBoostClassifier(
            verbose=False,
            allow_writing_files=False,
        )

        self.assertTrue(is_classifier(model))

    def test_candidate_spec_subset_is_respected(self) -> None:
        config = BenchmarkConfig(candidate_models=("logistic_regression", "random_forest"))

        specs = get_candidate_model_specs(config)

        self.assertEqual([spec.name for spec in specs], ["logistic_regression", "random_forest"])

    def test_leaderboard_is_sorted_by_cv_score(self) -> None:
        leaderboard = build_leaderboard(
            [
                {
                    "model_name": "model_b",
                    "cv_scores": {
                        "best_selection_score": 0.4,
                        "roc_auc": 0.9,
                        "average_precision": 0.4,
                        "fraud_f1": 0.5,
                    },
                    "threshold_selection": {
                        "metric": "fraud_f1",
                        "selected_threshold": 0.7,
                        "selected_metric_value": 0.4,
                    },
                    "validation_metrics": {
                        "fraud_f1": 0.4,
                        "fraud_precision": 0.5,
                        "fraud_recall": 0.6,
                        "average_precision": 0.3,
                    },
                    "holdout_metrics": {
                        "roc_auc": 0.8,
                        "average_precision": 0.3,
                        "fraud_f1": 0.4,
                        "fraud_precision": 0.5,
                        "fraud_recall": 0.6,
                        "threshold": 0.7,
                    },
                },
                {
                    "model_name": "model_a",
                    "cv_scores": {
                        "best_selection_score": 0.8,
                        "roc_auc": 0.95,
                        "average_precision": 0.8,
                        "fraud_f1": 0.7,
                    },
                    "threshold_selection": {
                        "metric": "fraud_f1",
                        "selected_threshold": 0.4,
                        "selected_metric_value": 0.8,
                    },
                    "validation_metrics": {
                        "fraud_f1": 0.8,
                        "fraud_precision": 0.7,
                        "fraud_recall": 0.9,
                        "average_precision": 0.5,
                    },
                    "holdout_metrics": {
                        "roc_auc": 0.85,
                        "average_precision": 0.5,
                        "fraud_f1": 0.6,
                        "fraud_precision": 0.7,
                        "fraud_recall": 0.8,
                        "threshold": 0.4,
                    },
                },
            ]
        )

        self.assertEqual(leaderboard["model_name"].tolist(), ["model_a", "model_b"])

    def test_split_train_validation_test_like_defaults_are_valid(self) -> None:
        config = BenchmarkConfig(validation_size=0.2, test_size=0.2)

        self.assertLess(config.validation_size + config.test_size, 1)

    def test_default_candidate_models_include_requested_families(self) -> None:
        config = BenchmarkConfig()

        self.assertIn("logistic_regression", config.candidate_models)
        self.assertIn("random_forest", config.candidate_models)
        self.assertIn("extra_trees", config.candidate_models)
        self.assertIn("gradient_boosting", config.candidate_models)
        self.assertIn("bagging_tree", config.candidate_models)
        self.assertIn("adaboost", config.candidate_models)
        self.assertIn("catboost", config.candidate_models)
        self.assertIn("lightgbm", config.candidate_models)
        self.assertIn("xgboost", config.candidate_models)
        self.assertEqual(config.pickle_output_path, Path("artifacts/best_fraud_model.pkl"))


if __name__ == "__main__":
    unittest.main()
