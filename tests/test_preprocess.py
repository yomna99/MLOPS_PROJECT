from __future__ import annotations

import unittest

import pandas as pd

from src.features.preprocess import prepare_training_frame


class PrepareTrainingFrameTest(unittest.TestCase):
    def test_drops_ids_and_extracts_target(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 100.0,
                    "nameOrig": "C1",
                    "oldbalanceOrg": 100.0,
                    "newbalanceOrig": 0.0,
                    "nameDest": "C2",
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "isFraud": 1,
                    "isFlaggedFraud": 0,
                }
            ]
        )

        features, target = prepare_training_frame(dataframe)

        self.assertNotIn("nameOrig", features.columns)
        self.assertNotIn("nameDest", features.columns)
        self.assertNotIn("isFraud", features.columns)
        self.assertEqual(target.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
