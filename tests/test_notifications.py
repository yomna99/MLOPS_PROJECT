from __future__ import annotations

import unittest

from src.notifications.feedback_tokens import FeedbackTokenService


class FeedbackTokenServiceTest(unittest.TestCase):
    def test_create_and_decode_token_round_trip(self) -> None:
        service = FeedbackTokenService(secret="unit-test-secret", ttl_seconds=3600)
        payload = {"transaction": {"customer_email": "client@example.com"}, "prediction": {"prediction_id": "pred-1"}}

        token = service.create_token(payload)
        decoded = service.decode_token(token)

        self.assertEqual(decoded, payload)


if __name__ == "__main__":
    unittest.main()
