from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.agent.n8n_client import AgentWorkflowError, N8nNotificationClient


class N8nNotificationClientTest(unittest.TestCase):
    def test_metadata_reflects_webhook_readiness(self) -> None:
        client = N8nNotificationClient(webhook_url="http://n8n.local/webhook/fraud")

        metadata = client.metadata()

        self.assertEqual(metadata["provider"], "n8n")
        self.assertTrue(metadata["ready"])
        self.assertEqual(metadata["webhook_url"], "http://n8n.local/webhook/fraud")

    @patch("src.agent.n8n_client.requests.post")
    def test_send_notification_posts_json_payload(self, mock_post: MagicMock) -> None:
        response = MagicMock()
        response.content = b'{"status":"accepted"}'
        response.json.return_value = {"status": "accepted"}
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        client = N8nNotificationClient(
            webhook_url="http://n8n.local/webhook/fraud",
            auth_header_name="X-Webhook-Token",
            auth_header_value="secret",
        )
        payload = {"prediction_id": "pred-1"}

        result = client.send_notification(payload)

        self.assertEqual(result, {"status": "accepted"})
        mock_post.assert_called_once_with(
            "http://n8n.local/webhook/fraud",
            json=payload,
            headers={"Content-Type": "application/json", "X-Webhook-Token": "secret"},
            timeout=20,
        )

    def test_send_notification_requires_webhook_url(self) -> None:
        client = N8nNotificationClient(webhook_url=None)

        with self.assertRaises(AgentWorkflowError):
            client.send_notification({"prediction_id": "pred-1"})


if __name__ == "__main__":
    unittest.main()
