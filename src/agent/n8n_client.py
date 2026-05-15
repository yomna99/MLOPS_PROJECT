from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests


N8N_NOTIFY_WEBHOOK_URL_ENV_VAR = "N8N_NOTIFY_WEBHOOK_URL"
N8N_WEBHOOK_AUTH_HEADER_NAME_ENV_VAR = "N8N_WEBHOOK_AUTH_HEADER_NAME"
N8N_WEBHOOK_AUTH_HEADER_VALUE_ENV_VAR = "N8N_WEBHOOK_AUTH_HEADER_VALUE"
N8N_WEBHOOK_TIMEOUT_SECONDS_ENV_VAR = "N8N_WEBHOOK_TIMEOUT_SECONDS"


class AgentWorkflowError(RuntimeError):
    """Raised when the n8n notification workflow cannot be executed."""


@dataclass(frozen=True)
class N8nNotificationClient:
    webhook_url: str | None
    timeout_seconds: int = 20
    auth_header_name: str | None = None
    auth_header_value: str | None = None

    @classmethod
    def from_environment(cls) -> "N8nNotificationClient":
        timeout_raw = os.getenv(N8N_WEBHOOK_TIMEOUT_SECONDS_ENV_VAR, "20")
        try:
            timeout_seconds = max(int(timeout_raw), 5)
        except ValueError:
            timeout_seconds = 20
        return cls(
            webhook_url=os.getenv(N8N_NOTIFY_WEBHOOK_URL_ENV_VAR),
            timeout_seconds=timeout_seconds,
            auth_header_name=os.getenv(N8N_WEBHOOK_AUTH_HEADER_NAME_ENV_VAR),
            auth_header_value=os.getenv(N8N_WEBHOOK_AUTH_HEADER_VALUE_ENV_VAR),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "n8n",
            "ready": bool(self.webhook_url),
            "webhook_url": self.webhook_url,
        }

    def send_notification(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.webhook_url:
            raise AgentWorkflowError("N8N_NOTIFY_WEBHOOK_URL is not configured.")

        headers = {"Content-Type": "application/json"}
        if self.auth_header_name and self.auth_header_value:
            headers[self.auth_header_name] = self.auth_header_value

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise AgentWorkflowError(f"n8n workflow call failed: {exc}") from exc

        if not response.content:
            return {"status": "accepted", "provider": "n8n"}

        try:
            return response.json()
        except ValueError:
            return {"status": "accepted", "provider": "n8n", "raw_response": response.text}
