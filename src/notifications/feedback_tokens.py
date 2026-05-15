from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
import time
from typing import Any


FEEDBACK_TOKEN_SECRET_ENV_VAR = "FRAUD_FEEDBACK_TOKEN_SECRET"
FEEDBACK_TOKEN_TTL_ENV_VAR = "FRAUD_FEEDBACK_TOKEN_TTL_SECONDS"


def _urlsafe_b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


@dataclass(frozen=True)
class FeedbackTokenService:
    secret: str
    ttl_seconds: int = 7 * 24 * 60 * 60

    @classmethod
    def from_environment(cls) -> "FeedbackTokenService":
        secret = os.getenv(FEEDBACK_TOKEN_SECRET_ENV_VAR, "dev-feedback-secret-change-me")
        ttl_raw = os.getenv(FEEDBACK_TOKEN_TTL_ENV_VAR, str(7 * 24 * 60 * 60))
        try:
            ttl_seconds = max(int(ttl_raw), 60)
        except ValueError:
            ttl_seconds = 7 * 24 * 60 * 60
        return cls(secret=secret, ttl_seconds=ttl_seconds)

    def create_token(self, payload: dict[str, Any]) -> str:
        envelope = {
            "iat": int(time.time()),
            "exp": int(time.time()) + self.ttl_seconds,
            "payload": payload,
        }
        body = json.dumps(envelope, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        signature = hmac.new(self.secret.encode("utf-8"), body, hashlib.sha256).digest()
        return f"{_urlsafe_b64encode(body)}.{_urlsafe_b64encode(signature)}"

    def decode_token(self, token: str) -> dict[str, Any]:
        try:
            encoded_body, encoded_signature = token.split(".", 1)
        except ValueError as exc:
            raise ValueError("Invalid feedback token format.") from exc

        body = _urlsafe_b64decode(encoded_body)
        received_signature = _urlsafe_b64decode(encoded_signature)
        expected_signature = hmac.new(self.secret.encode("utf-8"), body, hashlib.sha256).digest()

        if not hmac.compare_digest(received_signature, expected_signature):
            raise ValueError("Invalid feedback token signature.")

        envelope = json.loads(body.decode("utf-8"))
        now = int(time.time())
        if int(envelope["exp"]) < now:
            raise ValueError("Feedback token has expired.")

        return envelope["payload"]

