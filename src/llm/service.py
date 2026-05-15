from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Literal

from openai import OpenAI, OpenAIError


LLM_DEFAULT_LANGUAGE_ENV_VAR = "FRAUD_LLM_DEFAULT_LANGUAGE"
GROQ_MODEL_ENV_VAR = "FRAUD_GROQ_MODEL"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SUPPORTED_LANGUAGES = {"fr", "en"}


@dataclass(frozen=True)
class FraudExplanationResult:
    provider: str
    model: str
    language: Literal["fr", "en"]
    risk_level: Literal["low", "medium", "high"]
    reasons: list[str]
    analyst_summary: str
    customer_message: str
    email_subject: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "language": self.language,
            "risk_level": self.risk_level,
            "reasons": self.reasons,
            "analyst_summary": self.analyst_summary,
            "customer_message": self.customer_message,
            "email_subject": self.email_subject,
            "recommended_action": self.recommended_action,
        }


class FraudNarrativeService:
    """Generate Groq-based narratives around model predictions."""

    def __init__(
        self,
        default_language: Literal["fr", "en"] = "fr",
        groq_model: str = "openai/gpt-oss-20b",
        groq_client: Any | None = None,
    ) -> None:
        self.default_language = default_language
        self.groq_model = groq_model
        self._groq_client = groq_client

    @classmethod
    def from_environment(cls) -> "FraudNarrativeService":
        language = os.getenv(LLM_DEFAULT_LANGUAGE_ENV_VAR, "fr").strip().lower() or "fr"
        groq_model = os.getenv(GROQ_MODEL_ENV_VAR, "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"
        if language not in SUPPORTED_LANGUAGES:
            language = "fr"
        return cls(default_language=language, groq_model=groq_model)

    def metadata(self) -> dict[str, Any]:
        return {
            "active_provider": "groq",
            "default_language": self.default_language,
            "model": self.groq_model,
            "ready": self._is_groq_ready(),
        }

    def explain_prediction(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        language: Literal["fr", "en"] | None = None,
    ) -> FraudExplanationResult:
        selected_language = language or self.default_language
        if selected_language not in SUPPORTED_LANGUAGES:
            selected_language = self.default_language
        return self._build_groq_explanation(transaction, prediction, selected_language)

    def _is_groq_ready(self) -> bool:
        return self._groq_client is not None or bool(os.getenv(GROQ_API_KEY_ENV_VAR))

    def _get_groq_client(self) -> Any:
        if self._groq_client is not None:
            return self._groq_client
        api_key = os.getenv(GROQ_API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")
        self._groq_client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        return self._groq_client

    def _build_groq_explanation(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        language: Literal["fr", "en"],
    ) -> FraudExplanationResult:
        client = self._get_groq_client()
        response = client.responses.create(
            model=self.groq_model,
            instructions=self._build_groq_instructions(language),
            input=self._build_groq_input(transaction, prediction, language),
        )
        parsed = self._parse_provider_response(response.output_text)

        return FraudExplanationResult(
            provider="groq",
            model=self.groq_model,
            language=language,
            risk_level=self._normalize_risk_level(parsed.get("risk_level")),
            reasons=self._normalize_reasons(parsed.get("reasons")),
            analyst_summary=str(parsed["analyst_summary"]).strip(),
            customer_message=str(parsed["customer_message"]).strip(),
            email_subject=str(parsed["email_subject"]).strip(),
            recommended_action=str(parsed["recommended_action"]).strip(),
        )

    def _build_groq_instructions(self, language: Literal["fr", "en"]) -> str:
        language_name = "French" if language == "fr" else "English"
        return (
            "You are a fraud operations communication assistant for a banking platform. "
            f"Respond only in {language_name}. "
            "Return valid JSON only, with no markdown and no extra text. "
            "Use exactly this schema: "
            '{"risk_level":"low|medium|high","reasons":["short reason"],'
            '"analyst_summary":"...","customer_message":"...",'
            '"email_subject":"...","recommended_action":"..."}'
        )

    def _build_groq_input(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        language: Literal["fr", "en"],
    ) -> str:
        payload = {
            "task": "Explain a fraud prediction for an analyst and a customer notification workflow.",
            "language": language,
            "constraints": {
                "max_reasons": 4,
                "reasons_style": "short and concrete",
                "customer_message_style": "clear, calm, and actionable",
                "recommended_action_options": [
                    "allow_and_monitor",
                    "request_customer_confirmation",
                    "contact_customer_and_hold_transaction",
                ],
            },
            "transaction": transaction,
            "prediction": prediction,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _parse_provider_response(self, raw_text: str) -> dict[str, Any]:
        candidate = raw_text.strip()
        if candidate.startswith("```"):
            candidate = candidate.strip("`")
            if "\n" in candidate:
                candidate = candidate.split("\n", 1)[1]
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1:
            candidate = candidate[start : end + 1]

        parsed = json.loads(candidate)
        required_keys = {
            "risk_level",
            "reasons",
            "analyst_summary",
            "customer_message",
            "email_subject",
            "recommended_action",
        }
        missing = required_keys.difference(parsed)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"Groq explanation is missing keys: {missing_keys}")
        return parsed

    def _normalize_risk_level(self, value: Any) -> Literal["low", "medium", "high"]:
        normalized = str(value).strip().lower()
        if normalized not in {"low", "medium", "high"}:
            raise ValueError(f"Unsupported risk level returned by provider: {value}")
        return normalized

    def _normalize_reasons(self, value: Any) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("Provider reasons must be a non-empty list.")
        normalized = [str(reason).strip() for reason in value if str(reason).strip()]
        if not normalized:
            raise ValueError("Provider reasons must contain at least one non-empty item.")
        return normalized


__all__ = ["FraudNarrativeService", "FraudExplanationResult", "OpenAIError"]
