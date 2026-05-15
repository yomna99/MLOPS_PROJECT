from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


SUPPORTED_LANGUAGES = {"fr", "en"}


@dataclass(frozen=True)
class AgentExplanationResult:
    provider: str
    model: str | None
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


class AgentExplanationService:
    """Build deterministic explanation payloads without calling an LLM."""

    def __init__(self, default_language: Literal["fr", "en"] = "fr") -> None:
        self.default_language = default_language if default_language in SUPPORTED_LANGUAGES else "fr"

    def metadata(self) -> dict[str, Any]:
        return {
            "active_provider": "template",
            "model": None,
            "ready": True,
            "default_language": self.default_language,
        }

    def explain_prediction(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        language: Literal["fr", "en"] | None = None,
    ) -> AgentExplanationResult:
        selected_language = language or self.default_language
        if selected_language not in SUPPORTED_LANGUAGES:
            selected_language = self.default_language

        probability = float(prediction["fraud_probability"])
        is_fraud = int(prediction["prediction"]) == 1
        if probability >= 0.8:
            risk_level: Literal["low", "medium", "high"] = "high"
        elif probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        reasons = self._build_reasons(transaction, prediction, selected_language)
        analyst_summary = self._build_analyst_summary(is_fraud, selected_language)
        customer_message = self._build_customer_message(transaction, selected_language)
        email_subject = self._build_email_subject(selected_language)
        recommended_action = "request_customer_confirmation" if is_fraud else "allow_and_monitor"

        return AgentExplanationResult(
            provider="template",
            model=None,
            language=selected_language,
            risk_level=risk_level,
            reasons=reasons,
            analyst_summary=analyst_summary,
            customer_message=customer_message,
            email_subject=email_subject,
            recommended_action=recommended_action,
        )

    def _build_reasons(
        self,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        language: Literal["fr", "en"],
    ) -> list[str]:
        amount = float(transaction["amount"])
        fraud_probability = float(prediction["fraud_probability"])
        reasons: list[str] = []

        if language == "fr":
            reasons.append(f"Probabilite de fraude estimee a {fraud_probability * 100:.0f} %.")
            if transaction.get("type") in {"TRANSFER", "CASH_OUT"}:
                reasons.append("Type de transaction souvent surveille en fraude bancaire.")
            if float(transaction.get("oldbalanceOrg", 0.0)) > 0 and float(transaction.get("newbalanceOrig", 0.0)) == 0:
                reasons.append("Debit important du compte d origine apres l operation.")
            if float(transaction.get("oldbalanceDest", 0.0)) == 0 and float(transaction.get("newbalanceDest", 0.0)) == 0:
                reasons.append("Compte destinataire peu informatif ou sans solde observe.")
            if amount >= 1000:
                reasons.append("Montant relativement eleve par rapport aux cas usuels.")
        else:
            reasons.append(f"Fraud probability estimated at {fraud_probability * 100:.0f}%.")
            if transaction.get("type") in {"TRANSFER", "CASH_OUT"}:
                reasons.append("Transaction type frequently monitored for fraud.")
            if float(transaction.get("oldbalanceOrg", 0.0)) > 0 and float(transaction.get("newbalanceOrig", 0.0)) == 0:
                reasons.append("Large debit from the origin account after the transaction.")
            if float(transaction.get("oldbalanceDest", 0.0)) == 0 and float(transaction.get("newbalanceDest", 0.0)) == 0:
                reasons.append("Destination account carries little balance information.")
            if amount >= 1000:
                reasons.append("Amount is relatively high versus common activity.")

        return reasons[:4]

    def _build_analyst_summary(self, is_fraud: bool, language: Literal["fr", "en"]) -> str:
        if language == "fr":
            if is_fraud:
                return (
                    "La transaction presente plusieurs signaux de risque et devrait etre validee "
                    "par le client avant d etre consideree comme legitime."
                )
            return "La transaction reste proche des comportements normaux observes dans les donnees de reference."
        if is_fraud:
            return "The transaction shows multiple risk signals and should be validated by the customer."
        return "The transaction remains close to normal behaviour seen in reference data."

    def _build_customer_message(self, transaction: dict[str, Any], language: Literal["fr", "en"]) -> str:
        amount = float(transaction["amount"])
        if language == "fr":
            return (
                f"Bonjour, nous avons detecte une activite inhabituelle concernant une transaction de {amount:,.2f} EUR. "
                "Merci de confirmer s il s agit bien d une operation autorisee."
            )
        return (
            f"Hello, we detected unusual activity related to a transaction of {amount:,.2f} EUR. "
            "Please confirm whether this operation was authorized."
        )

    def _build_email_subject(self, language: Literal["fr", "en"]) -> str:
        if language == "fr":
            return "Verification d une transaction bancaire"
        return "Bank transaction verification"
