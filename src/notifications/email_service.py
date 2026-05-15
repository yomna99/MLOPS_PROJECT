from __future__ import annotations

from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from typing import Any


SMTP_HOST_ENV_VAR = "SMTP_HOST"
SMTP_PORT_ENV_VAR = "SMTP_PORT"
SMTP_USERNAME_ENV_VAR = "SMTP_USERNAME"
SMTP_PASSWORD_ENV_VAR = "SMTP_PASSWORD"
SMTP_FROM_EMAIL_ENV_VAR = "SMTP_FROM_EMAIL"
SMTP_USE_TLS_ENV_VAR = "SMTP_USE_TLS"
PUBLIC_BASE_URL_ENV_VAR = "FRAUD_PUBLIC_BASE_URL"
RESERVED_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "invalid",
    "localhost",
}


def format_currency(amount: float, language: str) -> str:
    normalized = f"{amount:,.2f}"
    if language == "fr":
        return normalized.replace(",", " ").replace(".", ",") + " EUR"
    return normalized + " EUR"


def is_deliverable_customer_email(email: str | None) -> bool:
    if not email:
        return False
    normalized = email.strip().lower()
    if "@" not in normalized:
        return False
    domain = normalized.rsplit("@", 1)[-1]
    return domain not in RESERVED_EMAIL_DOMAINS


@dataclass(frozen=True)
class FraudNotificationLinks:
    confirm_legit_url: str
    report_fraud_url: str


class EmailDeliveryError(RuntimeError):
    """Raised when SMTP delivery fails with a user-facing explanation."""


class NotificationEmailService:
    """Send fraud alert emails with one-click feedback links."""

    def __init__(
        self,
        smtp_host: str | None,
        smtp_port: int,
        smtp_username: str | None,
        smtp_password: str | None,
        from_email: str | None,
        use_tls: bool,
        public_base_url: str,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.use_tls = use_tls
        self.public_base_url = public_base_url.rstrip("/")

    @classmethod
    def from_environment(cls) -> "NotificationEmailService":
        smtp_port_raw = os.getenv(SMTP_PORT_ENV_VAR, "587")
        try:
            smtp_port = int(smtp_port_raw)
        except ValueError:
            smtp_port = 587
        use_tls_raw = os.getenv(SMTP_USE_TLS_ENV_VAR, "true").strip().lower()
        use_tls = use_tls_raw in {"1", "true", "yes", "on"}
        return cls(
            smtp_host=os.getenv(SMTP_HOST_ENV_VAR),
            smtp_port=smtp_port,
            smtp_username=os.getenv(SMTP_USERNAME_ENV_VAR),
            smtp_password=os.getenv(SMTP_PASSWORD_ENV_VAR),
            from_email=os.getenv(SMTP_FROM_EMAIL_ENV_VAR),
            use_tls=use_tls,
            public_base_url=os.getenv(PUBLIC_BASE_URL_ENV_VAR, "http://127.0.0.1:8000"),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "ready": self.is_configured(),
            "from_email": self.from_email,
            "public_base_url": self.public_base_url,
        }

    def is_configured(self) -> bool:
        return bool(self.smtp_host and self.from_email)

    def send_feedback_email(
        self,
        *,
        customer_email: str,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        explanation: dict[str, Any],
        links: FraudNotificationLinks,
    ) -> None:
        if not self.is_configured():
            raise RuntimeError("SMTP email delivery is not configured.")

        subject = explanation["email_subject"]
        html_body = self._build_html_body(
            customer_email=customer_email,
            transaction=transaction,
            prediction=prediction,
            explanation=explanation,
            links=links,
        )
        text_body = self._build_text_body(
            transaction=transaction,
            prediction=prediction,
            explanation=explanation,
            links=links,
        )

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.from_email or ""
        message["To"] = customer_email
        message.attach(MIMEText(text_body, "plain", "utf-8"))
        message.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=20) as server:
            try:
                server.ehlo()
                if self.use_tls:
                    server.starttls()
                    server.ehlo()
                if self.smtp_username:
                    server.login(self.smtp_username, self.smtp_password or "")
                server.sendmail(self.from_email or "", [customer_email], message.as_string())
            except smtplib.SMTPAuthenticationError as exc:
                raise EmailDeliveryError(
                    "SMTP authentication failed. If you use Gmail, set SMTP_USERNAME to your Gmail address "
                    "and SMTP_PASSWORD to a Google App Password, not your regular account password."
                ) from exc
            except (smtplib.SMTPException, OSError) as exc:
                raise EmailDeliveryError(f"SMTP delivery failed: {exc}") from exc

    def _build_html_body(
        self,
        *,
        customer_email: str,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        explanation: dict[str, Any],
        links: FraudNotificationLinks,
    ) -> str:
        amount_text = format_currency(float(transaction["amount"]), explanation["language"])
        reasons_html = "".join(f"<li>{reason}</li>" for reason in explanation["reasons"])
        score_text = f"{float(prediction['fraud_probability']) * 100:.0f}%"
        return f"""
        <html>
          <body style="font-family:Arial,sans-serif;line-height:1.5;color:#1f2937;">
            <p>Bonjour {customer_email},</p>
            <p>{explanation['customer_message']}</p>
            <p><strong>Montant</strong>: {amount_text}<br/>
               <strong>Probabilite fraude</strong>: {score_text}</p>
            <p><strong>Why the model reacted this way:</strong></p>
            <ul>{reasons_html}</ul>
            <p><strong>Recommended action</strong>: {explanation['recommended_action']}</p>
            <p>
              <a href="{links.confirm_legit_url}" style="display:inline-block;padding:12px 18px;margin-right:12px;background:#1d9f5f;color:#ffffff;text-decoration:none;border-radius:8px;">
                Confirm legitimate transaction
              </a>
              <a href="{links.report_fraud_url}" style="display:inline-block;padding:12px 18px;background:#c0392b;color:#ffffff;text-decoration:none;border-radius:8px;">
                Report confirmed fraud
              </a>
            </p>
            <p>Si vous preferez, vous pouvez aussi nous contacter directement.</p>
          </body>
        </html>
        """.strip()

    def _build_text_body(
        self,
        *,
        transaction: dict[str, Any],
        prediction: dict[str, Any],
        explanation: dict[str, Any],
        links: FraudNotificationLinks,
    ) -> str:
        amount_text = format_currency(float(transaction["amount"]), explanation["language"])
        score_text = f"{float(prediction['fraud_probability']) * 100:.0f}%"
        reasons_text = "\n".join(f"- {reason}" for reason in explanation["reasons"])
        return (
            f"{explanation['email_subject']}\n\n"
            f"{explanation['customer_message']}\n\n"
            f"Montant: {amount_text}\n"
            f"Probabilite fraude: {score_text}\n"
            f"Recommended action: {explanation['recommended_action']}\n\n"
            f"Why the model reacted this way:\n{reasons_text}\n\n"
            f"Confirm legitimate transaction: {links.confirm_legit_url}\n"
            f"Report confirmed fraud: {links.report_fraud_url}\n"
        )
