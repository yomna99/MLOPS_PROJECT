from __future__ import annotations

import base64
from datetime import datetime, timezone
from functools import lru_cache
from html import escape
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent.explainer import AgentExplanationService
from src.agent.n8n_client import AgentWorkflowError, N8nNotificationClient
from src.inference.service import DEFAULT_ARTIFACT_PATH, FraudPredictionService
from src.monitoring.evidently_service import EvidentlyMonitoringService
from src.notifications.email_service import is_deliverable_customer_email
from src.notifications.feedback_tokens import FeedbackTokenService
from src.production.store import ProductionFeedbackStore

DEFAULT_PUBLIC_BASE_URL = "http://127.0.0.1:8000"
BANK_BRAND_NAME = "MyBank"
BRAND_LOGO_PATH = Path(__file__).resolve().parent.parent / "branding" / "mybank_logo_email.png"


def _format_currency_fr(amount: float | int | str | None) -> str:
    try:
        numeric_amount = float(amount or 0.0)
    except (TypeError, ValueError):
        numeric_amount = 0.0
    return f"{numeric_amount:,.2f}".replace(",", " ").replace(".", ",") + " EUR"


def _derive_feedback_reasons(
    transaction: dict[str, object],
    prediction: dict[str, object],
) -> list[str]:
    reasons: list[str] = []
    probability = float(prediction.get("fraud_probability") or 0.0)
    transaction_type = str(transaction.get("type") or "")
    amount = float(transaction.get("amount") or 0.0)
    oldbalance_org = float(transaction.get("oldbalanceOrg") or 0.0)
    oldbalance_dest = float(transaction.get("oldbalanceDest") or 0.0)

    reasons.append(f"Probabilite de fraude estimee a {round(probability * 100)} %.")
    if transaction_type in {"TRANSFER", "CASH_OUT"}:
        reasons.append("Type de transaction souvent surveille en fraude bancaire.")
    if oldbalance_org > 0 and amount >= oldbalance_org:
        reasons.append("Debit important du compte d'origine apres l'operation.")
    if oldbalance_dest == 0:
        reasons.append("Compte destinataire peu informatif ou sans solde observe.")
    if len(reasons) < 4:
        reasons.append("Comportement detecte comme atypique par rapport aux transactions de reference.")
    return reasons[:4]


@lru_cache(maxsize=1)
def _load_brand_logo_data_uri() -> str:
    if not BRAND_LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(BRAND_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_feedback_status_card(
    *,
    label: str,
    background: str,
    border: str,
    color: str,
    note: str,
) -> str:
    return f"""
    <div style="background:{background};border:1px solid {border};border-radius:18px;padding:18px 20px;margin-bottom:22px;">
      <div style="font-size:13px;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;color:{color};margin-bottom:8px;">Statut de votre retour</div>
      <div style="font-size:22px;font-weight:700;color:{color};margin-bottom:8px;">{escape(label)}</div>
      <div style="font-size:14px;line-height:1.7;color:{color};opacity:0.92;">{escape(note)}</div>
    </div>
    """


def _render_feedback_page(
    *,
    title: str,
    subtitle: str,
    badge_label: str,
    badge_background: str,
    badge_color: str,
    amount: float | int | str | None,
    fraud_probability: float | int | str | None,
    prediction_id: str,
    customer_email: str | None,
    body_message: list[str],
    reasons: list[str],
    status_card_html: str,
) -> HTMLResponse:
    safe_title = escape(title)
    safe_subtitle = escape(subtitle)
    safe_badge = escape(badge_label)
    safe_prediction_id = escape(prediction_id)
    safe_email = escape(customer_email or "-")
    safe_amount = escape(_format_currency_fr(amount))
    safe_probability = f"{round(float(fraud_probability or 0.0) * 100)} %"
    body_markup = "".join(
        f'<p style="margin:0 0 14px 0;color:#334155;font-size:15px;line-height:1.8;">{escape(paragraph)}</p>'
        for paragraph in body_message
    )
    reasons_markup = "".join(
        f'<li style="margin:0 0 10px 0;">{escape(reason)}</li>'
        for reason in reasons
    )
    logo_data_uri = _load_brand_logo_data_uri()
    logo_markup = (
        f'<img src="{logo_data_uri}" alt="{escape(BANK_BRAND_NAME)}" '
        'style="display:block;width:88px;height:auto;margin-bottom:14px;">'
        if logo_data_uri
        else ""
    )
    html_page = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{safe_title}</title>
      <style>
        body {{
          margin: 0;
          font-family: Arial, sans-serif;
          background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 26%),
            linear-gradient(180deg, #eef4ff 0%, #f8fbff 100%);
          color: #102647;
        }}
        .shell {{
          max-width: 760px;
          margin: 0 auto;
          padding: 48px 18px;
        }}
        .card {{
          background: #ffffff;
          border: 1px solid rgba(15, 23, 42, 0.08);
          border-radius: 28px;
          overflow: hidden;
          box-shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
        }}
        .hero {{
          background: linear-gradient(135deg, #102647 0%, #1d4ed8 100%);
          color: #ffffff;
          padding: 30px 34px;
        }}
        .hero-kicker {{
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.12em;
          opacity: 0.8;
        }}
        .hero-title {{
          font-size: 30px;
          font-weight: 700;
          line-height: 1.2;
          margin-top: 10px;
        }}
        .hero-subtitle {{
          font-size: 15px;
          line-height: 1.7;
          opacity: 0.92;
          margin-top: 12px;
          max-width: 560px;
        }}
        .content {{
          padding: 30px 34px 34px 34px;
        }}
        .alert-box {{
          background: #fff7ed;
          border: 1px solid #fdba74;
          border-radius: 18px;
          padding: 18px 20px;
          margin-bottom: 22px;
        }}
        .alert-kicker {{
          color: #9a3412;
          font-size: 13px;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 10px;
        }}
        .alert-title {{
          color: #7c2d12;
          font-size: 18px;
          font-weight: 700;
          margin-bottom: 8px;
        }}
        .alert-meta {{
          color: #c2410c;
          font-size: 14px;
        }}
        .reason-box {{
          background: #f8fafc;
          border: 1px solid #dbe4f0;
          border-radius: 18px;
          padding: 20px 22px;
          margin-top: 8px;
        }}
        .reason-title {{
          color: #102647;
          font-size: 15px;
          font-weight: 700;
          margin-bottom: 12px;
        }}
        .reason-list {{
          margin: 0;
          padding-left: 18px;
          color: #475569;
          font-size: 14px;
          line-height: 1.7;
        }}
        .meta-row {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 14px;
          margin-top: 22px;
        }}
        .meta-card {{
          background: #f8fbff;
          border: 1px solid #dbe7f5;
          border-radius: 16px;
          padding: 16px 18px;
        }}
        .meta-label {{
          color: #64748b;
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-bottom: 7px;
        }}
        .meta-value {{
          color: #102647;
          font-size: 16px;
          font-weight: 700;
          line-height: 1.45;
          word-break: break-word;
        }}
        .footnote {{
          margin-top: 22px;
          color: #64748b;
          font-size: 12px;
          line-height: 1.7;
        }}
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="card">
          <div class="hero">
            {logo_markup}
            <div class="hero-kicker">{escape(BANK_BRAND_NAME)}</div>
            <div class="hero-title">Verification de transaction</div>
            <div class="hero-subtitle">{safe_subtitle}</div>
          </div>
          <div class="content">
            {status_card_html}
            <div class="alert-box">
              <div class="alert-kicker">{safe_badge}</div>
              <div class="alert-title">Transaction de {safe_amount}</div>
              <div class="alert-meta">Probabilite de fraude : {safe_probability}</div>
            </div>
            {body_markup}
            <div class="reason-box">
              <div class="reason-title">Pourquoi cette transaction a ete signalee</div>
              <ul class="reason-list">{reasons_markup}</ul>
            </div>
            <div class="meta-row">
              <div class="meta-card">
                <div class="meta-label">Prediction ID</div>
                <div class="meta-value">{safe_prediction_id}</div>
              </div>
              <div class="meta-card">
                <div class="meta-label">Email client</div>
                <div class="meta-value">{safe_email}</div>
              </div>
            </div>
            <div class="footnote">Cette reponse a ete enregistree de facon securisee et alimente le suivi du systeme {escape(BANK_BRAND_NAME)}.</div>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html_page)


class FraudPredictionRequest(BaseModel):
    step: int = Field(..., ge=0)
    type: str
    amount: float = Field(..., ge=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = Field(default=0, ge=0, le=1)
    customer_email: str | None = None


class FraudPredictionResponse(BaseModel):
    prediction_id: str
    prediction: Literal[0, 1]
    predicted_label: Literal["fraud", "not_fraud"]
    fraud_probability: float
    threshold: float
    model_name: str


class FraudBatchPredictionRequest(BaseModel):
    transactions: list[FraudPredictionRequest] = Field(..., min_length=1)


class FraudBatchSummary(BaseModel):
    total_transactions: int
    fraud_predictions: int
    non_fraud_predictions: int
    average_fraud_probability: float


class FraudBatchPredictionResponse(BaseModel):
    summary: FraudBatchSummary
    predictions: list[FraudPredictionResponse]


class FraudFeedbackRequest(BaseModel):
    transaction: FraudPredictionRequest
    prediction: FraudPredictionResponse
    user_feedback: Literal["confirmed_legit", "reported_fraud"]
    feedback_notes: str | None = Field(default=None, max_length=500)


class FraudFeedbackResponse(BaseModel):
    status: Literal["saved"]
    prediction_id: str
    user_feedback: Literal["confirmed_legit", "reported_fraud"]
    ground_truth_label: Literal[0, 1]
    production_data_path: str


class FraudFeedbackSummaryResponse(BaseModel):
    production_data_path: str
    total_feedback: int
    confirmed_legit: int
    reported_fraud: int
    last_feedback_timestamp: str | None


class FraudFeedbackRecord(BaseModel):
    prediction_id: str
    feedback_timestamp: str
    step: int | str
    type: str
    amount: float | str
    oldbalanceOrg: float | str
    newbalanceOrig: float | str
    oldbalanceDest: float | str
    newbalanceDest: float | str
    isFlaggedFraud: int | str
    customer_email: str | None = None
    prediction: int | str
    predicted_label: str
    fraud_probability: float | str
    threshold: float | str
    model_name: str
    user_feedback: str
    ground_truth_label: int | str
    feedback_notes: str | None = None


class FraudExplanationRequest(BaseModel):
    transaction: FraudPredictionRequest
    prediction: FraudPredictionResponse
    language: Literal["fr", "en"] = "fr"


class FraudExplanationResponse(BaseModel):
    provider: str
    model: str | None
    language: Literal["fr", "en"]
    risk_level: Literal["low", "medium", "high"]
    reasons: list[str]
    analyst_summary: str
    customer_message: str
    email_subject: str
    recommended_action: str


class FraudNotificationRequest(BaseModel):
    transaction: FraudPredictionRequest
    prediction: FraudPredictionResponse
    explanation: FraudExplanationResponse | None = None


class FraudNotificationResponse(BaseModel):
    status: str
    recipient_email: str
    email_subject: str
    confirm_legit_url: str
    report_fraud_url: str
    workflow_provider: str
    workflow_status: str | None = None


class MonitoringReportsResponse(BaseModel):
    generated_at: str
    reference_rows: int
    production_rows: int
    data_drift_report_path: str
    classification_report_path: str
    data_drift_report_url: str
    classification_report_url: str
    drift_detected: bool | None = None
    drift_threshold: float | None = None
    total_columns: int | None = None
    drifted_columns: int | None = None
    share_drifted_columns: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None


@lru_cache(maxsize=1)
def get_prediction_service() -> FraudPredictionService:
    return FraudPredictionService.from_environment()


@lru_cache(maxsize=1)
def get_feedback_store() -> ProductionFeedbackStore:
    return ProductionFeedbackStore.from_environment()


@lru_cache(maxsize=1)
def get_explanation_service() -> AgentExplanationService:
    return AgentExplanationService()


@lru_cache(maxsize=1)
def get_feedback_token_service() -> FeedbackTokenService:
    return FeedbackTokenService.from_environment()


@lru_cache(maxsize=1)
def get_n8n_notification_client() -> N8nNotificationClient:
    return N8nNotificationClient.from_environment()


@lru_cache(maxsize=1)
def get_monitoring_service() -> EvidentlyMonitoringService:
    feedback_store = get_feedback_store()
    public_base_url = os.getenv("FRAUD_PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL)
    return EvidentlyMonitoringService.from_environment(
        prediction_service=get_prediction_service(),
        production_data_path=feedback_store.output_path,
        public_base_url=public_base_url,
    )


app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.get("/health")
def healthcheck(
    service: FraudPredictionService = Depends(get_prediction_service),
    feedback_store: ProductionFeedbackStore = Depends(get_feedback_store),
    explanation_service: AgentExplanationService = Depends(get_explanation_service),
    n8n_client: N8nNotificationClient = Depends(get_n8n_notification_client),
    monitoring_service: EvidentlyMonitoringService = Depends(get_monitoring_service),
) -> dict:
    metadata = service.metadata()
    explanation_metadata = explanation_service.metadata()
    workflow_metadata = n8n_client.metadata()
    return {
        "status": "ok",
        "artifact_path": metadata["artifact_path"] or str(Path(DEFAULT_ARTIFACT_PATH)),
        "model_name": metadata["model_name"],
        "threshold": metadata["threshold"],
        "production_data_path": str(feedback_store.output_path),
        "explanation_provider": explanation_metadata["active_provider"],
        "explanation_model": explanation_metadata["model"],
        "agent_provider": workflow_metadata["provider"],
        "agent_ready": workflow_metadata["ready"],
        "n8n_webhook_url": workflow_metadata["webhook_url"],
        "monitoring_reference_data_path": str(monitoring_service.reference_data_path),
        "monitoring_reports_dir": str(monitoring_service.output_dir),
        "groq_in_fastapi_enabled": False,
    }


@app.post("/predict", response_model=FraudPredictionResponse)
def predict(
    request: FraudPredictionRequest,
    service: FraudPredictionService = Depends(get_prediction_service),
) -> FraudPredictionResponse:
    result = service.predict(request.model_dump())
    return FraudPredictionResponse(**result.to_dict())


@app.post("/predict-batch", response_model=FraudBatchPredictionResponse)
def predict_batch(
    request: FraudBatchPredictionRequest,
    service: FraudPredictionService = Depends(get_prediction_service),
) -> FraudBatchPredictionResponse:
    results = service.predict_batch([transaction.model_dump() for transaction in request.transactions])
    prediction_rows = [
        FraudPredictionResponse(
            prediction_id=row["prediction_id"],
            prediction=int(row["prediction"]),
            predicted_label=row["predicted_label"],
            fraud_probability=float(row["fraud_probability"]),
            threshold=float(row["threshold"]),
            model_name=row["model_name"],
        )
        for row in results.to_dict(orient="records")
    ]
    fraud_predictions = int(results["prediction"].sum())
    summary = FraudBatchSummary(
        total_transactions=int(len(results)),
        fraud_predictions=fraud_predictions,
        non_fraud_predictions=int(len(results) - fraud_predictions),
        average_fraud_probability=float(results["fraud_probability"].mean()),
    )
    return FraudBatchPredictionResponse(summary=summary, predictions=prediction_rows)


@app.post("/feedback", response_model=FraudFeedbackResponse)
def save_feedback(
    request: FraudFeedbackRequest,
    feedback_store: ProductionFeedbackStore = Depends(get_feedback_store),
) -> FraudFeedbackResponse:
    record = feedback_store.append_feedback(
        transaction=request.transaction.model_dump(),
        prediction=request.prediction.model_dump(),
        user_feedback=request.user_feedback,
        feedback_notes=request.feedback_notes,
    )
    return FraudFeedbackResponse(
        status="saved",
        prediction_id=record["prediction_id"],
        user_feedback=record["user_feedback"],
        ground_truth_label=record["ground_truth_label"],
        production_data_path=str(feedback_store.output_path),
    )


@app.get("/feedback-summary", response_model=FraudFeedbackSummaryResponse)
def feedback_summary(
    feedback_store: ProductionFeedbackStore = Depends(get_feedback_store),
) -> FraudFeedbackSummaryResponse:
    return FraudFeedbackSummaryResponse(**feedback_store.summary())


@app.get("/feedback-records", response_model=list[FraudFeedbackRecord])
def feedback_records(
    limit: int = Query(default=50, ge=1, le=500),
    feedback_store: ProductionFeedbackStore = Depends(get_feedback_store),
) -> list[FraudFeedbackRecord]:
    return [FraudFeedbackRecord(**record) for record in feedback_store.records(limit=limit)]


@app.get("/monitoring/reports/generate", response_model=MonitoringReportsResponse)
def generate_monitoring_reports(
    force_refresh: bool = Query(default=False),
    monitoring_service: EvidentlyMonitoringService = Depends(get_monitoring_service),
) -> MonitoringReportsResponse:
    try:
        bundle = monitoring_service.generate_reports(force=force_refresh)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MonitoringReportsResponse(**bundle.to_dict())


@app.get("/monitoring/reports/summary", response_model=MonitoringReportsResponse | None)
def monitoring_reports_summary(
    monitoring_service: EvidentlyMonitoringService = Depends(get_monitoring_service),
) -> MonitoringReportsResponse | None:
    bundle = monitoring_service.read_summary()
    if bundle is None:
        return None
    return MonitoringReportsResponse(**bundle.to_dict())


@app.get("/monitoring/reports/{report_name}", response_class=HTMLResponse)
def serve_monitoring_report(
    report_name: Literal["data-drift", "classification"],
    monitoring_service: EvidentlyMonitoringService = Depends(get_monitoring_service),
) -> HTMLResponse:
    try:
        html_report = monitoring_service.read_report_html(report_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return HTMLResponse(html_report)


@app.post("/explain-prediction", response_model=FraudExplanationResponse)
def explain_prediction(
    request: FraudExplanationRequest,
    explanation_service: AgentExplanationService = Depends(get_explanation_service),
) -> FraudExplanationResponse:
    explanation = explanation_service.explain_prediction(
        transaction=request.transaction.model_dump(),
        prediction=request.prediction.model_dump(),
        language=request.language,
    )
    return FraudExplanationResponse(**explanation.to_dict())


@app.post("/notify-customer", response_model=FraudNotificationResponse)
@app.post("/notify_user", response_model=FraudNotificationResponse)
def notify_customer(
    request: FraudNotificationRequest,
    token_service: FeedbackTokenService = Depends(get_feedback_token_service),
    explanation_service: AgentExplanationService = Depends(get_explanation_service),
    n8n_client: N8nNotificationClient = Depends(get_n8n_notification_client),
) -> FraudNotificationResponse:
    customer_email = (request.transaction.customer_email or "").strip()
    if not customer_email:
        raise HTTPException(status_code=400, detail="customer_email is required to send a notification email.")
    if not is_deliverable_customer_email(customer_email):
        raise HTTPException(
            status_code=400,
            detail="customer_email must be a real deliverable address. Placeholder domains like example.com are blocked.",
        )

    shared_payload = {
        "transaction": request.transaction.model_dump(),
        "prediction": request.prediction.model_dump(),
    }
    confirm_token = token_service.create_token(shared_payload)
    report_token = token_service.create_token(shared_payload)
    public_base_url = os.getenv("FRAUD_PUBLIC_BASE_URL", DEFAULT_PUBLIC_BASE_URL).rstrip("/")
    confirm_legit_url = f"{public_base_url}/feedback-action?{urlencode({'token': confirm_token, 'action': 'confirmed_legit'})}"
    report_fraud_url = f"{public_base_url}/feedback-action?{urlencode({'token': report_token, 'action': 'reported_fraud'})}"

    explanation_result = explanation_service.explain_prediction(
        transaction=request.transaction.model_dump(),
        prediction=request.prediction.model_dump(),
        language=request.explanation.language if request.explanation else "fr",
    )
    explanation_payload = FraudExplanationResponse(**explanation_result.to_dict())

    notification_payload = {
        "prediction_id": request.prediction.prediction_id,
        "brand_name": BANK_BRAND_NAME,
        "brand_logo_data_uri": _load_brand_logo_data_uri(),
        "customer_email": customer_email,
        "amount": request.transaction.amount,
        "prediction": request.prediction.prediction,
        "predicted_label": request.prediction.predicted_label,
        "fraud_probability": request.prediction.fraud_probability,
        "explanatory_factors": explanation_payload.reasons,
        "confirm_url": confirm_legit_url,
        "reject_url": report_fraud_url,
        "email_subject": explanation_payload.email_subject,
        "customer_message_preview": explanation_payload.customer_message,
        "recommended_action": explanation_payload.recommended_action,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_context": {
            "transaction": request.transaction.model_dump(),
            "prediction": request.prediction.model_dump(),
        },
    }
    try:
        workflow_response = n8n_client.send_notification(notification_payload)
    except AgentWorkflowError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FraudNotificationResponse(
        status=str(workflow_response.get("status", "accepted")),
        recipient_email=customer_email,
        email_subject=str(workflow_response.get("email_subject", explanation_payload.email_subject)),
        confirm_legit_url=confirm_legit_url,
        report_fraud_url=report_fraud_url,
        workflow_provider="n8n",
        workflow_status=str(workflow_response.get("workflow_status", workflow_response.get("status", "accepted"))),
    )


@app.get("/feedback-action", response_class=HTMLResponse)
def feedback_action(
    token: str = Query(...),
    action: Literal["confirmed_legit", "reported_fraud"] = Query(...),
    token_service: FeedbackTokenService = Depends(get_feedback_token_service),
    feedback_store: ProductionFeedbackStore = Depends(get_feedback_store),
) -> HTMLResponse:
    try:
        payload = token_service.decode_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction = payload["prediction"]
    transaction = payload["transaction"]
    prediction_id = prediction["prediction_id"]
    customer_email = transaction.get("customer_email")
    amount = transaction.get("amount")
    fraud_probability = prediction.get("fraud_probability")
    reasons = _derive_feedback_reasons(transaction, prediction)

    if feedback_store.has_feedback(prediction_id):
        return _render_feedback_page(
            title="Reponse deja enregistree",
            subtitle="Une validation etait requise pour proteger votre compte. Cette transaction a deja ete revue.",
            badge_label="Alerte de securite",
            badge_background="#fff7ed",
            badge_color="#9a3412",
            amount=amount,
            fraud_probability=fraud_probability,
            prediction_id=prediction_id,
            customer_email=customer_email,
            body_message=[
                "Bonjour,",
                "Cette transaction a deja ete examinee a partir d'une reponse precedente.",
                "Aucune action supplementaire n'est necessaire de votre cote.",
                "Cordialement,",
                "Votre banque",
            ],
            reasons=reasons,
            status_card_html=_render_feedback_status_card(
                label="Action deja traitee",
                background="#fff7ed",
                border="#fdba74",
                color="#9a3412",
                note="Nous avons conserve la premiere reponse recue pour cette transaction afin d'assurer l'integrite du suivi.",
            ),
        )

    record = feedback_store.append_feedback(
        transaction=transaction,
        prediction=prediction,
        user_feedback=action,
        feedback_notes="Saved from emailed feedback link.",
    )
    if action == "confirmed_legit":
        return _render_feedback_page(
            title="Transaction confirmee comme legitime",
            subtitle="Une validation est requise pour proteger votre compte.",
            badge_label="Alerte de securite",
            badge_background="#fff7ed",
            badge_color="#9a3412",
            amount=amount,
            fraud_probability=fraud_probability,
            prediction_id=record["prediction_id"],
            customer_email=customer_email,
            body_message=[
                "Bonjour,",
                f"Nous avons bien enregistre votre confirmation pour la transaction de {_format_currency_fr(amount)}.",
                "Cette operation est maintenant marquee comme legitime dans notre systeme.",
                "Aucune action supplementaire n'est requise de votre part.",
                "Cordialement,",
                "Votre banque",
            ],
            reasons=reasons,
            status_card_html=_render_feedback_status_card(
                label="Transaction legitime confirmee",
                background="#dcfce7",
                border="#86efac",
                color="#166534",
                note="Merci pour votre retour. Cette validation est prise en compte dans les donnees de production pour ameliorer le systeme.",
            ),
        )
    return _render_feedback_page(
        title="Fraude signalee avec succes",
        subtitle="Une validation est requise pour proteger votre compte.",
        badge_label="Alerte de securite",
        badge_background="#fff7ed",
        badge_color="#9a3412",
        amount=amount,
        fraud_probability=fraud_probability,
        prediction_id=record["prediction_id"],
        customer_email=customer_email,
        body_message=[
            "Bonjour,",
            f"Nous avons bien enregistre votre signalement concernant la transaction de {_format_currency_fr(amount)}.",
            "Par mesure de securite, cette operation est maintenant marquee comme potentiellement frauduleuse.",
            "Si vous ne reconnaissez pas cette transaction, contactez immediatement votre service client.",
            "Cordialement,",
            "Votre banque",
        ],
        reasons=reasons,
        status_card_html=_render_feedback_status_card(
            label="Fraude confirmee signalee",
            background="#fee2e2",
            border="#fca5a5",
            color="#b91c1c",
            note="Merci pour votre vigilance. Votre signalement a bien ete enregistre pour le suivi du risque et l'amelioration du modele.",
        ),
    )
