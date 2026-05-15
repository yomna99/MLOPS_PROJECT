from __future__ import annotations

import base64
from datetime import datetime, timedelta
import html
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    from app.client import FraudApiClient
except ModuleNotFoundError:
    from client import FraudApiClient
from src.notifications.email_service import is_deliverable_customer_email


APP_NAME = "FraudGuard AI"
BANK_BRAND_NAME = "MyBank"
RETRAIN_TARGET = 20
DEFAULT_API_HINT = "http://localhost:8080"
REFERENCE_DATA_PATH = Path("data/raw/AIML Dataset.csv")
PRODUCTION_DATA_PATH = Path("data/production/prod_data.csv")
BRAND_LOGO_PATH = Path("src/branding/mybank_logo_email.png")
MODEL_INPUT_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

SIDEBAR_PAGES = [
    ("dashboard", "Dashboard", "📊"),
    ("prediction", "Nouvelle prédiction", "🧾"),
    ("agent", "Agent IA", "🤖"),
    ("monitoring", "Monitoring", "🛰️"),
]

FEEDBACK_LABELS = {
    "Confirm legitimate transaction": "confirmed_legit",
    "Report confirmed fraud": "reported_fraud",
}

EXPLANATION_LANGUAGE_LABELS = {
    "Français": "fr",
    "English": "en",
}

TRANSACTION_STATUS_LABELS = {
    "not_fraud": "Normale",
    "fraud": "Suspecte",
}

STATUS_DISPLAY = {
    "Normale": "🟢 Normale",
    "Suspecte": "🟠 Suspecte",
    "Fraude confirmée": "🔴 Fraude confirmée",
    "Transaction normale confirmée": "🟢 Transaction normale confirmée",
    "Email envoyé": "📨 Email envoyé",
    "En attente": "⏳ En attente",
    "Aucune notification": "— Aucune notification",
    "Répondu": "✅ Répondu",
    "Échec": "❌ Échec",
}


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(70, 120, 255, 0.10), transparent 18%),
            linear-gradient(180deg, #f3f6fb 0%, #f3f6fb 100%);
    }
    [data-testid="stHeader"] {
        position: static !important;
        background: transparent !important;
        height: auto !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    .stAppViewContainer > .main > div {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3rem !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071427 0%, #09203c 45%, #0b2444 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #eef4ff !important;
    }
    .brand-shell {
        padding: 0.2rem 0 1.25rem 0;
    }
    .brand-title {
        font-size: 1.26rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .brand-subtitle {
        color: rgba(238, 244, 255, 0.74);
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.35rem;
    }
    .page-header {
        background: rgba(255,255,255,0.96);
        border-radius: 26px;
        border: 1px solid rgba(17, 36, 65, 0.08);
        box-shadow: 0 22px 48px rgba(12, 29, 53, 0.13);
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .page-kicker {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(135deg, #2d6bff, #5a8dff);
        color: white;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .page-title {
        font-size: 2rem;
        font-weight: 800;
        color: #102647;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        color: #5d6d84;
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .metric-card {
        background: white;
        border-radius: 22px;
        border: 1px solid rgba(16, 38, 71, 0.08);
        box-shadow: 0 14px 30px rgba(12, 29, 53, 0.08);
        padding: 1.15rem 1rem;
        min-height: 142px;
        margin-bottom: 0.7rem;
    }
    .metric-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 0.8rem;
    }
    .metric-label {
        color: #6a7890;
        font-size: 0.84rem;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        color: #112949;
        font-size: 1.72rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    .metric-delta {
        margin-top: 0.4rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .panel {
        background: rgba(255,255,255,0.98);
        border-radius: 22px;
        border: 1px solid rgba(17, 36, 65, 0.08);
        box-shadow: 0 14px 30px rgba(12, 29, 53, 0.08);
        padding: 1.15rem 1.2rem;
        margin-bottom: 1.35rem;
    }
    .panel-title {
        color: #112949;
        font-size: 1.04rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .panel-caption {
        color: #72839b;
        font-size: 0.88rem;
        margin-bottom: 0.95rem;
    }
    .result-card {
        border-radius: 24px;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        border: 1px solid rgba(17, 36, 65, 0.08);
        box-shadow: 0 18px 34px rgba(12, 29, 53, 0.09);
        margin-bottom: 1.35rem;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(233, 250, 240, 1), rgba(255,255,255,0.98));
    }
    .result-risk {
        background: linear-gradient(135deg, rgba(255, 240, 229, 1), rgba(255,255,255,0.98));
    }
    .result-title {
        color: #112949;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }
    .result-copy {
        color: #63748b;
        margin-top: 0.35rem;
        line-height: 1.5;
    }
    .score-label {
        color: #6c7d94;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }
    .score-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.05em;
    }
    .mini-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.62rem;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .pill-green {
        background: #dcfce7;
        color: #15803d;
    }
    .pill-orange {
        background: #ffedd5;
        color: #c2410c;
    }
    .pill-red {
        background: #fee2e2;
        color: #b91c1c;
    }
    .pill-blue {
        background: #dbeafe;
        color: #1d4ed8;
    }
    .status-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.9rem;
        margin: -0.15rem 0 1.2rem 0;
    }
    .status-tile {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(17, 36, 65, 0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 10px 24px rgba(12, 29, 53, 0.06);
    }
    .status-tile-label {
        color: #7a889d;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    .status-tile-value {
        color: #112949;
        font-size: 1rem;
        font-weight: 700;
    }
    .workflow-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.25rem 0 1.2rem 0;
    }
    .workflow-step {
        position: relative;
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(17, 36, 65, 0.08);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        min-height: 92px;
    }
    .workflow-step.is-active {
        border-color: rgba(37, 99, 235, 0.24);
        box-shadow: 0 12px 28px rgba(37, 99, 235, 0.12);
    }
    .workflow-step.is-done {
        border-color: rgba(22, 163, 74, 0.24);
        background: linear-gradient(135deg, rgba(240, 253, 244, 0.96), rgba(255,255,255,0.98));
    }
    .workflow-step.is-waiting {
        background: linear-gradient(135deg, rgba(255, 247, 237, 0.98), rgba(255,255,255,0.98));
    }
    .workflow-step-title {
        color: #112949;
        font-size: 0.96rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .workflow-step-copy {
        color: #67778f;
        font-size: 0.84rem;
        line-height: 1.45;
    }
    .workflow-step-state {
        display: inline-block;
        margin-top: 0.55rem;
        font-size: 0.75rem;
        font-weight: 700;
        color: #2563eb;
    }
    .quick-actions {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.2rem 0 1.2rem 0;
    }
    .action-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid rgba(17, 36, 65, 0.08);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 14px 30px rgba(12, 29, 53, 0.08);
        min-height: 124px;
    }
    .action-card-title {
        color: #112949;
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .action-card-copy {
        color: #67778f;
        font-size: 0.86rem;
        line-height: 1.5;
    }
    .soft-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(255,255,255,0.96));
        border: 1px solid rgba(37, 99, 235, 0.14);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        margin: 0.15rem 0 1rem 0;
    }
    .soft-banner-title {
        color: #0f2d57;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .soft-banner-copy {
        color: #5f738f;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .section-meta {
        color: #6f8098;
        font-size: 0.86rem;
        margin: -0.25rem 0 0.8rem 0;
    }
    .detail-label {
        color: #7a889d;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .detail-value {
        color: #112949;
        font-weight: 700;
        margin-top: 0.18rem;
        margin-bottom: 0.6rem;
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        border: 0;
    }
    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
    }
    @media (max-width: 1100px) {
        .status-strip,
        .workflow-strip,
        .quick-actions {
            grid-template-columns: 1fr;
        }
    }
    div[data-testid="stVerticalBlock"] > div:has(> div > .panel),
    div[data-testid="stVerticalBlock"] > div:has(> div > .page-header) {
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "selected_page": "dashboard",
        "api_available": False,
        "recent_predictions": [],
        "prediction_history": [],
        "last_prediction": None,
        "ai_send_history": [],
        "last_agent_response": None,
        "notification_status_message": None,
        "monitoring_action_message": None,
        "monitoring_reports": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-kicker">{APP_NAME}</div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_request_exception(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
    return str(exc)


def open_panel(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{title}</div>
            <div class="panel-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def close_panel() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_card(icon: str, background: str, label: str, value: str, delta: str, delta_color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon" style="background:{background};">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{delta_color};">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_strip(health: dict[str, Any] | None) -> None:
    if health is None:
        tiles = [
            ("API", "Indisponible", "#dc2626", "Backend non joignable"),
            ("Agent IA", "Mode demo", "#ea580c", "n8n / SMTP non verifies"),
            ("Modele", "Non connecte", "#64748b", "Aucune metadonnee disponible"),
            ("Production", "CSV local", "#2563eb", "Historique de demonstration"),
        ]
    else:
        agent_ready = "Pret" if health.get("agent_ready") else "A configurer"
        tiles = [
            ("API", "Connectee", "#16a34a", "Service FastAPI operationnel"),
            ("Agent IA", f"{health.get('agent_provider', 'n8n')} - {agent_ready}", "#2563eb", "Workflow de notification"),
            ("Modele", f"{health.get('model_name', '-')} @ {health.get('threshold', 0):.2f}", "#7c3aed", "Modele de scoring charge"),
            ("Production", Path(str(health.get('production_data_path', 'prod_data.csv'))).name, "#0f766e", "Collecte des retours utilisateurs"),
        ]
    columns = st.columns(len(tiles))
    for column, (label, value, color, caption) in zip(columns, tiles, strict=False):
        with column:
            st.markdown(
                f"""
                <div class="status-tile">
                    <div class="status-tile-label">{label}</div>
                    <div class="status-tile-value" style="color:{color};">{value}</div>
                    <div class="section-meta" style="margin:0.45rem 0 0 0;font-size:0.78rem;">{caption}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_workflow_stepper(last_prediction: dict[str, Any] | None) -> None:
    steps = [
        ("1. Prediction", "Transaction analysee par le modele.", "En attente", "is-waiting"),
        ("2. Alerte", "Transaction marquee pour revue utilisateur.", "En attente", "is-waiting"),
        ("3. Notification", "Demande de confirmation transmise.", "En attente", "is-waiting"),
        ("4. Feedback", "Retour utilisateur collecte.", "En attente", "is-waiting"),
    ]
    if last_prediction is not None:
        steps[0] = (steps[0][0], steps[0][1], "Complete", "is-done")
        if last_prediction["prediction"]["prediction"] == 1:
            steps[1] = (steps[1][0], steps[1][1], "A revoir", "is-active")
        else:
            steps[1] = (steps[1][0], steps[1][1], "Non necessaire", "is-done")

        notification_status = str(last_prediction.get("notification_status", ""))
        feedback_status = str(last_prediction.get("feedback_status", ""))
        if notification_status in {"En attente", "Email envoyé"}:
            steps[1] = (steps[1][0], steps[1][1], "Alerte prete", "is-done")
            steps[2] = (steps[2][0], steps[2][1], notification_status, "is-active" if notification_status == "En attente" else "is-done")
        if feedback_status not in {"", "-"}:
            steps[2] = (steps[2][0], steps[2][1], "Email envoye", "is-done")
            steps[3] = (steps[3][0], steps[3][1], feedback_status, "is-done")

    columns = st.columns(len(steps))
    for column, (title, copy, state, css_class) in zip(columns, steps, strict=False):
        with column:
            st.markdown(
                f"""
                <div class="workflow-step {css_class}">
                    <div class="workflow-step-title">{title}</div>
                    <div class="workflow-step-copy">{copy}</div>
                    <div class="workflow-step-state">{state}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_quick_actions() -> None:
    st.markdown(
        """
        <div class="soft-banner">
            <div class="soft-banner-title">Parcours recommande</div>
            <div class="soft-banner-copy">1. Detecter une transaction  2. Rediger une alerte  3. Lancer l agent IA  4. Collecter le feedback pour le monitoring.</div>
        </div>
        <div class="quick-actions">
            <div class="action-card">
                <div class="action-card-title">Detection en temps reel</div>
                <div class="action-card-copy">Lance une prediction unitaire puis bascule vers l Agent IA si une verification est necessaire.</div>
            </div>
            <div class="action-card">
                <div class="action-card-title">Notification orchestree</div>
                <div class="action-card-copy">Le backend delegue maintenant l email et le message IA a n8n, sans appel Groq direct dans FastAPI.</div>
            </div>
            <div class="action-card">
                <div class="action-card-title">Boucle de feedback</div>
                <div class="action-card-copy">Chaque confirmation utilisateur alimente prod_data.csv pour le monitoring et le futur reentrainement.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: Any) -> str:
    try:
        return f"{float(value):,.2f} €".replace(",", " ").replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def format_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f} %"
    except (TypeError, ValueError):
        return "-"


def format_datetime_label(value: Any) -> str:
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return "-"


def ensure_history_score(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = frame.copy()
    if "score" not in normalized.columns:
        if "fraud_probability" in normalized.columns:
            normalized["score"] = pd.to_numeric(normalized["fraud_probability"], errors="coerce").fillna(0.0)
        else:
            normalized["score"] = 0.0
    else:
        normalized["score"] = pd.to_numeric(normalized["score"], errors="coerce").fillna(0.0)
    return normalized


def build_demo_email_alias(base_email: str | None, tag: str) -> str:
    if not is_deliverable_customer_email(base_email):
        return ""
    normalized = (base_email or "").strip().lower()
    local_part, domain = normalized.split("@", 1)
    safe_tag = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-") or "demo"
    return f"{local_part}+fraudguard-{safe_tag}@{domain}"


@st.cache_data(show_spinner=False)
def build_sample_batch_csv(demo_inbox_email: str | None = None) -> bytes:
    rows = [
        {
            "step": 45211,
            "type": "PAYMENT",
            "amount": 149.62,
            "oldbalanceOrg": 2140.0,
            "newbalanceOrig": 1990.38,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 149.62,
            "isFlaggedFraud": 0,
            "customer_email": build_demo_email_alias(demo_inbox_email, "client01"),
        },
        {
            "step": 45214,
            "type": "TRANSFER",
            "amount": 980.0,
            "oldbalanceOrg": 980.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
            "customer_email": build_demo_email_alias(demo_inbox_email, "client02"),
        },
        {
            "step": 45218,
            "type": "TRANSFER",
            "amount": 1250.50,
            "oldbalanceOrg": 1250.50,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
            "customer_email": build_demo_email_alias(demo_inbox_email, "client03"),
        },
        {
            "step": 45222,
            "type": "CASH_OUT",
            "amount": 45.0,
            "oldbalanceOrg": 80.0,
            "newbalanceOrig": 35.0,
            "oldbalanceDest": 500.0,
            "newbalanceDest": 545.0,
            "isFlaggedFraud": 0,
            "customer_email": build_demo_email_alias(demo_inbox_email, "client04"),
        },
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def load_brand_logo_data_uri() -> str:
    if not BRAND_LOGO_PATH.exists():
        return ""
    encoded = base64.b64encode(BRAND_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def count_csv_rows(path_value: str) -> int:
    path = Path(path_value)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def demo_dashboard_rates() -> pd.DataFrame:
    start = datetime.now() - timedelta(days=11)
    values = [1.6, 1.3, 1.7, 1.5, 1.9, 1.6, 1.7, 1.4, 1.6, 1.5, 1.8, 1.91]
    dates = [(start + timedelta(days=offset)).strftime("%d/%m") for offset in range(len(values))]
    return pd.DataFrame({"Date": dates, "Taux de fraude": values}).set_index("Date")


def demo_monitoring_curve() -> pd.DataFrame:
    start = datetime.now() - timedelta(days=5)
    dates = [(start + timedelta(days=offset)).strftime("%d/%m") for offset in range(6)]
    return pd.DataFrame(
        {
            "Date": dates,
            "F1-score": [0.82, 0.84, 0.83, 0.85, 0.86, 0.85],
            "Recall": [0.78, 0.79, 0.80, 0.81, 0.80, 0.82],
        }
    ).set_index("Date")


def demo_latest_predictions() -> pd.DataFrame:
    now = datetime.now()
    rows = [
        {"Date": now - timedelta(minutes=20), "Montant": 149.62, "Prediction": "Normale", "Score": 0.04},
        {"Date": now - timedelta(minutes=16), "Montant": 980.0, "Prediction": "Suspecte", "Score": 0.91},
        {"Date": now - timedelta(minutes=11), "Montant": 45.0, "Prediction": "Normale", "Score": 0.02},
        {"Date": now - timedelta(minutes=7), "Montant": 1250.50, "Prediction": "Suspecte", "Score": 0.87},
    ]
    frame = pd.DataFrame(rows)
    frame["Date"] = frame["Date"].apply(format_datetime_label)
    frame["Montant"] = frame["Montant"].apply(format_currency)
    frame["Score"] = frame["Score"].apply(format_score)
    frame["Prediction"] = frame["Prediction"].map(
        {"Normale": STATUS_DISPLAY["Normale"], "Suspecte": STATUS_DISPLAY["Suspecte"]}
    )
    return frame


def build_dashboard_chart_frame(history_frame: pd.DataFrame) -> pd.DataFrame:
    if history_frame.empty or len(history_frame) < 8:
        return demo_dashboard_rates()

    working = history_frame.copy()
    working["Date"] = working["analysis_date"].dt.strftime("%d/%m")
    working["is_suspect"] = working["status_label"].isin(["Suspecte", "Fraude confirmée"]).astype(int)
    grouped = (
        working.groupby("Date", as_index=False)["is_suspect"]
        .mean()
        .rename(columns={"is_suspect": "Taux de fraude"})
    )
    grouped["Taux de fraude"] = grouped["Taux de fraude"] * 100
    grouped = grouped.tail(12)
    return grouped.set_index("Date")


def build_dashboard_latest_frame(
    history_frame: pd.DataFrame,
    status_filter: str = "Tous",
) -> pd.DataFrame:
    if history_frame.empty:
        return demo_latest_predictions()

    latest = ensure_history_score(history_frame)
    latest["Date"] = latest["analysis_date"].apply(format_datetime_label)
    latest["Montant"] = latest["amount"].apply(format_currency)
    latest["Prédiction"] = latest["status_label"].map(
        {
            "Normale": "Normale",
            "Transaction normale confirmée": "Normale",
            "Suspecte": "Suspecte",
            "Fraude confirmée": "Suspecte",
        }
    ).fillna(latest["status_label"])
    latest["Prédiction"] = latest["Prédiction"].map(
        {"Normale": STATUS_DISPLAY["Normale"], "Suspecte": STATUS_DISPLAY["Suspecte"]}
    )
    latest["Score"] = latest["score"].apply(format_score)
    return latest[["Date", "Montant", "Prédiction", "Score"]].head(5)


def build_dashboard_latest_frame_filtered(
    history_frame: pd.DataFrame,
    status_filter: str = "Tous",
) -> pd.DataFrame:
    if history_frame.empty:
        latest = demo_latest_predictions().copy()
        if status_filter != "Tous":
            latest["_dashboard_status"] = latest["Prediction"].astype(str).map(
                lambda value: "Normale" if "Normale" in value else "Suspecte"
            )
            if status_filter == "Fraude confirmée":
                latest = latest.iloc[0:0]
            else:
                mapped_status = "Normale" if status_filter == "Normale" else "Suspecte"
                latest = latest[latest["_dashboard_status"] == mapped_status]
            latest = latest.drop(columns=["_dashboard_status"], errors="ignore")
        return latest.reset_index(drop=True)

    latest = ensure_history_score(history_frame)
    latest["Date"] = latest["analysis_date"].apply(format_datetime_label)
    latest["Montant"] = latest["amount"].apply(format_currency)
    latest["_dashboard_status"] = latest["status_label"].map(
        {
            "Normale": "Normale",
            "Transaction normale confirmée": "Normale",
            "Suspecte": "Suspecte",
            "Fraude confirmée": "Fraude confirmée",
        }
    ).fillna("Suspecte")
    latest["Prediction"] = latest["_dashboard_status"].map(
        {
            "Normale": STATUS_DISPLAY["Normale"],
            "Suspecte": STATUS_DISPLAY["Suspecte"],
            "Fraude confirmée": STATUS_DISPLAY["Fraude confirmée"],
        }
    )
    latest["Score"] = latest["score"].apply(format_score)
    if status_filter != "Tous":
        latest = latest[latest["_dashboard_status"] == status_filter]
    latest = latest[["Date", "Montant", "Prediction", "Score"]]
    return latest.reset_index(drop=True)


def demo_history_frame() -> pd.DataFrame:
    now = datetime.now()
    rows = [
        {
            "prediction_id": "demo-001",
            "analysis_date": now - timedelta(minutes=30),
            "amount": 149.62,
            "status_label": "Normale",
            "fraud_probability": 0.04,
            "notification_status": "Aucune notification",
            "feedback_status": "-",
            "customer_email": "clientA@example.org",
        },
        {
            "prediction_id": "demo-002",
            "analysis_date": now - timedelta(minutes=24),
            "amount": 980.0,
            "status_label": "Suspecte",
            "fraud_probability": 0.91,
            "notification_status": "Email envoyé",
            "feedback_status": "En attente",
            "customer_email": "clientB@example.org",
        },
        {
            "prediction_id": "demo-003",
            "analysis_date": now - timedelta(minutes=18),
            "amount": 1250.50,
            "status_label": "Fraude confirmée",
            "fraud_probability": 0.87,
            "notification_status": "Email envoyé",
            "feedback_status": "Fraude confirmée",
            "customer_email": "clientC@example.org",
        },
        {
            "prediction_id": "demo-004",
            "analysis_date": now - timedelta(minutes=12),
            "amount": 20.0,
            "status_label": "Transaction normale confirmée",
            "fraud_probability": 0.01,
            "notification_status": "Aucune notification",
            "feedback_status": "Transaction normale confirmée",
            "customer_email": "clientD@example.org",
        },
    ]
    return pd.DataFrame(rows)


def record_transaction_payload(
    *,
    step: int,
    transaction_type: str,
    amount: float,
    oldbalance_org: float,
    newbalance_orig: float,
    oldbalance_dest: float,
    newbalance_dest: float,
    is_flagged_fraud: int,
    customer_email: str,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org),
        "newbalanceOrig": float(newbalance_orig),
        "oldbalanceDest": float(oldbalance_dest),
        "newbalanceDest": float(newbalance_dest),
        "isFlaggedFraud": int(is_flagged_fraud),
        "customer_email": customer_email.strip() or None,
    }


def build_simple_explanation(prediction: dict[str, Any]) -> dict[str, Any]:
    is_fraud = prediction["prediction"] == 1
    if not is_fraud:
        return {
            "summary": "Cette transaction ressemble aux transactions normales observées dans les données de référence.",
            "detail": "Le modèle ne détecte pas d'anomalie significative.",
            "decision_points": [
                "Le montant est dans une plage habituelle.",
                "Le comportement correspond au profil normal.",
                "Le score de risque est faible.",
            ],
        }
    return {
        "summary": "Cette transaction présente un comportement différent des transactions normales et nécessite une validation utilisateur.",
        "detail": "Le modèle recommande une vérification utilisateur.",
        "decision_points": [
            "Montant inhabituel par rapport aux habitudes.",
            "Comportement différent des transactions de référence.",
            "Score de risque élevé détecté par le modèle.",
        ],
    }


def load_feedback_records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    if "feedback_timestamp" in frame.columns:
        frame["feedback_timestamp"] = pd.to_datetime(frame["feedback_timestamp"], errors="coerce")
    for column in ["amount", "fraud_probability", "prediction", "ground_truth_label"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def bootstrap_history_from_feedback(records_frame: pd.DataFrame) -> None:
    if st.session_state["prediction_history"] or records_frame.empty:
        return
    history_rows: list[dict[str, Any]] = []
    for _, row in records_frame.sort_values("feedback_timestamp").iterrows():
        feedback_status = (
            "Fraude confirmée" if row.get("user_feedback") == "reported_fraud" else "Transaction normale confirmée"
        )
        status_label = feedback_status if pd.notna(row.get("ground_truth_label")) else TRANSACTION_STATUS_LABELS.get(
            str(row.get("predicted_label")), "Normale"
        )
        history_rows.append(
            {
                "prediction_id": str(row.get("prediction_id")),
                "analysis_date": row.get("feedback_timestamp"),
                "amount": float(row.get("amount") or 0.0),
                "score": float(row.get("fraud_probability") or 0.0),
                "status_label": status_label,
                "notification_status": "Email envoyé" if row.get("customer_email") else "Aucune notification",
                "feedback_status": feedback_status,
                "customer_email": row.get("customer_email") or "",
                "date_label": format_datetime_label(row.get("feedback_timestamp")),
            }
        )
    st.session_state["prediction_history"] = history_rows
    st.session_state["recent_predictions"] = history_rows[-5:]


def append_prediction_to_state(
    payload: dict[str, Any],
    prediction: dict[str, Any],
    explanation: dict[str, Any],
) -> None:
    timestamp = datetime.now()
    status_label = TRANSACTION_STATUS_LABELS.get(prediction["predicted_label"], "Normale")
    history_entry = {
        "prediction_id": prediction["prediction_id"],
        "analysis_date": timestamp,
        "date_label": timestamp.strftime("%d/%m/%Y %H:%M"),
        "amount": payload["amount"],
        "score": float(prediction["fraud_probability"]),
        "status_label": status_label,
        "notification_status": "En attente" if prediction["prediction"] == 1 else "Aucune notification",
        "feedback_status": "-",
        "customer_email": payload.get("customer_email") or "",
        "payload": payload,
        "prediction": prediction,
        "explanation": explanation,
    }
    st.session_state["prediction_history"].append(history_entry)
    st.session_state["recent_predictions"] = st.session_state["prediction_history"][-6:]
    st.session_state["last_prediction"] = history_entry


def update_history_entry(prediction_id: str, **updates: Any) -> None:
    for row in st.session_state["prediction_history"]:
        if row["prediction_id"] == prediction_id:
            row.update(updates)
    if st.session_state.get("last_prediction") and st.session_state["last_prediction"]["prediction_id"] == prediction_id:
        st.session_state["last_prediction"].update(updates)


def current_history_frame() -> pd.DataFrame:
    history = st.session_state.get("prediction_history", [])
    if not history:
        return pd.DataFrame()
    frame = ensure_history_score(pd.DataFrame(history))
    if "analysis_date" in frame.columns:
        frame["analysis_date"] = pd.to_datetime(frame["analysis_date"], errors="coerce")
    return frame.sort_values("analysis_date", ascending=False)


def render_sidebar(health: dict[str, Any] | None) -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-shell">
                <div class="brand-title">{APP_NAME}</div>
                <div class="brand-subtitle">
                    Détection intelligente de fraude bancaire<br/>
                    démonstration MLOps académique.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if health is not None:
            st.success(f"API connectée • {health['model_name']}")
            st.caption(f"Threshold {health['threshold']:.2f}")
        else:
            st.error("Mode démo : API FastAPI indisponible.")

        page_labels = [f"{icon} {label}" for _, label, icon in SIDEBAR_PAGES]
        keys_by_label = {f"{icon} {label}": key for key, label, icon in SIDEBAR_PAGES}
        default_label = next(
            label for key, label, icon in SIDEBAR_PAGES if key == st.session_state.get("selected_page", "dashboard")
        )
        selected_label = st.radio(
            "Navigation",
            options=page_labels,
            index=page_labels.index(next(label for label in page_labels if default_label in label)),
            label_visibility="collapsed",
        )
        selected_page = keys_by_label[selected_label]
        st.session_state["selected_page"] = selected_page

        st.markdown("---")
        st.caption("Intégration future")
        st.code("POST /predict\nPOST /notify_user\nn8n webhook\nEvidently report", language="text")
        st.caption(f"URL API démo attendue : {DEFAULT_API_HINT}")

    return selected_page


def render_dashboard(health: dict[str, Any] | None, history_frame: pd.DataFrame) -> None:
    page_header("Dashboard", "Vue globale du système de détection de fraude bancaire.")
    render_status_strip(health)
    render_quick_actions()

    use_real_history = not history_frame.empty and len(history_frame) >= 20
    if use_real_history:
        total_transactions = int(len(history_frame))
        suspicious_count = int(history_frame["status_label"].isin(["Suspecte", "Fraude confirmée"]).sum())
        fraud_rate = suspicious_count / max(total_transactions, 1)
        model_status = "Stable" if health is not None else "Actif"
    else:
        total_transactions = 12450
        suspicious_count = 238
        fraud_rate = 0.0191
        model_status = "Stable"

    metrics = [
        ("🧾", "linear-gradient(135deg,#2f7cff,#5aa0ff)", "Transactions analysées", f"{total_transactions:,}".replace(",", " "), "+245 ce mois", "#16a34a"),
        ("⚠️", "linear-gradient(135deg,#fb7185,#ff9a7a)", "Fraudes détectées", str(suspicious_count), "+12 ce mois", "#dc2626"),
        ("📈", "linear-gradient(135deg,#22c55e,#4ade80)", "Taux de fraude", f"{fraud_rate * 100:.2f} %", "+0.15 % vs mois dernier", "#16a34a"),
        ("🤖", "linear-gradient(135deg,#8b5cf6,#a78bfa)", "Statut du modèle", model_status, "Modèle actif", "#2563eb"),
    ]
    metric_columns = st.columns(4)
    for column, metric in zip(metric_columns, metrics, strict=False):
        with column:
            render_metric_card(*metric)

    action_columns = st.columns(3)
    with action_columns[0]:
        if st.button("Nouvelle prediction", width="stretch"):
            st.session_state["selected_page"] = "prediction"
            st.rerun()
    with action_columns[1]:
        if st.button("Ouvrir Agent IA", width="stretch"):
            st.session_state["selected_page"] = "agent"
            st.rerun()
    with action_columns[2]:
        if st.button("Voir monitoring", width="stretch"):
            st.session_state["selected_page"] = "monitoring"
            st.rerun()

    chart_column, table_column = st.columns([0.95, 1.05], gap="large")
    with chart_column:
        open_panel("Évolution du taux de fraude", "Suivi de l’évolution du taux de fraude dans le temps.")
        st.line_chart(build_dashboard_chart_frame(history_frame), height=230)
        close_panel()

    with table_column:
        open_panel("Dernières prédictions", "Les plus récentes analyses disponibles dans le système.")
        st.markdown(
            '<div class="section-meta">Lecture rapide des derniers cas, utile pour suivre les alertes, les notifications et le niveau de risque.</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            build_dashboard_latest_frame_filtered(history_frame, "Tous"),
            width="stretch",
            hide_index=True,
            height=188,
        )
        close_panel()


def render_model_feature_reference() -> None:
    open_panel(
        "Variables utilisées à l’entraînement",
        "Le formulaire reprend exactement les features du modèle supervisé utilisé en production.",
    )
    st.code(", ".join(MODEL_INPUT_FEATURES), language="text")
    close_panel()


def render_prediction_result(last_prediction: dict[str, Any]) -> None:
    prediction = last_prediction["prediction"]
    payload = last_prediction["payload"]
    simple_explanation = last_prediction["simple_explanation"]
    render_workflow_stepper(last_prediction)
    is_suspect = prediction["prediction"] == 1
    card_class = "result-card result-risk" if is_suspect else "result-card result-safe"
    title = "Transaction suspecte" if is_suspect else "Transaction normale"
    copy = (
        "Le modèle recommande une vérification utilisateur."
        if is_suspect
        else "Le modèle ne détecte pas d'anomalie significative."
    )
    score_color = "#ea580c" if is_suspect else "#16a34a"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-end;">
                <div>
                    <div class="result-title">{title}</div>
                    <div class="result-copy">{copy}</div>
                </div>
                <div style="text-align:right;">
                    <div class="score-label">Probabilité de fraude</div>
                    <div class="score-value" style="color:{score_color};">{prediction['fraud_probability'] * 100:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    progress_value = max(min(float(prediction["fraud_probability"]), 1.0), 0.0)
    st.progress(progress_value)

    detail_left, detail_right = st.columns(2, gap="large")
    with detail_left:
        open_panel("Détails de la transaction", "Synthèse des informations transmises au modèle.")
        detail_pairs = [
            ("Montant", format_currency(payload["amount"])),
            ("Temps", str(payload["step"])),
            ("Email utilisateur", payload.get("customer_email") or "-"),
        ]
        for label, value in detail_pairs:
            st.markdown(f'<div class="detail-label">{label}</div><div class="detail-value">{value}</div>', unsafe_allow_html=True)
        close_panel()

    with detail_right:
        open_panel("Résultat de la prédiction", "Décision du modèle et contexte de scoring.")
        detail_pairs = [
            ("Date de l'analyse", last_prediction["analysis_date"].strftime("%d/%m/%Y %H:%M")),
            ("ID de transaction", prediction["prediction_id"]),
            ("Statut", title),
        ]
        for label, value in detail_pairs:
            st.markdown(f'<div class="detail-label">{label}</div><div class="detail-value">{value}</div>', unsafe_allow_html=True)
        close_panel()

    open_panel("Explication de la décision", "Message simplifié pour une démonstration claire du raisonnement du système.")
    st.info(simple_explanation["summary"])
    for point in simple_explanation["decision_points"]:
        st.markdown(f"- {point}")
    close_panel()


def render_prediction_page(client: FraudApiClient, api_available: bool) -> None:
    page_header("Nouvelle prédiction", "Saisissez les informations d’une transaction pour obtenir une prédiction.")
    prediction_health = None
    if api_available:
        prediction_health = {
            "model_name": "Service actif",
            "threshold": 0.0,
            "agent_provider": "n8n",
            "agent_ready": True,
            "production_data_path": str(PRODUCTION_DATA_PATH),
        }
    render_status_strip(prediction_health)
    render_workflow_stepper(st.session_state.get("last_prediction"))
    if not api_available:
        st.error("Impossible de contacter l'API FastAPI. Vérifie que le service de prédiction est démarré.")

    render_model_feature_reference()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Formulaire d\'analyse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-caption">Les champs ci-dessous correspondent exactement aux variables utilisées pour '
        'l’entraînement du modèle de fraude. L’email utilisateur sert uniquement à la notification.</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form", clear_on_submit=False):
        row_one = st.columns(4)
        step = row_one[0].number_input("step", min_value=0, value=45214, step=1)
        transaction_type = row_one[1].selectbox("type", options=["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"], index=1)
        amount = row_one[2].number_input("amount", min_value=0.0, value=980.0, step=1.0)
        customer_email = row_one[3].text_input("Email utilisateur", value="", placeholder="client@banque.com")

        row_two = st.columns(4)
        oldbalance_org = row_two[0].number_input("oldbalanceOrg", value=980.0, step=1.0)
        new_balance_orig = row_two[1].number_input("newbalanceOrig", value=0.0, step=1.0)
        oldbalance_dest = row_two[2].number_input("oldbalanceDest", value=0.0, step=1.0)
        newbalance_dest = row_two[3].number_input("newbalanceDest", value=0.0, step=1.0)

        row_three = st.columns(2)
        is_flagged_fraud = row_three[0].selectbox("isFlaggedFraud", options=[0, 1], index=0)
        explanation_language = row_three[1].selectbox(
            "Langue de l'explication",
            options=list(EXPLANATION_LANGUAGE_LABELS.keys()),
            index=0,
        )

        analyze = st.form_submit_button("Analyser la transaction", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not api_available:
            st.error("Analyse impossible : l'API de prédiction n'est pas disponible.")
        else:
            st.session_state["last_agent_response"] = None
            payload = record_transaction_payload(
                step=int(step),
                transaction_type=transaction_type,
                amount=amount,
                oldbalance_org=oldbalance_org,
                newbalance_orig=new_balance_orig,
                oldbalance_dest=oldbalance_dest,
                newbalance_dest=newbalance_dest,
                is_flagged_fraud=is_flagged_fraud,
                customer_email=customer_email,
            )
            try:
                prediction = client.predict(payload)
                try:
                    explanation = client.explain_prediction(
                        transaction=payload,
                        prediction=prediction,
                        language=EXPLANATION_LANGUAGE_LABELS[explanation_language],
                    )
                except requests.RequestException:
                    explanation = {
                        "analyst_summary": "LLM indisponible, explication simplifiée affichée.",
                        "customer_message": "",
                        "email_subject": "Validation de transaction",
                        "recommended_action": "request_customer_confirmation",
                        "reasons": [],
                    }
                simple_explanation = build_simple_explanation(prediction)
                append_prediction_to_state(
                    payload,
                    prediction,
                    explanation,
                )
                st.session_state["last_prediction"]["simple_explanation"] = simple_explanation
            except requests.RequestException as exc:
                st.error(f"Erreur API : {format_request_exception(exc)}")

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction is None:
        return

    render_prediction_result(last_prediction)

    prediction = last_prediction["prediction"]
    is_suspect = prediction["prediction"] == 1
    if is_suspect:
        if st.button("Rédiger une alerte à l’utilisateur", width="stretch"):
            update_history_entry(
                prediction["prediction_id"],
                notification_status="En attente",
            )
            st.session_state["selected_page"] = "agent"
            st.rerun()
    else:
        action_columns = st.columns(2)
        with action_columns[0]:
            if st.button("Retour", width="stretch"):
                st.session_state["selected_page"] = "dashboard"
                st.rerun()
        with action_columns[1]:
            if st.button("Analyser une autre transaction", width="stretch"):
                st.session_state["last_prediction"] = None
                st.rerun()


def merged_history_frame() -> pd.DataFrame:
    session_history = current_history_frame()
    if not session_history.empty:
        return session_history
    return demo_history_frame()


def render_agent_page(client: FraudApiClient, api_available: bool) -> None:
    page_header("Agent IA", "Génération et envoi de messages de validation à l’utilisateur.")
    agent_health = None
    if api_available:
        agent_health = {
            "model_name": "Notification active",
            "threshold": 0.0,
            "agent_provider": "n8n",
            "agent_ready": True,
            "production_data_path": str(PRODUCTION_DATA_PATH),
        }
    render_status_strip(agent_health)
    last_prediction = st.session_state.get("last_prediction")
    send_history = st.session_state.get("ai_send_history", [])
    render_workflow_stepper(last_prediction)

    st.markdown(
        """
        <style>
        .agent-message-box {
            background: #ffffff;
            border: 1px solid rgba(17, 36, 65, 0.08);
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            min-height: 330px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
        }
        .agent-message-title {
            font-size: 1rem;
            font-weight: 700;
            color: #102647;
            margin-bottom: 1rem;
        }
        .agent-message-copy {
            color: #334155;
            font-size: 1rem;
            line-height: 1.75;
        }
        .agent-email-shell {
            background: #f5f8ff;
            border: 1px solid rgba(59, 130, 246, 0.12);
            border-radius: 18px;
            overflow: hidden;
        }
        .agent-email-header {
            background: linear-gradient(135deg, #102647 0%, #1d4ed8 100%);
            color: #ffffff;
            padding: 1.25rem 1.4rem;
        }
        .agent-email-kicker {
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.82;
            margin-bottom: 0.45rem;
        }
        .agent-email-subject {
            font-size: 1.32rem;
            font-weight: 700;
            line-height: 1.3;
        }
        .agent-email-body {
            padding: 1.35rem 1.4rem 1.45rem 1.4rem;
        }
        .agent-alert-card {
            background: linear-gradient(180deg, #fff7ed 0%, #fffbeb 100%);
            border: 1px solid #fed7aa;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
        }
        .agent-alert-label {
            color: #9a3412;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .agent-alert-value {
            color: #7c2d12;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        .agent-alert-meta {
            color: #9a3412;
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }
        .agent-message-copy p {
            margin: 0 0 0.9rem 0;
        }
        .agent-reason-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            margin-top: 1rem;
        }
        .agent-reason-title {
            color: #102647;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .agent-reason-list {
            margin: 0;
            padding-left: 1.1rem;
            color: #475569;
            line-height: 1.65;
        }
        .agent-cta-row {
            display: flex;
            gap: 0.8rem;
            flex-wrap: wrap;
            margin-top: 1.1rem;
        }
        .agent-cta-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.95rem 1.15rem;
            border-radius: 999px;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(135deg, #0f9f4b 0%, #16a34a 100%);
            min-width: 248px;
            box-shadow: 0 14px 28px rgba(22, 163, 74, 0.22);
            border: 1px solid rgba(255, 255, 255, 0.22);
            letter-spacing: -0.01em;
        }
        .agent-cta-button.secondary {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            box-shadow: 0 14px 28px rgba(220, 38, 38, 0.18);
        }
        .agent-email-footnote {
            color: #64748b;
            font-size: 0.83rem;
            line-height: 1.6;
            margin-top: 1rem;
        }
        .agent-side-box {
            background: #ffffff;
            border: 1px solid rgba(17, 36, 65, 0.08);
            border-radius: 18px;
            padding: 1.2rem 1.2rem 0.9rem 1.2rem;
        }
        .agent-side-title {
            font-size: 1rem;
            font-weight: 700;
            color: #102647;
            margin-bottom: 1rem;
        }
        .agent-field {
            margin-bottom: 1rem;
        }
        .agent-field-label {
            color: #6b7280;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }
        .agent-field-value {
            color: #102647;
            font-size: 1rem;
            font-weight: 600;
        }
        .agent-score {
            color: #ef4444;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .agent-empty {
            color: #64748b;
            line-height: 1.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if last_prediction is None:
        open_panel("Agent IA", "Aucune transaction suspecte n’est prête pour la génération de message.")
        st.info("Analyse d’abord une transaction puis clique sur « Rédiger une alerte à l’utilisateur ».")
        close_panel()
    else:
        prediction = last_prediction["prediction"]
        payload = last_prediction["payload"]
        explanation = last_prediction.get("explanation", {})
        email_subject = explanation.get(
            "email_subject",
            f"Action requise concernant votre transaction de {int(payload['amount'])} EUR",
        )
        message = explanation.get(
            "customer_message",
            (
                "Bonjour,\n\n"
                f"Nous avons detecte une activite inhabituelle concernant une transaction de {format_currency(payload['amount'])} "
                f"effectuee le {last_prediction['analysis_date'].strftime('%d/%m/%Y a %H:%M')}.\n\n"
                "Merci de confirmer si cette operation est bien de votre initiative.\n\n"
                "Cordialement,\n"
                "L'equipe MyBank"
            ),
        )
        message_lines = [
            f"<p>{html.escape(line)}</p>"
            for line in message.splitlines()
            if line.strip()
        ]
        reason_items = explanation.get("reasons") or [
            "Probabilite de fraude elevee detectee par le modele.",
            "Transaction atypique par rapport aux operations observees.",
        ]
        reason_markup = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reason_items[:4])
        brand_logo_data_uri = load_brand_logo_data_uri()
        brand_logo_markup = (
            f'<img src="{brand_logo_data_uri}" alt="{html.escape(BANK_BRAND_NAME)}" style="display:block;width:76px;height:auto;margin-bottom:14px;">'
            if brand_logo_data_uri
            else ""
        )

        layout_left, layout_right = st.columns([1.75, 1.0], gap="large")
        with layout_left:
            open_panel("Message généré par l’IA", "Génération et envoi de message à l’utilisateur.")
            st.markdown(
                f"""
                <div class="agent-message-box">
                    <div class="agent-message-title">Apercu de l'email client</div>
                    <div class="agent-email-shell">
                        <div class="agent-email-header">
                            {brand_logo_markup}
                            <div class="agent-email-kicker">{html.escape(BANK_BRAND_NAME)}</div>
                            <div class="agent-email-subject">{html.escape(email_subject)}</div>
                        </div>
                        <div class="agent-email-body">
                            <div class="agent-alert-card">
                                <div class="agent-alert-label">Alerte de securite</div>
                                <div class="agent-alert-value">Transaction de {html.escape(format_currency(payload["amount"]))}</div>
                                <div class="agent-alert-meta">Probabilite de fraude : {html.escape(format_percent(prediction["fraud_probability"]))}</div>
                            </div>
                            <div class="agent-message-copy">{''.join(message_lines)}</div>
                            <div class="agent-reason-box">
                                <div class="agent-reason-title">Pourquoi cette transaction a ete signalee</div>
                                <ul class="agent-reason-list">{reason_markup}</ul>
                            </div>
                            <div class="agent-cta-row">
                                <span class="agent-cta-button">Confirmer une transaction legitime</span>
                                <span class="agent-cta-button secondary">Signaler une fraude confirmee</span>
                            </div>
                            <div class="agent-email-footnote">L'email final utilise des boutons d'action. Aucun lien brut ne doit apparaitre dans le texte du message.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            close_panel()

        with layout_right:
            open_panel("Destinataire", "Bloc opérationnel d’envoi de confirmation.")
            st.markdown(
                f"""
                <div class="agent-side-box">
                    <div class="agent-side-title">Destinataire</div>
                    <div class="agent-field">
                        <div class="agent-field-label">Email utilisateur</div>
                        <div class="agent-field-value">{payload.get("customer_email") or "-"}</div>
                    </div>
                    <div class="agent-side-title" style="margin-top:1.2rem;">Transaction</div>
                    <div class="agent-field">
                        <div class="agent-field-label">Montant</div>
                        <div class="agent-field-value">{format_currency(payload["amount"])}</div>
                    </div>
                    <div class="agent-field">
                        <div class="agent-field-label">Date</div>
                        <div class="agent-field-value">{last_prediction['analysis_date'].strftime("%d/%m/%Y %H:%M")}</div>
                    </div>
                    <div class="agent-field">
                        <div class="agent-field-label">Probabilité de fraude</div>
                        <div class="agent-score">{format_percent(prediction["fraud_probability"])}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("✈️ Envoyer la demande de confirmation", width="stretch"):
                if prediction["prediction"] == 0:
                    st.info("Aucune notification nécessaire pour une transaction normale.")
                elif not api_available:
                    st.error("Impossible de contacter l’agent IA : API indisponible.")
                elif not is_deliverable_customer_email(payload.get("customer_email")):
                    st.error("Adresse email utilisateur invalide ou non livrable.")
                else:
                    try:
                        response = client.notify_user(
                            transaction=payload,
                            prediction=prediction,
                        )
                        st.session_state["last_agent_response"] = response
                        update_history_entry(
                            prediction["prediction_id"],
                            notification_status="Email envoyé",
                        )
                        st.session_state["ai_send_history"].append(
                            {
                                "Date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                                "Email": response["recipient_email"],
                                "Montant": format_currency(payload["amount"]),
                                "Statut": "Envoyé",
                                "Réponse utilisateur": "En attente",
                            }
                        )
                        st.success("Email envoyé avec succès")
                    except requests.RequestException as exc:
                        st.session_state["last_agent_response"] = None
                        st.error(f"Impossible de contacter l’agent IA : {format_request_exception(exc)}")
            close_panel()

        last_agent_response = st.session_state.get("last_agent_response")
        if last_agent_response:
            open_panel("Retour du workflow", "Trace resumee du dernier appel n8n reçu par l application.")
            meta_left, meta_right, meta_third = st.columns(3)
            meta_left.metric("Statut", str(last_agent_response.get("status", "-")))
            meta_right.metric("Provider", str(last_agent_response.get("workflow_provider", "n8n")))
            meta_third.metric("Workflow", str(last_agent_response.get("workflow_status", "-")))
            st.caption(f"Destinataire : {last_agent_response.get('recipient_email', '-')}")
            st.caption(f"Sujet : {last_agent_response.get('email_subject', '-')}")
            close_panel()

    open_panel("Historique des envois", "Journal des demandes de confirmation envoyées à l’utilisateur.")
    if send_history:
        history_frame = pd.DataFrame(send_history).iloc[::-1].copy()
        history_frame["Statut"] = history_frame["Statut"].map(
            {
                "Envoyé": STATUS_DISPLAY["Email envoyé"],
                "Répondu": "🟢 Répondu",
                "En attente": STATUS_DISPLAY["En attente"],
                "Échec": STATUS_DISPLAY["Échec"],
            }
        ).fillna(history_frame["Statut"])
        st.dataframe(history_frame, width="stretch", hide_index=True)
    else:
        st.info("Aucun envoi n'a encore été déclenché dans cette session.")
    close_panel()


def render_monitoring_page(health: dict[str, Any] | None, history_frame: pd.DataFrame) -> None:
    page_header("Monitoring du modèle", "Suivi de la performance et de la santé du modèle.")
    history_frame = ensure_history_score(history_frame)
    actual_feedback = int((history_frame["feedback_status"] != "-").sum()) if not history_frame.empty else 18
    feedback_display = f"{actual_feedback} / {RETRAIN_TARGET}"
    remaining_feedback = max(RETRAIN_TARGET - actual_feedback, 0)
    drift_status = "Faible" if history_frame.empty or history_frame["score"].mean() < 0.7 else "Moyen"
    performance_status = "Stable" if health is not None else "Dégradée"

    metric_columns = st.columns(4)
    metric_data = [
        ("🧠", "linear-gradient(135deg,#2f7cff,#57a4ff)", "Feedbacks collectés", feedback_display, "objectif suivi", "#2563eb"),
        ("♻️", "linear-gradient(135deg,#16a34a,#4ade80)", "Prochain réentraînement", f"dans {remaining_feedback} feedbacks", "seuil : 20", "#16a34a"),
        ("📡", "linear-gradient(135deg,#8b5cf6,#a78bfa)", "Data drift", drift_status, "surveillance continue", "#7c3aed"),
        ("📉", "linear-gradient(135deg,#14b8a6,#2dd4bf)", "Performance actuelle", performance_status, "modèle en production", "#0f766e"),
    ]
    for column, metric in zip(metric_columns, metric_data, strict=False):
        with column:
            render_metric_card(*metric)

    left, right = st.columns([1.2, 0.95], gap="large")
    with left:
        open_panel("Évolution de la performance", "Suivi simulé ou réel de métriques comme le F1-score et le recall.")
        if len(history_frame) >= 3:
            perf = history_frame.sort_values("analysis_date").copy()
            perf["Date"] = perf["analysis_date"].dt.strftime("%d/%m")
            perf["F1-score"] = perf["score"].rolling(window=2, min_periods=1).mean()
            perf["Recall"] = perf["score"].rolling(window=3, min_periods=1).mean() * 0.92
            st.line_chart(perf.set_index("Date")[["F1-score", "Recall"]], height=280)
        else:
            st.line_chart(demo_monitoring_curve(), height=280)
        close_panel()

    with right:
        open_panel("Données", "Vue de synthèse sur les jeux de données de référence et de production.")
        st.markdown(f"**Nombre de lignes dans ref_data.csv** : {count_csv_rows(str(REFERENCE_DATA_PATH))}")
        st.markdown(f"**Nombre de lignes dans prod_data.csv** : {count_csv_rows(str(PRODUCTION_DATA_PATH))}")
        st.markdown(f"**Dernière mise à jour** : {format_datetime_label(datetime.now())}")
        st.markdown("**Dernier réentraînement** : 12/01/2026 14:30")
        close_panel()

    open_panel("Accès rapide", "Raccourcis pour les outils de monitoring et d’audit du pipeline MLOps.")
    access = st.columns(3)
    with access[0]:
        st.link_button("Voir le rapport Evidently", "http://localhost:8082", use_container_width=True)
    with access[1]:
        if st.button("Détails du data drift", use_container_width=True):
            st.session_state["monitoring_action_message"] = "Le data drift reste faible sur cette démo."
    with access[2]:
        if st.button("Historique des réentraînements", use_container_width=True):
            st.session_state["monitoring_action_message"] = "Dernier réentraînement simulé : 12/01/2026 14:30."
    if st.session_state.get("monitoring_action_message"):
        st.info(st.session_state["monitoring_action_message"])
    close_panel()


def render_monitoring_page_v2(
    client: FraudApiClient,
    health: dict[str, Any] | None,
    history_frame: pd.DataFrame,
) -> None:
    page_header("Monitoring du modÃ¨le", "Suivi de la performance et de la santÃ© du modÃ¨le.")
    history_frame = ensure_history_score(history_frame)
    actual_feedback = int((history_frame["feedback_status"] != "-").sum()) if not history_frame.empty else 18
    feedback_display = f"{actual_feedback} / {RETRAIN_TARGET}"
    remaining_feedback = max(RETRAIN_TARGET - actual_feedback, 0)
    drift_status = "Faible" if history_frame.empty or history_frame["score"].mean() < 0.7 else "Moyen"
    performance_status = "Stable" if health is not None else "DÃ©gradÃ©e"

    metric_columns = st.columns(4)
    metric_data = [
        ("ðŸ§ ", "linear-gradient(135deg,#2f7cff,#57a4ff)", "Feedbacks collectÃ©s", feedback_display, "objectif suivi", "#2563eb"),
        ("â™»ï¸", "linear-gradient(135deg,#16a34a,#4ade80)", "Prochain rÃ©entraÃ®nement", f"dans {remaining_feedback} feedbacks", "seuil : 20", "#16a34a"),
        ("ðŸ“¡", "linear-gradient(135deg,#8b5cf6,#a78bfa)", "Data drift", drift_status, "surveillance continue", "#7c3aed"),
        ("ðŸ“‰", "linear-gradient(135deg,#14b8a6,#2dd4bf)", "Performance actuelle", performance_status, "modÃ¨le en production", "#0f766e"),
    ]
    for column, metric in zip(metric_columns, metric_data, strict=False):
        with column:
            render_metric_card(*metric)

    left, right = st.columns([1.2, 0.95], gap="large")
    with left:
        open_panel("Ã‰volution de la performance", "Suivi simulÃ© ou rÃ©el de mÃ©triques comme le F1-score et le recall.")
        if len(history_frame) >= 3:
            perf = history_frame.sort_values("analysis_date").copy()
            perf["Date"] = perf["analysis_date"].dt.strftime("%d/%m")
            perf["F1-score"] = perf["score"].rolling(window=2, min_periods=1).mean()
            perf["Recall"] = perf["score"].rolling(window=3, min_periods=1).mean() * 0.92
            st.line_chart(perf.set_index("Date")[["F1-score", "Recall"]], height=280)
        else:
            st.line_chart(demo_monitoring_curve(), height=280)
        close_panel()

    with right:
        open_panel("DonnÃ©es", "Vue de synthÃ¨se sur les jeux de donnÃ©es de rÃ©fÃ©rence et de production.")
        reference_path = str(health.get("monitoring_reference_data_path", REFERENCE_DATA_PATH)) if health else str(REFERENCE_DATA_PATH)
        st.markdown(f"**Nombre de lignes dans ref_data.csv** : {count_csv_rows(reference_path)}")
        st.markdown(f"**Nombre de lignes dans prod_data.csv** : {count_csv_rows(str(PRODUCTION_DATA_PATH))}")
        st.markdown(f"**DerniÃ¨re mise Ã  jour** : {format_datetime_label(datetime.now())}")
        st.markdown("**Dernier rÃ©entraÃ®nement** : 12/01/2026 14:30")
        close_panel()

    open_panel("Rapports Evidently", "GÃ©nÃ©rez puis ouvrez les rapports HTML de Data Drift et de Classification.")
    if st.session_state["api_available"]:
        if st.button("GÃ©nÃ©rer les rapports Evidently", use_container_width=True):
            try:
                st.session_state["monitoring_reports"] = client.generate_monitoring_reports(force_refresh=True)
                st.success("Rapports Evidently gÃ©nÃ©rÃ©s avec succÃ¨s.")
            except requests.RequestException as exc:
                st.error(f"Impossible de gÃ©nÃ©rer les rapports Evidently : {exc}")
    else:
        st.info("Le mode dÃ©mo n'a pas accÃ¨s Ã  FastAPI. Connecte l'API pour gÃ©nÃ©rer les rapports Evidently.")

    report_bundle = st.session_state.get("monitoring_reports")
    if report_bundle:
        report_meta_left, report_meta_right = st.columns(2)
        report_meta_left.caption(f"DerniÃ¨re gÃ©nÃ©ration : {report_bundle.get('generated_at', '-')}")
        report_meta_right.caption(
            f"RÃ©fÃ©rence : {report_bundle.get('reference_rows', 0)} lignes | Production : {report_bundle.get('production_rows', 0)} lignes"
        )
        report_links = st.columns(2)
        with report_links[0]:
            st.link_button("Ouvrir Data Drift Report", report_bundle["data_drift_report_url"], use_container_width=True)
        with report_links[1]:
            st.link_button("Ouvrir Classification Report", report_bundle["classification_report_url"], use_container_width=True)
    close_panel()

    open_panel("AccÃ¨s rapide", "Raccourcis pour les outils de monitoring et d'audit du pipeline MLOps.")
    access = st.columns(2)
    with access[0]:
        if st.button("DÃ©tails du data drift", use_container_width=True):
            st.session_state["monitoring_action_message"] = "Le data drift reste faible sur cette dÃ©mo."
    with access[1]:
        if st.button("Historique des rÃ©entraÃ®nements", use_container_width=True):
            st.session_state["monitoring_action_message"] = "Dernier rÃ©entraÃ®nement simulÃ© : 12/01/2026 14:30."
    if st.session_state.get("monitoring_action_message"):
        st.info(st.session_state["monitoring_action_message"])
    close_panel()


def render_monitoring_page_v3(
    client: FraudApiClient,
    health: dict[str, Any] | None,
    history_frame: pd.DataFrame,
) -> None:
    page_header("Monitoring du modèle", "Suivi de la performance et de la santé du modèle.")
    history_frame = ensure_history_score(history_frame)

    if st.session_state["api_available"] and st.session_state.get("monitoring_reports") is None:
        try:
            st.session_state["monitoring_reports"] = client.monitoring_reports_summary()
        except requests.RequestException:
            pass

    report_bundle = st.session_state.get("monitoring_reports") or {}
    actual_feedback = int((history_frame["feedback_status"] != "-").sum()) if not history_frame.empty else 0
    feedback_display = f"{actual_feedback} / {RETRAIN_TARGET}"
    remaining_feedback = max(RETRAIN_TARGET - actual_feedback, 0)

    drift_threshold = float(report_bundle.get("drift_threshold", 0.5) or 0.5)
    share_drifted = report_bundle.get("share_drifted_columns")
    if share_drifted is None:
        drift_status = "En attente"
        drift_note = "générez un rapport pour calculer le drift"
    else:
        share_drifted = float(share_drifted)
        if share_drifted >= 0.75:
            drift_status = "Élevé"
        elif share_drifted >= drift_threshold:
            drift_status = "Moyen"
        else:
            drift_status = "Faible"
        drift_note = (
            f"{int(report_bundle.get('drifted_columns', 0))}/"
            f"{int(report_bundle.get('total_columns', 0))} colonnes en drift"
        )

    f1_value = report_bundle.get("f1_score")
    if f1_value is None:
        performance_status = "Stable" if health is not None else "À surveiller"
        performance_note = "modèle en production"
    else:
        f1_value = float(f1_value)
        if f1_value >= 0.75:
            performance_status = "Stable"
        elif f1_value >= 0.55:
            performance_status = "À surveiller"
        else:
            performance_status = "Dégradée"
        performance_note = f"F1-score : {f1_value:.2f}"

    metric_columns = st.columns(4)
    metric_data = [
        ("🧠", "linear-gradient(135deg,#2f7cff,#57a4ff)", "Feedbacks collectés", feedback_display, "objectif suivi", "#2563eb"),
        ("♻️", "linear-gradient(135deg,#16a34a,#4ade80)", "Prochain réentraînement", f"dans {remaining_feedback} feedbacks", "seuil : 20", "#16a34a"),
        ("📡", "linear-gradient(135deg,#8b5cf6,#a78bfa)", "Data drift", drift_status, drift_note, "#7c3aed"),
        ("📉", "linear-gradient(135deg,#14b8a6,#2dd4bf)", "Performance actuelle", performance_status, performance_note, "#0f766e"),
    ]
    for column, metric in zip(metric_columns, metric_data, strict=False):
        with column:
            render_metric_card(*metric)

    left, right = st.columns([1.2, 0.95], gap="large")
    with left:
        open_panel("Évolution de la performance", "Suivi simulé ou réel de métriques comme le F1-score et le recall.")
        if len(history_frame) >= 3:
            perf = history_frame.sort_values("analysis_date").copy()
            perf["Date"] = perf["analysis_date"].dt.strftime("%d/%m")
            perf["F1-score"] = perf["score"].rolling(window=2, min_periods=1).mean()
            perf["Recall"] = perf["score"].rolling(window=3, min_periods=1).mean() * 0.92
            st.line_chart(perf.set_index("Date")[["F1-score", "Recall"]], height=280)
        else:
            st.line_chart(demo_monitoring_curve(), height=280)
        close_panel()

    with right:
        open_panel("Données", "Vue de synthèse sur les jeux de données de référence et de production.")
        reference_path = str(health.get("monitoring_reference_data_path", REFERENCE_DATA_PATH)) if health else str(REFERENCE_DATA_PATH)
        st.markdown(f"**Nombre de lignes dans ref_data.csv** : {count_csv_rows(reference_path)}")
        st.markdown(f"**Nombre de lignes dans prod_data.csv** : {count_csv_rows(str(PRODUCTION_DATA_PATH))}")
        st.markdown(f"**Dernière mise à jour** : {report_bundle.get('generated_at', format_datetime_label(datetime.now()))}")
        st.markdown("**Dernier réentraînement** : 12/01/2026 14:30")
        close_panel()

    open_panel("Rapports Evidently", "Générez puis ouvrez les rapports HTML de Data Drift et de Classification.")
    if st.session_state["api_available"]:
        if st.button("Générer les rapports Evidently", width="stretch"):
            try:
                st.session_state["monitoring_reports"] = client.generate_monitoring_reports(force_refresh=True)
                st.success("Rapports Evidently générés avec succès.")
            except requests.RequestException as exc:
                st.error(f"Impossible de générer les rapports Evidently : {format_request_exception(exc)}")
    else:
        st.info("Le mode démo n'a pas accès à FastAPI. Connecte l'API pour générer les rapports Evidently.")

    report_bundle = st.session_state.get("monitoring_reports") or {}
    if report_bundle:
        report_meta_left, report_meta_right = st.columns(2)
        report_meta_left.caption(f"Dernière génération : {report_bundle.get('generated_at', '-')}")
        report_meta_right.caption(
            f"Référence : {report_bundle.get('reference_rows', 0)} lignes | Production : {report_bundle.get('production_rows', 0)} lignes"
        )
        report_links = st.columns(2)
        with report_links[0]:
            st.link_button("Ouvrir Data Drift Report", report_bundle["data_drift_report_url"], width="stretch")
        with report_links[1]:
            st.link_button("Ouvrir Classification Report", report_bundle["classification_report_url"], width="stretch")
    close_panel()

    open_panel("Accès rapide", "Raccourcis pour les outils de monitoring et d'audit du pipeline MLOps.")
    access = st.columns(2)
    with access[0]:
        if st.button("Détails du data drift", width="stretch"):
            if report_bundle and report_bundle.get("share_drifted_columns") is not None:
                st.session_state["monitoring_action_message"] = (
                    f"{int(report_bundle.get('drifted_columns', 0))} colonnes en drift sur "
                    f"{int(report_bundle.get('total_columns', 0))} "
                    f"({float(report_bundle.get('share_drifted_columns', 0.0)):.0%})."
                )
            else:
                st.session_state["monitoring_action_message"] = (
                    "Générez d'abord un rapport Evidently pour afficher le détail du drift."
                )
    with access[1]:
        if st.button("Historique des réentraînements", width="stretch"):
            st.session_state["monitoring_action_message"] = "Dernier réentraînement simulé : 12/01/2026 14:30."
    if st.session_state.get("monitoring_action_message"):
        st.info(st.session_state["monitoring_action_message"])
    close_panel()


def try_load_backend(client: FraudApiClient) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    try:
        health = client.healthcheck()
        st.session_state["api_available"] = True
    except requests.RequestException:
        st.session_state["api_available"] = False
        return None, pd.DataFrame()

    try:
        feedback_records = client.feedback_records(limit=200)
    except requests.RequestException:
        feedback_records = []
    return health, load_feedback_records_frame(feedback_records)


def render_top_mode_banner(api_available: bool) -> None:
    if api_available:
        st.success("Mode connecté : l'application communique avec FastAPI pour la prédiction et la notification.")
    else:
        st.warning("Mode démo : les pages Dashboard, Historique et Monitoring utilisent des données simulées.")


def main() -> None:
    initialize_state()
    client = FraudApiClient.from_environment()
    health, feedback_records_frame = try_load_backend(client)
    bootstrap_history_from_feedback(feedback_records_frame)
    page = render_sidebar(health)
    render_top_mode_banner(st.session_state["api_available"])
    history_frame = merged_history_frame()

    if page == "dashboard":
        render_dashboard(health, history_frame)
    elif page == "prediction":
        render_prediction_page(client, st.session_state["api_available"])
    elif page == "agent":
        render_agent_page(client, st.session_state["api_available"])
    elif page == "monitoring":
        render_monitoring_page_v3(client, health, history_frame)


if __name__ == "__main__":
    main()
