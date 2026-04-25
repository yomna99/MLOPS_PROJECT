from __future__ import annotations

from io import StringIO

import pandas as pd
import requests
import streamlit as st

from app.client import FraudApiClient


REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
]

TRANSACTION_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]


st.set_page_config(
    page_title="FraudShield Console",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 224, 163, 0.45), transparent 30%),
            radial-gradient(circle at top right, rgba(155, 210, 255, 0.35), transparent 28%),
            linear-gradient(180deg, #f5efe3 0%, #f7f7f4 45%, #eef2f5 100%);
    }
    .hero-card {
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(44, 62, 80, 0.08);
        box-shadow: 0 18px 35px rgba(44, 62, 80, 0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #16324f;
        margin-bottom: 0.35rem;
    }
    .hero-caption {
        color: #425466;
        font-size: 1rem;
        line-height: 1.6;
    }
    .risk-card {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid rgba(22, 50, 79, 0.08);
        background: rgba(255,255,255,0.75);
        box-shadow: 0 12px 24px rgba(44, 62, 80, 0.06);
    }
    .risk-high {
        background: linear-gradient(135deg, rgba(255, 108, 92, 0.14), rgba(255,255,255,0.86));
        border: 1px solid rgba(210, 63, 49, 0.20);
    }
    .risk-low {
        background: linear-gradient(135deg, rgba(88, 184, 135, 0.14), rgba(255,255,255,0.86));
        border: 1px solid rgba(55, 121, 87, 0.20);
    }
    .section-note {
        color: #4d5b6a;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def build_sample_batch_csv() -> bytes:
    sample = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 7,
                "type": "PAYMENT",
                "amount": 9839.64,
                "oldbalanceOrg": 170136.0,
                "newbalanceOrig": 160296.36,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
            {
                "step": 1,
                "type": "CASH_OUT",
                "amount": 181.0,
                "oldbalanceOrg": 181.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 21182.0,
                "newbalanceDest": 0.0,
                "isFlaggedFraud": 0,
            },
        ]
    )
    return sample.to_csv(index=False).encode("utf-8")


def render_metric_cards(health: dict | None) -> None:
    metric_columns = st.columns(3)
    model_name = health["model_name"] if health else "Unavailable"
    threshold = f"{health['threshold']:.2f}" if health else "-"
    artifact_path = health["artifact_path"] if health else "-"

    metric_columns[0].metric("Model in production", model_name)
    metric_columns[1].metric("Decision threshold", threshold)
    metric_columns[2].metric("Artifact path", artifact_path)


def validate_batch_frame(dataframe: pd.DataFrame) -> tuple[bool, str]:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    return True, ""


def normalize_batch_records(dataframe: pd.DataFrame) -> list[dict]:
    cleaned = dataframe[REQUIRED_COLUMNS].copy()
    cleaned["step"] = cleaned["step"].astype(int)
    cleaned["amount"] = cleaned["amount"].astype(float)
    cleaned["oldbalanceOrg"] = cleaned["oldbalanceOrg"].astype(float)
    cleaned["newbalanceOrig"] = cleaned["newbalanceOrig"].astype(float)
    cleaned["oldbalanceDest"] = cleaned["oldbalanceDest"].astype(float)
    cleaned["newbalanceDest"] = cleaned["newbalanceDest"].astype(float)
    cleaned["isFlaggedFraud"] = cleaned["isFlaggedFraud"].astype(int)
    cleaned["type"] = cleaned["type"].astype(str)
    return cleaned.to_dict(orient="records")


def render_single_result(result: dict) -> None:
    high_risk = result["prediction"] == 1
    card_class = "risk-card risk-high" if high_risk else "risk-card risk-low"
    label = "Suspicious transaction" if high_risk else "Looks safe"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div style="font-size:0.85rem; letter-spacing:0.08em; text-transform:uppercase; color:#51606f;">
                Decision
            </div>
            <div style="font-size:1.7rem; font-weight:700; color:#14324a; margin-top:0.25rem;">
                {label}
            </div>
            <div style="margin-top:0.45rem; color:#415262;">
                Probability score: <strong>{result['fraud_probability']:.4f}</strong> |
                Threshold: <strong>{result['threshold']:.2f}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


client = FraudApiClient.from_environment()

try:
    health = client.healthcheck()
except requests.RequestException:
    health = None

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">FraudShield Console</div>
        <div class="hero-caption">
            Review a single banking transaction, upload a CSV batch for screening,
            and inspect fraud scores from the production model in one place.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_metric_cards(health)

with st.sidebar:
    st.subheader("Platform status")
    if health is not None:
        st.success(f"Connected to API: {health['model_name']}")
        st.caption(f"Threshold {health['threshold']:.2f}")
    else:
        st.error("API unavailable. Start the API or the Docker Compose stack first.")

    st.markdown('<div class="section-note">Recommended launch command:</div>', unsafe_allow_html=True)
    st.code("docker compose -f deployment/docker/compose.yaml up --build", language="powershell")
    st.download_button(
        "Download sample batch CSV",
        data=build_sample_batch_csv(),
        file_name="sample_batch_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )


single_tab, batch_tab, about_tab = st.tabs(
    ["Single transaction", "Batch screening", "Model details"]
)

with single_tab:
    st.markdown("### Screen one transaction")
    st.caption("Use this workflow for analyst demos or manual review.")

    left_col, right_col = st.columns([1.15, 0.85], gap="large")
    with left_col:
        with st.form("fraud-prediction-form", clear_on_submit=False):
            step = st.number_input("Step", min_value=0, value=1, step=1)
            transaction_type = st.selectbox(
                "Transaction type",
                options=TRANSACTION_TYPES,
                index=1,
            )
            amount = st.number_input("Amount", min_value=0.0, value=181.0, step=1.0)
            oldbalance_org = st.number_input(
                "Origin balance before transaction",
                value=181.0,
                step=1.0,
            )
            newbalance_orig = st.number_input(
                "Origin balance after transaction",
                value=0.0,
                step=1.0,
            )
            oldbalance_dest = st.number_input(
                "Destination balance before transaction",
                value=0.0,
                step=1.0,
            )
            newbalance_dest = st.number_input(
                "Destination balance after transaction",
                value=0.0,
                step=1.0,
            )
            is_flagged_fraud = st.selectbox("Flagged by business rule", options=[0, 1], index=0)
            submitted = st.form_submit_button("Run prediction", use_container_width=True)

    with right_col:
        st.markdown("#### Analyst hints")
        st.markdown(
            """
            - `TRANSFER` and `CASH_OUT` are typically the riskiest transaction types.
            - Large gaps between source and destination balances can be informative.
            - The API applies the production threshold automatically.
            """
        )

    if submitted:
        payload = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalance_org),
            "newbalanceOrig": float(newbalance_orig),
            "oldbalanceDest": float(oldbalance_dest),
            "newbalanceDest": float(newbalance_dest),
            "isFlaggedFraud": int(is_flagged_fraud),
        }

        try:
            result = client.predict(payload)
            render_single_result(result)
            st.json({"request_payload": payload, "api_response": result})
        except requests.RequestException as exc:
            st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.markdown("### Batch screening")
    st.caption("Upload a CSV batch, send it to the API, and export enriched results.")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing one or more transactions",
        type=["csv"],
        help="The file must include the required columns shown below.",
    )
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

    if uploaded_file is not None:
        try:
            batch_frame = pd.read_csv(uploaded_file)
            is_valid, message = validate_batch_frame(batch_frame)
            if not is_valid:
                st.error(message)
            else:
                preview = batch_frame[REQUIRED_COLUMNS].head(10)
                st.dataframe(preview, use_container_width=True)
                if st.button("Run batch prediction", type="primary", use_container_width=True):
                    payloads = normalize_batch_records(batch_frame)
                    result = client.predict_batch(payloads)
                    summary = result["summary"]
                    predictions = pd.DataFrame(result["predictions"])
                    enriched = batch_frame.reset_index(drop=True).copy()
                    enriched = pd.concat([enriched, predictions], axis=1)

                    metric_cols = st.columns(4)
                    metric_cols[0].metric("Transactions", summary["total_transactions"])
                    metric_cols[1].metric("Fraud alerts", summary["fraud_predictions"])
                    metric_cols[2].metric("Non-fraud", summary["non_fraud_predictions"])
                    metric_cols[3].metric(
                        "Avg score",
                        f"{summary['average_fraud_probability']:.4f}",
                    )

                    chart_data = pd.DataFrame(
                        {
                            "label": ["Fraud", "Not fraud"],
                            "count": [
                                summary["fraud_predictions"],
                                summary["non_fraud_predictions"],
                            ],
                        }
                    ).set_index("label")
                    st.bar_chart(chart_data)
                    st.dataframe(enriched, use_container_width=True)
                    st.download_button(
                        "Download enriched predictions",
                        data=enriched.to_csv(index=False).encode("utf-8"),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
        except (ValueError, requests.RequestException) as exc:
            st.error(f"Batch prediction failed: {exc}")

with about_tab:
    st.markdown("### Production model snapshot")
    if health is not None:
        st.json(health)
    st.markdown(
        """
        This console currently exposes:
        - a single-transaction decision path for manual review
        - a CSV batch screening path for operational usage
        - a download flow for enriched batch predictions

        Next natural upgrades:
        - explanation messages with an LLM layer
        - human feedback capture
        - production drift monitoring
        """
    )
