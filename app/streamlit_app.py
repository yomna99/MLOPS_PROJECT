from __future__ import annotations

import requests
import streamlit as st

from app.client import FraudApiClient


st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="",
    layout="centered",
)

client = FraudApiClient.from_environment()

st.title("Fraud Detection Demo")
st.caption("Simulate a banking transaction and ask the fraud API for a prediction.")

with st.sidebar:
    st.subheader("API Status")
    if st.button("Check API health", use_container_width=True):
        try:
            health = client.healthcheck()
            st.success(
                f"API ready: {health['model_name']} with threshold {health['threshold']:.2f}"
            )
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

with st.form("fraud-prediction-form"):
    step = st.number_input("Step", min_value=0, value=1, step=1)
    transaction_type = st.selectbox(
        "Transaction type",
        options=["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        index=1,
    )
    amount = st.number_input("Amount", min_value=0.0, value=181.0, step=1.0)
    oldbalance_org = st.number_input("Origin balance before transaction", value=181.0, step=1.0)
    newbalance_orig = st.number_input("Origin balance after transaction", value=0.0, step=1.0)
    oldbalance_dest = st.number_input("Destination balance before transaction", value=0.0, step=1.0)
    newbalance_dest = st.number_input("Destination balance after transaction", value=0.0, step=1.0)
    is_flagged_fraud = st.selectbox("Flagged by business rule", options=[0, 1], index=0)
    submitted = st.form_submit_button("Predict", use_container_width=True)

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
        if result["prediction"] == 1:
            st.error(
                f"Prediction: {result['predicted_label']} "
                f"(score={result['fraud_probability']:.4f}, threshold={result['threshold']:.2f})"
            )
        else:
            st.success(
                f"Prediction: {result['predicted_label']} "
                f"(score={result['fraud_probability']:.4f}, threshold={result['threshold']:.2f})"
            )

        st.json(
            {
                "request_payload": payload,
                "api_response": result,
            }
        )
    except requests.RequestException as exc:
        st.error(f"Prediction failed: {exc}")
