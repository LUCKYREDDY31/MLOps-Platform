"""Streamlit monitoring dashboard - model performance, drift status, live predictions."""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")

log = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "reports")
LOGS_DIR = os.environ.get("LOGS_DIR", "logs")

st.set_page_config(
    page_title="MLOps Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def api_get(path: str) -> dict | None:
    try:
        res = requests.get(f"{API_URL}{path}", timeout=5)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None


def api_post(path: str, data: dict) -> dict:
    try:
        res = requests.post(f"{API_URL}{path}", json=data, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}


st.sidebar.title("MLOps Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Churn Prediction", "Fraud Detection", "Drift Reports", "Alerts & Logs"],
)
st.sidebar.markdown("---")

health = api_get("/health")
if health:
    st.sidebar.success("API: Online")
    st.sidebar.caption(f"Models loaded: {', '.join(health.get('models_loaded', []))}")
else:
    st.sidebar.error("API: Offline")
    st.sidebar.caption("Start the API server first.")


if page == "Overview":
    st.title("Platform Overview")
    st.markdown("Real-time status of all models and the monitoring pipeline.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if health:
            st.metric("API Status", "Online")
        else:
            st.metric("API Status", "Offline")

    summary = load_json(os.path.join(MODELS_DIR, "training_summary.json"))
    if summary:
        churn_auc = summary.get("churn", {}).get("metrics", {}).get("roc_auc", 0)
        fraud_auc = summary.get("fraud", {}).get("metrics", {}).get("roc_auc", 0)
        with col2:
            st.metric("Churn Model AUC", f"{churn_auc:.4f}")
        with col3:
            st.metric("Fraud Model AUC", f"{fraud_auc:.4f}")
    else:
        with col2:
            st.metric("Churn Model AUC", "N/A")
        with col3:
            st.metric("Fraud Model AUC", "N/A")

    drift_check = load_json(os.path.join(REPORTS_DIR, "drift_check_result.json"))
    retrain_needed = drift_check.get("retrain_needed", {}) if drift_check else {}
    drift_any = any(retrain_needed.values())
    with col4:
        st.metric("Drift Status", "Drift Detected" if drift_any else "No Drift")

    st.markdown("---")

    st.subheader("Live Prediction Metrics")
    live_metrics = api_get("/metrics")
    if live_metrics and live_metrics.get("total", 0) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Churn Predictions**")
            churn_m = live_metrics.get("churn", {})
            if churn_m:
                st.dataframe(pd.DataFrame([churn_m]).T.rename(columns={0: "Value"}))
        with col2:
            st.markdown("**Fraud Predictions**")
            fraud_m = live_metrics.get("fraud", {})
            if fraud_m:
                st.dataframe(pd.DataFrame([fraud_m]).T.rename(columns={0: "Value"}))
    else:
        st.info("No live predictions yet. Use the prediction pages to generate some.")

    if summary:
        st.markdown("---")
        st.subheader("Training Performance")
        col1, col2 = st.columns(2)
        with col1:
            churn_metrics = summary.get("churn", {}).get("metrics", {})
            if churn_metrics:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(churn_metrics.keys(), churn_metrics.values(), color="#4C72B0")
                ax.set_title("Churn Model Metrics")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        with col2:
            fraud_metrics = summary.get("fraud", {}).get("metrics", {})
            if fraud_metrics:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(fraud_metrics.keys(), fraud_metrics.values(), color="#DD8452")
                ax.set_title("Fraud Model Metrics")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()


elif page == "Churn Prediction":
    st.title("Customer Churn Prediction")
    st.markdown("Enter customer details to get a real-time churn prediction.")

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Tenure (months)", 1, 72, 24)
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 60.0)
            total_charges = st.number_input("Total Charges ($)", value=float(monthly_charges * tenure))
        with col2:
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4, 5])
            support_calls = st.slider("Support Calls", 0, 10, 1)
            payment_delays = st.slider("Payment Delays", 0, 10, 0)
        with col3:
            contract_type = st.selectbox(
                "Contract Type", [
                    0, 1, 2], format_func=lambda x: [
                    "Month-to-Month", "1 Year", "2 Year"][x])
            internet_service = st.selectbox(
                "Internet Service", [
                    0, 1, 2], format_func=lambda x: [
                    "None", "DSL", "Fiber"][x])
            online_security = st.selectbox("Online Security", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            tech_support = st.selectbox("Tech Support", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            paperless_billing = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: ["No", "Yes"][x])

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        payload = {
            "tenure": tenure, "monthly_charges": monthly_charges,
            "total_charges": total_charges, "num_products": num_products,
            "support_calls": support_calls, "payment_delays": payment_delays,
            "contract_type": contract_type, "internet_service": internet_service,
            "online_security": online_security, "tech_support": tech_support,
            "paperless_billing": paperless_billing, "senior_citizen": senior_citizen,
        }
        result = api_post("/predict/churn", payload)
        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                label = result["label"]
                if label == "Churn":
                    st.error(f"Prediction: **{label}**")
                else:
                    st.success(f"Prediction: **{label}**")
            with col2:
                st.metric("Churn Probability", f"{result['probability']:.2%}")
            with col3:
                st.metric("Latency", f"{result['latency_ms']:.1f} ms")


elif page == "Fraud Detection":
    st.title("Transaction Fraud Detection")
    st.markdown("Enter transaction details to get a real-time fraud prediction.")

    with st.form("fraud_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Transaction Amount ($)", value=150.0, min_value=0.01)
            hour = st.slider("Hour of Day", 0, 23, 14)
            day_of_week = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: [
                                       "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x])
            merchant_category = st.slider("Merchant Category (0-19)", 0, 19, 5)
        with col2:
            distance_from_home = st.number_input("Distance from Home (km)", value=15.0, min_value=0.0)
            distance_from_last_txn = st.number_input("Distance from Last Txn (km)", value=5.0, min_value=0.0)
            ratio_to_median = st.number_input("Ratio to Median Spend", value=1.0, min_value=0.0)
        with col3:
            used_chip = st.selectbox("Used Chip", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            used_pin = st.selectbox("Used PIN", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            online_order = st.selectbox("Online Order", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            num_txn_last_hour = st.slider("Transactions in Last Hour", 0, 10, 1)

        submitted = st.form_submit_button("Check for Fraud")

    if submitted:
        payload = {
            "amount": amount, "hour": hour, "day_of_week": day_of_week,
            "merchant_category": merchant_category, "distance_from_home": distance_from_home,
            "distance_from_last_txn": distance_from_last_txn, "ratio_to_median": ratio_to_median,
            "used_chip": used_chip, "used_pin": used_pin, "online_order": online_order,
            "num_txn_last_hour": num_txn_last_hour,
        }
        result = api_post("/predict/fraud", payload)
        if "error" in result:
            st.error(f"Prediction failed: {result['error']}")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                label = result["label"]
                if label == "Fraud":
                    st.error(f"Prediction: **{label}**")
                else:
                    st.success(f"Prediction: **{label}**")
            with col2:
                st.metric("Fraud Probability", f"{result['probability']:.2%}")
            with col3:
                st.metric("Latency", f"{result['latency_ms']:.1f} ms")


elif page == "Drift Reports":
    st.title("Data Drift Reports")

    for task in ["churn", "fraud"]:
        st.subheader(f"{task.capitalize()} Model")
        summary = load_json(os.path.join(REPORTS_DIR, f"{task}_drift_summary.json"))
        if summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                drifted = summary.get("dataset_drift_detected", False)
                if drifted:
                    st.error("Drift Detected")
                else:
                    st.success("No Drift")
            with col2:
                st.metric("Drift Share", f"{summary.get('drift_share', 0):.1%}")
            with col3:
                st.metric("Drifted Columns", summary.get("drifted_column_count", 0))

            report_path = summary.get("report_path", "")
            if os.path.exists(report_path):
                st.markdown(f"[View Full HTML Report]({report_path})")
        else:
            st.info(f"No drift report found for {task}. Run the monitoring pipeline first.")
        st.markdown("---")


elif page == "Alerts & Logs":
    st.title("Alerts & Logs")

    alert_log = os.path.join(LOGS_DIR, "alerts.jsonl")
    if os.path.exists(alert_log):
        alerts = []
        with open(alert_log) as f:
            for line in f:
                if line.strip():
                    alerts.append(json.loads(line))

        if alerts:
            df = pd.DataFrame(alerts[::-1])  # most recent first
            st.dataframe(df[["timestamp", "level", "task", "message"]], width="stretch")
        else:
            st.info("No alerts logged yet.")
    else:
        st.info("No alert log found. Run the monitoring pipeline to generate alerts.")

    st.markdown("---")
    st.subheader("Latest Business Report")
    report_path = os.path.join(REPORTS_DIR, "business_summary.md")
    if os.path.exists(report_path):
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.info("No business report found. Run the report generator first.")
