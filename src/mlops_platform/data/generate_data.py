"""Synthetic dataset generator for churn and fraud models."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

np.random.seed(42)


def make_churn_data(n: int = 50000) -> pd.DataFrame:
    """
    Generate a synthetic customer churn dataset.
    Features loosely mimic telecom/subscription churn patterns.
    """
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges = np.clip(total_charges, 0, None)

    num_products = np.random.randint(1, 6, n)
    support_calls = np.random.poisson(2, n)
    payment_delays = np.random.poisson(0.5, n)
    contract_type = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # month, 1yr, 2yr
    internet_service = np.random.choice([0, 1, 2], n)
    online_security = np.random.randint(0, 2, n)
    tech_support = np.random.randint(0, 2, n)
    paperless_billing = np.random.randint(0, 2, n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.85, 0.15])

    churn_score = (
        -0.03 * tenure
        + 0.008 * monthly_charges
        + 0.3 * (contract_type == 0).astype(int)
        - 0.2 * (contract_type == 2).astype(int)
        + 0.15 * (support_calls > 3).astype(int)
        + 0.2 * (payment_delays > 1).astype(int)
        - 0.1 * online_security
        + 0.05 * senior_citizen
        + np.random.normal(0, 0.3, n)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churn = (churn_prob > 0.5).astype(int)

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_products": num_products,
        "support_calls": support_calls,
        "payment_delays": payment_delays,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "paperless_billing": paperless_billing,
        "senior_citizen": senior_citizen,
        "churn": churn,
    })


def make_fraud_data(n: int = 100000) -> pd.DataFrame:
    """
    Generate a synthetic transaction fraud dataset.
    Fraud rate is kept low (~19%) to reflect skewed class distribution.
    """
    amount = np.round(np.random.lognormal(mean=4.0, sigma=1.5, size=n), 2)
    hour = np.random.randint(0, 24, n)
    day_of_week = np.random.randint(0, 7, n)
    merchant_category = np.random.randint(0, 20, n)
    distance_from_home = np.round(np.random.exponential(scale=30, size=n), 2)
    distance_from_last_txn = np.round(np.random.exponential(scale=10, size=n), 2)
    ratio_to_median = np.round(np.random.lognormal(0, 0.5, n), 4)
    used_chip = np.random.randint(0, 2, n)
    used_pin = np.random.randint(0, 2, n)
    online_order = np.random.randint(0, 2, n)
    num_txn_last_hour = np.random.poisson(2, n)

    fraud_score = (
        -3.5
        + 0.0003 * amount
        + 0.04 * distance_from_home
        + 0.06 * distance_from_last_txn
        + 0.5 * ratio_to_median
        - 0.4 * used_chip
        - 0.3 * used_pin
        + 0.6 * online_order
        + 0.2 * (num_txn_last_hour > 4).astype(int)
        + np.random.normal(0, 0.5, n)
    )
    fraud_prob = 1 / (1 + np.exp(-fraud_score))
    fraud = (fraud_prob > 0.5).astype(int)

    return pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "day_of_week": day_of_week,
        "merchant_category": merchant_category,
        "distance_from_home": distance_from_home,
        "distance_from_last_txn": distance_from_last_txn,
        "ratio_to_median": ratio_to_median,
        "used_chip": used_chip,
        "used_pin": used_pin,
        "online_order": online_order,
        "num_txn_last_hour": num_txn_last_hour,
        "fraud": fraud,
    })


def make_drift_data(df: pd.DataFrame, drift_factor: float = 0.15) -> pd.DataFrame:
    """
    Create a drifted version of a dataset by shifting numeric feature columns.
    Used to simulate production data drift for monitoring tests.
    """
    drifted = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = "churn" if "churn" in drifted.columns else "fraud"
    feature_cols = [c for c in numeric_cols if c != label_col]

    for col in feature_cols:
        shift = df[col].std() * drift_factor
        drifted[col] = drifted[col] + np.random.normal(shift, shift * 0.3, len(drifted))

    return drifted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser(description="Generate synthetic ML datasets")
    parser.add_argument("--output-dir", default=str(_project_root / "data" / "raw"))
    parser.add_argument("--churn-rows", type=int, default=50000)
    parser.add_argument("--fraud-rows", type=int, default=100000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating churn dataset...")
    churn_df = make_churn_data(args.churn_rows)
    churn_df.to_csv(output_dir / "churn_raw.csv", index=False)
    log.info("  Saved %d rows | churn rate: %.2f%%", len(churn_df), churn_df["churn"].mean() * 100)

    log.info("Generating fraud dataset...")
    fraud_df = make_fraud_data(args.fraud_rows)
    fraud_df.to_csv(output_dir / "fraud_raw.csv", index=False)
    log.info("  Saved %d rows | fraud rate: %.2f%%", len(fraud_df), fraud_df["fraud"].mean() * 100)

    log.info("Generating drifted versions for monitoring tests...")
    make_drift_data(churn_df).to_csv(output_dir / "churn_drifted.csv", index=False)
    make_drift_data(fraud_df).to_csv(output_dir / "fraud_drifted.csv", index=False)

    log.info("All datasets saved to: %s", output_dir)
