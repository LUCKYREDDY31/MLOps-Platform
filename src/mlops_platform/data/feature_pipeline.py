"""Feature engineering pipeline for churn and fraud datasets."""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def engineer_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["charges_per_month"] = df["total_charges"] / (df["tenure"] + 1)
    df["support_call_rate"] = df["support_calls"] / (df["tenure"] + 1)
    df["delay_rate"] = df["payment_delays"] / (df["tenure"] + 1)

    df["tenure_bucket"] = pd.cut(
        df["tenure"].clip(lower=1),
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],
    ).cat.add_categories([-1]).fillna(-1).astype(int)

    df["high_value"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)

    df["engagement_score"] = (
        df["num_products"] + df["online_security"] + df["tech_support"]
    )

    # month-to-month contract with high support call volume
    df["at_risk"] = (
        (df["contract_type"] == 0) & (df["support_calls"] > 3)
    ).astype(int)

    return df


def prepare_churn(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    df = engineer_churn_features(df)

    feature_cols = [
        "tenure", "monthly_charges", "total_charges", "num_products",
        "support_calls", "payment_delays", "contract_type", "internet_service",
        "online_security", "tech_support", "paperless_billing", "senior_citizen",
        "charges_per_month", "support_call_rate", "delay_rate",
        "tenure_bucket", "high_value", "engagement_score", "at_risk",
    ]

    X = df[feature_cols].values
    y = df["churn"].values

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, feature_cols, scaler


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["total_distance"] = df["distance_from_home"] + df["distance_from_last_txn"]
    df["distance_ratio"] = df["distance_from_last_txn"] / (df["distance_from_home"] + 1)
    df["high_velocity"] = (df["num_txn_last_hour"] > 4).astype(int)
    df["security_score"] = df["used_chip"] + df["used_pin"]

    # online transaction with no chip and no pin is the highest-risk combination
    df["suspicious_online"] = (
        (df["online_order"] == 1) & (df["used_chip"] == 0) & (df["used_pin"] == 0)
    ).astype(int)

    return df


def prepare_fraud(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    df = engineer_fraud_features(df)

    feature_cols = [
        "amount", "log_amount", "hour", "day_of_week", "merchant_category",
        "distance_from_home", "distance_from_last_txn", "ratio_to_median",
        "used_chip", "used_pin", "online_order", "num_txn_last_hour",
        "is_night", "is_weekend", "total_distance", "distance_ratio",
        "high_velocity", "security_score", "suspicious_online",
    ]

    X = df[feature_cols].values
    y = df["fraud"].values

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, feature_cols, scaler


def run_pipeline(raw_dir: str, processed_dir: str) -> None:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    log.info("Processing churn data...")
    churn_raw = pd.read_csv(Path(raw_dir) / "churn_raw.csv")
    X_c, y_c, churn_cols, churn_scaler = prepare_churn(churn_raw)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
    )
    np.save(processed_path / "churn_X_train.npy", X_train_c)
    np.save(processed_path / "churn_X_test.npy", X_test_c)
    np.save(processed_path / "churn_y_train.npy", y_train_c)
    np.save(processed_path / "churn_y_test.npy", y_test_c)
    with open(processed_path / "churn_scaler.pkl", "wb") as f:
        pickle.dump(churn_scaler, f)
    with open(processed_path / "churn_feature_cols.pkl", "wb") as f:
        pickle.dump(churn_cols, f)
    log.info("  Churn: train=%d, test=%d, features=%d", len(X_train_c), len(X_test_c), len(churn_cols))

    log.info("Processing fraud data...")
    fraud_raw = pd.read_csv(Path(raw_dir) / "fraud_raw.csv")
    X_f, y_f, fraud_cols, fraud_scaler = prepare_fraud(fraud_raw)
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_f, y_f, test_size=0.2, random_state=42, stratify=y_f
    )
    np.save(processed_path / "fraud_X_train.npy", X_train_f)
    np.save(processed_path / "fraud_X_test.npy", X_test_f)
    np.save(processed_path / "fraud_y_train.npy", y_train_f)
    np.save(processed_path / "fraud_y_test.npy", y_test_f)
    with open(processed_path / "fraud_scaler.pkl", "wb") as f:
        pickle.dump(fraud_scaler, f)
    with open(processed_path / "fraud_feature_cols.pkl", "wb") as f:
        pickle.dump(fraud_cols, f)
    log.info("  Fraud: train=%d, test=%d, features=%d", len(X_train_f), len(X_test_f), len(fraud_cols))

    log.info("Processed data saved to: %s", processed_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _project_root = Path(__file__).resolve().parents[3]

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(_project_root / "data" / "raw"))
    parser.add_argument("--processed-dir", default=str(_project_root / "data" / "processed"))
    args = parser.parse_args()
    run_pipeline(args.raw_dir, args.processed_dir)
