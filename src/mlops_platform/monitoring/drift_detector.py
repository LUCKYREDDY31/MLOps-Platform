"""
Data and model drift detection using Evidently AI.
Compares reference (training) data against current (production) data.
Generates HTML reports and a JSON drift summary.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report

log = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve(base: str, project_root: Path) -> str:
    p = Path(base)
    return str(p) if p.is_absolute() else str(project_root / p)


def load_reference_data(processed_dir: str, task: str) -> pd.DataFrame:
    """Load training split as the reference set for drift comparison."""
    X = np.load(os.path.join(processed_dir, f"{task}_X_train.npy"))
    y = np.load(os.path.join(processed_dir, f"{task}_y_train.npy"))
    with open(os.path.join(processed_dir, f"{task}_feature_cols.pkl"), "rb") as f:
        cols = pickle.load(f)
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y
    return df


def load_current_data(raw_dir: str, task: str, processed_dir: str) -> pd.DataFrame:
    """
    Load the drifted production data for comparison.
    In production this would be a rolling window of recent predictions.
    """
    drifted_path = os.path.join(raw_dir, f"{task}_drifted.csv")
    if not os.path.exists(drifted_path):
        raise FileNotFoundError(f"Drifted data not found: {drifted_path}")

    df = pd.read_csv(drifted_path)
    label_col = "churn" if task == "churn" else "fraud"

    project_root = get_project_root()
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    if task == "churn":
        from mlops_platform.data.feature_pipeline import engineer_churn_features
        df = engineer_churn_features(df)
    else:
        from mlops_platform.data.feature_pipeline import engineer_fraud_features
        df = engineer_fraud_features(df)

    with open(os.path.join(processed_dir, f"{task}_feature_cols.pkl"), "rb") as f:
        cols = pickle.load(f)
    with open(os.path.join(processed_dir, f"{task}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X = scaler.transform(df[cols].values)
    result = pd.DataFrame(X, columns=cols)
    result["target"] = df[label_col].values
    return result.head(5000)


def run_drift_report(
    task: str,
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    reports_dir: str = "reports",
) -> dict:
    project_root = get_project_root()
    processed_dir = resolve(processed_dir, project_root)
    raw_dir = resolve(raw_dir, project_root)
    reports_dir = resolve(reports_dir, project_root)

    os.makedirs(reports_dir, exist_ok=True)

    log.info("Running drift detection for: %s", task)
    reference = load_reference_data(processed_dir, task)
    current = load_current_data(raw_dir, task, processed_dir)

    feature_cols = [c for c in reference.columns if c != "target"]
    ref_features = reference[feature_cols]
    cur_features = current[feature_cols]

    drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    drift_report.run(reference_data=ref_features, current_data=cur_features)

    report_path = os.path.join(reports_dir, f"{task}_drift_report.html")
    drift_report.save_html(report_path)

    result = drift_report.as_dict()
    drift_metrics = result.get("metrics", [])

    dataset_drift = False
    drifted_columns = 0
    drift_share = 0.0

    for m in drift_metrics:
        if m.get("metric") == "DatasetDriftMetric":
            r = m.get("result", {})
            dataset_drift = r.get("dataset_drift", False)
            drift_share = r.get("share_of_drifted_columns", 0.0)
            drifted_columns = r.get("number_of_drifted_columns", 0)

    summary = {
        "task": task,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_drift_detected": dataset_drift,
        "drift_share": round(drift_share, 4),
        "drifted_column_count": drifted_columns,
        "total_columns": len(feature_cols),
        "report_path": report_path,
        "reference_rows": len(ref_features),
        "current_rows": len(cur_features),
    }

    summary_path = os.path.join(reports_dir, f"{task}_drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    status = "DRIFT DETECTED" if dataset_drift else "No drift"
    log.info("  [%s] drift_share=%.2f%% | report: %s", status, drift_share * 100, report_path)
    return summary


def check_all_tasks(
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    reports_dir: str = "reports",
    drift_threshold: float = 0.2,
) -> dict:
    """Run drift detection for all tasks and determine which models need retraining."""
    results = {}
    for task in ["churn", "fraud"]:
        try:
            summary = run_drift_report(task, processed_dir, raw_dir, reports_dir)
            results[task] = summary
        except Exception as e:
            log.error("Error checking %s: %s", task, e)
            results[task] = {"error": str(e), "dataset_drift_detected": False}

    retrain_needed = {
        task: info.get("dataset_drift_detected", False)
        or info.get("drift_share", 0) > drift_threshold
        for task, info in results.items()
    }

    project_root = get_project_root()
    reports_dir_abs = resolve(reports_dir, project_root)
    os.makedirs(reports_dir_abs, exist_ok=True)

    final = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "retrain_needed": retrain_needed,
    }
    with open(os.path.join(reports_dir_abs, "drift_check_result.json"), "w") as f:
        json.dump(final, f, indent=2)

    log.info("Retrain needed: %s", retrain_needed)
    return final


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["churn", "fraud", "all"], default="all")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    if args.task == "all":
        check_all_tasks(args.processed_dir, args.raw_dir, args.reports_dir)
    else:
        run_drift_report(args.task, args.processed_dir, args.raw_dir, args.reports_dir)
