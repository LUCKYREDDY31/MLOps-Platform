"""
Auto-retraining pipeline using Prefect 3.
Checks for data drift and retrains whichever models need it.

Run manually:   python src/mlops_platform/training/retrain_flow.py
Schedule:       prefect server start  (in one terminal)
                prefect deploy src/mlops_platform/training/retrain_flow.py:retraining_pipeline \\
                    --name "daily-retrain" --cron "0 2 * * *"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

PROJECT_DIR = Path(__file__).resolve().parents[3]
PYTHON = sys.executable


def run_script(script: str, extra_args: list[str] | None = None) -> int:
    """Run a project script as a subprocess and return its exit code."""
    import subprocess
    cmd = [PYTHON, str(PROJECT_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    return result.returncode


@task(name="generate-data", retries=1)
def generate_data() -> None:
    logger = get_run_logger()
    logger.info("Generating synthetic datasets...")
    rc = run_script("src/mlops_platform/data/generate_data.py")
    if rc != 0:
        raise RuntimeError("Data generation failed")
    logger.info("Data generation complete.")


@task(name="feature-engineering", retries=1)
def feature_engineering() -> None:
    logger = get_run_logger()
    logger.info("Running feature engineering pipeline...")
    rc = run_script("src/mlops_platform/data/feature_pipeline.py")
    if rc != 0:
        raise RuntimeError("Feature engineering failed")
    logger.info("Feature engineering complete.")


@task(name="drift-detection", retries=1)
def drift_detection() -> dict:
    logger = get_run_logger()
    logger.info("Running drift detection...")
    rc = run_script("src/mlops_platform/monitoring/drift_detector.py")
    if rc != 0:
        raise RuntimeError("Drift detection failed")

    result_path = PROJECT_DIR / "reports" / "drift_check_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Drift result not found at {result_path}")

    with open(result_path) as f:
        result = json.load(f)

    retrain_needed = result.get("retrain_needed", {})
    logger.info("Retrain needed: %s", retrain_needed)
    return retrain_needed


@task(name="train-churn-model", retries=1)
def train_churn_model() -> None:
    logger = get_run_logger()
    logger.info("Retraining churn model (XGBoost)...")
    rc = run_script("src/mlops_platform/training/churn_model.py")
    if rc != 0:
        raise RuntimeError("Churn model training failed")
    logger.info("Churn model training complete.")


@task(name="train-fraud-model", retries=1)
def train_fraud_model() -> None:
    logger = get_run_logger()
    logger.info("Retraining fraud model (PyTorch)...")
    rc = run_script("src/mlops_platform/training/fraud_model.py", ["--epochs", "15"])
    if rc != 0:
        raise RuntimeError("Fraud model training failed")
    logger.info("Fraud model training complete.")


@task(name="register-models")
def register_models() -> None:
    logger = get_run_logger()
    logger.info("Registering models in MLflow registry...")
    rc = run_script("src/mlops_platform/registry/model_registry.py")
    if rc != 0:
        raise RuntimeError("Model registration failed")
    logger.info("Model registration complete.")


@task(name="reload-api")
def reload_api() -> None:
    """Ask the FastAPI server to hot-reload its models."""
    logger = get_run_logger()
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/reload", method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info("API reload response: %s", resp.status)
    except Exception as exc:
        logger.warning("API reload skipped (server may not be running): %s", exc)


@flow(
    name="mlops-retraining-pipeline",
    description="End-to-end model retraining pipeline triggered by drift detection.",
)
def retraining_pipeline() -> None:
    logger = get_run_logger()

    generate_data()
    feature_engineering()

    retrain_needed = drift_detection()
    any_retrained = False

    if retrain_needed.get("churn"):
        logger.info("[CHURN] Drift detected - retraining.")
        train_churn_model()
        any_retrained = True
    else:
        logger.info("[CHURN] No drift - skipping.")

    if retrain_needed.get("fraud"):
        logger.info("[FRAUD] Drift detected - retraining.")
        train_fraud_model()
        any_retrained = True
    else:
        logger.info("[FRAUD] No drift - skipping.")

    if any_retrained:
        register_models()
        reload_api()
    else:
        logger.info("No models retrained. Pipeline complete.")


if __name__ == "__main__":
    retraining_pipeline()
