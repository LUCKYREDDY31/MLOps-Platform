"""Entry point for the training phase. Runs both models and prints a comparison table."""

from mlops_platform.training.fraud_model import train_fraud_model
from mlops_platform.training.churn_model import train_churn_model
import argparse
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def print_table(title: str, metrics: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*30}")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:>10.4f}")
    print(f"{'='*55}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all training experiments")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  MLOps Platform - Training Pipeline")
    print("=" * 55)

    print("\n[1/2] Training Churn Model (XGBoost)...")
    _, churn_metrics, churn_run = train_churn_model(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
    )
    print_table("Churn Model Results", churn_metrics)

    print("\n[2/2] Training Fraud Model (PyTorch)...")
    _, fraud_metrics, fraud_run = train_fraud_model(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
    )
    print_table("Fraud Model Results", fraud_metrics)

    summary = {
        "churn": {"run_id": churn_run, "metrics": churn_metrics},
        "fraud": {"run_id": fraud_run, "metrics": fraud_metrics},
    }

    _project_root = Path(__file__).resolve().parents[3]
    models_dir_abs = args.models_dir if os.path.isabs(args.models_dir) else str(_project_root / args.models_dir)
    summary_path = os.path.join(models_dir_abs, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete. Summary saved to: {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
