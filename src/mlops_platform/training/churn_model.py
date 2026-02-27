"""Churn prediction model - XGBoost. Trains on processed churn data and logs to MLflow."""

from mlops_platform.mlflow_setup import setup_mlflow, get_project_root
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    confusion_matrix,
)
import mlflow.xgboost
import mlflow
import matplotlib.pyplot as plt
import argparse
import logging
import os
import pickle
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")

log = logging.getLogger(__name__)


def load_data(processed_dir: str):
    X_train = np.load(os.path.join(processed_dir, "churn_X_train.npy"))
    X_test = np.load(os.path.join(processed_dir, "churn_X_test.npy"))
    y_train = np.load(os.path.join(processed_dir, "churn_y_train.npy"))
    y_test = np.load(os.path.join(processed_dir, "churn_y_test.npy"))
    with open(os.path.join(processed_dir, "churn_feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return X_train, X_test, y_train, y_test, feature_cols


def plot_feature_importance(model: XGBClassifier, feature_cols: list, output_path: str) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(indices)), importances[indices])
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Top 15 Feature Importances - Churn Model")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Churn")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def train_churn_model(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    experiment_name: str = "churn-prediction",
):
    project_root = get_project_root()

    if not os.path.isabs(processed_dir):
        processed_dir = str(project_root / processed_dir)
    if not os.path.isabs(models_dir):
        models_dir = str(project_root / models_dir)
    logs_dir = str(project_root / "logs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    setup_mlflow(experiment_name)

    X_train, X_test, y_train, y_test, feature_cols = load_data(processed_dir)
    log.info("Training churn model | train=%d, test=%d", len(X_train), len(X_test))

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float((y_train == 0).sum()) / float((y_train == 1).sum()),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-churn"):
        mlflow.log_params(xgb_params)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(X_train))

        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)

        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
        mlflow.log_metric("cv_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_auc_std", float(cv_scores.std()))

        fi_path = os.path.join(logs_dir, "churn_feature_importance.png")
        cm_path = os.path.join(logs_dir, "churn_confusion_matrix.png")
        report_path = os.path.join(logs_dir, "churn_classification_report.txt")

        plot_feature_importance(model, feature_cols, fi_path)
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(cm_path)

        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        mlflow.xgboost.log_model(model, artifact_path="churn_model")

        model_path = os.path.join(models_dir, "churn_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        run_id = mlflow.active_run().info.run_id

        log.info("  XGBoost Churn: roc_auc ~%.3f", metrics["roc_auc"])
        log.info("  CV AUC: %.4f +/- %.4f", cv_scores.mean(), cv_scores.std())
        log.info("  MLflow run_id: %s", run_id)

    meta = {
        "model_type": "XGBoost",
        "task": "churn",
        "feature_cols": feature_cols,
        "metrics": metrics,
        "run_id": run_id,
        "model_path": model_path,
    }
    with open(os.path.join(models_dir, "churn_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Churn model saved to: %s", model_path)
    return model, metrics, run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    train_churn_model(args.processed_dir, args.models_dir)
