"""
Fraud detection model - PyTorch neural network.
Handles class imbalance with weighted BCE loss. Logs to MLflow.
"""

from mlops_platform.mlflow_setup import setup_mlflow, get_project_root
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import argparse
import json
import logging
import os
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")

log = logging.getLogger(__name__)


class FraudNet(nn.Module):
    """Feed-forward network: input -> 128 -> 64 -> 32 -> 1 with BatchNorm and Dropout."""

    def __init__(self, input_dim: int, hidden_dims: tuple = (128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(y)
    weights = 1.0 / counts[y]
    return WeightedRandomSampler(weights=weights, num_samples=len(y), replacement=True)


def evaluate(model: FraudNet,
             loader: DataLoader,
             device: torch.device) -> Tuple[dict,
                                            np.ndarray,
                                            np.ndarray,
                                            np.ndarray]:
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            probs = torch.sigmoid(model(X_b)).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }, y_true, y_pred, y_prob


def plot_training_curves(train_losses: list, val_losses: list, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fraud Model - Training Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"])
    ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Fraud")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def train_fraud_model(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    experiment_name: str = "fraud-detection",
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
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

    X_train = np.load(os.path.join(processed_dir, "fraud_X_train.npy")).astype(np.float32)
    X_test = np.load(os.path.join(processed_dir, "fraud_X_test.npy")).astype(np.float32)
    y_train = np.load(os.path.join(processed_dir, "fraud_y_train.npy")).astype(np.int64)
    y_test = np.load(os.path.join(processed_dir, "fraud_y_test.npy")).astype(np.int64)
    with open(os.path.join(processed_dir, "fraud_feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)

    log.info("Training fraud model | train=%d, test=%d", len(X_train), len(X_test))
    log.info("  Fraud rate train: %.3f%%", y_train.mean() * 100)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info("  Device: %s", device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=make_weighted_sampler(y_train))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = FraudNet(input_dim=input_dim).to(device)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    hparams = {
        "model_type": "PyTorch FraudNet",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dims": "128-64-32",
        "dropout": 0.3,
        "optimizer": "Adam",
        "n_features": input_dim,
    }

    with mlflow.start_run(run_name="pytorch-fraud"):
        mlflow.log_params(hparams)

        train_losses, val_losses = [], []
        best_auc, best_state = 0.0, None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_b)

            avg_loss = epoch_loss / len(train_loader.dataset)
            train_losses.append(avg_loss)

            val_m, _, _, _ = evaluate(model, test_loader, device)
            val_losses.append(1 - val_m["roc_auc"])
            scheduler.step(1 - val_m["roc_auc"])

            if val_m["roc_auc"] > best_auc:
                best_auc = val_m["roc_auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 5 == 0:
                log.info("  Epoch %d/%d | loss=%.4f | auc=%.4f", epoch + 1, epochs, avg_loss, val_m["roc_auc"])
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                mlflow.log_metric("val_auc", val_m["roc_auc"], step=epoch)

        model.load_state_dict(best_state)
        final_metrics, y_true, y_pred, y_prob = evaluate(model, test_loader, device)
        mlflow.log_metrics(final_metrics)

        curve_path = os.path.join(logs_dir, "fraud_training_curves.png")
        cm_path = os.path.join(logs_dir, "fraud_confusion_matrix.png")
        report_path = os.path.join(logs_dir, "fraud_classification_report.txt")

        plot_training_curves(train_losses, val_losses, curve_path)
        plot_confusion_matrix(y_true, y_pred, cm_path)
        mlflow.log_artifact(curve_path)
        mlflow.log_artifact(cm_path)

        report = classification_report(y_true, y_pred, target_names=["Legit", "Fraud"])
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        mlflow.pytorch.log_model(model, artifact_path="fraud_model")

        model_path = os.path.join(models_dir, "fraud_model.pt")
        torch.save({"model_state": model.state_dict(), "input_dim": input_dim}, model_path)

        run_id = mlflow.active_run().info.run_id

        log.info("  PyTorch Fraud: roc_auc ~%.3f, Device: %s", final_metrics["roc_auc"], device)
        log.info("  MLflow run_id: %s", run_id)

    meta = {
        "model_type": "PyTorch FraudNet",
        "task": "fraud",
        "feature_cols": feature_cols,
        "metrics": final_metrics,
        "run_id": run_id,
        "model_path": model_path,
        "input_dim": input_dim,
    }
    with open(os.path.join(models_dir, "fraud_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Fraud model saved to: %s", model_path)
    return model, final_metrics, run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_fraud_model(
        args.processed_dir, args.models_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
