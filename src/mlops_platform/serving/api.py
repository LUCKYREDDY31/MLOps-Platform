"""
FastAPI model serving.
Serves both churn (XGBoost) and fraud (PyTorch) models.
Includes /health, prediction endpoints, rolling metrics, and hot-reload.
"""

from mlops_platform.training.fraud_model import FraudNet
import json
import logging
import math
import os
import pickle
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="MLOps Platform - Model API",
    description="Serves churn prediction (XGBoost) and fraud detection (PyTorch) models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models: dict = {}
prediction_log: deque = deque(maxlen=1000)  # rolling window for monitoring


def _project_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def load_models() -> None:
    default_models_dir = os.path.join(_project_root(), "models")
    base = os.environ.get("MODELS_DIR", default_models_dir)

    churn_path = os.path.join(base, "churn_model.pkl")
    churn_meta = os.path.join(base, "churn_meta.json")
    if os.path.exists(churn_path):
        with open(churn_path, "rb") as f:
            models["churn"] = pickle.load(f)
        with open(churn_meta) as f:
            models["churn_meta"] = json.load(f)
        log.info("Loaded churn model from %s", churn_path)
    else:
        log.warning("Churn model not found at %s", churn_path)

    fraud_path = os.path.join(base, "fraud_model.pt")
    fraud_meta_path = os.path.join(base, "fraud_meta.json")
    if os.path.exists(fraud_path):
        checkpoint = torch.load(fraud_path, map_location="cpu", weights_only=True)
        input_dim = checkpoint["input_dim"]
        fraud_net = FraudNet(input_dim=input_dim)
        fraud_net.load_state_dict(checkpoint["model_state"])
        fraud_net.eval()
        models["fraud"] = fraud_net
        with open(fraud_meta_path) as f:
            models["fraud_meta"] = json.load(f)
        log.info("Loaded fraud model from %s", fraud_path)
    else:
        log.warning("Fraud model not found at %s", fraud_path)


class ChurnRequest(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float
    num_products: int = Field(ge=1, le=10)
    support_calls: int = Field(ge=0)
    payment_delays: int = Field(ge=0)
    contract_type: int = Field(ge=0, le=2, description="0=month-to-month, 1=1yr, 2=2yr")
    internet_service: int = Field(ge=0, le=2)
    online_security: int = Field(ge=0, le=1)
    tech_support: int = Field(ge=0, le=1)
    paperless_billing: int = Field(ge=0, le=1)
    senior_citizen: int = Field(ge=0, le=1)


class FraudRequest(BaseModel):
    amount: float
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    merchant_category: int = Field(ge=0, le=19)
    distance_from_home: float
    distance_from_last_txn: float
    ratio_to_median: float
    used_chip: int = Field(ge=0, le=1)
    used_pin: int = Field(ge=0, le=1)
    online_order: int = Field(ge=0, le=1)
    num_txn_last_hour: int = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    model_version: str
    latency_ms: float
    timestamp: str


def build_churn_features(req: ChurnRequest) -> np.ndarray:
    tenure = req.tenure
    monthly_charges = req.monthly_charges
    total_charges = req.total_charges
    support_calls = req.support_calls
    payment_delays = req.payment_delays
    contract_type = req.contract_type

    charges_per_month = total_charges / (tenure + 1)
    support_call_rate = support_calls / (tenure + 1)
    delay_rate = payment_delays / (tenure + 1)

    if tenure <= 12:
        tenure_bucket = 0
    elif tenure <= 24:
        tenure_bucket = 1
    elif tenure <= 48:
        tenure_bucket = 2
    else:
        tenure_bucket = 3

    high_value = 1 if monthly_charges > 80 else 0
    engagement_score = req.num_products + req.online_security + req.tech_support
    at_risk = 1 if (contract_type == 0 and support_calls > 3) else 0

    return np.array([[
        tenure, monthly_charges, total_charges, req.num_products,
        support_calls, payment_delays, contract_type, req.internet_service,
        req.online_security, req.tech_support, req.paperless_billing, req.senior_citizen,
        charges_per_month, support_call_rate, delay_rate,
        tenure_bucket, high_value, engagement_score, at_risk,
    ]])


def build_fraud_features(req: FraudRequest) -> np.ndarray:
    log_amount = math.log1p(req.amount)
    is_night = 1 if (req.hour < 6 or req.hour > 22) else 0
    is_weekend = 1 if req.day_of_week >= 5 else 0
    total_distance = req.distance_from_home + req.distance_from_last_txn
    distance_ratio = req.distance_from_last_txn / (req.distance_from_home + 1)
    high_velocity = 1 if req.num_txn_last_hour > 4 else 0
    security_score = req.used_chip + req.used_pin
    suspicious_online = 1 if (req.online_order == 1 and req.used_chip == 0 and req.used_pin == 0) else 0

    return np.array([[
        req.amount, log_amount, req.hour, req.day_of_week, req.merchant_category,
        req.distance_from_home, req.distance_from_last_txn, req.ratio_to_median,
        req.used_chip, req.used_pin, req.online_order, req.num_txn_last_hour,
        is_night, is_weekend, total_distance, distance_ratio,
        high_velocity, security_score, suspicious_online,
    ]])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": [k for k in models if "_meta" not in k],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict/churn", response_model=PredictionResponse)
def predict_churn(req: ChurnRequest):
    if "churn" not in models:
        raise HTTPException(status_code=503, detail="Churn model not loaded")

    t0 = time.time()
    X = build_churn_features(req)
    model = models["churn"]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    latency = (time.time() - t0) * 1000

    label = "Churn" if pred == 1 else "No Churn"
    version = models.get("churn_meta", {}).get("run_id", "unknown")[:8]

    entry = {
        "task": "churn", "prediction": pred, "probability": prob,
        "timestamp": datetime.now(timezone.utc).isoformat(), "latency_ms": latency,
    }
    prediction_log.append(entry)

    return PredictionResponse(
        prediction=pred, probability=round(prob, 4), label=label,
        model_version=version, latency_ms=round(latency, 2),
        timestamp=entry["timestamp"],
    )


@app.post("/predict/fraud", response_model=PredictionResponse)
def predict_fraud(req: FraudRequest):
    if "fraud" not in models:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")

    t0 = time.time()
    X = build_fraud_features(req).astype(np.float32)
    model = models["fraud"]

    with torch.no_grad():
        logit = model(torch.from_numpy(X))
        prob = float(torch.sigmoid(logit).item())

    pred = 1 if prob > 0.5 else 0
    latency = (time.time() - t0) * 1000

    label = "Fraud" if pred == 1 else "Legitimate"
    version = models.get("fraud_meta", {}).get("run_id", "unknown")[:8]

    entry = {
        "task": "fraud", "prediction": pred, "probability": prob,
        "timestamp": datetime.now(timezone.utc).isoformat(), "latency_ms": latency,
    }
    prediction_log.append(entry)

    return PredictionResponse(
        prediction=pred, probability=round(prob, 4), label=label,
        model_version=version, latency_ms=round(latency, 2),
        timestamp=entry["timestamp"],
    )


@app.get("/metrics")
def get_metrics():
    """Return rolling prediction stats for the monitoring dashboard."""
    if not prediction_log:
        return {"message": "No predictions yet"}

    logs = list(prediction_log)
    churn_logs = [log for log in logs if log["task"] == "churn"]
    fraud_logs = [log for log in logs if log["task"] == "fraud"]

    def stats(task_logs: list) -> dict:
        if not task_logs:
            return {}
        probs = [log["probability"] for log in task_logs]
        preds = [log["prediction"] for log in task_logs]
        latencies = [log["latency_ms"] for log in task_logs]
        return {
            "total_predictions": len(task_logs),
            "positive_rate": round(sum(preds) / len(preds), 4),
            "avg_probability": round(sum(probs) / len(probs), 4),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        }

    return {
        "churn": stats(churn_logs),
        "fraud": stats(fraud_logs),
        "total": len(logs),
    }


@app.post("/reload")
def reload_models():
    """Hot-reload models from disk - called after auto-retraining completes."""
    models.clear()
    load_models()
    return {"status": "reloaded", "models": [k for k in models if "_meta" not in k]}
