# MLOps Platform

End-to-end machine learning operations platform covering the full ML lifecycle: data generation, feature engineering, model training, experiment tracking, drift monitoring, auto-retraining, and live serving. Runs entirely locally.

## What This Project Does

This platform trains two production models on synthetic datasets, monitors them for data drift, and automatically retrains them when the data distribution shifts. A FastAPI REST API serves live predictions, an MLflow server tracks every experiment run, and a Streamlit dashboard gives a visual overview of model health and recent alerts. The entire pipeline - from raw data to live predictions - runs with a single command.

## Why It Matters

Models degrade over time. Customer behaviour changes, fraud patterns evolve, and a model trained six months ago can quietly become unreliable without anyone noticing. This platform solves that by treating retraining as an automated process, not a manual task. Drift is detected on every run, and only the models that actually need retraining are retrained. That means lower compute cost, faster recovery from data shift, and a system that stays accurate without human intervention.

## Models

| Model | Algorithm | Dataset | AUC |
|---|---|---|---|
| Churn prediction | XGBoost | 50k telecom customer records | ~0.955 |
| Fraud detection | PyTorch FraudNet | 100k transaction records | ~0.982 |

## Requirements

- Python 3.11 or later
- 4 GB RAM minimum (8 GB recommended for PyTorch training)
- ~3 GB disk space

## Quickstart

```bash
cd mlops-platform
bash scripts/setup.sh
bash scripts/run.sh
```

After the pipeline completes (~5 minutes on first run), three services start:

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://127.0.0.1:5001 |

Stop all services:

```bash
bash scripts/stop.sh
```

Skip training on subsequent runs (uses existing models):

```bash
bash scripts/run.sh --skip-train
```


## Pipeline Steps

`bash scripts/run.sh` executes in order:

1. **Data generation** - `data/raw/churn_raw.csv` (50k rows), `data/raw/fraud_raw.csv` (100k rows), plus drifted variants for monitoring tests
2. **Feature engineering** - 19 engineered features per model, saved as numpy arrays in `data/processed/`
3. **Model training** - XGBoost (churn) and PyTorch FraudNet (fraud), all params and metrics logged to MLflow
4. **Model registry** - both models registered in MLflow and promoted to production
5. **Drift detection** - Evidently AI compares training vs. drifted production data, generates HTML reports
6. **Alerting** - structured alerts written to `logs/alerts.jsonl`
7. **Report generation** - markdown business summary written to `reports/business_summary.md`

## Retraining Pipeline

```bash
source venv/bin/activate
python src/mlops_platform/training/retrain_flow.py
```

To schedule (daily at 2am):

```bash
prefect server start
prefect deploy src/mlops_platform/training/retrain_flow.py:retraining_pipeline \
    --name "daily-retrain" --cron "0 2 * * *"
```

## Anomaly Investigation

```bash
python src/mlops_platform/agents/anomaly_agent.py
```

## API Usage

Interactive docs: http://localhost:8000/docs

```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{"tenure": 5, "monthly_charges": 95.0, "total_charges": 475.0,
       "num_products": 1, "support_calls": 6, "payment_delays": 3,
       "contract_type": 0, "internet_service": 2, "online_security": 0,
       "tech_support": 0, "paperless_billing": 1, "senior_citizen": 0}'
```

## Tests

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

## Compatibility Notes

- **MLflow port** - runs on 5001 (macOS AirPlay Receiver occupies 5000 by default)
- **Evidently** - pinned to 0.4.30; later versions use typing constructs incompatible with Python 3.11+ stdlib
- **PyTorch** - uses `weights_only=True` in `torch.load` to suppress deprecation warnings
- **FastAPI** - uses `lifespan` context manager instead of deprecated `on_event`
