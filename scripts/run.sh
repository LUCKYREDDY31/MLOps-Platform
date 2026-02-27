#!/usr/bin/env bash
# Run the full MLOps pipeline, then start all services.
#
# Usage:
#   bash scripts/run.sh              - full pipeline + start services
#   bash scripts/run.sh --skip-train - skip training + start services
#   bash scripts/run.sh --train-only - run pipeline only, don't start services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SKIP_TRAIN=false
TRAIN_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --train-only) TRAIN_ONLY=true ;;
    esac
done

if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[ERROR] Virtual environment not found. Run 'bash scripts/setup.sh' first."
    exit 1
fi

PYTHON="${VIRTUAL_ENV}/bin/python"

echo ""
echo "======================================================"
echo "  MLOps Platform - Full Pipeline Run"
echo "======================================================"
echo ""

if [ "$SKIP_TRAIN" = false ]; then

    # mlflow.db stores absolute paths - delete before training so a fresh
    # DB is created with the correct paths for this machine.
    echo "[Pre-flight] Clearing stale MLflow state..."
    rm -f mlflow.db
    rm -rf mlruns
    echo ""

    echo "[Step 1/5] Generating synthetic datasets..."
    $PYTHON src/mlops_platform/data/generate_data.py
    echo ""

    echo "[Step 2/5] Running feature engineering pipeline..."
    $PYTHON src/mlops_platform/data/feature_pipeline.py
    echo ""

    echo "[Step 3/5] Training models (XGBoost + PyTorch)..."
    $PYTHON src/mlops_platform/training/run_experiments.py
    echo ""

    echo "[Step 4/5] Registering models in MLflow registry..."
    $PYTHON src/mlops_platform/registry/model_registry.py 2>&1 | grep -v "FutureWarning\|deprecated\|migration guide" || true
    echo ""

    echo "[Step 5/5] Running drift detection and generating reports..."
    $PYTHON src/mlops_platform/monitoring/drift_detector.py 2>&1 | grep -v "RuntimeWarning\|invalid value\|c /=" || true
    $PYTHON src/mlops_platform/monitoring/alerting.py
    $PYTHON src/mlops_platform/agents/report_generator.py
    echo ""

else
    echo "[Skipping training - using existing models]"
    echo ""
fi

if [ "$TRAIN_ONLY" = true ]; then
    echo "Training pipeline complete. Exiting (--train-only mode)."
    exit 0
fi

# ── Start services ─────────────────────────────────────────────────────────────

echo "======================================================"
echo "  Starting Services"
echo "======================================================"
echo ""

# Kill any existing processes on these ports.
# Uses lsof (macOS/Linux compatible) - fuser is Linux-only.
echo "Clearing ports 8000, 5001, 8501..."
kill_port() {
    local port=$1
    lsof -ti:"$port" | xargs kill -9 2>/dev/null || \
    fuser -k "$port"/tcp 2>/dev/null || true
}
kill_port 8000
kill_port 5001
kill_port 8501
sleep 1

mkdir -p logs

echo "Starting FastAPI server on port 8000..."
MODELS_DIR="$(pwd)/models" \
PYTHONPATH="$(pwd)/src" \
    uvicorn mlops_platform.serving.api:app \
    --host 0.0.0.0 --port 8000 \
    --log-level warning > logs/api.log 2>&1 &
echo "  API PID: $!"

echo "Starting MLflow UI on port 5001..."
rm -f mlflow.db
mlflow ui \
    --backend-store-uri "sqlite:///$(pwd)/mlflow.db" \
    --default-artifact-root "$(pwd)/mlruns" \
    --port 5001 --host 0.0.0.0 > logs/mlflow.log 2>&1 &
echo "  MLflow PID: $!"

echo "Starting Streamlit dashboard on port 8501..."
MODELS_DIR="$(pwd)/models" \
REPORTS_DIR="$(pwd)/reports" \
LOGS_DIR="$(pwd)/logs" \
    streamlit run src/mlops_platform/serving/dashboard.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0 > logs/streamlit.log 2>&1 &
echo "  Streamlit PID: $!"

sleep 5

echo ""
echo "======================================================"
echo "  All services are running!"
echo ""
echo "  Streamlit Dashboard:  http://localhost:8501"
echo "  FastAPI Docs:         http://localhost:8000/docs"
echo "  MLflow Experiment UI: http://127.0.0.1:5001"
echo ""
echo "  To stop all services: bash scripts/stop.sh"
echo ""
echo "  To run the retraining pipeline manually:"
echo "    source venv/bin/activate"
echo "    python src/mlops_platform/training/retrain_flow.py"
echo "======================================================"
echo ""

echo "Tailing logs (Ctrl+C to stop watching logs - services keep running)..."
tail -f logs/api.log logs/streamlit.log
