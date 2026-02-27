#!/usr/bin/env bash
# One-time environment setup. Run once before scripts/run.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "======================================================"
echo "  MLOps Platform - Environment Setup"
echo "======================================================"
echo ""

PYTHON=$(which python3.11 2>/dev/null || which python3 2>/dev/null || which python 2>/dev/null)

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python not found. Install Python 3.11+ first."
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[OK] Using Python $PY_VERSION at: $PYTHON"

$PYTHON -c "
import sys
if sys.version_info < (3, 10):
    print('[WARN] Python 3.10+ is recommended.')
"

echo ""
if [ ! -d "venv" ]; then
    echo "[1/4] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "[1/4] Virtual environment already exists, skipping."
fi

source venv/bin/activate
pip install --upgrade pip --quiet

echo ""
echo "[2/4] Installing dependencies..."
echo "      (first run takes 3-5 minutes - PyTorch is large)"
echo ""

pip install torch --quiet 2>/dev/null || \
pip install torch --quiet \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt --quiet

echo ""
echo "[OK] All dependencies installed."

echo ""
echo "[3/4] Creating project directories..."
mkdir -p data/raw data/processed models logs reports

# mlflow.db stores absolute paths from the machine that created it.
# Always delete on setup so a fresh DB is created with correct paths on first run.
echo ""
echo "[4/4] Clearing any stale MLflow state..."
rm -f mlflow.db
rm -rf mlruns
echo "      MLflow will create a fresh database on first run."

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Run the full pipeline with:"
echo "    bash scripts/run.sh"
echo ""
echo "  Or skip training if models already exist:"
echo "    bash scripts/run.sh --skip-train"
echo "======================================================"
echo ""
