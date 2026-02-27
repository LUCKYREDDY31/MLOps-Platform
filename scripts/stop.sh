#!/usr/bin/env bash
# Stop all running MLOps platform services.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Stopping MLOps platform services..."

kill_port() {
    local port=$1
    lsof -ti:"$port" | xargs kill -9 2>/dev/null || \
    fuser -k "$port"/tcp 2>/dev/null || true
}

kill_port 8000 && echo "  Cleared port 8000" || true
kill_port 5001 && echo "  Cleared port 5001" || true
kill_port 8501 && echo "  Cleared port 8501" || true

echo "Done."
