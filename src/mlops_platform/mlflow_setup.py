"""Central MLflow configuration - tracking URI, artifact root, and experiment setup."""

import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_mlflow(experiment_name: str) -> str:
    project_root = get_project_root()
    mlruns_dir = project_root / "mlruns"
    db_path = project_root / "mlflow.db"

    os.environ["MLFLOW_ARTIFACT_ROOT"] = str(mlruns_dir)
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    client = MlflowClient()
    artifact_location = str(mlruns_dir / experiment_name)
    try:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    except Exception:
        pass  # experiment already exists

    mlflow.set_experiment(experiment_name)
    return experiment_name
