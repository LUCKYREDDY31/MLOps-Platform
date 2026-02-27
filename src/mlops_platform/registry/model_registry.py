"""MLflow model registry wrapper - registration, versioning, and promotion."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class ModelRegistry:
    def __init__(self) -> None:
        project_root = get_project_root()
        mlruns_dir = project_root / "mlruns"
        db_path = project_root / "mlflow.db"
        os.environ["MLFLOW_ARTIFACT_ROOT"] = str(mlruns_dir)
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        self.client = MlflowClient()

    def register(self, run_id: str, model_name: str, artifact_path: str) -> str:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        log.info("Registered '%s' version %s from run %s...", model_name, result.version, run_id[:8])
        return result.version

    def promote(self, model_name: str, version: str, stage: str = "Production") -> None:
        alias = stage.lower()
        try:
            self.client.set_registered_model_alias(model_name, alias, version)
            log.info("Promoted '%s' v%s -> alias '%s'", model_name, version, alias)
        except Exception:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production"),
            )
            log.info("Promoted '%s' v%s to %s", model_name, version, stage)

    def get_production_version(self, model_name: str) -> dict | None:
        try:
            mv = self.client.get_model_version_by_alias(model_name, "production")
            return {"name": mv.name, "version": mv.version, "run_id": mv.run_id}
        except Exception:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return None
            v = sorted(versions, key=lambda x: int(x.version))[-1]
            return {"name": v.name, "version": v.version, "run_id": v.run_id}

    def list_versions(self, model_name: str) -> list[dict]:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        rows = [{"version": v.version, "run_id": v.run_id[:8] + "..."} for v in versions]
        return sorted(rows, key=lambda x: int(x["version"]))

    def compare_runs(self, experiment_name: str, metric: str = "roc_auc") -> list[dict]:
        exp = self.client.get_experiment_by_name(experiment_name)
        if not exp:
            return []
        runs = self.client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
        )
        return [
            {
                "run_id": r.info.run_id[:8] + "...",
                "run_name": r.data.tags.get("mlflow.runName", ""),
                metric: r.data.metrics.get(metric),
            }
            for r in runs
        ]


def register_all_models(models_dir: str = None) -> None:
    project_root = get_project_root()
    if models_dir is None:
        models_dir = str(project_root / "models")

    registry = ModelRegistry()

    tasks = [
        {
            "meta_file": os.path.join(models_dir, "churn_meta.json"),
            "model_name": "churn-predictor",
            "artifact_path": "churn_model",
        },
        {
            "meta_file": os.path.join(models_dir, "fraud_meta.json"),
            "model_name": "fraud-detector",
            "artifact_path": "fraud_model",
        },
    ]

    for task in tasks:
        if not os.path.exists(task["meta_file"]):
            log.warning("Skipping %s - meta file not found", task["model_name"])
            continue
        with open(task["meta_file"]) as f:
            meta = json.load(f)
        run_id = meta["run_id"]
        log.info("Registering %s...", task["model_name"])
        version = registry.register(run_id, task["model_name"], task["artifact_path"])
        registry.promote(task["model_name"], version, stage="Production")
        info = registry.get_production_version(task["model_name"])
        log.info("  %s promoted to production", task["model_name"])
        log.info("  Production version: %s", info)

    log.info("Model registry updated.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()
    register_all_models(args.models_dir)
