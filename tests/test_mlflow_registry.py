"""
Quick end-to-end test confirming MLflow file:
Run: python tests/test_mlflow_registry.py
"""

import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project_root = Path(__file__).resolve().parents[1]
tracking_uri = (project_root / "mlruns").as_uri()
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

# Create a test artifact file
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
test_file = str(logs_dir / "test_artifact.txt")
with open(test_file, "w") as f:
    f.write("test content")


class DummyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


mlflow.set_experiment("test-registry")
with mlflow.start_run(run_name="reg-test"):
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact(test_file)
    mlflow.pyfunc.log_model("model", python_model=DummyModel())
    artifact_uri = mlflow.active_run().info.artifact_uri
    run_id = mlflow.active_run().info.run_id
    print(f"artifact_uri = {artifact_uri}")
    assert "/home/ubuntu" not in artifact_uri or str(project_root) in artifact_uri, \
        f"artifact_uri contains wrong path: {artifact_uri}"
    print("log_artifact: OK")
mlflow.end_run()

result = mlflow.register_model(f"runs:/{run_id}/model", "test-model-v2")
print(f"Registered version: {result.version}")
client.set_registered_model_alias("test-model-v2", "production", result.version)
print("Alias set: OK")
print("\nAll MLflow tests passed.")
