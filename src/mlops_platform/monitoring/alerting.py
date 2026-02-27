"""
Alerting module. Checks drift reports and writes structured alerts.
Extend by dropping in a Slack/email webhook in write_alert().
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

LEVEL_PREFIX = {"INFO": "[i]", "WARNING": "[!]", "CRITICAL": "[!!!]"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_alert_log() -> str:
    return str(get_project_root() / "logs" / "alerts.jsonl")


def write_alert(level: str, task: str, message: str, data: dict = None) -> dict:
    alert_log = get_alert_log()
    os.makedirs(os.path.dirname(alert_log), exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "task": task,
        "message": message,
        "data": data or {},
    }
    with open(alert_log, "a") as f:
        f.write(json.dumps(entry) + "\n")

    prefix = LEVEL_PREFIX.get(level, "[?]")
    log.info("%s [%s] %s", prefix, task.upper(), message)
    return entry


def check_and_alert(drift_result_path: str = None) -> list:
    project_root = get_project_root()
    if drift_result_path is None:
        drift_result_path = str(project_root / "reports" / "drift_check_result.json")
    elif not os.path.isabs(drift_result_path):
        drift_result_path = str(project_root / drift_result_path)

    if not os.path.exists(drift_result_path):
        log.warning("No drift result found. Run drift_detector.py first.")
        return []

    with open(drift_result_path) as f:
        result = json.load(f)

    alerts = []
    for task, info in result.get("results", {}).items():
        if info.get("error"):
            a = write_alert("WARNING", task, f"Drift check failed: {info['error']}")
            alerts.append(a)
            continue

        drift_share = info.get("drift_share", 0)
        drift_detected = info.get("dataset_drift_detected", False)

        if drift_detected and drift_share > 0.4:
            a = write_alert(
                "CRITICAL", task,
                f"Severe data drift detected ({drift_share:.0%} of features drifted). Retraining required.",
                {"drift_share": drift_share},
            )
            alerts.append(a)
        elif drift_detected:
            a = write_alert(
                "WARNING", task,
                f"Moderate data drift detected ({drift_share:.0%} of features drifted).",
                {"drift_share": drift_share},
            )
            alerts.append(a)
        else:
            write_alert("INFO", task, "No significant drift detected.")

    return alerts


def read_recent_alerts(n: int = 20) -> list:
    alert_log = get_alert_log()
    if not os.path.exists(alert_log):
        return []
    alerts = []
    with open(alert_log) as f:
        for line in f:
            line = line.strip()
            if line:
                alerts.append(json.loads(line))
    return alerts[-n:]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    alerts = check_and_alert()
    log.info("%d alert(s) generated.", len(alerts))
