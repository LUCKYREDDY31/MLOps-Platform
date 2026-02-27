"""Rule-based anomaly investigation agent. Checks drift reports, live API metrics, and alert logs."""

import json
import logging
import os
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class AnomalyAgent:
    def __init__(self, reports_dir: str = None, api_url: str = "http://localhost:8000"):
        project_root = get_project_root()
        self.reports_dir = reports_dir or str(project_root / "reports")
        self.logs_dir = str(project_root / "logs")
        self.api_url = api_url
        self.findings: list = []

    def _add_finding(self, level: str, title: str, details: str) -> None:
        self.findings.append({"level": level, "title": title, "details": details})
        log.info("  [%s] %s", level, title)

    def investigate(self, task: str, trigger: str) -> str:
        """Run a full investigation and return a markdown report."""
        self.findings = []
        log.info("Starting investigation for [%s] | Trigger: %s", task.upper(), trigger)

        self._check_drift_report(task)
        self._check_api_metrics(task)
        self._check_recent_alerts(task)

        report = self._generate_report(task, trigger)
        os.makedirs(self.reports_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        report_path = os.path.join(self.reports_dir, f"investigation_{task}_{ts}.md")
        with open(report_path, "w") as f:
            f.write(report)

        log.info("Investigation complete. Report saved to: %s", report_path)
        return report

    def _check_drift_report(self, task: str) -> None:
        path = os.path.join(self.reports_dir, f"{task}_drift_summary.json")
        if not os.path.exists(path):
            self._add_finding("INFO", "Drift Report Status", "No drift report found.")
            return

        with open(path) as f:
            summary = json.load(f)

        if summary.get("dataset_drift_detected"):
            share = summary.get("drift_share", 0)
            level = "CRITICAL" if share > 0.4 else "WARNING"
            self._add_finding(
                level,
                "Data Drift Detected",
                f"Drift detected in {share:.1%} of features. "
                "This is a likely root cause. Recommend retraining.",
            )
        else:
            self._add_finding("INFO", "Drift Report Status", "No significant data drift detected.")

    def _check_api_metrics(self, task: str) -> None:
        try:
            res = requests.get(f"{self.api_url}/metrics", timeout=5)
            res.raise_for_status()
            metrics = res.json().get(task, {})
        except requests.RequestException as e:
            self._add_finding("WARNING", "API Metrics", f"Could not fetch live metrics: {e}")
            return

        if not metrics:
            self._add_finding("INFO", "API Metrics", "No live prediction metrics available yet.")
            return

        positive_rate = metrics.get("positive_rate", 0)
        if task == "churn" and positive_rate > 0.35:
            self._add_finding(
                "WARNING", "High Churn Rate",
                f"Live churn rate is {positive_rate:.1%}, above the typical threshold.",
            )
        elif task == "fraud" and positive_rate > 0.05:
            self._add_finding(
                "CRITICAL", "High Fraud Rate",
                f"Live fraud rate is {positive_rate:.1%}, indicating a potential spike.",
            )
        else:
            self._add_finding(
                "INFO", "API Metrics",
                f"Live positive rate is {positive_rate:.1%}, within normal limits.",
            )

    def _check_recent_alerts(self, task: str) -> None:
        alert_log = os.path.join(self.logs_dir, "alerts.jsonl")
        if not os.path.exists(alert_log):
            return

        recent_alerts = []
        with open(alert_log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                alert = json.loads(line)
                if alert.get("task") == task and alert.get("level") == "CRITICAL":
                    ts = datetime.fromisoformat(alert["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) - ts < timedelta(hours=24):
                        recent_alerts.append(alert)

        if recent_alerts:
            msg = (
                f"Found {len(recent_alerts)} recent critical alert(s). "
                f"Example: \"{recent_alerts[0]['message']}\""
            )
            self._add_finding("WARNING", "Recent Critical Alerts", msg)

    def _generate_report(self, task: str, trigger: str) -> str:
        report = f"# Investigation Report: {task.upper()}\n\n"
        report += f"- **Timestamp**: {datetime.now(timezone.utc).isoformat()} UTC\n"
        report += f"- **Trigger**: {trigger}\n\n"
        report += "## Summary of Findings\n\n"

        for finding in self.findings:
            report += f"### {finding['level']}: {finding['title']}\n\n"
            report += f"> {finding['details']}\n\n"

        report += "## Recommended Actions\n\n"
        if any(f["level"] == "CRITICAL" for f in self.findings):
            report += "- **Immediate action required.** A critical issue was detected.\n"
            report += "- **Recommendation**: Trigger the auto-retraining pipeline immediately.\n"
        elif any(f["level"] == "WARNING" for f in self.findings):
            report += "- **Monitor closely.** A potential issue was detected.\n"
            report += "- **Recommendation**: Keep an eye on the dashboard metrics. "
            report += "If the situation worsens, consider a manual retraining run.\n"
        else:
            report += "- **No immediate action required.** "
            report += "The system appears to be operating within normal parameters.\n"

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    agent = AnomalyAgent()
    agent.investigate("fraud", trigger="Manual run from CLI")
    agent.investigate("churn", trigger="Scheduled daily check")
