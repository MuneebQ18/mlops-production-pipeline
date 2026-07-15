import json
import os
import time
from prometheus_client import Gauge, Counter, start_http_server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "metrics_state.json")

# Store previously exported counter values
last_feature_added = 0
last_feature_removed = 0
last_datalake_unavailable = 0
last_retrain_count_total = 0

model_accuracy = Gauge("model_accuracy", "Current model validation accuracy")
model_version = Gauge("model_version", "Current model version")
distribution_drift_detected = Gauge("distribution_drift_detected", "Distribution drift status")
feature_added = Counter("feature_added", "Feature additions detected")
feature_removed = Counter("feature_removed", "Feature removals detected")
datalake_unavailable = Counter("datalake_unavailable", "Data lake unavailable events")
retrain_count_total = Counter("retrain_count_total", "Total retraining runs")


def refresh_metrics():
    if not os.path.exists(STATE_FILE):
        return

    with open(STATE_FILE, "r") as f:
        data = json.load(f)

        global last_feature_added
        global last_feature_removed
        global last_datalake_unavailable
        global last_retrain_count_total

        model_accuracy.set(data.get("model_accuracy", 0))
        model_version.set(data.get("model_version", 0))
        distribution_drift_detected.set(data.get("distribution_drift_detected", 0))

        current = data.get("feature_added", 0)
        if current > last_feature_added:
            feature_added.inc(current - last_feature_added)
        last_feature_added = current

        current = data.get("feature_removed", 0)
        if current > last_feature_removed:
            feature_removed.inc(current - last_feature_removed)
        last_feature_removed = current

        current = data.get("datalake_unavailable", 0)
        if current > last_datalake_unavailable:
            datalake_unavailable.inc(current - last_datalake_unavailable)
        last_datalake_unavailable = current

        current = data.get("retrain_count_total", 0)
        if current > last_retrain_count_total:
            retrain_count_total.inc(current - last_retrain_count_total)
        last_retrain_count_total = current


if __name__ == "__main__":
    start_http_server(8001)
    print("Metrics exporter running on :8001")

    while True:
        refresh_metrics()
        time.sleep(5)