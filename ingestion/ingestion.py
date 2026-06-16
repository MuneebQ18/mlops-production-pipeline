import os
import time
import requests
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge
from drift_detector import SimpleDriftDetector
import subprocess
import json
import sys
# Append project root to path to allow seamless cross-directory sub-module loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.retrain_trigger import trigger_retraining_pipeline


# ==========================================
# METRICS CONFIGURATION (Task 3.3 Reference)
# ==========================================
# Task 3.3: Prometheus counter for feature additions
feature_added_counter = Counter('feature_added', 'Number of features added to the schema since startup')

# Task 3.3: Prometheus counter for feature removals
feature_removed_counter = Counter('feature_removed', 'Number of features removed from the schema since startup')

# Task 3.3: Prometheus gauge for binary distribution drift tracking (1 = drift, 0 = stable)
drift_gauge = Gauge('distribution_drift_detected', 'Set to 1 when drift is detected in the current batch, 0 otherwise')

# Task 3.3: Prometheus counter for tracking data source/datalake infrastructure unavailability
api_unavailable_counter = Counter('datalake_unavailable_total', 'Total number of times the data lake API was unavailable')

# ==========================================
# SYSTEM PARAMETERS & SECURITY CONFIGURATION
# ==========================================
url = "http://149.40.228.124:6500/records"
csv_path = "ingested_records.csv"
previous_schema = None
baseline_df = None 

# Task 4.2 Accuracy Monitoring Configuration
ACCURACY_THRESHOLD = 0.80

# In-memory tracking state to prevent repeated retraining loops for the same bad model version
last_evaluated_model_version = None

# SECURE CREDENTIAL MANAGEMENT: Prevents exposed keys in git logs/source code files
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

if not SLACK_WEBHOOK_URL:
    print("Warning: SLACK_WEBHOOK_URL environment variable is not set. Slack alerts will fail.")

# ==========================================
# UTILITY ALERTS & RETRAINING TRIGGER FUNCTIONS
# ==========================================
def send_slack_alert(message):
    """Sends a real-time notification payload to the configured Slack webhook channel."""
    if not SLACK_WEBHOOK_URL:
        return
    try:
        payload = {"text": message}
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send Slack notification: {e}")

# ==========================================
# MAIN LOOP ENGINE EXECUTION ENTRYPOINT
# ==========================================
# Expose the metric scraping endpoint externally on port 8000
start_http_server(8000)
print("Starting continuous ingestion loop. Metrics exposed at http://localhost:8000/metrics")

# Instantiate our custom drift detection component with a Z-Score threshold of 2.0
detector = SimpleDriftDetector(threshold=2.0)

# Task 4.1: Load initial values from the metadata file when ingestion.py boots up
from model.retrain_trigger import update_model_prometheus_metrics
update_model_prometheus_metrics()

while True:
    try:
        response = requests.get(url, timeout=10)
        
        # Guard against API Unavailability (Task 3.3 Service Failures)
        if response.status_code == 503:
            err_msg = "API Unavailable: Received status code 503 Service Unavailable from the Datalake."
            print(err_msg)
            api_unavailable_counter.inc()
            send_slack_alert(err_msg)
            time.sleep(30)
            continue
            
        if response.status_code == 200:
            raw_data = response.json()
            
            # Map dynamic incoming JSON dictionary entries into flat data structures
            records = []
            for item in raw_data:
                row = {f"feature_{i}": val for i, val in enumerate(item["features"])}
                row["label"] = item["label"]
                records.append(row)
                
            df = pd.DataFrame(records)
            current_schema = set(df.columns)
            
            # --------------------------------------------------
            # TASK 3.3 IMPLEMENTATION: SCHEMA MONITORING LOGIC
            # --------------------------------------------------
            if previous_schema is not None:
                added_columns = current_schema - previous_schema
                removed_columns = previous_schema - current_schema
                
                if added_columns or removed_columns:
                    msg = f"Schema change detected. Added: {list(added_columns)}, Removed: {list(removed_columns)}"
                    print(msg)
                    send_slack_alert(msg)
                    
                    # Trigger the retraining pipeline hook due to feature layout mutations
                    trigger_retraining_pipeline(f"Schema structural anomaly detected. Details: {msg}")
                    
                    if added_columns:
                        feature_added_counter.inc(len(added_columns))
                    if removed_columns:
                        feature_removed_counter.inc(len(removed_columns))

            # --------------------------------------------------
            # TASK 3.3 IMPLEMENTATION: DRIFT TRACKING LOGIC
            # --------------------------------------------------
            drift_status = 0
            
            if baseline_df is None:
                # Capture and freeze the first successful data batch as the baseline reference
                baseline_df = df.copy()
                detector.set_baseline(baseline_df)
                print("Baseline distribution checkpoint established and statistics cached.")
            else:
                # Execute drift check across common numeric keys using optimized cache
                drifted_cols = detector.get_drifted_columns(df)
                if drifted_cols:
                    drift_status = 1  # Alert Prometheus
                    send_slack_alert(f"Distribution drift detected in columns: {drifted_cols}")
                    
                    # Trigger the retraining pipeline hook due to data distribution changes
                    trigger_retraining_pipeline(f"Statistical distribution drift detected in features: {drifted_cols}")
            
            # Push current drift status to Prometheus
            drift_gauge.set(drift_status)
            previous_schema = current_schema
            
            # Write batch file to disk repository
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
            else:
                df.to_csv(csv_path, mode='a', header=False, index=False)
                
            print(f"Successfully saved {len(df)} records. Current schema tracking: {list(current_schema)}")

            # ----------------------------------------------------------------------
            # ACCURACY PERFORMANCE MONITORING (Task 4.2)
            # ----------------------------------------------------------------------
            metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "latest_model_metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        meta_data = json.load(f)
                        
                    current_model_ver = meta_data.get("model_version")
                    current_model_acc = meta_data.get("validation_accuracy", 1.0)
                    
                    # Only evaluate if we haven't already processed this model version
                    if current_model_ver != last_evaluated_model_version:
                        if current_model_acc < ACCURACY_THRESHOLD:
                            acc_msg = f"Performance Drop! Model v{current_model_ver} accuracy ({current_model_acc}) fell below threshold of {ACCURACY_THRESHOLD}."
                            print(acc_msg)
                            send_slack_alert(acc_msg)
                            
                            # Fire retraining pipeline immediately
                            trigger_retraining_pipeline(f"Accuracy performance degradation in production: {acc_msg}")
                            
                            # CRITICAL SAFETY UPDATE: Immediately sync the lock with the newly trained version
                            if os.path.exists(metadata_path):
                                with open(metadata_path, "r") as f_new:
                                    meta_data_new = json.load(f_new)
                                    current_model_ver = meta_data_new.get("model_version", current_model_ver)
                        
                        # Lock this model version to ensure we don't trigger again next cycle
                        last_evaluated_model_version = current_model_ver
                        
                except Exception as e:
                    print(f"Error reading model performance metadata file: {e}")
            # ----------------------------------------------------------------------

        else:
            print(f"Received unexpected status code: {response.status_code}")
            
    # Guard against Network Exceptions (Task 3.3 Transport Drops)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
        err_msg = f"Network Exception: Data source connection failed or timed out. Details: {e}"
        print(err_msg)
        api_unavailable_counter.inc()
        send_slack_alert(err_msg)
        
    time.sleep(30)