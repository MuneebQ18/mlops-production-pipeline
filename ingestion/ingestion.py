import os
import time
import requests
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge
from drift_detector import SimpleDriftDetector
import subprocess


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

# Task 4.2: Prometheus counter tracking the cumulative number of retraining pipeline invocations
retrain_total_counter = Counter('retrain_total', 'Total number of times the retraining pipeline has been triggered')


# ==========================================
# SYSTEM PARAMETERS & SECURITY CONFIGURATION
# ==========================================
url = "http://149.40.228.124:6500/records"
csv_path = "ingested_records.csv"
previous_schema = None
baseline_df = None 

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

def trigger_retraining_pipeline(reason):
    """
    Task 4.2: Invocated by structural anomalies or statistical drift events.
    Increments metrics, spins up the training subprocess, and captures evaluation logs.
    """
    print(f"\n[RETRAINING INITIATED] Reason: {reason}")
    
    # 1. Increment the cumulative execution metric counter
    retrain_total_counter.inc()
    
    # Resolve the path to train.py dynamically based on execution context
    script_path = "model/train.py"
    if not os.path.exists(script_path) and os.path.exists("../model/train.py"):
        script_path = "../model/train.py"

    print(f"Executing training pipeline process via shell: {script_path}")
    
    try:
        # 2. Execute model/train.py as a distinct isolated system subprocess
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 3. Stream and display stdout metrics emitted from the training execution
        print("----- SUBPROCESS TRAINING OUTPUT -----")
        print(result.stdout.strip())
        print("--------------------------------------")
        
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: Retraining subprocess execution failed with exit code {e.returncode}")
        print(f"Subprocess Error Output:\n{e.stderr.strip()}")
    except Exception as e:
        print(f"An unexpected error occurred while launching training subprocess: {e}")


# ==========================================
# MAIN LOOP ENGINE EXECUTION ENTRYPOINT
# ==========================================
# Expose the metric scraping endpoint externally on port 8000
start_http_server(8000)
print("Starting continuous ingestion loop. Metrics exposed at http://localhost:8000/metrics")

# Instantiate our custom drift detection component with a Z-Score threshold of 2.0
detector = SimpleDriftDetector(threshold=2.0)

# trigger_retraining_pipeline("manual test")

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
        else:
            print(f"Received unexpected status code: {response.status_code}")
            
    # Guard against Network Exceptions (Task 3.3 Transport Drops)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
        err_msg = f"Network Exception: Data source connection failed or timed out. Details: {e}"
        print(err_msg)
        api_unavailable_counter.inc()
        send_slack_alert(err_msg)
        
    time.sleep(30)