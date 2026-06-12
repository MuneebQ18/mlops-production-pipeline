import os
import json
import subprocess
import requests
from prometheus_client import Counter
from prometheus_client import Counter, Gauge  # Update import

# Preserve your exact Prometheus metric reference
retrain_total_counter = Counter('retrain_total', 'Total number of times the retraining pipeline has been triggered')

# Fetch Slack configuration securely from environment variables
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

# Task 4.1: Metrics recording and exposing active model state
model_version_gauge = Gauge('model_version', 'The version integer of the currently deployed model')
model_accuracy_gauge = Gauge('model_validation_accuracy', 'The validation accuracy of the currently deployed model')

def update_model_prometheus_metrics():
    """Reads latest_model_metadata.json and refreshes the Prometheus Gauges."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "latest_model_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta_data = json.load(f)
            
            # Extract and update values in Prometheus
            version = meta_data.get("model_version")
            accuracy = meta_data.get("validation_accuracy")
            
            if version is not None:
                model_version_gauge.set(version)
            if accuracy is not None:
                model_accuracy_gauge.set(accuracy)
                
            print(f"[PROMETHEUS UPDATE] Synchronized state to Model v{version} (Acc: {accuracy})")
        except Exception as e:
            print(f"Failed to refresh model metrics to Prometheus: {e}")


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
    Task 4.2 Deliverable: Orchestrates subprocess training, updates counters, 
    parses metadata, and dispatches structural status alerts to Slack.
    """
    print(f"\n[RETRAINING INITIATED] Reason: {reason}")
    retrain_total_counter.inc()
    
    # Resolve the path to train.py relative to this orchestrator file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "train.py")

    print(f"Executing training pipeline process via shell: {script_path}")
    
    try:
        result = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        print("----- SUBPROCESS TRAINING OUTPUT -----")
        print(result.stdout.strip())
        print("--------------------------------------")
        
        # REFRESH PROMETHEUS METRICS IMMEDIATELY ON SUCCESSFUL TRAINING
        update_model_prometheus_metrics()
        
        
        # Resolve path to the unversioned single metadata file
        metadata_path = os.path.join(current_dir, "latest_model_metadata.json")
        
        version = "Unknown"
        accuracy = "Unknown"
        metadata_retrieved = False
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    meta_data = json.load(f)
                version = meta_data.get("model_version", "Unknown")
                accuracy = meta_data.get("validation_accuracy", "Unknown")
                metadata_retrieved = True
            except Exception as e:
                print(f"Error parsing metadata file inside trigger pipeline: {e}")

        if metadata_retrieved:
            slack_msg = (
                f"Retraining completed successfully.\n"
                f"Reason: {reason}\n"
                f"Model Version: {version}\n"
                f"Validation Accuracy: {accuracy}"
            )
        else:
            slack_msg = (
                f"Retraining completed successfully, but metadata retrieval failed.\n"
                f"Reason: {reason}"
            )
            
        send_slack_alert(slack_msg)
        
    except subprocess.CalledProcessError as e:
        print(f"CRITICAL: Retraining subprocess execution failed with exit code {e.returncode}")
        print(f"Subprocess Error Output:\n{e.stderr.strip()}")
        send_slack_alert(f"CRITICAL: Retraining pipeline failed execution for Reason: {reason}")
    except Exception as e:
        print(f"An unexpected error occurred while launching training subprocess: {e}")