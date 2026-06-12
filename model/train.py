import os
import re
import json      # For saving model metadata metrics
import datetime  # For generating tracking timestamps
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_next_version(directory, prefix="model_v", extension=".pkl"):
    """
    Scans the model directory to find existing model files and
    calculates the next sequential version integer.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1
        
    max_version = 0
    pattern = re.compile(rf"^{prefix}(\d+){extension}$")
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            max_version = max(max_version, int(match.group(1)))
            
    return max_version + 1

def run_minimal_training(accuracy_threshold=0.80, max_attempts=5):
    # --------------------------------------------------
    # STATIC PATH DETERMINATION
    # --------------------------------------------------
    # Resolve the absolute folder containing train.py (top-level model/ directory)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve the project root path (one folder level up from train.py)
    project_root = os.path.dirname(model_dir)
    
    # Anchor the data target to the project root directory
    csv_path = os.path.join(project_root, "ingestion", "ingested_records.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Training source file data not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Error: Ingested training dataset is empty.")
        return

    X = df.drop(columns=["label"])
    y = df["label"]

    # --------------------------------------------------
    # ITERATIVE TRAINING & THRESHOLD LOGIC
    # --------------------------------------------------
    best_model = None
    best_accuracy = -1.0
    threshold_achieved = False
    
    print(f"Starting training loop. Target Threshold: {accuracy_threshold}")
    
    for attempt in range(1, max_attempts + 1):
        # Pass dynamic random seeds to introduce execution variations per loop pass
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42 + attempt, stratify=y if y.nunique() > 1 else None
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42 + attempt)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        print(f"Attempt {attempt}: Accuracy = {accuracy:.4f}")

        # Update absolute historical best performer references
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Check early stopping threshold criteria
        if accuracy >= accuracy_threshold:
            print(f"Success! Accuracy threshold met on attempt {attempt}.")
            threshold_achieved = True
            break

    # --------------------------------------------------
    # WARNING HANDLING & FIXED VERSIONED SERIALIZATION
    # --------------------------------------------------
    if not threshold_achieved:
        print(f"WARNING: Maximum training attempts ({max_attempts}) reached without meeting the target threshold of {accuracy_threshold}.")

    # Calculate sequential version index based strictly on the fixed model_dir path
    next_ver = get_next_version(model_dir)
    filename = f"model_v{next_ver}.pkl"
    filepath = os.path.join(model_dir, filename)
    
    # Write the champion iteration binary module to the absolute target directory location
    joblib.dump(best_model, filepath)

    # --------------------------------------------------
    # FIXED METADATA SERIALIZATION (Task 4.2 Single File Design)
    # --------------------------------------------------
    # Generate metadata dictionary payload exactly as requested
    metadata = {
        "model_version": next_ver,
        "validation_accuracy": round(best_accuracy, 4),
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Target a static, unversioned file that gets overwritten every run
    metadata_filename = "latest_model_metadata.json"
    metadata_filepath = os.path.join(model_dir, metadata_filename)
    
    # Write metadata payload to disk as JSON (overwriting previous contents)
    with open(metadata_filepath, "w") as f:
        json.dump(metadata, f, indent=4)

    # Output Execution Metrics Summary Table
    print("\n" + "="*40)
    print(f"Final Selected Accuracy: {best_accuracy:.4f}")
    print(f"Threshold Achieved: {threshold_achieved}")
    print(f"Saved Model Location: {filepath}")
    print(f"Saved Metadata Location: {metadata_filepath}")
    print("="*40)

if __name__ == "__main__":
    run_minimal_training(accuracy_threshold=0.80, max_attempts=5)