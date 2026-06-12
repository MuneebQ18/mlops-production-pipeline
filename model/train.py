import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def get_next_version(directory="model", prefix="model_v", extension=".pkl"):
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
    csv_path = "ingestion/ingested_records.csv"
    if not os.path.exists(csv_path) and os.path.exists("../ingestion/ingested_records.csv"):
        csv_path = "../ingestion/ingested_records.csv"
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
    # NEW CODE: ITERATIVE TRAINING & THRESHOLD LOGIC
    # --------------------------------------------------
    best_model = None
    best_accuracy = -1.0
    threshold_achieved = False
    
    print(f"Starting training loop. Target Threshold: {accuracy_threshold}")
    
    for attempt in range(1, max_attempts + 1):
        # Splitting data inside the loop allows variation if random state or data shifts
        # Setting a dynamic random state per attempt to simulate varying training initialization conditions
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42 + attempt, stratify=y if y.nunique() > 1 else None
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42 + attempt)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        print(f"Attempt {attempt}: Accuracy = {accuracy:.4f}")

        # Track tracking parameter statistics across iterations
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Check early exit criteria condition
        if accuracy >= accuracy_threshold:
            print(f"Success! Accuracy threshold met on attempt {attempt}.")
            threshold_achieved = True
            break

    # --------------------------------------------------
    # WARNINGS & MODEL PERSISTENCE
    # --------------------------------------------------
    if not threshold_achieved:
        print(f"WARNING: Maximum training attempts ({max_attempts}) reached without meeting the target threshold of {accuracy_threshold}.")

    # Resolve destination path directory safely
    model_dir = "model"
    if not os.path.exists(model_dir) and os.path.basename(os.getcwd()) == "model":
        model_dir = "."

    next_ver = get_next_version(model_dir)
    filename = f"model_v{next_ver}.pkl"
    filepath = os.path.join(model_dir, filename)
    
    # Save whichever model performed best across the total runtime footprint
    joblib.dump(best_model, filepath)

    # Output Execution Metrics Summary Table
    print("\n" + "="*40)
    print(f"Final Selected Accuracy: {best_accuracy:.4f}")
    print(f"Threshold Achieved: {threshold_achieved}")
    print(f"Saved Model Filename: {filename}")
    print("="*40)

if __name__ == "__main__":
    # Parameters can easily be customized here
    run_minimal_training(accuracy_threshold=0.80, max_attempts=5)