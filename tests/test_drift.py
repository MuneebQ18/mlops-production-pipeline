import pandas as pd
from ingestion.drift_detector import SimpleDriftDetector


def test_no_drift_detected():
    detector = SimpleDriftDetector(threshold=2.0)

    baseline = pd.DataFrame({
        "feature_0": [1.0, 1.1, 0.9, 1.0],
        "feature_1": [5.0, 5.2, 4.8, 5.1]
    })

    detector.set_baseline(baseline)

    current = pd.DataFrame({
        "feature_0": [1.0, 1.05, 0.95, 1.0],
        "feature_1": [5.1, 5.0, 4.9, 5.2]
    })

    assert detector.get_drifted_columns(current) == []


def test_drift_detected():
    detector = SimpleDriftDetector(threshold=2.0)

    baseline = pd.DataFrame({
        "feature_0": [1.0, 1.1, 0.9, 1.0],
        "feature_1": [5.0, 5.2, 4.8, 5.1]
    })

    detector.set_baseline(baseline)

    shifted = pd.DataFrame({
        "feature_0": [10.0, 10.5, 9.8, 10.2],
        "feature_1": [5.0, 5.1, 5.2, 5.0]
    })

    drifted = detector.get_drifted_columns(shifted)

    assert "feature_0" in drifted