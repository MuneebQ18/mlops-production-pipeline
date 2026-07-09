import os
import sys
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient

# Create a temporary model before importing the app
os.makedirs("model", exist_ok=True)

X = pd.DataFrame(
    [[1.0, 2.0],
     [2.0, 3.0]],
    columns=["feature_0", "feature_1"]
)

y = [0, 1]

model = RandomForestClassifier(
    n_estimators=2,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model/model_v999.pkl")

from serving.app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    payload = {
        "feature_0": 1.0,
        "feature_1": 2.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "prediction" in body
    assert "confidence" in body
    assert "model_version" in body


def test_predict_bad_schema():
    payload = {
        "feature_0": 1.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422