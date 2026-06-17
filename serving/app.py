import os
import re
import joblib
import pandas as pd
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
from prometheus_client import make_asgi_app, Counter, Histogram

# ==========================================
# SYSTEM & PATH CONFIGURATION
# ==========================================
app = FastAPI(title="MLOps Dynamic Inference Service", version="2.0.0")

# Resolve absolute pathing contexts to match structural deliverables
SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SERVING_DIR, "..", "model"))

# ==========================================
# PROMETHEUS METRICS INSTRUMENTATION
# ==========================================
inference_request_total = Counter(
    'inference_requests_total', 
    'Total number of prediction requests processed',
    ['status']
)
inference_latency_seconds = Histogram(
    'response_delay_seconds', 
    'Time spent processing model inference'
)

# Route Prometheus client exposition metrics path cleanly via ASGI middleware
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ==========================================
# STATE MANAGEMENT ENGINE
# ==========================================
class ModelRuntimeState:
    """
    Manages active state contexts safely inside memory cache.
    Decoupled to handle on-demand dynamic reloading easily in future milestones.
    """
    def __init__(self):
        self.model = None
        self.version = None
        self.feature_names = []
        self.validator_schema = None

    def locate_latest_model(self, directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Model directory does not exist at: {directory}")
            
        max_version = 0
        latest_filename = None
        pattern = re.compile(r"^model_v(\d+)\.pkl$")
        
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                version_num = int(match.group(1))
                if version_num > max_version:
                    max_version = version_num
                    latest_filename = filename
                    
        if not latest_filename:
            raise FileNotFoundError(f"No versioned model artifacts (model_vN.pkl) found in {directory}")
            
        return os.path.join(directory, latest_filename), max_version

    def load_active_champion(self):
        try:
            model_path, version = self.locate_latest_model(MODEL_DIR)
            loaded_model = joblib.load(model_path)
            
            # Extract historical structural count parameters straight from the model's metadata properties
            if hasattr(loaded_model, "n_features_in_"):
                num_features = loaded_model.n_features_in_
            else:
                raise AttributeError("Loaded classifier model lacks a verifiable 'n_features_in_' attribute.")
                
            # Reconstruct exact deterministic feature columns matrix mapping strings
            self.feature_names = [f"feature_{i}" for i in range(num_features)]
            
            # Generate dynamic runtime evaluation models based on discovered sizing
            fields = {feat: (float, ...) for feat in self.feature_names}
            self.validator_schema = create_model("DynamicInferenceRequest", **fields)
            
            self.model = loaded_model
            self.version = version
            
            print(f"[BOOT SUCCESS] Loaded Model v{self.version} with {num_features} expected input features.")
        except Exception as e:
            print(f"[CRITICAL] Operational failure establishing dynamic schema mappings: {e}")
            self.model = None
            self.version = None
            self.feature_names = []
            # Build a fallback empty validation container to prevent runtime compilation errors
            self.validator_schema = create_model("FallbackInferenceRequest")

# Initialize global runtime state engine context
state = ModelRuntimeState()
state.load_active_champion()

# ==========================================
# API ENDPOINT ROUTING DEFINITIONS
# ==========================================
@app.get("/health")
def health_check():
    """System health check endpoint verifying model load status."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Inference service unhealthy: Model tracking context lost.")
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Dict[str, float]):
    """
    Accepts arbitrary JSON dictionaries payloads, executes dynamic key alignment
    verifications, maps elements into standard DataFrames, and yields predictions.
    """
    if state.model is None:
        inference_request_total.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Inference model is uninitialized.")
        
    try:
        # 1. Dynamic Key Inspection (Validates exact feature sets matches model baseline)
        received_keys = set(payload.keys())
        expected_keys = set(state.feature_names)
        
        missing_keys = expected_keys - received_keys
        extra_keys = received_keys - expected_keys
        
        if missing_keys or extra_keys:
            inference_request_total.labels(status="error").inc()
            raise HTTPException(
                status_code=422, 
                detail={
                    "error": "Schema misalignment matrix error.",
                    "missing_features": list(missing_keys),
                    "unexpected_features": list(extra_keys)
                }
            )
            
        # 2. Convert incoming structural sequence maps to ordered layout DataFrames
        ordered_row = [payload[feat] for feat in state.feature_names]
        df_input = pd.DataFrame([ordered_row], columns=state.feature_names)
        
        # 3. Instrument timing contexts safe from execution jitter variations
        with inference_latency_seconds.time():
            prediction = state.model.predict(df_input)
            
            # --- FIXED CONFIDENCE SCORE COMPUTATION FOR ARBITRARY LABELS ---
            if hasattr(state.model, "predict_proba"):
                probabilities = state.model.predict_proba(df_input)
                # Map the predicted label value to its exact tracking position inside model.classes_
                predicted_label = prediction[0]
                class_position = list(state.model.classes_).index(predicted_label)
                confidence = float(probabilities[0][class_position])
            elif hasattr(state.model, "decision_function"):
                scores = state.model.decision_function(df_input)
                confidence = float(scores[0])
            else:
                confidence = 1.0
            # ----------------------------------------------------------------

        inference_request_total.labels(status="success").inc()
        
        return {
            "prediction": int(prediction[0]),
            "confidence": round(confidence, 4),
            "model_version": state.version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        inference_request_total.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=f"Prediction mapping crashed: {str(e)}")