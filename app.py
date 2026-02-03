import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

# ==============================
# Load Model
# ==============================

MODEL_PATH = "model"
model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# Prometheus Metrics
# ==============================

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

# ==============================
# FastAPI App
# ==============================

app = FastAPI(title="Wine Quality Model API")

# ==============================
# Input Schema (Wine Features)
# ==============================

class WineRequest(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    ph: float
    sulphates: float
    alcohol: float

# ==============================
# Health Check
# ==============================

@app.get("/")
def health_check():
    return {"status": "Model is running"}

# ==============================
# Metrics Endpoint
# ==============================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

# ==============================
# Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(data: WineRequest):

    REQUEST_COUNT.inc()

    start_time = time.time()

    input_dict = {
        key: np.array([[getattr(data, key)]])
        for key in data.dict().keys()
    }

    prediction = model.predict(input_dict)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    REQUEST_LATENCY.observe(time.time() - start_time)

    return {
        "prediction_probabilities": prediction.tolist(),
        "predicted_class": predicted_class
    }
