"""
main.py

A small FastAPI application that serves predictions from the model trained
by src/train.py.

Run it from the project root (train a model first if you haven't already):

    python src/train.py
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API documentation,
where you can try each endpoint directly in the browser.
"""

import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.utils import setup_logging

MODEL_PATH = "models/model.joblib"

logger = setup_logging("logs/api.log")

# .env holds secrets — here, a simple API key used to protect /predict.
load_dotenv()
API_KEY = os.getenv("API_KEY", "not-set")

# Populated once at startup by the lifespan handler below, then reused for
# every request — the model is only loaded from disk a single time.
model_artifact: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once when the API starts up."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"No trained model found at '{MODEL_PATH}'. "
            "Run `python src/train.py` first to create one."
        )
    model_artifact["data"] = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    yield
    model_artifact.clear()
    logger.info("API shutting down")


app = FastAPI(
    title="AI Mini Project API",
    description="Serves predictions from the model trained in src/train.py",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """The input features needed to make one prediction."""

    features: list[float] = Field(
        ...,
        description="Feature values, in the same order as feature_names "
        "returned by GET /model-info.",
        examples=[[0.5, -1.2, 0.3, 1.1, -0.4]],
    )


class PredictionResponse(BaseModel):
    """The model's prediction for one set of input features."""

    prediction: int
    probability: float


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    A simple dependency that checks the X-API-Key header against the value
    loaded from .env. Real projects would use a proper auth scheme, but
    this demonstrates the same config/.env pattern from training in the
    context of an API.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def read_root() -> dict:
    """Basic info about the API. No authentication required."""
    return {
        "name": "AI Mini Project API",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict"],
    }


@app.get("/health")
def health_check() -> dict:
    """Simple health check. No authentication required."""
    model_loaded = "data" in model_artifact
    return {"status": "ok" if model_loaded else "model not loaded"}


@app.get("/model-info")
def model_info() -> dict:
    """Describe the currently loaded model. No authentication required."""
    artifact = model_artifact["data"]
    return {
        "feature_names": artifact["feature_names"],
        "target_column": artifact["target_column"],
        "random_state": artifact["random_state"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict a label for one set of feature values.

    Requires a valid X-API-Key header (see .env / .env.example).
    """
    artifact = model_artifact["data"]
    feature_names = artifact["feature_names"]

    if len(request.features) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {len(feature_names)} features "
                f"({feature_names}), got {len(request.features)}."
            ),
        )

    sk_model = artifact["model"]
    row = pd.DataFrame([request.features], columns=feature_names)
    prediction = sk_model.predict(row)[0]
    probability = max(sk_model.predict_proba(row)[0])

    logger.info(f"Prediction served: {prediction} (probability={probability:.4f})")

    return PredictionResponse(
        prediction=int(prediction), probability=float(probability)
    )
