"""
train.py

The main entry point for this project. Run it from the project root:

    python src/train.py

Demonstrates: configuration management (config.yaml + .env), JSON output,
logging, exception handling, reproducibility, and model persistence with
joblib \u2014 all in one place. The saved model is what main.py (the FastAPI
app) loads to serve predictions.
"""

import json
import os
import random
import sys
from pathlib import Path

import joblib
import yaml
from dotenv import load_dotenv

# Allow this file to be run directly as `python src/train.py` from the
# project root, as well as with `python -m src.train`. Without this, Python
# would not recognize `src` as an importable package when run directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import DataValidationError, load_dataset, split_features_target
from src.model import RandomForestModel, TrainingConfig
from src.utils import setup_logging

MODEL_PATH = "models/model.joblib"


def load_app_config(path: str = "configs/config.yaml") -> dict:
    """Load non-secret settings (paths, hyperparameters) from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    logger = setup_logging("logs/train.log")

    # .env holds secrets (never committed). config.yaml holds everything else.
    load_dotenv()
    api_key = os.getenv("API_KEY", "not-set")
    masked_key = api_key[:3] + "*" * max(len(api_key) - 3, 0)
    logger.info(f"Loaded API_KEY from .env: {masked_key}")

    app_config = load_app_config()

    # Fixing the random seed makes this run reproducible.
    random_state = app_config["training"]["random_state"]
    random.seed(random_state)

    logger.info("Training run started")

    try:
        df = load_dataset(app_config["data_path"])
    except (FileNotFoundError, DataValidationError) as e:
        logger.error(f"Failed to load data: {e}")
        raise

    target_col = app_config["data"]["target_column"]
    X, y = split_features_target(df, target_col=target_col)

    training_config = TrainingConfig(
        n_estimators=app_config["model"]["n_estimators"],
        random_state=random_state,
        test_size=app_config["training"]["test_size"],
    )
    model = RandomForestModel(training_config)
    model.train(X, y)

    accuracy = model.score(X, y)
    logger.info(f"Training complete. Accuracy: {accuracy:.4f}")
    print(f"Training accuracy: {accuracy:.4f}")

    # Persist the trained model with joblib, alongside the feature names it
    # expects, so main.py can load it later without needing to retrain.
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model.model,
        "feature_names": list(X.columns),
        "target_column": target_col,
        "random_state": random_state,
    }
    joblib.dump(artifact, MODEL_PATH)
    logger.info(f"Saved trained model to {MODEL_PATH}")

    # Save results as JSON so they can be read by other tools or teammates.
    metrics = {"accuracy": round(accuracy, 4), "random_state": random_state}
    with open("logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to logs/metrics.json")


if __name__ == "__main__":
    main()
