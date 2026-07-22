import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "experiments.json"

def log_experiment(
    model_name: str, 
    params: dict, 
    metrics: dict, 
    train_time: float, 
    inference_time: float, 
    dataset_version: str, 
    random_seed: int
):
    """
    Saves the results of a training and evaluation run.
    Stores records in logs/experiments.json.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "parameters": params,
        "metrics": metrics,
        "training_time_seconds": float(train_time),
        "inference_latency_seconds_per_sample": float(inference_time),
        "dataset_version": dataset_version,
        "random_seed": int(random_seed)
    }

    experiments = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r") as f:
                experiments = json.load(f)
        except Exception:
            pass

    experiments.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(experiments, f, indent=4)

    print(f"[Tracker] Logged experiment details for model: {model_name}")
