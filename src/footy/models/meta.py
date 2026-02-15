from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

META_MODEL_VERSION = "v2_meta_stack"
MODEL_PATH = Path("data/models") / f"{META_MODEL_VERSION}.joblib"

FEATURE_NAMES = [
    "pE_home","pE_draw","pE_away",
    "pP_home","pP_draw","pP_away",
    "elo_diff",
    "eg_home","eg_away","eg_diff"
]

def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=500,
            C=1.0,
        )),
    ])

def save(model: Pipeline) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)

def load() -> Pipeline | None:
    if not MODEL_PATH.exists():
        return None
    obj = joblib.load(MODEL_PATH)
    return obj["model"]

def predict_proba(model: Pipeline, X: np.ndarray) -> np.ndarray:
    # returns shape (n, 3) for [home, draw, away]
    return model.predict_proba(X)
