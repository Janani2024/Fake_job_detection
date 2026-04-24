import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
RESULTS_PATH = os.path.join(MODELS_DIR, "results.joblib")

_tfidf = None
_model = None


def _load():
    global _tfidf, _model
    if _tfidf is None:
        _tfidf = joblib.load(TFIDF_PATH)
    if _model is None:
        _model = joblib.load(BEST_MODEL_PATH)


def models_exist() -> bool:
    return os.path.exists(TFIDF_PATH) and os.path.exists(BEST_MODEL_PATH)


def predict(clean_text: str) -> dict:
    _load()
    vec = _tfidf.transform([clean_text])
    prediction = int(_model.predict(vec)[0])
    label = "Fake" if prediction == 1 else "Real"

    confidence = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(vec)[0]
        confidence = round(float(proba[prediction]) * 100, 1)
    elif hasattr(_model, "decision_function"):
        score = _model.decision_function(vec)[0]
        # Convert decision score to a rough confidence percentage
        confidence = round(min(99.9, max(50.1, 50 + abs(float(score)) * 10)), 1)

    return {"label": label, "prediction": prediction, "confidence": confidence}


def load_results() -> dict:
    if not os.path.exists(RESULTS_PATH):
        return {}
    return joblib.load(RESULTS_PATH)
