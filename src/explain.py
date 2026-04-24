import os
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")

_tfidf = None
_model = None
_explainer = LimeTextExplainer(class_names=["Real", "Fake"])


def _load():
    global _tfidf, _model
    if _tfidf is None:
        _tfidf = joblib.load(TFIDF_PATH)
    if _model is None:
        _model = joblib.load(BEST_MODEL_PATH)


def _predict_proba_wrapper(texts):
    _load()
    vec = _tfidf.transform(texts)
    if hasattr(_model, "predict_proba"):
        return _model.predict_proba(vec)
    # For models without predict_proba (e.g. LinearSVC), use decision function
    scores = _model.decision_function(vec)
    if scores.ndim == 1:
        # Binary: convert to two-column probability-like array
        pos = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - pos, pos])
    return scores


def explain(clean_text: str, num_features: int = 10) -> dict:
    _load()
    exp = _explainer.explain_instance(
        clean_text,
        _predict_proba_wrapper,
        num_features=num_features,
        num_samples=500,
    )
    # Returns list of (word, weight) sorted by abs weight descending
    word_weights = exp.as_list()
    word_weights_sorted = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)

    fake_words = [(w, round(wt, 4)) for w, wt in word_weights_sorted if wt > 0]
    real_words = [(w, round(abs(wt), 4)) for w, wt in word_weights_sorted if wt < 0]

    return {
        "all_weights": word_weights_sorted,
        "fake_indicators": fake_words,
        "real_indicators": real_words,
        "html": exp.as_html(),
    }
