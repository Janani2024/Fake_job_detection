import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from src.preprocess import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.joblib")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
RESULTS_PATH = os.path.join(MODELS_DIR, "results.joblib")


def _get_classifiers():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "SVM": LinearSVC(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def train(csv_path: str) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_and_preprocess(csv_path)
    X_raw = df["clean_text"].to_numpy(dtype=str)
    y = df["label"].to_numpy(dtype=int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)

    # Apply SMOTE only on training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    joblib.dump(tfidf, TFIDF_PATH)

    classifiers = _get_classifiers()
    results = {}
    best_recall = -1
    best_name = None

    for name, clf in classifiers.items():
        # Naive Bayes needs non-negative input; SMOTE can produce negatives with TF-IDF
        if name == "Naive Bayes":
            X_tr = abs(X_train_res)
            X_te = abs(X_test)
        else:
            X_tr = X_train_res
            X_te = X_test

        clf.fit(X_tr, y_train_res)

        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        report = classification_report(y_test, y_pred, target_names=["Real", "Fake"])

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "report": report,
            "model": clf,
        }

        if recall > best_recall:
            best_recall = recall
            best_name = name

    joblib.dump(results[best_name]["model"], BEST_MODEL_PATH)
    joblib.dump(
        {k: {m: v for m, v in r.items() if m != "model"} for k, r in results.items()},
        RESULTS_PATH,
    )

    print(f"\nBest model: {best_name} (Recall={best_recall*100:.1f}%)")
    for name, r in results.items():
        print(f"{name}: Acc={r['accuracy']}%  Recall={r['recall']}%  F1={r['f1']}%")

    return results


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/fake_job_postings.csv"
    train(csv)
