"""
Export trained artifacts for the Streamlit app.

Runs ONLY the notebook cells required before export (sections 7–9),
then executes the export logic from section (9b).

Uses the identical code, hyperparameters, and random_state=42 from
fake_news_modeling.ipynb. Does NOT run EDA, exploratory TF-IDF, or
downstream evaluation/visualization cells.

If artifacts already exist, skips refitting and only verifies paths.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

MODELS_DIR = ROOT / "models"
SVM_PATH = MODELS_DIR / "linear_svm_model.joblib"
VEC_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


def artifacts_exist() -> bool:
    return SVM_PATH.exists() and VEC_PATH.exists()


def verify_artifacts() -> None:
    model = joblib.load(SVM_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print(f"✅ {SVM_PATH}")
    print(f"   size: {SVM_PATH.stat().st_size:,} bytes")
    print(f"✅ {VEC_PATH}")
    print(f"   size: {VEC_PATH.stat().st_size:,} bytes")
    print(f"   vectorizer features: {len(vectorizer.get_feature_names_out()):,}")
    print(f"   model classes: {model.classes_.tolist()}")


def build_and_export() -> None:
    print("Loading preprocessed data...")
    df = pd.read_csv(ROOT / "fake_news_eda_preprocessed.csv")

    expected = ["content", "clean_text", "label", "text_length", "word_count"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["clean_text", "label"]).copy()
    df["clean_text"] = df["clean_text"].astype(str).str.strip()
    df = df[df["clean_text"] != ""].copy()

    X_text = df["clean_text"].astype(str)
    y = df["label"]

    print("Train/test split (same as notebook section 7)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Fitting TF-IDF on training set only (section 8)...")
    tfidf_train = TfidfVectorizer(
        max_features=10_000,
        min_df=5,
        max_df=0.90,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf_train.fit_transform(X_train_text)
    X_test_tfidf = tfidf_train.transform(X_test_text)

    print("Training models (section 9 — export uses Linear SVM only)...")
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=30,
            early_stopping=True,
            random_state=42,
        ),
    }

    predictions = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            predictions[name] = model.predict(X_test_tfidf)

    f1 = f1_score(y_test, predictions["Linear SVM"])
    print(f"Verification — Linear SVM test F1: {f1:.4f} (expected 0.9981)")

    linear_svm_model = models["Linear SVM"]
    tfidf_vectorizer = tfidf_train

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(linear_svm_model, SVM_PATH)
    joblib.dump(tfidf_vectorizer, VEC_PATH)

    print("=" * 60)
    print("💾 Artifacts exported (section 9b)")
    print("=" * 60)
    verify_artifacts()
    print(f"\n▶️  streamlit run app.py")


if __name__ == "__main__":
    if artifacts_exist():
        print("Artifacts already present — verifying only (no refit).")
        verify_artifacts()
    else:
        build_and_export()
