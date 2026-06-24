"""
Fake News Detection — Streamlit demo (inference only).

Loads pre-trained Linear SVM + TF-IDF vectorizer exported from fake_news_modeling.ipynb.
"""
import os
from pathlib import Path

import joblib
import streamlit as st

from preprocessing import clean_text

APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)

PROJECT_METRICS = {
    "Accuracy": "99.80%",
    "Precision": "99.81%",
    "Recall": "99.81%",
    "F1 Score": "99.81%",
}

LIMITATION = (
    "This model does not verify facts. It predicts whether an article resembles "
    "the linguistic patterns of real or fake news articles learned from the training dataset."
)


@st.cache_resource
def load_artifacts():
    svm_path = APP_DIR / "models" / "linear_svm_model.joblib"
    vec_path = APP_DIR / "models" / "tfidf_vectorizer.joblib"
    if not svm_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run Phase 3 + export cell (9b) in "
            "fake_news_modeling.ipynb to create models/linear_svm_model.joblib and "
            "models/tfidf_vectorizer.joblib."
        )
    model = joblib.load("models/linear_svm_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    return model, vectorizer


def predict_article(text: str, model, vectorizer):
    cleaned = clean_text(text)
    if not cleaned:
        return None, cleaned, None
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])
    decision_score = float(model.decision_function(features)[0])
    return prediction, cleaned, decision_score


def main():
    st.set_page_config(
        page_title="Fake News Detection System",
        page_icon="📰",
        layout="centered",
    )

    st.title("Fake News Detection System")
    st.subheader("Machine Learning Final Project")

    st.markdown("---")

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    article = st.text_area(
        "Paste a news article here",
        height=220,
        placeholder="Enter the full news article text (title + body)...",
    )

    if st.button("Predict", type="primary"):
        if not article.strip():
            st.warning("Please paste a news article before predicting.")
        else:
            prediction, cleaned, decision_score = predict_article(article, model, vectorizer)
            if not cleaned:
                st.warning(
                    "The text could not be processed after cleaning. "
                    "Please provide a longer article with meaningful English words."
                )
            else:
                st.markdown("---")
                if prediction == 0:
                    st.success("✅ Real News")
                else:
                    st.error("❌ Fake News")

                st.metric(
                    label="Decision function score",
                    value=f"{decision_score:.4f}",
                    help=(
                        "Signed distance from the Linear SVM decision boundary. "
                        "Higher positive → more Fake-like; more negative → more Real-like. "
                        "This is NOT a calibrated probability."
                    ),
                )
                st.caption(
                    "The decision function score indicates direction and margin from the "
                    "class boundary — it is **not** a probability of being fake or real."
                )

    st.markdown("---")
    st.header("Project Information")

    st.subheader("Project Pipeline")
    st.code(
        "Article\n"
        "→ Text Cleaning\n"
        "→ TF-IDF\n"
        "→ Linear SVM\n"
        "→ Prediction",
        language=None,
    )

    st.subheader("Model Performance (Test Set)")
    cols = st.columns(4)
    for col, (name, value) in zip(cols, PROJECT_METRICS.items()):
        col.metric(name, value)

    st.subheader("Important Limitation")
    st.info(LIMITATION)


if __name__ == "__main__":
    main()
