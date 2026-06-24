# System Architecture — Fake News Detection Streamlit App

## Overview

This application is an **inference-only** demo built on top of the completed university project. It loads pre-trained artifacts exported from `fake_news_modeling.ipynb` and classifies user-pasted news articles as **Real** or **Fake**.

No training, retraining, or dataset access occurs at runtime.

---

## System Architecture

```
User
 ↓
Streamlit Interface (app.py)
 ↓
Text Cleaning (preprocessing.py)
 ↓
TF-IDF Vectorizer (tfidf_vectorizer.joblib)
 ↓
Linear SVM Model (linear_svm_model.joblib)
 ↓
Prediction (Real / Fake)
```

---

## Component Descriptions

### 1. User

The end user pastes a news article (title + body) into the Streamlit text area and clicks **Predict**.

**Input:** Raw free-text article  
**Output:** Classification label and decision function score

---

### 2. Streamlit Interface (`app.py`)

Provides the web UI for the final project demonstration.

**Responsibilities:**
- Load and cache model artifacts with `@st.cache_resource`
- Collect user input
- Orchestrate preprocessing → vectorization → prediction
- Display results, project metrics, pipeline diagram, and limitations

**Technologies:** Streamlit

---

### 3. Text Cleaning (`preprocessing.py`)

Applies the **same** cleaning logic used in `fake_news_eda_improved.ipynb`:

| Step | Operation |
|------|-----------|
| 1 | Convert to lowercase |
| 2 | Remove URLs (`http://`, `www.`) |
| 3 | Keep English letters and spaces only |
| 4 | Normalize whitespace |

**Input:** Raw article text  
**Output:** `clean_text` string ready for TF-IDF

Consistency between training and inference is critical — any deviation would degrade prediction quality.

---

### 4. TF-IDF Vectorizer (`models/tfidf_vectorizer.joblib`)

Serialized `TfidfVectorizer` fitted **only on the training set** in the notebook (Phase 3).

**Configuration (from notebook):**
- `max_features=10_000`
- `ngram_range=(1, 2)`
- `min_df=5`, `max_df=0.90`
- `stop_words="english"`
- `sublinear_tf=True`

**Input:** Cleaned text  
**Output:** Sparse numerical feature vector (1 × 10,000)

At runtime the vectorizer performs **`transform` only** — never `fit`.

---

### 5. Linear SVM Model (`models/linear_svm_model.joblib`)

Serialized `LinearSVC` — the best-performing model from project evaluation.

**Training configuration:**
- `LinearSVC(max_iter=2000, random_state=42)`
- Trained on TF-IDF features of the 80% training split

**Test set performance (unchanged from notebook):**

| Metric | Value |
|--------|-------|
| Accuracy | 99.80% |
| Precision | 99.81% |
| Recall | 99.81% |
| F1 Score | 99.81% |

**Input:** TF-IDF feature vector  
**Output:** Binary label — `0` = Real News, `1` = Fake News

The app also displays `decision_function()` — a signed distance from the decision boundary. **This is not a probability.**

---

### 6. Prediction

The final label is mapped to user-facing output:

| Label | Display |
|-------|---------|
| `0` | ✅ Real News |
| `1` | ❌ Fake News |

---

## Artifact Provenance

Artifacts are created in notebook section **(9b) Export Model Artifacts**:

```python
joblib.dump(models["Linear SVM"], "models/linear_svm_model.joblib")
joblib.dump(tfidf_train, "models/tfidf_vectorizer.joblib")
```

These are the **exact objects** evaluated in Phase 3 — not retrained copies.

---

## File Structure

```
fake_news_modelingFInal/
├── app.py                          # Streamlit application
├── preprocessing.py                # Shared text cleaning
├── requirements.txt                # Python dependencies
├── README_APP.md                   # App documentation
├── ARCHITECTURE.md                 # This file
├── fake_news_modeling.ipynb        # Training + export notebook
├── fake_news_eda_preprocessed.csv  # Preprocessed dataset
└── models/
    ├── linear_svm_model.joblib     # Exported Linear SVM
    └── tfidf_vectorizer.joblib     # Exported TF-IDF vectorizer
```

---

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| No retraining | App loads joblib artifacts only |
| Reproducibility | Same cleaning + same vectorizer + same model as notebook |
| Simplicity | Single-page Streamlit UI, no external services |
| Transparency | Displays metrics, pipeline, and limitation disclaimer |
