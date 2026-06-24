# Fake News Detection — Streamlit Application

**Course:** Advanced Tools in Data Science  
**Project:** Fake News Detection Using Machine Learning

---

## 1. Project Overview

This project builds a machine learning pipeline to classify news articles as **Real** or **Fake** based on text content alone. The workflow includes:

- Exploratory Data Analysis (EDA)
- Text preprocessing
- TF-IDF feature representation
- Training and comparison of four classifiers
- Evaluation on a held-out test set

**Best model:** Linear SVM — F1 Score **0.9981** on 8,978 test articles.

This Streamlit app is a lightweight **demonstration interface** for the final project presentation.

---

## 2. Application Purpose

The app allows anyone to:

1. Paste a news article
2. Receive an instant prediction: **Real News** or **Fake News**

The application performs **inference only**. It does not train models, modify data, or connect to external services.

---

## 3. Architecture

```
Article → Text Cleaning → TF-IDF → Linear SVM → Prediction
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed component breakdown.

---

## 4. Technologies Used

| Technology | Role |
|------------|------|
| **Python** | Core language |
| **scikit-learn** | TF-IDF vectorizer, Linear SVM |
| **joblib** | Model serialization and loading |
| **Streamlit** | Web demo interface |
| **pandas / numpy** | Data handling (notebook phase) |

---

## 5. Installation Instructions

### Prerequisites

- Python 3.10+
- Model artifacts exported from the notebook (see step 1 below)

### Step 1 — Export model artifacts (one-time)

Open `fake_news_modeling.ipynb` and run all cells through:

1. **Phase 3** — model training (sections 7–9)
2. **Section (9b)** — Export Model Artifacts

This creates:

```
models/linear_svm_model.joblib
models/tfidf_vectorizer.joblib
```

> **Important:** The export cell saves the exact in-memory objects from Phase 3. It does **not** retrain.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 6. How to Run

From the project directory:

```bash
streamlit run app.py
```

The app opens in your browser (default: `http://localhost:8501`).

### Usage

1. Paste a news article into the text area
2. Click **Predict**
3. View the result:
   - ✅ **Real News** — label 0
   - ❌ **Fake News** — label 1
4. Review the **decision function score** (signed margin — not a probability)

---

## 7. Limitations

- **No fact-checking** — the model detects linguistic patterns similar to the training data, not factual truth
- **US political bias** — trained on American political news (Kaggle dataset)
- **English only** — preprocessing keeps `[a-z]` characters only
- **TF-IDF limitations** — no semantic understanding, sarcasm, or context
- **Not production-ready** — no external validation, cross-validation, or monitoring

> *"This model does not verify facts. It predicts whether an article resembles the linguistic patterns of real or fake news articles learned from the training dataset."*

---

## 8. Future Improvements

- Replace TF-IDF with transformer models (BERT / DistilBERT)
- Add K-Fold cross-validation and hyperparameter tuning
- Manual error analysis on misclassified test articles
- Support for Hebrew and multilingual news
- Explainability with SHAP / LIME
- Deploy as a hosted demo with model versioning

---

## 9. Demo Articles

Use these examples to test the app during your presentation.

### Example A — Likely classified as **Real News**

Paste this into the app:

```
WASHINGTON (Reuters) - The Federal Reserve held interest rates steady on Wednesday
and said it would continue to monitor economic data before deciding on future policy
adjustments. Chair Jerome Powell told reporters that inflation has moderated but
remains above the central bank's two percent target. Markets reacted calmly as
investors had largely priced in the decision. Analysts at major banks said the
statement was consistent with prior guidance and noted that officials emphasized
a data-dependent approach going forward.
```

**Why Real:** Reuters-style dateline, neutral factual tone, policy reporting vocabulary (`Federal Reserve`, `inflation`, `analysts`).

---

### Example B — Likely classified as **Fake News**

Paste this into the app:

```
BREAKING: Shocking leaked emails prove the entire election was a massive fraud
orchestrated by the deep state! Mainstream media will NEVER report this because
they are all corrupt liars protecting the elite. Share this before they delete
it! This is the biggest scandal in history and the dishonest fake news media is
trying to cover it up. You won't believe what they found in the secret documents!!!
```

**Why Fake:** Sensational language (`BREAKING`, `Shocking`, `deep state`), emotional appeals, conspiracy framing, patterns common in fake political articles from the training set.

---

## 10. Project Results (Reference)

| Metric | Linear SVM |
|--------|------------|
| Accuracy | 99.80% |
| Precision | 99.81% |
| Recall | 99.81% |
| F1 Score | 99.81% |

Test set errors: **18 / 8,978** articles (9 false positives, 9 false negatives).

---

## Team

- אחמד אלדדה — 326087244
- אבראהים אבו קוש — 326501905
- עבד אלרחמן אלנבארי — 214042517
