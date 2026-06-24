# Fake News Detection

**Course:** Advanced Tools in Data Science  
**Project:** Classifying news articles as Real or Fake using machine learning

| Team member | ID |
|-------------|-----|
| Ahmad Eldada | 326087244 |
| Ibrahim Abu Kosh | 326501905 |
| Abdulrahman Al-Nabari | 214042517 |

## Quick start — run the app

Model artifacts are included in `models/`. From the project root:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, paste an article, and click **Predict**.

## Project structure

```
├── app.py                          # Streamlit demo (inference only)
├── preprocessing.py                # Text cleaning (matches EDA logic)
├── requirements.txt
├── export_artifacts.py             # Re-export joblib files if notebook kernel was reset
├── fake_news_modeling.ipynb        # Training, evaluation, model export
├── fake_news_eda_improved (1).ipynb # EDA and preprocessing
├── models/
│   ├── linear_svm_model.joblib     # Best model (Linear SVM)
│   └── tfidf_vectorizer.joblib
├── README_APP.md                   # Full app documentation and demo examples
├── ARCHITECTURE.md                 # Pipeline architecture
└── FakeNews_Installation_Guide_RTL.pdf  # Hebrew installation guide
```

## Results

**Best model:** Linear SVM — Accuracy **99.80%**, F1 **0.9981** on 8,978 test articles.

## Reproducing the pipeline

1. Run `fake_news_eda_improved (1).ipynb` to produce `fake_news_eda_preprocessed.csv` (not in repo — file exceeds GitHub size limit).
2. Run `fake_news_modeling.ipynb` through section **(9b) Export Model Artifacts**, or run `python export_artifacts.py` as a fallback.

## Documentation

- [README_APP.md](README_APP.md) — installation, usage, limitations, demo articles
- [ARCHITECTURE.md](ARCHITECTURE.md) — component breakdown
- [FakeNews_Installation_Guide_RTL.pdf](FakeNews_Installation_Guide_RTL.pdf) — Hebrew setup guide

## Limitations

The model detects linguistic patterns from the training data; it does not verify facts. English only; trained on US political news.
