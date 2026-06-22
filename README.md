# 🫀 Heart Disease Prediction App

> An end-to-end machine learning web app that predicts cardiovascular disease risk using clinical data — built with Scikit-learn and deployed via Streamlit.

---

## 🔗 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-predictor-klcswckn9brltyxpnqfx4k.streamlit.app/)

---

## 📌 Overview

Heart disease is the leading cause of death globally. Early screening using routine clinical measurements can significantly improve outcomes. This project builds a full ML pipeline — from raw data to a deployed interactive web application — that estimates a patient's likelihood of heart disease based on 11 clinical features.

The app lets users input patient data via a form, returns a real-time prediction with a confidence gauge, and contextualises results against population-level statistics from the training dataset.

---

## 📊 Dataset

**Source:** [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

| Property | Value |
|---|---|
| Rows | 918 patients |
| Features | 11 clinical inputs |
| Target | Heart Disease (binary: 0 / 1) |
| Class balance | 55.3% positive (heart disease) |
| Mean age | 53.5 years |
| Mean cholesterol | 198.8 mg/dL |

The dataset combines five independent heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog) — making it the largest publicly available heart disease dataset for research.

---

## 🧪 Model Comparison

Rather than picking KNN arbitrarily, I trained and evaluated four models on the same 80/20 stratified train-test split.

**Why recall matters here:** In medical screening, a false negative (telling a sick patient they're fine) is far costlier than a false alarm. So recall is the primary metric, not accuracy.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN (K=11) | 87.0% | 87.5% | 89.2% | 88.3% | 0.922 |
| Logistic Regression | 88.6% | 87.2% | 93.1% | 90.0% | 0.929 |
| **Random Forest** ✅ | **89.1%** | **89.4%** | 91.2% | **90.3%** | 0.932 |
| SVM | 88.6% | 87.2% | **93.1%** | 90.0% | **0.942** |

**Decision:** Random Forest was selected for deployment — it achieves the best F1 and accuracy, with strong recall and ROC-AUC. SVM edges it on recall and ROC-AUC, but Random Forest offers better interpretability via feature importances, which matters for a medical context.

### KNN Hyperparameter Sweep (K=3 to 21)

| K | Accuracy | Recall |
|---|---|---|
| 3 | 87.5% | 93.1% |
| **5** | **91.8%** | **96.1%** |
| 7 | 89.7% | 93.1% |
| 9 | 87.5% | 90.2% |
| 11 | 87.0% | 89.2% |

K=5 is significantly better than K=11 — worth noting if KNN simplicity is preferred.

---

## 🔍 Feature Importance

Using permutation importance from Random Forest, the most predictive clinical features are:

| Rank | Feature | Importance |
|---|---|---|
| 1 | ST_Slope (Up) | 0.1590 |
| 2 | Oldpeak (ST depression) | 0.1009 |
| 3 | Cholesterol | 0.0932 |
| 4 | ST_Slope (Flat) | 0.0926 |
| 5 | Max Heart Rate | 0.0923 |
| 6 | Chest Pain Type (Asymptomatic) | 0.0862 |
| 7 | Age | 0.0715 |
| 8 | Resting BP | 0.0662 |

ST_Slope is the dominant predictor — an "Up" slope post-exercise is strongly protective, while "Flat" is a major risk signal. This aligns with clinical cardiology literature.

---

## 🏗️ Architecture

```
heart-disease-predictor/
│
├── app.py                  # Main Streamlit entry point
├── pages/
│   ├── 1_Predict.py        # Patient input form + prediction
│   └── 2_About.py          # Model performance + methodology
│
├── model/
│   ├── train.py            # Training script
│   ├── model.pkl           # Serialised Random Forest
│   └── scaler.pkl          # Fitted StandardScaler
│
├── utils.py                # build_input_row(), percentile helpers
├── notebooks/
│   └── model_comparison.ipynb   # Full EDA + model comparison
│
├── tests/
│   └── test_utils.py       # pytest unit tests
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### Preprocessing Pipeline

1. **One-hot encoding** — categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) are encoded, with column alignment enforced at inference time to prevent feature mismatch bugs
2. **StandardScaler** — all numeric features scaled before model input; the fitted scaler is serialised with `joblib` and loaded at inference
3. **Train/test split** — 80/20 stratified split (preserves class balance)

### Inference Flow

```
User inputs → build_input_row() → one-hot encode → align columns → scale → model.predict_proba() → confidence gauge
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/gupta-0603/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit==1.x
scikit-learn==1.x
pandas==2.x
numpy==1.x
joblib==1.x
matplotlib==3.x
plotly==5.x
```

---

## ⚠️ Limitations

- **Dataset size:** 918 patients is sufficient for a proof-of-concept but small for clinical use
- **No external validation:** Model has not been tested on out-of-sample hospital data
- **Not a medical device:** This tool is for educational and portfolio purposes only — it should not be used to make real clinical decisions
- **Selection bias:** The dataset combines multiple legacy studies with inconsistent collection protocols
- **No calibration:** `predict_proba` outputs are not calibrated probabilities; they should not be interpreted as clinical risk percentages

---

## 🔮 What I'd Do With More Time

- Cross-validation (k-fold) instead of a single train/test split for more reliable evaluation
- Probability calibration (Platt scaling or isotonic regression) so confidence scores are meaningful
- SHAP values for per-prediction explainability
- Larger, more recent dataset (e.g. UK Biobank or MIMIC-IV)
- Docker containerisation for reproducible deployment

---

## 🧠 What I Learned

- Why **preprocessing must happen at inference time** with the same fitted objects — not just at training time. This is one of the most common production ML bugs.
- That **accuracy is a misleading metric for medical tasks** — a model predicting everyone has heart disease would score 55% accuracy on this dataset.
- That **KNN is outperformed by tree-based models here**, even at its optimal K=5. This is typical when feature scales differ and categorical encodings dominate.
- That **feature importance tells a clinical story** — ST_Slope and Oldpeak dominating the model aligns exactly with what cardiologists use in practice.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**[Aditya Kumar]**
  [@GitHub](https://github.com/gupta-0603)
  [@LinkedIn](https://www.linkedin.com/in/aditya-kumar-827616291/)
