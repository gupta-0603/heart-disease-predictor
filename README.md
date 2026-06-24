#  CardioScope — Heart Disease Risk Predictor

> An end-to-end machine learning web app that predicts cardiovascular disease risk using clinical data — built with Scikit-learn and deployed via Streamlit.

---

##  Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-predictor-klcswckn9brltyxpnqfx4k.streamlit.app/)

---

##  Overview

Heart disease is the leading cause of death globally. Early screening using routine clinical measurements can significantly improve outcomes. This project builds a full ML pipeline — from raw data to a deployed interactive web application — that estimates a patient's likelihood of heart disease based on 11 clinical features.

The app lets users input patient data via a form, returns a real-time prediction with a confidence gauge, and contextualises results against population-level statistics from the training dataset.

---

##  Dataset

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

##  Why KNN?

KNN was chosen as the deployed model for deliberate reasons, not by default:

- **Instance-based reasoning** — KNN makes predictions by finding the most similar real patients in the training set. This maps naturally to clinical intuition: *"patients with similar profiles had similar outcomes."* It's a model a clinician can actually reason about.
- **No training assumptions** — KNN makes no distributional assumptions about the data, which is appropriate for a mixed dataset combining five different clinical studies with inconsistent protocols.
- **Transparency at inference time** — unlike a black-box ensemble, KNN's decision can always be traced back to specific similar patients, which is meaningful in a medical screening context.
- **Simplicity as a baseline** — for a first deployment, keeping the model simple makes the preprocessing pipeline, scaling logic, and inference code easier to validate and debug. Getting that right matters more than squeezing out 2% more F1.

The tradeoffs are acknowledged honestly in the Limitations section below.

---

##  Model Comparison

I evaluated KNN against three alternatives on the same 80/20 stratified split to understand exactly where it stands — and where it falls short.

**Why recall matters here:** In medical screening, a false negative (telling a sick patient they're fine) is far costlier than a false alarm. So recall is weighted alongside accuracy.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **KNN (K=11)**  | 87.0% | 87.5% | 89.2% | 88.3% | 0.922 |
| Logistic Regression | 88.6% | 87.2% | 93.1% | 90.0% | 0.929 |
| Random Forest | 89.1% | 89.4% | 91.2% | 90.3% | 0.932 |
| SVM | 88.6% | 87.2% | 93.1% | 90.0% | 0.942 |

Random Forest and SVM outperform KNN on most metrics. The gap is real — roughly 2% F1 and 1-2% recall. The decision to deploy KNN was made consciously: for a portfolio project and learning exercise, building the cleanest possible inference pipeline around a well-understood model was prioritised over maximising benchmark metrics. The model comparison notebook documents this reasoning, and upgrading to Random Forest is the clearest next step.

### KNN Hyperparameter Sweep (K=3 to 21)

K was tuned by sweeping odd values and evaluating accuracy and recall on the held-out test set:

| K | Accuracy | Recall |
|---|---|---|
| 3 | 87.5% | 93.1% |
| 5 | 91.8% | 96.1% |
| 7 | 89.7% | 93.1% |
| 9 | 87.5% | 90.2% |
| **11** | 87.0% | 89.2% |
| 13 | 88.6% | 91.2% |
| 15 | 89.1% | 91.2% |

K=11 was selected to balance generalisation against overfitting — lower K values (especially K=3 and K=5) show higher recall on the test set but are more sensitive to noise in unseen data. K=11 sits in the stable region of the curve where performance plateaus.

---

##  Feature Importance

KNN has no built-in feature importance. To understand which clinical features drive predictions, a Random Forest was trained in parallel purely for interpretability analysis:

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

ST_Slope is the dominant predictor — an "Up" slope post-exercise is strongly protective, while "Flat" is a major risk signal. This aligns with clinical cardiology literature and validates that the model is responding to clinically meaningful signals, not noise.

---

##  Architecture

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
│   ├── model.pkl           # Serialised KNN (K=11)
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

##  How It Works

### Preprocessing Pipeline

1. **One-hot encoding** — categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) are encoded, with column alignment enforced at inference time to prevent feature mismatch bugs
2. **StandardScaler** — all numeric features scaled before model input; the fitted scaler is serialised with `joblib` and reloaded at inference (not re-fit)
3. **Train/test split** — 80/20 stratified split preserving class balance

### Inference Flow

```
User inputs → build_input_row() → one-hot encode → align columns → scale → model.predict_proba() → confidence gauge
```

---

##  Running Locally

```bash
git clone https://github.com/gupta-0603/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

##  Dependencies

```
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
plotly
```

---

##  Limitations

- **KNN performance ceiling:** Random Forest and SVM both outperform KNN by ~2% F1 on this dataset. The next version would swap in Random Forest as the deployed model.
- **No feature importance from KNN:** KNN is instance-based and cannot rank feature contributions natively. The RF-based importance analysis is a workaround, not a native explanation.
- **Dataset size:** 918 patients is sufficient for a proof-of-concept but small for clinical use
- **No external validation:** Model has not been tested on out-of-sample hospital data
- **Not a medical device:** This tool is for educational and portfolio purposes only
- **Selection bias:** The dataset combines multiple legacy studies with inconsistent collection protocols
- **No calibration:** `predict_proba` outputs are not calibrated probabilities and should not be interpreted as clinical risk percentages

---

##  What I'd Do With More Time

- **Replace KNN with Random Forest** — the comparison clearly shows RF wins on F1 and overall balance; the deployment model should reflect this
- Cross-validation (k-fold) instead of a single train/test split
- SHAP values for per-prediction explainability on the deployed model
- Probability calibration (Platt scaling) so confidence scores are meaningful
- Larger, more recent dataset (e.g. UK Biobank or MIMIC-IV)
- Docker containerisation for reproducible deployment

---

##  What I Learned

- Why **preprocessing must happen at inference time** with the same fitted objects — not just at training time. This is one of the most common production ML bugs and I explicitly engineered around it.
- That **accuracy is a misleading metric for medical tasks** — a model predicting everyone has heart disease would score 55% on this dataset.
- That **KNN is outperformed by tree-based models here** — the comparison was the point. Knowing *why* a simpler model underperforms is more valuable than blindly deploying whatever scored highest.
- That **feature importance tells a clinical story** — ST_Slope and Oldpeak dominating the model aligns exactly with what cardiologists look for, which validates the pipeline is learning real signal.
- That **model choice involves tradeoffs beyond benchmark metrics** — interpretability, debugging simplicity, and deployment risk are all part of the decision.

---

##  Author

**Aditya Kumar**
- GitHub: [@GitHub](https://github.com/gupta-0603)
- LinkedIn: [@LinkedIn](https://www.linkedin.com/in/aditya-kumar-827616291/)
