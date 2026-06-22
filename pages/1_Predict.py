"""pages/1_Predict.py — the risk assessment form."""

import streamlit as st
from utils import (
    inject_global_css, sidebar_brand, COLORS,
    load_artifacts, build_input_row, risk_gauge,
)

st.set_page_config(page_title="Predict · CardioScope", page_icon="🫀", layout="wide")
inject_global_css()
sidebar_brand()

model, scaler, expected_columns, artifacts_ok = load_artifacts()

st.markdown('<div class="eyebrow">Step 1 of 1</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title" style="font-size:2.1rem;">Heart Risk Assessment</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Fill in the clinical details below. Hover the ⓘ on any '
    'field if you\'re unsure what it means.</p>',
    unsafe_allow_html=True,
)

if not artifacts_ok:
    st.warning(
        "Model files weren't found, so this page can't make live predictions "
        "right now. Add `knn_heart_model.pkl`, `heart_scaler.pkl`, and "
        "`heart_columns.pkl` next to `Home.py` to enable it. The form below "
        "still works so you can see the layout.",
        icon="⚠️",
    )

st.write("")
form_col, gauge_col = st.columns([1.4, 1], gap="large")

with form_col:
    st.markdown('<div class="section-label" style="margin-top:0;">Demographics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 100, 40, help="Patient's age in years.")
    with c2:
        sex = st.selectbox("Sex", ["M", "F"], help="Biological sex as recorded in the dataset.")

    st.markdown('<div class="section-label">Vitals & Labs</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", 80, 200, 120,
            help="Blood pressure measured at rest, in mm Hg.",
        )
    with c4:
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 100, 600, 200,
            help="Serum cholesterol level in mg/dL.",
        )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="1 if fasting blood sugar exceeds 120 mg/dL, otherwise 0.",
    )

    st.markdown('<div class="section-label">Cardiac Findings</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        chest_pain = st.selectbox(
            "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
            help="ATA = Atypical Angina · NAP = Non-Anginal Pain · TA = Typical Angina · ASY = Asymptomatic",
        )
    with c6:
        resting_ecg = st.selectbox(
            "Resting ECG", ["Normal", "ST", "LVH"],
            help="ST = ST-T wave abnormality · LVH = Left Ventricular Hypertrophy",
        )
    max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved during exercise.")
    c7, c8 = st.columns(2)
    with c7:
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", ["Y", "N"],
            help="Whether exercise triggers chest pain.",
        )
    with c8:
        st_slope = st.selectbox(
            "ST Slope", ["Up", "Flat", "Down"],
            help="Slope of the peak exercise ST segment.",
        )
    oldpeak = st.slider(
        "Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1,
        help="ST depression induced by exercise relative to rest.",
    )

    st.write("")
    predict_clicked = st.button("Run Prediction  →", use_container_width=True, disabled=not artifacts_ok)

with gauge_col:
    st.markdown('<div class="section-label" style="margin-top:0;">Risk Output</div>', unsafe_allow_html=True)

    if "last_proba" not in st.session_state:
        st.session_state.last_proba = None
        st.session_state.last_pred = None
        st.session_state.last_inputs = None

    if predict_clicked and artifacts_ok:
        input_df = build_input_row(
            age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
            resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,
            expected_columns,
        )
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_input)[0][1]
        else:
            proba = 1.0 if prediction == 1 else 0.0

        st.session_state.last_proba = proba
        st.session_state.last_pred = prediction
        # Save the raw, human-readable inputs (not the encoded row) so the
        # Insights page can compare this person against the population.
        st.session_state.last_inputs = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope,
        }

    if st.session_state.last_proba is not None:
        st.plotly_chart(
            risk_gauge(st.session_state.last_proba),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if st.session_state.last_pred == 1:
            st.markdown(
                f"""
                <div class="result-risk">
                    <div class="result-title" style="color:{COLORS['risk']};">⚠ Elevated Risk</div>
                    <div style="color:{COLORS['text_dim']};">
                        The model flags this profile as higher risk for heart disease.
                        This is a screening signal, not a diagnosis — please discuss
                        these numbers with a doctor.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-safe">
                    <div class="result-title" style="color:{COLORS['safe']};">✓ Lower Risk</div>
                    <div style="color:{COLORS['text_dim']};">
                        The model doesn't flag elevated risk for this profile.
                        Keep up with regular checkups regardless.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("")
        if st.button("See how you compare →", use_container_width=True):
            st.switch_page("pages/2_Insights.py")
    else:
        st.markdown(
            f"""
            <div class="card" style="text-align:center; color:{COLORS['text_dim']};
                        padding:3rem 1.5rem;">
                Fill in the form and click <b style="color:{COLORS['text']};">Run Prediction</b>
                to see your risk gauge here.
            </div>
            """,
            unsafe_allow_html=True,
        )
