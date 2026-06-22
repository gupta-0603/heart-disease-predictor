"""pages/3_About.py — model & dataset documentation."""

import streamlit as st
from utils import inject_global_css, sidebar_brand, COLORS

st.set_page_config(page_title="About · CardioScope", page_icon="ℹ️", layout="wide")
inject_global_css()
sidebar_brand()

st.markdown('<div class="eyebrow">Documentation</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title" style="font-size:2.1rem;">About this model</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">What CardioScope is, what it isn\'t, and how the '
    'underlying model was built.</p>',
    unsafe_allow_html=True,
)

st.write("")
c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="section-label" style="margin-top:0;">The model</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <p style="margin-top:0;">
                CardioScope uses a <b>K-Nearest Neighbors (KNN)</b> classifier
                trained on eleven clinical features. For a new patient, the
                model looks at the closest matching patients in the training
                data (by scaled distance) and predicts based on how many of
                those neighbors had heart disease.
            </p>
            <p>
                Inputs are standardized with a saved <code>StandardScaler</code>
                before scoring, and categorical fields (sex, chest pain type,
                resting ECG, exercise angina, ST slope) are one-hot encoded to
                match the columns the model was trained on.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Input features</div>', unsafe_allow_html=True)
    features = [
        ("Age", "Patient age in years"),
        ("Sex", "M / F"),
        ("ChestPainType", "ATA, NAP, TA, ASY"),
        ("RestingBP", "Resting blood pressure (mm Hg)"),
        ("Cholesterol", "Serum cholesterol (mg/dL)"),
        ("FastingBS", "1 if fasting blood sugar > 120 mg/dL"),
        ("RestingECG", "Normal, ST, LVH"),
        ("MaxHR", "Maximum heart rate achieved"),
        ("ExerciseAngina", "Y / N"),
        ("Oldpeak", "ST depression induced by exercise"),
        ("ST_Slope", "Up, Flat, Down"),
    ]
    rows = "".join(
        f"""<tr style="border-bottom:1px solid {COLORS['border']};">
                <td style="padding:0.5rem 0.8rem; font-weight:600;">{f}</td>
                <td style="padding:0.5rem 0.8rem; color:{COLORS['text_dim']};">{d}</td>
            </tr>"""
        for f, d in features
    )
    st.markdown(
        f"""<div class="card" style="padding:0.5rem 0;">
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">{rows}</table>
            </div>""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown('<div class="section-label" style="margin-top:0;">Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <ul style="margin:0; padding-left:1.1rem; color:{COLORS['text_dim']}; line-height:1.7;">
                <li>This is a <b style="color:{COLORS['text']};">screening demo</b>,
                    not a certified medical device.</li>
                <li>KNN predictions reflect patterns in the training dataset and
                    may not generalize to every population.</li>
                <li>Results should never replace a consultation with a
                    qualified healthcare professional.</li>
                <li>The model has no awareness of family history, medication,
                    or other conditions not in the eleven input fields.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Files this app expects</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="card">
            <p style="margin-top:0; color:{COLORS['text_dim']};">
                Place these three files in the same folder as
                <code>Home.py</code>:
            </p>
            <ul style="padding-left:1.1rem; line-height:1.8;">
                <li><code>knn_heart_model.pkl</code> — trained KNN classifier</li>
                <li><code>heart_scaler.pkl</code> — fitted StandardScaler</li>
                <li><code>heart_columns.pkl</code> — list of expected columns,
                    in training order</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Tech stack</div>', unsafe_allow_html=True)
    tags = ["Streamlit", "scikit-learn", "Plotly", "pandas", "joblib"]
    tag_html = "".join(
        f"""<span style="display:inline-block; background:{COLORS['panel']};
                border:1px solid {COLORS['border']}; border-radius:8px;
                padding:0.3rem 0.8rem; margin:0 0.4rem 0.4rem 0; font-size:0.85rem;">
                {t}</span>"""
        for t in tags
    )
    st.markdown(f'<div class="card">{tag_html}</div>', unsafe_allow_html=True)

st.write("")
st.markdown(
    f"""
    <div style="text-align:center; color:{COLORS['text_dim']}; font-size:0.82rem; padding-top:1rem;
                border-top:1px solid {COLORS['border']};">
        Built with Streamlit · Educational use only
    </div>
    """,
    unsafe_allow_html=True,
)
