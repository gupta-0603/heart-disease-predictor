"""
Home.py — entry point for the multipage app.
Run with:  streamlit run Home.py
"""

import streamlit as st
from utils import inject_global_css, sidebar_brand, COLORS, load_artifacts

st.set_page_config(
    page_title="CardioScope · Heart Stroke Risk",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
sidebar_brand()

_, _, _, artifacts_ok = load_artifacts()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.markdown('<div class="eyebrow">KNN-Powered Screening Tool</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-title">Know your heart<br>risk in 60 seconds.</p>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <p class="hero-sub">
        CardioScope screens for elevated heart disease risk using a
        K-Nearest Neighbors model trained on eleven clinical indicators —
        from resting blood pressure to ST slope. Built as a learning aid,
        not a diagnosis.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Start Risk Check  →", use_container_width=True):
            st.switch_page("pages/1_Predict.py")
    with c2:
        if st.button("View Dataset Insights", use_container_width=True):
            st.switch_page("pages/2_Insights.py")

    if not artifacts_ok:
        st.warning(
            "Model files (`knn_heart_model.pkl`, `heart_scaler.pkl`, "
            "`heart_columns.pkl`) weren't found next to `Home.py`. The "
            "Predict page will show this same notice until they're added.",
            icon="⚠️",
        )

with right:
    st.markdown(
        f"""
        <div class="card" style="margin-top:1rem;">
            <div class="section-label" style="margin-top:0;">At a glance</div>
            <div style="display:flex; flex-direction:column; gap:0.7rem;">
                <div class="stat-tile">
                    <div class="stat-num">11</div>
                    <div class="stat-label">Clinical input features</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-num">KNN</div>
                    <div class="stat-label">Model: K-Nearest Neighbors</div>
                </div>
                <div class="stat-tile">
                    <div class="stat-num">3</div>
                    <div class="stat-label">Pages — Predict, Insights, About</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)

s1, s2, s3 = st.columns(3, gap="medium")
steps = [
    ("01", "Enter your details", "Age, blood pressure, cholesterol, ECG results, and a few more — same fields a cardiologist would ask for."),
    ("02", "Model scores your risk", "Your inputs are scaled and compared against the nearest patient profiles the model learned from."),
    ("03", "See a clear result", "A risk gauge and plain-language summary — plus where to go for dataset context."),
]
for col, (num, title, desc) in zip([s1, s2, s3], steps):
    with col:
        st.markdown(
            f"""
            <div class="card" style="min-height:170px;">
                <div style="color:{COLORS['gold']}; font-family:'Space Grotesk',sans-serif;
                            font-weight:700; font-size:1.3rem;">{num}</div>
                <div style="font-weight:600; margin:0.3rem 0 0.4rem 0;">{title}</div>
                <div style="color:{COLORS['text_dim']}; font-size:0.9rem; line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
st.markdown(
    f"""
    <div style="text-align:center; color:{COLORS['text_dim']}; font-size:0.82rem; padding-top:1rem;
                border-top:1px solid {COLORS['border']};">
        CardioScope is an educational project, not a medical device. It does not diagnose
        or replace consultation with a healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)
