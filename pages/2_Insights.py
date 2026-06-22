"""
pages/2_Insights.py — Insights, rebuilt around what a visitor actually has:
no dataset of their own. Three sections:
  1. Population snapshot   — baked-in stats from the training data
  2. How you compare       — uses the last Predict-page result, if any
  3. What each number means — plain-language feature explainer
"""

import streamlit as st
import plotly.graph_objects as go
from utils import (
    inject_global_css, sidebar_brand, COLORS,
    POPULATION_STATS, percentile_from_normal,
)

st.set_page_config(page_title="Insights · CardioScope", page_icon="📊", layout="wide")
inject_global_css()
sidebar_brand()

st.markdown('<div class="eyebrow">Insights</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-title" style="font-size:2.1rem;">What the data says</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">No file to upload — this page draws on the same '
    'dataset the model was trained on, and on your last risk check if you\'ve '
    'run one.</p>',
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter"),
    margin=dict(l=10, r=10, t=10, b=10),
)

# ---------------------------------------------------------------------------
# 1. Population snapshot
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label" style="margin-top:0;">Population snapshot</div>', unsafe_allow_html=True)

n = POPULATION_STATS["n"]
pos_rate = POPULATION_STATS["positive_rate"]

k1, k2, k3, k4 = st.columns(4)
tiles = [
    ("Training records", f"{n:,}"),
    ("Flagged high-risk", f"{pos_rate*100:.0f}%"),
    ("Avg. age", f"{POPULATION_STATS['numeric']['Age'][0]:.0f} yrs"),
    ("Avg. cholesterol", f"{POPULATION_STATS['numeric']['Cholesterol'][0]:.0f} mg/dL"),
]
for col, (label, val) in zip([k1, k2, k3, k4], tiles):
    with col:
        st.markdown(
            f"""<div class="stat-tile"><div class="stat-num">{val}</div>
            <div class="stat-label">{label}</div></div>""",
            unsafe_allow_html=True,
        )

st.write("")
chart_a, chart_b = st.columns(2, gap="large")

with chart_a:
    st.markdown('<div class="section-label" style="margin-top:0;">Chest pain type, across all patients</div>', unsafe_allow_html=True)
    cp = POPULATION_STATS["categorical"]["ChestPainType"]
    fig = go.Figure(go.Bar(
        x=list(cp.keys()), y=[v * 100 for v in cp.values()],
        marker_color=[COLORS["safe"], COLORS["gold"], COLORS["risk"], "#5B8BB0"],
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280, yaxis_title="% of patients")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<p style="color:{COLORS["text_dim"]}; font-size:0.85rem;">'
        "ASY (asymptomatic) is the most common chest-pain category in the "
        "training data — and also the category most associated with a "
        "positive diagnosis, which is why it's worth tracking even without pain.</p>",
        unsafe_allow_html=True,
    )

with chart_b:
    st.markdown('<div class="section-label" style="margin-top:0;">ST slope, across all patients</div>', unsafe_allow_html=True)
    slope = POPULATION_STATS["categorical"]["ST_Slope"]
    fig = go.Figure(go.Pie(
        labels=list(slope.keys()), values=list(slope.values()), hole=0.55,
        marker_colors=[COLORS["gold"], COLORS["risk"], COLORS["safe"]],
        textfont=dict(color=COLORS["bg"]),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=True,
                       legend=dict(font=dict(color=COLORS["text"])))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<p style="color:{COLORS["text_dim"]}; font-size:0.85rem;">'
        "A flat ST slope shows up in roughly half of patients and is one of "
        "the strongest single predictors the model relies on.</p>",
        unsafe_allow_html=True,
    )

st.markdown(
    f'<p style="color:{COLORS["text_dim"]}; font-size:0.78rem; padding-top:0.4rem;">'
    "Figures summarize the dataset the KNN model was trained on. Swap in your "
    "own numbers in <code>utils.py → POPULATION_STATS</code> if you're using a "
    "different dataset.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 2. Personal comparison
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">How you compare</div>', unsafe_allow_html=True)

last_inputs = st.session_state.get("last_inputs")

if last_inputs is None:
    st.markdown(
        f"""
        <div class="card" style="text-align:center; padding:2.4rem 1.5rem; color:{COLORS['text_dim']};">
            You haven't run a risk check yet. Head to <b style="color:{COLORS['text']};">Predict</b>
            and click <b style="color:{COLORS['text']};">Run Prediction</b> — come back here
            afterward to see exactly where your numbers sit against the training population.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Go to Predict →"):
        st.switch_page("pages/1_Predict.py")
else:
    cols = st.columns(3, gap="large")
    compare_fields = [
        ("Age", "Age", "yrs"),
        ("Cholesterol", "Cholesterol", "mg/dL"),
        ("RestingBP", "RestingBP", "mm Hg"),
    ]
    numeric_stats = POPULATION_STATS["numeric"]

    for col, (key, stat_key, unit) in zip(cols, compare_fields):
        value = last_inputs.get(key)
        if stat_key in numeric_stats and value is not None:
            mean, std, lo, hi = numeric_stats[stat_key]
            pct = percentile_from_normal(value, mean, std)
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="stat-label">{key}</div>
                        <div class="stat-num" style="font-size:1.6rem;">{value} {unit}</div>
                        <div style="color:{COLORS['text_dim']}; font-size:0.85rem; margin-top:0.3rem;">
                            Higher than <b style="color:{COLORS['text']};">{pct:.0f}%</b> of
                            patients in the training data (avg. {mean:.0f} {unit})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.write("")
    cat_cols = st.columns(3, gap="large")
    cat_fields = [
        ("ChestPainType", "Chest pain type"),
        ("ST_Slope", "ST slope"),
        ("ExerciseAngina", "Exercise angina"),
    ]
    cat_stats = POPULATION_STATS["categorical"]

    for col, (key, label) in zip(cat_cols, cat_fields):
        value = last_inputs.get(key)
        if key in cat_stats and value in cat_stats[key]:
            share = cat_stats[key][value] * 100
            with col:
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="stat-label">{label}</div>
                        <div class="stat-num" style="font-size:1.6rem;">{value}</div>
                        <div style="color:{COLORS['text_dim']}; font-size:0.85rem; margin-top:0.3rem;">
                            <b style="color:{COLORS['text']};">{share:.0f}%</b> of training
                            patients share this value
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    proba = st.session_state.get("last_proba")
    if proba is not None:
        st.markdown(
            f"""
            <p style="color:{COLORS['text_dim']}; font-size:0.85rem; padding-top:0.6rem;">
            Your model score was <b style="color:{COLORS['text']};">{proba*100:.1f}%</b>,
            against a baseline rate of <b style="color:{COLORS['text']};">{pos_rate*100:.0f}%</b>
            positive cases across the whole training set.
            </p>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# 3. What each number means
# ---------------------------------------------------------------------------
st.markdown('<div class="section-label">What each number means</div>', unsafe_allow_html=True)

explainers = [
    ("Age", "Risk climbs steadily with age — arteries stiffen and plaque builds up over decades."),
    ("Resting blood pressure", "Sustained high pressure strains artery walls and the heart muscle itself."),
    ("Cholesterol", "High LDL cholesterol contributes to the plaque buildup behind most heart attacks."),
    ("Fasting blood sugar", "Levels above 120 mg/dL can signal diabetes or prediabetes, both linked to heart disease."),
    ("Resting ECG", "An abnormal resting heart-rhythm reading (ST or LVH) can point to existing heart strain."),
    ("Max heart rate", "A lower achievable max heart rate during exertion can indicate reduced heart capacity."),
    ("Exercise angina", "Chest pain triggered specifically by exertion is a classic sign of restricted blood flow."),
    ("Oldpeak (ST depression)", "A larger ST depression during exercise is one of the strongest single warning signs."),
    ("ST slope", "A flat or downward slope during peak exercise often reflects reduced blood flow to the heart."),
]

ex_cols = st.columns(3, gap="medium")
for i, (term, desc) in enumerate(explainers):
    with ex_cols[i % 3]:
        st.markdown(
            f"""
            <div class="card" style="min-height:130px; margin-bottom:1rem;">
                <div style="font-weight:600; margin-bottom:0.3rem;">{term}</div>
                <div style="color:{COLORS['text_dim']}; font-size:0.85rem; line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
