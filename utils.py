"""
Shared styling, constants, and helper functions for the Heart Stroke
Prediction app. Imported by Home.py and every page in /pages.
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Design tokens — keep every color/spacing decision in one place
# ---------------------------------------------------------------------------
COLORS = {
    "bg":        "#0B1F33",  # deep clinical navy — page background
    "bg_alt":    "#11293F",  # slightly lighter navy — card background
    "panel":     "#16334C",  # input/panel surface
    "border":    "#234058",  # hairline borders on navy
    "paper":     "#F7F9FB",  # paper-white for high-contrast cards
    "text":      "#E8EEF4",  # primary text on navy
    "text_dim":  "#8FA6BC",  # secondary / caption text on navy
    "ink":       "#0B1F33",  # primary text on paper
    "risk":      "#E0566B",  # coral-red — elevated risk
    "risk_dim":  "#3A2230",
    "safe":      "#3FA796",  # sage-teal — low risk
    "safe_dim":  "#1E3A38",
    "gold":      "#D4A24C",  # warm gold accent — highlights / brand mark
}

FONT_IMPORT = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
"""

EXPECTED_COLUMNS_FALLBACK = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_F", "Sex_M",
    "ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_LVH", "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_N", "ExerciseAngina_Y",
    "ST_Slope_Down", "ST_Slope_Flat", "ST_Slope_Up",
]

# ---------------------------------------------------------------------------
# Population reference stats — baked in so Insights works with zero uploads.
# These are summary statistics (mean/std + category rates), not row-level
# data, drawn from the publicly documented profile of the UCI/Kaggle "Heart
# Failure Prediction" dataset commonly used for this exact KNN exercise
# (918 records). If you have your own heart.csv, replace these numbers with
# df.describe() / value_counts() output from your actual training data —
# see the comment block at the bottom of this dict for how.
# ---------------------------------------------------------------------------
POPULATION_STATS = {
    "n": 918,
    "positive_rate": 0.553,  # share of records with HeartDisease == 1
    "numeric": {
        # feature: (mean, std, min, max)
        "Age":         (53.5, 9.4, 28, 77),
        "RestingBP":   (132.4, 18.5, 0, 200),
        "Cholesterol": (198.8, 109.4, 0, 603),
        "MaxHR":       (136.8, 25.5, 60, 202),
        "Oldpeak":     (0.89, 1.07, -2.6, 6.2),
    },
    "categorical": {
        "Sex": {"M": 0.79, "F": 0.21},
        "ChestPainType": {"ASY": 0.54, "NAP": 0.22, "ATA": 0.19, "TA": 0.05},
        "RestingECG": {"Normal": 0.60, "LVH": 0.20, "ST": 0.19},
        "ExerciseAngina": {"N": 0.59, "Y": 0.41},
        "ST_Slope": {"Flat": 0.50, "Up": 0.43, "Down": 0.07},
        "FastingBS": {0: 0.77, 1: 0.23},
    },
}
# To use your real data instead:
#   import pandas as pd; df = pd.read_csv("heart.csv")
#   df[["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"]].agg(["mean","std","min","max"])
#   df["ChestPainType"].value_counts(normalize=True)   # repeat per categorical column
#   (df["HeartDisease"] == 1).mean()                    # positive_rate


def percentile_from_normal(value, mean, std):
    """Approximate percentile rank of `value` within a normal distribution
    defined by (mean, std), using the error function — no scipy dependency."""
    import math
    if std == 0:
        return 50.0
    z = (value - mean) / std
    pct = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return round(pct * 100, 1)


def inject_global_css():
    """Injects the shared visual identity into every page. Call once per page,
    right after st.set_page_config()."""
    st.markdown(
        f"""
        <style>
        {FONT_IMPORT}

        /* ---------- base canvas ---------- */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background: {COLORS['bg']};
            color: {COLORS['text']};
        }}
        section[data-testid="stSidebar"] {{
            background: {COLORS['bg_alt']};
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {COLORS['text']} !important;
        }}

        /* ---------- headings use the display face ---------- */
        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
        }}

        /* ---------- the eyebrow / brand mark ---------- */
        .eyebrow {{
            font-family: 'Space Grotesk', sans-serif;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.72rem;
            color: {COLORS['gold']};
            font-weight: 600;
            margin-bottom: 0.3rem;
        }}

        /* ---------- hero ---------- */
        .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 3rem;
            line-height: 1.05;
            color: {COLORS['text']};
            margin: 0;
        }}
        .hero-sub {{
            color: {COLORS['text_dim']};
            font-size: 1.05rem;
            max-width: 640px;
            margin-top: 0.9rem;
            line-height: 1.55;
        }}

        /* ---------- card surfaces ---------- */
        .card {{
            background: {COLORS['bg_alt']};
            border: 1px solid {COLORS['border']};
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
        }}
        .card-paper {{
            background: {COLORS['paper']};
            color: {COLORS['ink']};
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
        }}

        /* ---------- stat tiles ---------- */
        .stat-tile {{
            background: {COLORS['panel']};
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 1rem 1.2rem;
        }}
        .stat-num {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.9rem;
            font-weight: 700;
            color: {COLORS['gold']};
        }}
        .stat-label {{
            color: {COLORS['text_dim']};
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        /* ---------- result banners ---------- */
        .result-risk {{
            background: linear-gradient(135deg, {COLORS['risk_dim']}, {COLORS['bg_alt']});
            border: 1px solid {COLORS['risk']};
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
        }}
        .result-safe {{
            background: linear-gradient(135deg, {COLORS['safe_dim']}, {COLORS['bg_alt']});
            border: 1px solid {COLORS['safe']};
            border-radius: 14px;
            padding: 1.6rem 1.8rem;
        }}
        .result-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}

        /* ---------- divider with label ---------- */
        .section-label {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {COLORS['text_dim']};
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 0.5rem;
            margin: 1.6rem 0 1rem 0;
        }}

        /* ---------- buttons ---------- */
        .stButton > button {{
            background: {COLORS['gold']};
            color: {COLORS['bg']};
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.6rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(212, 162, 76, 0.25);
            color: {COLORS['bg']};
        }}

        /* ---------- inputs ---------- */
        .stSlider [data-baseweb="slider"] {{
            padding-top: 0.3rem;
        }}
        div[data-baseweb="select"] > div, .stNumberInput input {{
            background: {COLORS['panel']} !important;
            border-color: {COLORS['border']} !important;
            color: {COLORS['text']} !important;
        }}

        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    """Loads model, scaler, and expected columns. Returns (model, scaler,
    columns, ok) — ok=False if files are missing, so pages can show a
    friendly notice instead of crashing."""
    try:
        model = joblib.load("knn_heart_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        columns = joblib.load("heart_columns.pkl")
        return model, scaler, columns, True
    except FileNotFoundError:
        return None, None, EXPECTED_COLUMNS_FALLBACK, False


def build_input_row(age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                     resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,
                     expected_columns):
    """Builds a one-hot-encoded, correctly-ordered dataframe row matching
    the columns the model was trained on."""
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[expected_columns]


def risk_gauge(probability: float, colors=COLORS):
    """The signature visual: a clinical-style risk gauge (0-100) rendered
    with Plotly, styled to match the app's navy/gold/coral palette."""
    pct = round(probability * 100, 1)

    if pct < 35:
        bar_color = colors["safe"]
    elif pct < 65:
        bar_color = colors["gold"]
    else:
        bar_color = colors["risk"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 40, "color": colors["text"], "family": "Space Grotesk"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": colors["text_dim"],
                    "tickfont": {"color": colors["text_dim"], "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": colors["safe_dim"]},
                    {"range": [35, 65], "color": "#3A331E"},
                    {"range": [65, 100], "color": colors["risk_dim"]},
                ],
                "threshold": {
                    "line": {"color": colors["text"], "width": 2},
                    "thickness": 0.8,
                    "value": pct,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=10),
        height=260,
        font={"color": colors["text"]},
    )
    return fig


def sidebar_brand():
    """Consistent sidebar header shown on every page."""
    st.sidebar.markdown(
        f"""
        <div style="padding: 0.4rem 0 1.2rem 0;">
            <div class="eyebrow">Clinical Decision Support</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-weight:700;
                        font-size:1.25rem; color:{COLORS['text']};">
                ❤ Cardio<span style="color:{COLORS['gold']};">Scope</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
