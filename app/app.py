from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

REGRESSION_MODEL_PATH = ARTIFACT_DIR / "regression_pipeline.joblib"
CLASSIFICATION_MODEL_PATH = ARTIFACT_DIR / "classification_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "classification_metadata.joblib"


# ---------------------------------------------------------
# Expected model features
# ---------------------------------------------------------

EXPECTED_FEATURES = [
    "road_type",
    "num_lanes",
    "curvature",
    "speed_limit",
    "lighting",
    "weather",
    "road_signs_present",
    "public_road",
    "time_of_day",
    "holiday",
    "school_season",
    "num_reported_accidents",
]

CATEGORICAL_VALUES = {
    "road_type": ["highway", "rural", "urban"],
    "lighting": ["daylight", "dim", "night"],
    "weather": ["clear", "foggy", "rainy"],
    "time_of_day": ["afternoon", "evening", "morning"],
}

BOOLEAN_FEATURES = [
    "road_signs_present",
    "public_road",
    "holiday",
    "school_season",
]

HIGH_PRIORITY_PROBABILITY = 0.75

PRIORITY_STYLES = {
    "HIGH": {
        "color": "#DC2626",
        "bg": "#FEF2F2",
        "icon": "🔴",
        "title": "HIGH PRIORITY",
        "message": "Flag for additional transportation-safety review.",
    },
    "MEDIUM": {
        "color": "#D97706",
        "bg": "#FFFBEB",
        "icon": "🟠",
        "title": "MEDIUM PRIORITY",
        "message": "Consider additional transportation-safety review.",
    },
    "LOW": {
        "color": "#16A34A",
        "bg": "#F0FDF4",
        "icon": "🟢",
        "title": "LOW PRIORITY",
        "message": "Lower model-estimated priority for additional review.",
    },
}


# ---------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="Road Safety Prioritization Tool",
    page_icon="🚦",
    layout="wide"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --ink: #0B1220;
    --ink-2: #1E293B;
    --muted: #64748B;
    --line: #E5E9F0;
    --amber: #F59E0B;
    --surface: #FFFFFF;
}

.stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(1200px 500px at 100% -10%, rgba(245,158,11,0.07), transparent 60%),
        radial-gradient(900px 400px at -10% 0%, rgba(59,130,246,0.06), transparent 60%),
        #F4F6FA;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1240px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    display: flex;
    gap: 1.8rem;
    align-items: center;
    background: linear-gradient(135deg, #0B1220 0%, #15213A 55%, #1F2E4D 100%);
    color: #F8FAFC;
    border-radius: 22px;
    padding: 2.2rem 2.4rem 2.6rem 2.4rem;
    box-shadow: 0 20px 45px -15px rgba(11, 18, 32, 0.45);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}

.hero::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 14px;
    height: 4px;
    background: repeating-linear-gradient(90deg, rgba(251,191,36,0.55) 0 38px, transparent 38px 70px);
    opacity: 0.6;
}

.hero::after {
    content: "";
    position: absolute;
    right: -80px;
    top: -80px;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,158,11,0.28), transparent 70%);
}

.hero-lights {
    display: flex;
    flex-direction: column;
    gap: 9px;
    background: #020617;
    padding: 14px 11px;
    border-radius: 16px;
    border: 1px solid #334155;
    box-shadow: inset 0 0 0 3px #0F172A, 0 8px 20px rgba(0,0,0,0.35);
    flex-shrink: 0;
    z-index: 1;
}

.hero-lights span {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: block;
}

.hero-lights .r { background: #EF4444; box-shadow: 0 0 14px #EF4444; }
.hero-lights .y { background: #F59E0B; box-shadow: 0 0 14px #F59E0B; }
.hero-lights .g { background: #22C55E; box-shadow: 0 0 14px #22C55E; }

.hero-body { z-index: 1; }

.hero-eyebrow {
    display: inline-block;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #FBBF24;
    font-weight: 700;
    margin-bottom: 0.6rem;
    padding: 0.25rem 0.65rem;
    border: 1px solid rgba(251,191,36,0.35);
    background: rgba(251,191,36,0.08);
    border-radius: 999px;
}

.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.12;
    margin: 0 0 0.55rem 0;
    color: #F8FAFC;
}

.hero-text {
    font-size: 1rem;
    color: #CBD5E1;
    max-width: 760px;
    margin: 0;
    line-height: 1.6;
}

.hero-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.hero-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #E2E8F0;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 0.3rem 0.65rem;
    border-radius: 8px;
}

.hero-chip b { color: #FBBF24; }

.notice {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid var(--amber);
    color: #78350F;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 1.1rem 0 1.4rem 0;
    font-size: 0.9rem;
    line-height: 1.55;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    background: #E6EAF0;
    padding: 0.35rem;
    border-radius: 14px;
    width: fit-content;
}

.stTabs [data-baseweb="tab"] {
    height: 2.6rem;
    padding: 0 1.3rem;
    border-radius: 10px;
    font-weight: 600;
    color: #475569;
    background: transparent;
    transition: all 0.15s ease;
}

.stTabs [data-baseweb="tab"]:hover { color: var(--ink); }

.stTabs [aria-selected="true"] {
    background: #FFFFFF;
    color: var(--ink);
    box-shadow: 0 1px 2px rgba(15,23,42,0.08), 0 4px 10px rgba(15,23,42,0.06);
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin: 1.6rem 0 0.25rem 0;
}

.section-title::before {
    content: "";
    width: 4px;
    height: 1.1rem;
    border-radius: 4px;
    background: linear-gradient(180deg, #F59E0B, #EF4444);
}

.section-sub {
    font-size: 0.9rem;
    color: var(--muted);
    margin: 0 0 1rem 0.95rem;
    line-height: 1.55;
    max-width: 820px;
}

.group-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 0.2rem 0 0.8rem 0;
}

.group-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--line);
}

[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.6rem 1.8rem 1.4rem 1.8rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 10px 30px -12px rgba(15,23,42,0.10);
}

[data-testid="stWidgetLabel"] p {
    font-weight: 600;
    font-size: 0.85rem;
    color: #334155;
}

[data-baseweb="select"] > div,
[data-testid="stNumberInput"] [data-baseweb="input"] {
    border-radius: 10px !important;
    border-color: #DCE2EA !important;
    background: #F9FAFB !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

[data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.18);
    background: #FFFFFF !important;
}

[data-testid="stNumberInput"] button {
    background: #F1F4F8;
    border: none;
}

[data-testid="stNumberInput"] button:hover {
    background: var(--amber);
    color: var(--ink);
}

[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #0B1220, #1E293B);
    color: #FFFFFF;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    letter-spacing: 0.01em;
    padding: 0.75rem 1rem;
    margin-top: 0.6rem;
    box-shadow: 0 6px 16px -6px rgba(11,18,32,0.5);
    transition: all 0.18s ease;
}

[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #F59E0B, #FBBF24);
    color: var(--ink);
    transform: translateY(-1px);
    box-shadow: 0 10px 22px -8px rgba(245,158,11,0.6);
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.15rem 1.25rem 1.05rem 1.25rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 8px 24px -14px rgba(15,23,42,0.12);
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    right: 0;
    height: 4px;
    background: var(--accent);
}

.metric-card::after {
    content: "";
    position: absolute;
    right: -30px;
    bottom: -30px;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    background: var(--accent);
    opacity: 0.06;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(15,23,42,0.05), 0 16px 30px -14px rgba(15,23,42,0.2);
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ink);
    margin-top: 0.35rem;
}

.metric-caption {
    font-size: 0.78rem;
    color: #94A3B8;
    margin-top: 0.2rem;
}

.priority-card {
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0 0.4rem 0;
    border: 1px solid var(--accent);
    background: linear-gradient(90deg, var(--bg), #FFFFFF);
    display: flex;
    gap: 1rem;
    align-items: center;
    box-shadow: 0 8px 22px -14px var(--accent);
}

.priority-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    background: #FFFFFF;
    border: 1px solid var(--accent);
    flex-shrink: 0;
}

.priority-text {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.priority-badge {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    color: #FFFFFF;
    background: var(--accent);
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    white-space: nowrap;
    width: fit-content;
}

.priority-message {
    color: var(--ink-2);
    font-size: 0.96rem;
    font-weight: 500;
}

.gauge-wrap {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.1rem 1.3rem 1rem 1.3rem;
    margin-top: 0.7rem;
    box-shadow: 0 8px 24px -16px rgba(15,23,42,0.15);
}

.gauge-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.7rem;
}

.gauge-head b {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    letter-spacing: 0;
}

.gauge {
    position: relative;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #DCFCE7 0%, #FEF3C7 60%, #FEE2E2 100%);
    box-shadow: inset 0 1px 2px rgba(15,23,42,0.08);
}

.gauge-fill {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 999px;
    box-shadow: 0 0 10px rgba(15,23,42,0.15);
}

.gauge-marker {
    position: absolute;
    top: -5px;
    bottom: -5px;
    width: 2px;
    border-radius: 2px;
    background: var(--ink);
    opacity: 0.6;
}

.gauge-legend {
    display: flex;
    gap: 1.4rem;
    flex-wrap: wrap;
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.8rem;
}

.gauge-legend b {
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
}

.dist-bar {
    display: flex;
    gap: 3px;
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: #E2E8F0;
    margin: 1.2rem 0 0.7rem 0;
}

.dist-bar > div:first-child { border-radius: 999px 0 0 999px; }
.dist-bar > div:last-child { border-radius: 0 999px 999px 0; }

.dist-legend {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    font-size: 0.84rem;
    color: #475569;
}

.dist-legend > span {
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
}

.dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 6px;
}

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--line) !important;
    border-radius: 14px;
    box-shadow: 0 4px 14px -10px rgba(15,23,42,0.12);
}

[data-testid="stExpander"] summary {
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 6px 18px -14px rgba(15,23,42,0.18);
}

[data-testid="stFileUploader"] section {
    border-radius: 16px;
    border: 2px dashed #CBD5E1;
    background: var(--surface);
    padding: 1.6rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--amber);
    background: #FFFDF6;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--line);
}

.footer span {
    color: var(--amber);
    margin: 0 0.3rem;
}
</style>
"""


def render_html(markup):
    st.markdown(
        " ".join(
            line.strip()
            for line in markup.splitlines()
            if line.strip()
        ),
        unsafe_allow_html=True
    )


render_html(CUSTOM_CSS)


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    regression_model = joblib.load(
        REGRESSION_MODEL_PATH
    )

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

    return (
        regression_model,
        classification_model,
        metadata
    )


regression_model, classification_model, metadata = (
    load_models()
)

probability_threshold = metadata[
    "classification_probability_threshold"
]

high_risk_threshold = metadata[
    "high_risk_threshold"
]


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_review_priority(probability):

    if probability >= HIGH_PRIORITY_PROBABILITY:
        return "HIGH"

    elif probability >= probability_threshold:
        return "MEDIUM"

    else:
        return "LOW"


def validate_batch_data(df):

    errors = []

    missing_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        errors.append(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        return errors

    for column, allowed_values in (
        CATEGORICAL_VALUES.items()
    ):

        invalid_values = (
            df.loc[
                ~df[column].isin(allowed_values),
                column
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if invalid_values:
            errors.append(
                f"Invalid values in '{column}': "
                f"{invalid_values}. "
                f"Allowed values: {allowed_values}"
            )

    for column in BOOLEAN_FEATURES:

        invalid_values = (
            df.loc[
                ~df[column].isin(
                    [0, 1, True, False]
                ),
                column
            ]
            .dropna()
            .unique()
            .tolist()
        )

        if invalid_values:
            errors.append(
                f"Invalid values in '{column}': "
                f"{invalid_values}. "
                "Use 0/1 or True/False."
            )

    if df[
        EXPECTED_FEATURES
    ].isna().any().any():

        errors.append(
            "The uploaded file contains missing "
            "values in required model features."
        )

    return errors


def score_batch_data(df):

    scoring_data = df[
        EXPECTED_FEATURES
    ].copy()

    for column in BOOLEAN_FEATURES:
        scoring_data[column] = (
            scoring_data[column].astype(int)
        )

    raw_risk_predictions = (
        regression_model.predict(
            scoring_data
        )
    )

    risk_predictions = (
        raw_risk_predictions.clip(
            min=0.0,
            max=1.0
        )
    )

    probabilities = (
        classification_model.predict_proba(
            scoring_data
        )[:, 1]
    )

    # Use selected probability threshold from metadata
    # instead of Logistic Regression's default 0.50.
    class_predictions = (
        probabilities >= probability_threshold
    ).astype(int)

    results = df.copy()

    results[
        "estimated_accident_risk"
    ] = risk_predictions

    results[
        "high_risk_probability"
    ] = probabilities

    results[
        "high_risk_flag"
    ] = class_predictions

    results[
        "review_priority"
    ] = [
        get_review_priority(probability)
        for probability in probabilities
    ]

    results = (
        results.sort_values(
            by="high_risk_probability",
            ascending=False
        )
        .reset_index(drop=True)
    )

    results.insert(
        0,
        "priority_rank",
        range(1, len(results) + 1)
    )

    return results


# ---------------------------------------------------------
# UI components
# ---------------------------------------------------------

def metric_card(
    label,
    value,
    caption="",
    accent="#0F172A"
):
    return (
        f'<div class="metric-card" '
        f'style="--accent:{accent}">'
        f'<div class="metric-label">'
        f'{label}</div>'
        f'<div class="metric-value">'
        f'{value}</div>'
        f'<div class="metric-caption">'
        f'{caption}</div>'
        "</div>"
    )


def section_header(
    title,
    subtitle=""
):
    markup = (
        f'<div class="section-title">'
        f'{title}</div>'
    )

    if subtitle:
        markup += (
            f'<div class="section-sub">'
            f'{subtitle}</div>'
        )

    render_html(markup)


def priority_card(priority):

    style = PRIORITY_STYLES[
        priority
    ]

    render_html(
        f'<div class="priority-card" '
        f'style="--accent:{style["color"]};'
        f'--bg:{style["bg"]}">'
        f'<div class="priority-icon">{style["icon"]}</div>'
        '<div class="priority-text">'
        f'<span class="priority-badge">'
        f'{style["title"]}</span>'
        f'<span class="priority-message">'
        f'{style["message"]}</span>'
        "</div></div>"
    )


def probability_gauge(probability):

    color = PRIORITY_STYLES[
        get_review_priority(
            probability
        )
    ]["color"]

    render_html(
        '<div class="gauge-wrap">'
        '<div class="gauge-head">'
        '<span>High-risk probability</span>'
        f'<span><b style="color:{color}">'
        f'{probability:.1%}</b></span>'
        '</div>'

        '<div class="gauge">'

        f'<div class="gauge-fill" '
        f'style="width:{probability * 100:.1f}%;'
        f'background:{color}"></div>'

        f'<div class="gauge-marker" '
        f'style="left:'
        f'{probability_threshold * 100:.1f}%">'
        '</div>'

        f'<div class="gauge-marker" '
        f'style="left:'
        f'{HIGH_PRIORITY_PROBABILITY * 100:.1f}%">'
        '</div>'

        '</div>'

        '<div class="gauge-legend">'

        f'<span>LOW &lt; '
        f'<b>{probability_threshold:.2f}'
        '</b></span>'

        f'<span>MEDIUM ≥ '
        f'<b>{probability_threshold:.2f}'
        '</b></span>'

        f'<span>HIGH ≥ '
        f'<b>{HIGH_PRIORITY_PROBABILITY:.2f}'
        '</b></span>'

        '</div></div>'
    )


def distribution_bar(
    counts,
    total
):

    segments = ""
    legend = ""

    for level in [
        "HIGH",
        "MEDIUM",
        "LOW"
    ]:

        color = (
            PRIORITY_STYLES[
                level
            ]["color"]
        )

        share = (
            counts[level]
            / total
            * 100
            if total
            else 0
        )

        if share > 0:
            segments += (
                f'<div style="width:'
                f'{share:.2f}%;'
                f'background:{color}" '
                f'title="{level}: '
                f'{counts[level]:,}">'
                '</div>'
            )

        legend += (
            f'<span>'
            f'<span class="dot" '
            f'style="background:{color}">'
            '</span>'
            f'{level.title()} — '
            f'<b>{counts[level]:,}</b> '
            f'({share:.1f}%)'
            '</span>'
        )

    render_html(
        f'<div class="dist-bar">'
        f'{segments}</div>'
        f'<div class="dist-legend">'
        f'{legend}</div>'
    )


def results_column_config():

    return {
        "priority_rank":
            st.column_config.NumberColumn(
                "Rank",
                format="%d"
            ),

        "estimated_accident_risk":
            st.column_config.ProgressColumn(
                "Estimated Accident Risk",
                min_value=0.0,
                max_value=1.0,
                format="%.4f"
            ),

        "high_risk_probability":
            st.column_config.ProgressColumn(
                "High-Risk Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.4f"
            ),

        "high_risk_flag":
            st.column_config.CheckboxColumn(
                "High-Risk Flag"
            ),

        "review_priority":
            st.column_config.TextColumn(
                "Review Priority"
            ),
    }


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

render_html(
    """
    <div class="hero">
        <div class="hero-lights">
            <span class="r"></span>
            <span class="y"></span>
            <span class="g"></span>
        </div>

        <div class="hero-body">
            <div class="hero-eyebrow">
                Proof-of-concept · Transportation safety analytics
            </div>

            <div class="hero-title">
                Road Safety Prioritization Tool
            </div>

            <p class="hero-text">
                This proof-of-concept tool estimates accident risk
                and identifies higher-risk road-condition observations
                that may benefit from additional transportation-safety
                review.
            </p>

        </div>
    </div>
    """
)

render_html(
    """
    <div class="notice">
        <span>⚠️</span>

        <span>
            This tool was developed using a synthetic dataset and is
            intended for demonstration and screening purposes only.
            Model outputs should not replace professional
            transportation-safety or engineering analysis.
        </span>
    </div>
    """
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

single_tab, batch_tab = st.tabs(
    [
        "🧭  Single Observation",
        "📂  Batch CSV Prioritization"
    ]
)


# =========================================================
# TAB 1 — SINGLE OBSERVATION
# =========================================================

with single_tab:

    section_header(
        "Road Condition Information",
        "Describe one road-condition observation, "
        "then run the screening models."
    )

    with st.form(
        "risk_input_form"
    ):

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            render_html(
                '<div class="group-label">'
                'Road characteristics'
                '</div>'
            )

            road_type = st.selectbox(
                "Road Type",
                [
                    "highway",
                    "rural",
                    "urban"
                ],
                format_func=lambda x: x.title()
            )

            num_lanes = st.number_input(
                "Number of Lanes",
                min_value=1,
                max_value=10,
                value=2,
                step=1
            )

            curvature = st.number_input(
                "Curvature",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.01,
                format="%.2f"
            )

            speed_limit = st.number_input(
                "Speed Limit",
                min_value=10,
                max_value=100,
                value=45,
                step=5
            )

            lighting = st.selectbox(
                "Lighting",
                [
                    "daylight",
                    "dim",
                    "night"
                ],
                format_func=lambda x: x.title()
            )

            weather = st.selectbox(
                "Weather",
                [
                    "clear",
                    "foggy",
                    "rainy"
                ],
                format_func=lambda x: x.title()
            )

        with col2:

            render_html(
                '<div class="group-label">'
                'Context &amp; history'
                '</div>'
            )

            road_signs_present = (
                st.selectbox(
                    "Road Signs Present",
                    [
                        "Yes",
                        "No"
                    ]
                )
            )

            public_road = st.selectbox(
                "Public Road",
                [
                    "Yes",
                    "No"
                ]
            )

            time_of_day = st.selectbox(
                "Time of Day",
                [
                    "afternoon",
                    "evening",
                    "morning"
                ],
                format_func=lambda x: x.title()
            )

            holiday = st.selectbox(
                "Holiday",
                [
                    "No",
                    "Yes"
                ]
            )

            school_season = (
                st.selectbox(
                    "School Season",
                    [
                        "No",
                        "Yes"
                    ]
                )
            )

            num_reported_accidents = (
                st.number_input(
                    "Previously Reported Accidents",
                    min_value=0,
                    value=0,
                    step=1
                )
            )

        analyze_button = (
            st.form_submit_button(
                "Analyze Risk",
                use_container_width=True
            )
        )

    if analyze_button:

        input_data = pd.DataFrame({
            "road_type": [
                road_type
            ],

            "num_lanes": [
                num_lanes
            ],

            "curvature": [
                curvature
            ],

            "speed_limit": [
                speed_limit
            ],

            "lighting": [
                lighting
            ],

            "weather": [
                weather
            ],

            "road_signs_present": [
                1
                if road_signs_present
                == "Yes"
                else 0
            ],

            "public_road": [
                1
                if public_road
                == "Yes"
                else 0
            ],

            "time_of_day": [
                time_of_day
            ],

            "holiday": [
                1
                if holiday
                == "Yes"
                else 0
            ],

            "school_season": [
                1
                if school_season
                == "Yes"
                else 0
            ],

            "num_reported_accidents": [
                num_reported_accidents
            ],
        })

        raw_estimated_risk = (
            regression_model.predict(
                input_data
            )[0]
        )

        estimated_risk = max(
            0.0,
            min(
                1.0,
                raw_estimated_risk
            )
        )

        high_risk_probability = (
            classification_model.predict_proba(
                input_data
            )[0, 1]
        )

        # Use selected probability threshold from metadata
        # instead of Logistic Regression's default 0.50.
        predicted_class = int(
            high_risk_probability
            >= probability_threshold
        )

        priority = (
            get_review_priority(
                high_risk_probability
            )
        )

        priority_color = (
            PRIORITY_STYLES[
                priority
            ]["color"]
        )

        section_header(
            "Risk Screening Results"
        )

        (
            metric1,
            metric2,
            metric3
        ) = st.columns(3)

        with metric1:

            render_html(
                metric_card(
                    "Estimated Accident Risk",
                    f"{estimated_risk:.3f}",
                    "Regression estimate (0–1)",
                    "#3B82F6"
                )
            )

        with metric2:

            render_html(
                metric_card(
                    "High-Risk Probability",
                    f"{high_risk_probability:.1%}",
                    "Classifier probability",
                    priority_color
                )
            )

        with metric3:

            flagged = (
                predicted_class == 1
            )

            render_html(
                metric_card(
                    "High-Risk Flag",
                    (
                        "YES"
                        if flagged
                        else "NO"
                    ),
                    (
                        f"Cutoff "
                        f"{probability_threshold:.2f}"
                    ),
                    (
                        "#DC2626"
                        if flagged
                        else "#16A34A"
                    )
                )
            )

        section_header(
            "Review Priority"
        )

        priority_card(
            priority
        )

        probability_gauge(
            high_risk_probability
        )

        st.write("")

        with st.expander(
            "Model Details"
        ):

            st.write(
                "**Training-derived "
                "high-risk accident-risk cutoff:** "
                f"{high_risk_threshold:.2f}"
            )

            st.write(
                "**Classification "
                "probability threshold:** "
                f"{probability_threshold:.2f}"
            )

            st.write(
                """
                HIGH, MEDIUM, and LOW
                review-priority categories are
                illustrative decision-support
                rules. They are not
                transportation-engineering
                thresholds learned directly
                from the dataset.
                """
            )

        if (
            raw_estimated_risk < 0
            or raw_estimated_risk > 1
        ):

            st.caption(
                "The raw Linear Regression "
                "prediction fell outside the "
                "0–1 range. The displayed "
                "value was bounded to 0–1 "
                "for presentation."
            )

        with st.expander(
            "View Input Data"
        ):

            st.dataframe(
                input_data,
                hide_index=True,
                use_container_width=True
            )


# =========================================================
# TAB 2 — BATCH CSV PRIORITIZATION
# =========================================================

with batch_tab:

    section_header(
        "Batch Road-Safety Prioritization",
        "Upload a CSV containing multiple "
        "road-condition observations. "
        "The tool will score each observation, "
        "assign a review priority, and rank the "
        "observations from highest to lowest "
        "predicted high-risk probability."
    )

    with st.expander(
        "📋  Required CSV Columns"
    ):

        req_col1, req_col2 = (
            st.columns(
                [1, 1.4],
                gap="large"
            )
        )

        with req_col1:

            st.dataframe(
                pd.DataFrame({
                    "Column":
                        EXPECTED_FEATURES
                }),
                hide_index=True,
                use_container_width=True
            )

        with req_col2:

            st.write(
                "Categorical values must "
                "use the same values as "
                "the training data:"
            )

            st.code(
                """
road_type: highway, rural, urban
lighting: daylight, dim, night
weather: clear, foggy, rainy
time_of_day: afternoon, evening, morning

road_signs_present: 0 or 1
public_road: 0 or 1
holiday: 0 or 1
school_season: 0 or 1
                """
            )

    uploaded_file = (
        st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(
                uploaded_file
            )

            st.write(
                "**Uploaded observations:** "
                f"{len(batch_df):,}"
            )

            with st.expander(
                "Preview Uploaded Data"
            ):

                st.dataframe(
                    batch_df.head(20),
                    hide_index=True,
                    use_container_width=True
                )

            validation_errors = (
                validate_batch_data(
                    batch_df
                )
            )

            if validation_errors:

                st.error(
                    "The uploaded file "
                    "cannot be scored."
                )

                for error in (
                    validation_errors
                ):
                    st.write(
                        f"- {error}"
                    )

            else:

                st.success(
                    "CSV validation passed."
                )

                results = (
                    score_batch_data(
                        batch_df
                    )
                )

                # Summary
                section_header(
                    "Batch Prioritization Summary"
                )

                total_observations = (
                    len(results)
                )

                priority_counts = {
                    level: int(
                        results[
                            "review_priority"
                        ]
                        .eq(level)
                        .sum()
                    )
                    for level in [
                        "HIGH",
                        "MEDIUM",
                        "LOW"
                    ]
                }

                high_risk_flag_count = int(
                    results[
                        "high_risk_flag"
                    ].sum()
                )

                (
                    summary1,
                    summary2,
                    summary3,
                    summary4
                ) = st.columns(4)

                with summary1:

                    render_html(
                        metric_card(
                            "Total Observations",
                            (
                                f"{total_observations:,}"
                            ),
                            "Rows scored",
                            "#0F172A"
                        )
                    )

                with summary2:

                    render_html(
                        metric_card(
                            "High Priority",
                            (
                                f"{priority_counts['HIGH']:,}"
                            ),
                            (
                                "Probability ≥ "
                                f"{HIGH_PRIORITY_PROBABILITY:.2f}"
                            ),
                            PRIORITY_STYLES[
                                "HIGH"
                            ]["color"]
                        )
                    )

                with summary3:

                    render_html(
                        metric_card(
                            "Medium Priority",
                            (
                                f"{priority_counts['MEDIUM']:,}"
                            ),
                            (
                                "Probability ≥ "
                                f"{probability_threshold:.2f}"
                            ),
                            PRIORITY_STYLES[
                                "MEDIUM"
                            ]["color"]
                        )
                    )

                with summary4:

                    render_html(
                        metric_card(
                            "High-Risk Flags",
                            (
                                f"{high_risk_flag_count:,}"
                            ),
                            "Classifier positives",
                            "#7C3AED"
                        )
                    )

                distribution_bar(
                    priority_counts,
                    total_observations
                )

                # Display-only formatting;
                # download keeps raw values

                display_results = (
                    results.copy()
                )

                display_results[
                    "estimated_accident_risk"
                ] = (
                    display_results[
                        "estimated_accident_risk"
                    ]
                    .round(4)
                )

                display_results[
                    "high_risk_probability"
                ] = (
                    display_results[
                        "high_risk_probability"
                    ]
                    .round(4)
                )

                display_results[
                    "high_risk_flag"
                ] = (
                    display_results[
                        "high_risk_flag"
                    ]
                    .astype(bool)
                )

                display_results[
                    "review_priority"
                ] = (
                    display_results[
                        "review_priority"
                    ]
                    .map(
                        lambda p:
                            f'{PRIORITY_STYLES[p]["icon"]} {p}'
                    )
                )

                section_header(
                    "Top 10 Observations for Review",
                    "Highest predicted high-risk "
                    "probability first."
                )

                st.dataframe(
                    display_results.head(10),
                    hide_index=True,
                    use_container_width=True,
                    column_config=(
                        results_column_config()
                    )
                )

                section_header(
                    "Prioritized Observations"
                )

                st.dataframe(
                    display_results,
                    hide_index=True,
                    use_container_width=True,
                    column_config=(
                        results_column_config()
                    )
                )

                csv_output = (
                    results
                    .to_csv(
                        index=False
                    )
                    .encode("utf-8")
                )

                st.download_button(
                    label=(
                        "⬇️  Download "
                        "Prioritized Results"
                    ),
                    data=csv_output,
                    file_name=(
                        "road_safety_"
                        "prioritized_results.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                "An error occurred while "
                "processing the uploaded CSV."
            )

            st.exception(
                error
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

render_html(
    '<div class="footer">'
    '🚦 Road Accident Risk Prediction '
    'and Safety Prioritization '
    '<span>•</span> '
    'Proof-of-Concept Analytics Project'
    '</div>'
)