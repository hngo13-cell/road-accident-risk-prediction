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

.stApp { font-family: 'Inter', sans-serif; background: #F6F7F9; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
    display: flex; gap: 1.6rem; align-items: center;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #334155 100%);
    color: #F8FAFC; border-radius: 20px; padding: 2rem 2.2rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
    position: relative; overflow: hidden;
}
.hero::after {
    content: ""; position: absolute; right: -60px; top: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(245,158,11,0.25), transparent 70%);
}
.hero-lights {
    display: flex; flex-direction: column; gap: 8px;
    background: #020617; padding: 12px 10px; border-radius: 14px;
    border: 1px solid #334155; flex-shrink: 0;
}
.hero-lights span { width: 18px; height: 18px; border-radius: 50%; display: block; }
.hero-lights .r { background: #EF4444; box-shadow: 0 0 12px #EF4444; }
.hero-lights .y { background: #F59E0B; box-shadow: 0 0 12px #F59E0B; }
.hero-lights .g { background: #22C55E; box-shadow: 0 0 12px #22C55E; }
.hero-eyebrow {
    font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #FBBF24; font-weight: 600; margin-bottom: 0.35rem;
}
.hero-title { font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 0 0 0.5rem 0; color: #F8FAFC; }
.hero-text { font-size: 1rem; color: #CBD5E1; max-width: 760px; margin: 0; line-height: 1.55; }

.notice {
    display: flex; gap: 0.8rem; align-items: flex-start;
    background: #FFFBEB; border: 1px solid #FDE68A; border-left: 4px solid #F59E0B;
    color: #78350F; border-radius: 12px; padding: 0.9rem 1.1rem;
    margin: 1.1rem 0 1.4rem 0; font-size: 0.92rem; line-height: 1.5;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem; background: #E9ECF1; padding: 0.35rem; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 2.6rem; padding: 0 1.2rem; border-radius: 9px;
    font-weight: 600; color: #475569; background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF; color: #0F172A; box-shadow: 0 1px 4px rgba(15,23,42,0.12);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.section-title { font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 1.4rem 0 0.2rem 0; }
.section-sub { font-size: 0.92rem; color: #64748B; margin-bottom: 0.9rem; line-height: 1.5; }
.group-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #94A3B8; margin: 0.2rem 0 0.6rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px dashed #E2E8F0;
}

[data-testid="stForm"] {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px;
    padding: 1.4rem 1.6rem; box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}
[data-testid="stFormSubmitButton"] button, [data-testid="stDownloadButton"] button {
    background: #0F172A; color: #FFFFFF; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.65rem 1rem;
}
[data-testid="stFormSubmitButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
    background: #F59E0B; color: #0F172A;
}

.metric-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1.1rem 1.2rem; border-top: 4px solid var(--accent);
    box-shadow: 0 2px 8px rgba(15,23,42,0.04); height: 100%;
}
.metric-label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #64748B;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 700;
    color: #0F172A; margin-top: 0.3rem;
}
.metric-caption { font-size: 0.8rem; color: #94A3B8; margin-top: 0.15rem; }

.priority-card {
    border-radius: 14px; padding: 1.1rem 1.3rem; margin: 1rem 0 0.4rem 0;
    border: 1px solid var(--accent); border-left: 6px solid var(--accent);
    background: var(--bg); display: flex; gap: 1rem; align-items: center;
}
.priority-badge {
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.8rem;
    color: #FFFFFF; background: var(--accent); padding: 0.35rem 0.7rem;
    border-radius: 999px; white-space: nowrap;
}
.priority-message { color: #1E293B; font-size: 0.98rem; font-weight: 500; }

.gauge-wrap {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
    padding: 1rem 1.2rem 0.9rem 1.2rem; margin-top: 0.6rem;
}
.gauge-head { display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748B; margin-bottom: 0.5rem; }
.gauge {
    position: relative; height: 14px; border-radius: 999px;
    background: linear-gradient(90deg, #DCFCE7 0%, #FEF3C7 60%, #FEE2E2 100%);
}
.gauge-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }
.gauge-marker { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #0F172A; opacity: 0.55; }
.gauge-legend { display: flex; gap: 1.2rem; flex-wrap: wrap; font-size: 0.78rem; color: #64748B; margin-top: 0.6rem; }
.gauge-legend b { color: #0F172A; font-family: 'JetBrains Mono', monospace; }

.dist-bar {
    display: flex; height: 16px; border-radius: 999px; overflow: hidden;
    background: #E2E8F0; margin: 1rem 0 0.5rem 0;
}
.dist-legend { display: flex; gap: 1.4rem; flex-wrap: wrap; font-size: 0.85rem; color: #475569; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }

[data-testid="stExpander"] { background: #FFFFFF; border-radius: 12px; }
[data-testid="stFileUploader"] section { border-radius: 14px; border: 2px dashed #CBD5E1; background: #FFFFFF; }

.footer { text-align: center; color: #94A3B8; font-size: 0.82rem; margin-top: 2.5rem; }
.footer span { color: #F59E0B; }
</style>
"""


def render_html(markup):
    st.markdown(
        "".join(line.strip() for line in markup.splitlines()),
        unsafe_allow_html=True
    )


render_html(CUSTOM_CSS)


# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    return regression_model, classification_model, metadata


regression_model, classification_model, metadata = load_models()

probability_threshold = metadata["classification_probability_threshold"]
high_risk_threshold = metadata["high_risk_threshold"]


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
        column for column in EXPECTED_FEATURES if column not in df.columns
    ]

    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))
        return errors

    for column, allowed_values in CATEGORICAL_VALUES.items():
        invalid_values = (
            df.loc[~df[column].isin(allowed_values), column]
            .dropna().unique().tolist()
        )

        if invalid_values:
            errors.append(
                f"Invalid values in '{column}': {invalid_values}. "
                f"Allowed values: {allowed_values}"
            )

    for column in BOOLEAN_FEATURES:
        invalid_values = (
            df.loc[~df[column].isin([0, 1, True, False]), column]
            .dropna().unique().tolist()
        )

        if invalid_values:
            errors.append(
                f"Invalid values in '{column}': {invalid_values}. "
                "Use 0/1 or True/False."
            )

    if df[EXPECTED_FEATURES].isna().any().any():
        errors.append(
            "The uploaded file contains missing values "
            "in required model features."
        )

    return errors


def score_batch_data(df):
    scoring_data = df[EXPECTED_FEATURES].copy()

    for column in BOOLEAN_FEATURES:
        scoring_data[column] = scoring_data[column].astype(int)

    raw_risk_predictions = regression_model.predict(scoring_data)
    risk_predictions = raw_risk_predictions.clip(min=0.0, max=1.0)

    probabilities = classification_model.predict_proba(scoring_data)[:, 1]
    class_predictions = classification_model.predict(scoring_data)

    results = df.copy()
    results["estimated_accident_risk"] = risk_predictions
    results["high_risk_probability"] = probabilities
    results["high_risk_flag"] = class_predictions
    results["review_priority"] = [
        get_review_priority(probability) for probability in probabilities
    ]

    results = results.sort_values(
        by="high_risk_probability", ascending=False
    ).reset_index(drop=True)

    results.insert(0, "priority_rank", range(1, len(results) + 1))

    return results


# ---------------------------------------------------------
# UI components
# ---------------------------------------------------------

def metric_card(label, value, caption="", accent="#0F172A"):
    return (
        f'<div class="metric-card" style="--accent:{accent}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-caption">{caption}</div>'
        "</div>"
    )


def section_header(title, subtitle=""):
    markup = f'<div class="section-title">{title}</div>'
    if subtitle:
        markup += f'<div class="section-sub">{subtitle}</div>'
    render_html(markup)


def priority_card(priority):
    style = PRIORITY_STYLES[priority]
    render_html(
        f'<div class="priority-card" style="--accent:{style["color"]};--bg:{style["bg"]}">'
        f'<span class="priority-badge">{style["title"]}</span>'
        f'<span class="priority-message">{style["message"]}</span>'
        "</div>"
    )


def probability_gauge(probability):
    color = PRIORITY_STYLES[get_review_priority(probability)]["color"]
    render_html(
        '<div class="gauge-wrap">'
        '<div class="gauge-head"><span>High-risk probability</span>'
        f'<span><b style="color:{color}">{probability:.1%}</b></span></div>'
        '<div class="gauge">'
        f'<div class="gauge-fill" style="width:{probability * 100:.1f}%;background:{color}"></div>'
        f'<div class="gauge-marker" style="left:{probability_threshold * 100:.1f}%"></div>'
        f'<div class="gauge-marker" style="left:{HIGH_PRIORITY_PROBABILITY * 100:.1f}%"></div>'
        "</div>"
        '<div class="gauge-legend">'
        f'<span>LOW &lt; <b>{probability_threshold:.2f}</b></span>'
        f'<span>MEDIUM ≥ <b>{probability_threshold:.2f}</b></span>'
        f'<span>HIGH ≥ <b>{HIGH_PRIORITY_PROBABILITY:.2f}</b></span>'
        "</div></div>"
    )


def distribution_bar(counts, total):
    segments = ""
    legend = ""

    for level in ["HIGH", "MEDIUM", "LOW"]:
        color = PRIORITY_STYLES[level]["color"]
        share = counts[level] / total * 100 if total else 0

        if share > 0:
            segments += (
                f'<div style="width:{share:.2f}%;background:{color}" '
                f'title="{level}: {counts[level]:,}"></div>'
            )

        legend += (
            f'<span><span class="dot" style="background:{color}"></span>'
            f"{level.title()} — <b>{counts[level]:,}</b> ({share:.1f}%)</span>"
        )

    render_html(
        f'<div class="dist-bar">{segments}</div>'
        f'<div class="dist-legend">{legend}</div>'
    )


def results_column_config():
    return {
        "priority_rank": st.column_config.NumberColumn("Rank", format="%d"),
        "estimated_accident_risk": st.column_config.ProgressColumn(
            "Estimated Accident Risk", min_value=0.0, max_value=1.0, format="%.4f"
        ),
        "high_risk_probability": st.column_config.ProgressColumn(
            "High-Risk Probability", min_value=0.0, max_value=1.0, format="%.4f"
        ),
        "high_risk_flag": st.column_config.CheckboxColumn("High-Risk Flag"),
        "review_priority": st.column_config.TextColumn("Review Priority"),
    }


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

render_html(
    """
    <div class="hero">
    <div class="hero-lights"><span class="r"></span><span class="y"></span><span class="g"></span></div>
    <div>
    <div class="hero-eyebrow">Proof-of-concept · Transportation safety analytics</div>
    <div class="hero-title">Road Safety Prioritization Tool</div>
    <p class="hero-text">This proof-of-concept tool estimates accident risk and identifies higher-risk road-condition observations that may benefit from additional transportation-safety review.</p>
    </div>
    </div>
    """
)

render_html(
    """
    <div class="notice">
    <span>⚠️</span>
    <span>This tool was developed using a synthetic dataset and is intended for demonstration and screening purposes only. Model outputs should not replace professional transportation-safety or engineering analysis.</span>
    </div>
    """
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

single_tab, batch_tab = st.tabs(
    ["🧭  Single Observation", "📂  Batch CSV Prioritization"]
)


# =========================================================
# TAB 1 — SINGLE OBSERVATION
# =========================================================

with single_tab:

    section_header(
        "Road Condition Information",
        "Describe one road-condition observation, then run the screening models."
    )

    with st.form("risk_input_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            render_html('<div class="group-label">Road characteristics</div>')

            road_type = st.selectbox(
                "Road Type", ["highway", "rural", "urban"],
                format_func=lambda x: x.title()
            )

            num_lanes = st.number_input(
                "Number of Lanes", min_value=1, max_value=10, value=2, step=1
            )

            curvature = st.number_input(
                "Curvature", min_value=0.0, max_value=1.0,
                value=0.30, step=0.01, format="%.2f"
            )

            speed_limit = st.number_input(
                "Speed Limit", min_value=10, max_value=100, value=45, step=5
            )

            lighting = st.selectbox(
                "Lighting", ["daylight", "dim", "night"],
                format_func=lambda x: x.title()
            )

            weather = st.selectbox(
                "Weather", ["clear", "foggy", "rainy"],
                format_func=lambda x: x.title()
            )

        with col2:
            render_html('<div class="group-label">Context &amp; history</div>')

            road_signs_present = st.selectbox("Road Signs Present", ["Yes", "No"])

            public_road = st.selectbox("Public Road", ["Yes", "No"])

            time_of_day = st.selectbox(
                "Time of Day", ["afternoon", "evening", "morning"],
                format_func=lambda x: x.title()
            )

            holiday = st.selectbox("Holiday", ["No", "Yes"])

            school_season = st.selectbox("School Season", ["No", "Yes"])

            num_reported_accidents = st.number_input(
                "Previously Reported Accidents", min_value=0, value=0, step=1
            )

        analyze_button = st.form_submit_button(
            "Analyze Risk", use_container_width=True
        )

    if analyze_button:

        input_data = pd.DataFrame({
            "road_type": [road_type],
            "num_lanes": [num_lanes],
            "curvature": [curvature],
            "speed_limit": [speed_limit],
            "lighting": [lighting],
            "weather": [weather],
            "road_signs_present": [1 if road_signs_present == "Yes" else 0],
            "public_road": [1 if public_road == "Yes" else 0],
            "time_of_day": [time_of_day],
            "holiday": [1 if holiday == "Yes" else 0],
            "school_season": [1 if school_season == "Yes" else 0],
            "num_reported_accidents": [num_reported_accidents],
        })

        raw_estimated_risk = regression_model.predict(input_data)[0]
        estimated_risk = max(0.0, min(1.0, raw_estimated_risk))

        high_risk_probability = classification_model.predict_proba(input_data)[0, 1]
        predicted_class = classification_model.predict(input_data)[0]

        priority = get_review_priority(high_risk_probability)
        priority_color = PRIORITY_STYLES[priority]["color"]

        section_header("Risk Screening Results")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            render_html(metric_card(
                "Estimated Accident Risk", f"{estimated_risk:.3f}",
                "Regression estimate (0–1)", "#3B82F6"
            ))

        with metric2:
            render_html(metric_card(
                "High-Risk Probability", f"{high_risk_probability:.1%}",
                "Classifier probability", priority_color
            ))

        with metric3:
            flagged = predicted_class == 1
            render_html(metric_card(
                "High-Risk Flag", "YES" if flagged else "NO",
                f"Cutoff {probability_threshold:.2f}",
                "#DC2626" if flagged else "#16A34A"
            ))

        section_header("Review Priority")
        priority_card(priority)
        probability_gauge(high_risk_probability)

        st.write("")

        with st.expander("Model Details"):
            st.write(
                "**Training-derived high-risk accident-risk "
                f"cutoff:** {high_risk_threshold:.2f}"
            )
            st.write(
                "**Classification probability threshold:** "
                f"{probability_threshold:.2f}"
            )
            st.write(
                """
                HIGH, MEDIUM, and LOW review-priority categories
                are illustrative decision-support rules. They
                are not transportation-engineering thresholds
                learned directly from the dataset.
                """
            )

        if raw_estimated_risk < 0 or raw_estimated_risk > 1:
            st.caption(
                "The raw Linear Regression prediction fell "
                "outside the 0–1 range. The displayed value "
                "was bounded to 0–1 for presentation."
            )

        with st.expander("View Input Data"):
            st.dataframe(input_data, hide_index=True, use_container_width=True)


# =========================================================
# TAB 2 — BATCH CSV PRIORITIZATION
# =========================================================

with batch_tab:

    section_header(
        "Batch Road-Safety Prioritization",
        "Upload a CSV containing multiple road-condition observations. "
        "The tool will score each observation, assign a review priority, "
        "and rank the observations from highest to lowest predicted "
        "high-risk probability."
    )

    with st.expander("📋  Required CSV Columns"):

        req_col1, req_col2 = st.columns([1, 1.4], gap="large")

        with req_col1:
            st.dataframe(
                pd.DataFrame({"Column": EXPECTED_FEATURES}),
                hide_index=True,
                use_container_width=True
            )

        with req_col2:
            st.write(
                "Categorical values must use the same values as the training data:"
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

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        try:
            batch_df = pd.read_csv(uploaded_file)

            st.write(f"**Uploaded observations:** {len(batch_df):,}")

            with st.expander("Preview Uploaded Data"):
                st.dataframe(
                    batch_df.head(20), hide_index=True, use_container_width=True
                )

            validation_errors = validate_batch_data(batch_df)

            if validation_errors:
                st.error("The uploaded file cannot be scored.")

                for error in validation_errors:
                    st.write(f"- {error}")

            else:
                st.success("CSV validation passed.")

                results = score_batch_data(batch_df)

                # Summary
                section_header("Batch Prioritization Summary")

                total_observations = len(results)
                priority_counts = {
                    level: int(results["review_priority"].eq(level).sum())
                    for level in ["HIGH", "MEDIUM", "LOW"]
                }
                high_risk_flag_count = int(results["high_risk_flag"].sum())

                summary1, summary2, summary3, summary4 = st.columns(4)

                with summary1:
                    render_html(metric_card(
                        "Total Observations", f"{total_observations:,}",
                        "Rows scored", "#0F172A"
                    ))

                with summary2:
                    render_html(metric_card(
                        "High Priority", f"{priority_counts['HIGH']:,}",
                        f"Probability ≥ {HIGH_PRIORITY_PROBABILITY:.2f}",
                        PRIORITY_STYLES["HIGH"]["color"]
                    ))

                with summary3:
                    render_html(metric_card(
                        "Medium Priority", f"{priority_counts['MEDIUM']:,}",
                        f"Probability ≥ {probability_threshold:.2f}",
                        PRIORITY_STYLES["MEDIUM"]["color"]
                    ))

                with summary4:
                    render_html(metric_card(
                        "High-Risk Flags", f"{high_risk_flag_count:,}",
                        "Classifier positives", "#7C3AED"
                    ))

                distribution_bar(priority_counts, total_observations)

                # Display-only formatting; the download keeps raw values
                display_results = results.copy()
                display_results["estimated_accident_risk"] = (
                    display_results["estimated_accident_risk"].round(4)
                )
                display_results["high_risk_probability"] = (
                    display_results["high_risk_probability"].round(4)
                )
                display_results["high_risk_flag"] = (
                    display_results["high_risk_flag"].astype(bool)
                )
                display_results["review_priority"] = (
                    display_results["review_priority"].map(
                        lambda p: f'{PRIORITY_STYLES[p]["icon"]} {p}'
                    )
                )

                section_header(
                    "Top 10 Observations for Review",
                    "Highest predicted high-risk probability first."
                )

                st.dataframe(
                    display_results.head(10),
                    hide_index=True,
                    use_container_width=True,
                    column_config=results_column_config()
                )

                section_header("Prioritized Observations")

                st.dataframe(
                    display_results,
                    hide_index=True,
                    use_container_width=True,
                    column_config=results_column_config()
                )

                csv_output = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️  Download Prioritized Results",
                    data=csv_output,
                    file_name="road_safety_prioritized_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:
            st.error("An error occurred while processing the uploaded CSV.")
            st.exception(error)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

render_html(
    '<div class="footer">🚦 Road Accident Risk Prediction and Safety Prioritization '
    '<span>•</span> Proof-of-Concept Analytics Project</div>'
)