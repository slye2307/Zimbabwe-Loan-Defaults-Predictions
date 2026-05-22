from pathlib import Path
from datetime import date
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
FEATURES_PATH = APP_DIR / "features.pkl"
TRAIN_PATH = APP_DIR / "Train.csv"
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"
DEMO_URL = "https://zimbabwe-loan-defaults-predictions-fndx9su4lrdlyen2xfuahg.streamlit.app/"

PRIMARY = "#2E7D32"
SECONDARY = "#FFC107"
TEXT = "#111827"
MUTED_TEXT = "#4B5563"
SURFACE = "#FFFFFF"
BORDER = "#D1D5DB"

ASSISTANT_SYSTEM_PROMPT = """
You are Brighty, a warm credit-risk assistant inside a Streamlit app for
Zimbabwe loan default prediction. Help users understand the app, interpret
default-risk scores, think through borrower risk factors, and prepare sensible
follow-up questions for a credit officer.

Be friendly, plain-spoken, and practical. Keep answers concise. Do not claim a
loan should be approved or rejected automatically. Make it clear that the model
is a screening signal and that final credit decisions need human review.
"""

INITIAL_ASSISTANT_MESSAGE = """
Hi, I am Brighty. Ask me about the loan-risk score, the dashboard charts, or
what a credit officer should review next. I will keep it practical and human.
"""

CAT_COLS = [
    "loan_purpose",
    "marital_status",
    "employment_sector",
    "disbursement_channel",
    "client_gender",
    "collateral_type",
    "province",
    "payment_frequency",
]

DATE_COLS = [
    "date_approved",
    "date_disbursed",
    "first_payment_due",
    "maturity_date",
    "client_dob",
]

RAW_INPUT_COLS = [
    "product_code",
    "date_approved",
    "date_disbursed",
    "first_payment_due",
    "maturity_date",
    "amount_usd",
    "annual_rate_pct",
    "term_months",
    "payment_frequency",
    "loan_purpose",
    "client_gender",
    "client_dob",
    "marital_status",
    "num_dependents",
    "employment_sector",
    "months_at_employer",
    "monthly_income_usd",
    "existing_obligations",
    "collateral_type",
    "disbursement_channel",
    "province",
]

NUMERIC_INPUT_COLS = [
    "product_code",
    "amount_usd",
    "annual_rate_pct",
    "term_months",
    "num_dependents",
    "months_at_employer",
    "monthly_income_usd",
    "existing_obligations",
]

DROP_WEAK = [
    "age",
    "client_dob_day",
    "client_dob_month",
    "client_dob_year",
    "annual_rate_pct_missing",
    "num_dependents_missing",
    "client_gender",
    "months_at_employer_missing",
    "disbursement_channel",
    "payment_frequency",
    "loan_purpose",
    "date_disbursed_year",
]


st.set_page_config(
    page_title="Zimbabwe Loan Default Prediction",
    page_icon=":bank:",
    layout="wide",
)


st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: light;
    }}
    .stApp {{
        background: #f8fafc;
        color: {TEXT};
    }}
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        background: #f8fafc;
    }}
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    label,
    label p {{
        color: {TEXT} !important;
    }}
    .main-title {{
        color: {PRIMARY};
        font-weight: 800;
        margin-bottom: 0.2rem;
    }}
    .subtitle {{
        color: {MUTED_TEXT};
        font-size: 1.05rem;
        margin-bottom: 1.4rem;
    }}
    .demo-link {{
        display: inline-block;
        background: {PRIMARY};
        color: white !important;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-weight: 800;
        text-decoration: none !important;
        margin: 0 0 1.2rem 0;
    }}
    .demo-link:hover {{
        background: #1b5e20;
        color: white !important;
    }}
    [data-testid="stForm"] {{
        background: {SURFACE};
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(17, 24, 39, 0.04);
    }}
    .stNumberInput input,
    .stDateInput input,
    .stTextInput input,
    textarea,
    div[data-baseweb="select"] > div {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border-color: {BORDER} !important;
        box-shadow: none !important;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input {{
        color: {TEXT} !important;
    }}
    .stNumberInput input:focus,
    .stDateInput input:focus,
    .stTextInput input:focus,
    textarea:focus,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 1px {PRIMARY} !important;
    }}
    .metric-card {{
        border: 1px solid #e5e7eb;
        border-left: 6px solid {SECONDARY};
        border-radius: 8px;
        padding: 1rem;
        background: {SURFACE};
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.08);
    }}
    .metric-label {{
        color: #6b7280;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .metric-value {{
        color: {PRIMARY};
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }}
    .insight-card {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background: {SURFACE};
        height: 100%;
        box-shadow: 0 4px 14px rgba(17, 24, 39, 0.04);
    }}
    .insight-card strong {{
        color: {PRIMARY};
    }}
    .insight-title {{
        color: {PRIMARY};
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }}
    .insight-card p,
    .insight-card li {{
        color: {MUTED_TEXT} !important;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    .review-box {{
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background: {SURFACE};
        box-shadow: 0 4px 14px rgba(17, 24, 39, 0.04);
        margin-top: 0.8rem;
    }}
    .review-box li {{
        color: {TEXT} !important;
        margin-bottom: 0.35rem;
    }}
    .disclaimer {{
        border-left: 5px solid {SECONDARY};
        background: #fffbeb;
        color: {TEXT};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }}
    .disclaimer strong {{
        color: #92400e;
    }}
    .risk-box {{
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }}
    .risk-box * {{
        color: inherit !important;
    }}
    .risk-caption {{
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.55rem;
    }}
    .risk-prob {{
        font-size: 3rem;
        font-weight: 900;
        line-height: 1;
    }}
    .risk-label {{
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }}
    .section-header {{
        color: {PRIMARY};
        font-weight: 800;
        margin-top: 1rem;
    }}
    div.stButton > button:first-child,
    div.stFormSubmitButton > button {{
        background: {PRIMARY};
        color: white !important;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.7rem 1.4rem;
    }}
    div.stButton > button:first-child:hover,
    div.stFormSubmitButton > button:hover {{
        background: #1b5e20;
        color: white !important;
        border: 0;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY} 0%, #1b5e20 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    .assistant-hero {{
        background: linear-gradient(135deg, #0f766e 0%, {PRIMARY} 58%, #f59e0b 100%);
        border-radius: 8px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(15, 118, 110, 0.18);
    }}
    .assistant-hero h1 {{
        color: white !important;
        font-size: 2rem;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
        font-weight: 850;
    }}
    .assistant-hero p {{
        color: rgba(255, 255, 255, 0.94) !important;
        margin: 0;
        max-width: 920px;
        font-size: 1rem;
    }}
    .assistant-panel {{
        background: {SURFACE};
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(17, 24, 39, 0.04);
    }}
    .assistant-panel strong {{
        color: {PRIMARY};
    }}
    .status-pill {{
        display: inline-block;
        border-radius: 999px;
        padding: 0.22rem 0.7rem;
        background: #ecfdf5;
        color: #065f46 !important;
        border: 1px solid #a7f3d0;
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 0.7rem;
    }}
    .status-pill.status-guide {{
        background: #fffbeb;
        color: #92400e !important;
        border-color: #fcd34d;
    }}
    .tiny-muted {{
        color: {MUTED_TEXT} !important;
        font-size: 0.9rem;
        line-height: 1.5;
    }}
    [data-testid="stChatMessage"] {{
        background: {SURFACE};
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.7rem;
        box-shadow: 0 3px 12px rgba(17, 24, 39, 0.03);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_and_features():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURES_PATH}")

    import pickle

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with FEATURES_PATH.open("rb") as f:
        features = pickle.load(f)
    return model, features


@st.cache_data(show_spinner=False)
def load_train_data() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing Train.csv beside app.py: {TRAIN_PATH}")
    return pd.read_csv(TRAIN_PATH)


@st.cache_resource(show_spinner=False)
def load_label_encoders() -> dict:
    train = load_train_data().copy()
    mode_cols = [
        "loan_purpose",
        "marital_status",
        "employment_sector",
        "disbursement_channel",
        "client_gender",
        "payment_frequency",
        "province",
    ]

    for col in mode_cols:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].mode()[0])
    if "collateral_type" in train.columns:
        train["collateral_type"] = train["collateral_type"].fillna("Unknown")

    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        values = train[col].astype(str).tolist() if col in train.columns else ["Unknown"]
        le.fit(pd.Series(values).astype(str))
        encoders[col] = le
    return encoders


def encode_value(value: str, encoder: LabelEncoder) -> int:
    value = str(value)
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    for fallback in ["Unknown", "Informal_Sector", "Personal", "Monthly", "Bank_Transfer", "Harare", "Single"]:
        if fallback in encoder.classes_:
            return int(encoder.transform([fallback])[0])
    return 0


def parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d-%b-%y"]:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing],
            format=fmt,
            errors="coerce",
        )

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing],
            format="mixed",
            errors="coerce",
        )
    return parsed


@st.cache_data(show_spinner=False)
def load_input_defaults() -> dict:
    try:
        train = load_train_data()
    except Exception:
        train = pd.DataFrame()

    numeric_defaults = {}
    for col in NUMERIC_INPUT_COLS:
        if col in train.columns:
            values = pd.to_numeric(train[col], errors="coerce")
            median = values.median()
            numeric_defaults[col] = float(median) if pd.notna(median) else 0.0
        else:
            numeric_defaults[col] = 0.0

    categorical_defaults = {}
    for col in CAT_COLS:
        if col in train.columns:
            values = train[col].dropna().astype(str)
            categorical_defaults[col] = values.mode().iloc[0] if not values.empty else "Unknown"
        else:
            categorical_defaults[col] = "Unknown"
    categorical_defaults["collateral_type"] = "Unknown"

    date_defaults = {}
    for col in DATE_COLS:
        if col in train.columns:
            values = parse_date_series(train[col]).dropna().sort_values()
            date_defaults[col] = values.iloc[len(values) // 2] if not values.empty else pd.Timestamp("2026-01-01")
        else:
            date_defaults[col] = pd.Timestamp("2026-01-01")

    return {
        "numeric": numeric_defaults,
        "categorical": categorical_defaults,
        "dates": date_defaults,
    }


def normalize_inputs(raw: dict) -> dict:
    employment_map = {
        "Retail": "Retail_Trade",
        "Health": "Healthcare",
        "Self Employed": "Informal_Sector",
        "Informal": "Informal_Sector",
        "Other": "Other",
    }
    province_map = {
        "Mashonaland Central": "Mashonaland_Central",
        "Mashonaland East": "Mashonaland_East",
        "Mashonaland West": "Mashonaland_West",
        "Matabeleland North": "Matabeleland_North",
        "Matabeleland South": "Matabeleland_South",
    }
    loan_purpose_map = {
        "Business": "Business_Expansion",
        "Agriculture": "Farming_Inputs",
        "Education": "School_Fees",
        "Housing": "Home_Improvement",
        "Other": "Other",
    }
    disbursement_map = {
        "Bank": "Bank_Transfer",
        "Mobile Money": "EcoCash",
        "Cheque": "Bank_Transfer",
    }

    raw["employment_sector"] = employment_map.get(
        raw["employment_sector"], raw["employment_sector"]
    )
    raw["province"] = province_map.get(raw["province"], raw["province"])
    raw["loan_purpose"] = loan_purpose_map.get(raw["loan_purpose"], raw["loan_purpose"])
    raw["disbursement_channel"] = disbursement_map.get(
        raw["disbursement_channel"], raw["disbursement_channel"]
    )
    if raw["collateral_type"] in ["None", "Equipment", "Other"]:
        raw["collateral_type"] = "Unknown"
    return raw


def normalize_input_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in RAW_INPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan

    mappings = {
        "employment_sector": {
            "Retail": "Retail_Trade",
            "Health": "Healthcare",
            "Self Employed": "Informal_Sector",
            "Informal": "Informal_Sector",
        },
        "province": {
            "Mashonaland Central": "Mashonaland_Central",
            "Mashonaland East": "Mashonaland_East",
            "Mashonaland West": "Mashonaland_West",
            "Matabeleland North": "Matabeleland_North",
            "Matabeleland South": "Matabeleland_South",
        },
        "loan_purpose": {
            "Business": "Business_Expansion",
            "Agriculture": "Farming_Inputs",
            "Education": "School_Fees",
            "Housing": "Home_Improvement",
        },
        "disbursement_channel": {
            "Bank": "Bank_Transfer",
            "Mobile Money": "EcoCash",
            "Cheque": "Bank_Transfer",
        },
        "collateral_type": {
            "None": "Unknown",
            "Equipment": "Unknown",
            "Other": "Unknown",
        },
    }

    for col, mapping in mappings.items():
        df[col] = df[col].replace(mapping)

    return df


def build_feature_frame(raw_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    encoders = load_label_encoders()
    defaults = load_input_defaults()
    df = normalize_input_frame(raw_df)

    missing_cols = [
        "collateral_type",
        "monthly_income_usd",
        "months_at_employer",
        "num_dependents",
        "annual_rate_pct",
    ]
    for col in missing_cols:
        df[f"{col}_missing"] = df[col].isna().astype(int)

    for col in NUMERIC_INPUT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(defaults["numeric"].get(col, 0.0))

    for col in CAT_COLS:
        df[col] = df[col].fillna(defaults["categorical"].get(col, "Unknown")).astype(str)

    for col in DATE_COLS:
        parsed = parse_date_series(df[col])
        df[col] = parsed.fillna(defaults["dates"].get(col, pd.Timestamp("2026-01-01")))

    for col in DATE_COLS:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day

    df["age"] = 2026 - df["client_dob_year"]
    df["loan_duration"] = (
        (df["maturity_date_year"] - df["date_approved_year"]) * 12
        + (df["maturity_date_month"] - df["date_approved_month"])
    )

    df["amount_to_income"] = df["amount_usd"] / (df["monthly_income_usd"] + 1)
    df["monthly_payment"] = df["amount_usd"] / (df["term_months"] + 1)
    df["payment_to_income"] = df["monthly_payment"] / (df["monthly_income_usd"] + 1)
    df["obligations_to_income"] = df["existing_obligations"] / (
        df["monthly_income_usd"] + 1
    )
    df["rate_to_income"] = df["annual_rate_pct"] / (df["monthly_income_usd"] + 1)
    df["log_income"] = np.log1p(df["monthly_income_usd"].clip(lower=0))
    df["log_amount"] = np.log1p(df["amount_usd"].clip(lower=0))
    df["log_obligations"] = np.log1p(df["existing_obligations"].clip(lower=0))
    df["credit_stress"] = df["existing_obligations"] + df["monthly_payment"] / (
        df["monthly_income_usd"] + 1
    )
    df["income_per_dependent"] = df["monthly_income_usd"] / (df["num_dependents"] + 1)
    df["job_stability"] = df["months_at_employer"] / (df["monthly_income_usd"] + 1)

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda value: encode_value(value, encoders[col]))

    df = df.drop(columns=[col for col in DATE_COLS if col in df.columns])
    df = df.drop(columns=[col for col in DROP_WEAK if col in df.columns])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


def build_features(raw_inputs: dict, feature_names: list[str]) -> pd.DataFrame:
    return build_feature_frame(pd.DataFrame([raw_inputs]), feature_names)


def risk_label(probability: float) -> tuple[str, str, str]:
    if probability < 0.35:
        level, color, text_color = "LOW RISK", "#2E7D32", "#FFFFFF"
    elif probability < 0.65:
        level, color, text_color = "MEDIUM RISK", "#F59E0B", TEXT
    else:
        level, color, text_color = "HIGH RISK", "#C62828", "#FFFFFF"
    return level, color, text_color


def risk_details(probability: float, inputs: dict) -> tuple[str, str, str, list[str]]:
    level, color, text_color = risk_label(probability)

    factors = []
    income = inputs["monthly_income_usd"]
    amount = inputs["amount_usd"]
    obligations = inputs["existing_obligations"]
    term = inputs["term_months"]

    if amount / (income + 1) > 8:
        factors.append("High loan amount relative to monthly income")
    if obligations / (income + 1) > 0.5:
        factors.append("Existing obligations are high for stated income")
    if inputs["annual_rate_pct"] > 80:
        factors.append("High annual interest rate")
    if inputs["months_at_employer"] < 12:
        factors.append("Short employment history")
    if term <= 3:
        factors.append("Short repayment term may increase repayment pressure")
    if inputs["collateral_type"] in ["Unknown", "None", "Other"]:
        factors.append("Limited or unclear collateral")
    if not factors:
        factors.append("No major risk concentration detected from entered fields")
    return level, color, text_color, factors


def review_recommendations(probability: float, inputs: dict, factors: list[str]) -> list[str]:
    level, _, _ = risk_label(probability)
    recommendations = [
        "Verify income, existing obligations, and identity documents before final approval.",
        "Compare the repayment amount with disposable income after other obligations.",
    ]

    if any("collateral" in factor.lower() for factor in factors):
        recommendations.append("Confirm collateral ownership, valuation, and enforceability.")
    if inputs["months_at_employer"] < 12:
        recommendations.append("Request employment confirmation or recent cash-flow evidence.")
    if inputs["annual_rate_pct"] > 80:
        recommendations.append("Check whether pricing is driving repayment pressure.")
    if inputs["amount_usd"] / (inputs["monthly_income_usd"] + 1) > 8:
        recommendations.append("Consider a lower amount, stronger guarantor, or staged disbursement.")
    if level == "HIGH RISK":
        recommendations.append("Escalate to a senior credit review before any approval decision.")
    elif level == "MEDIUM RISK":
        recommendations.append("Review borderline factors and document the reason for any decision.")
    else:
        recommendations.append("Keep the standard human review even when the model signal is low risk.")

    return list(dict.fromkeys(recommendations))


def format_prediction_context(context: dict | None) -> str:
    if not context:
        return ""

    inputs = context.get("inputs", {})
    factors = context.get("factors", [])
    recommendations = context.get("recommendations", [])
    details = [
        f"Risk level: {context.get('level', 'Unknown')}",
        f"Default probability: {context.get('probability_pct', 'Unknown')}",
        f"Amount: USD {inputs.get('amount_usd', 'Unknown')}",
        f"Monthly income: USD {inputs.get('monthly_income_usd', 'Unknown')}",
        f"Existing obligations: USD {inputs.get('existing_obligations', 'Unknown')}",
        f"Term months: {inputs.get('term_months', 'Unknown')}",
        f"Employment sector: {inputs.get('employment_sector', 'Unknown')}",
        f"Collateral: {inputs.get('collateral_type', 'Unknown')}",
        "Key risk factors: " + "; ".join(factors),
        "Recommended review steps: " + "; ".join(recommendations),
    ]
    return "\n".join(details)


def render_prediction_result(context: dict) -> None:
    probability = context["probability"]
    level = context["level"]
    color = context["color"]
    text_color = context["text_color"]
    factors = context["factors"]
    recommendations = context["recommendations"]

    st.markdown('<h3 class="section-header">Latest Prediction Result</h3>', unsafe_allow_html=True)
    result_col, detail_col = st.columns([1, 1.25])
    with result_col:
        st.markdown(
            f"""
            <div class="risk-box" style="background:{color}; color:{text_color};">
                <div class="risk-caption">Estimated default probability</div>
                <div class="risk-prob">{probability * 100:.1f}%</div>
                <div class="risk-label">{level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(probability)
    with detail_col:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Decision Support Summary</div>
                <p>
                This applicant is currently classified as <strong>{level.title()}</strong>.
                The model result should support a credit officer's review, not replace it.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<h3 class="section-header">Key Risk Factors</h3>', unsafe_allow_html=True)
    for factor in factors:
        st.write(f"- {factor}")

    recommendation_items = "".join(f"<li>{item}</li>" for item in recommendations)
    st.markdown(
        f"""
        <div class="review-box">
            <div class="insight-title">Recommended Human Review Steps</div>
            <ul>{recommendation_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Ask Brighty to explain this result", use_container_width=True):
            st.session_state.assistant_pending_prompt = (
                "Explain the latest prediction result in plain English. "
                "Mention the risk level, key risk factors, and what a credit officer should review next."
            )
            st.session_state.page = "AI Assistant"
            st.rerun()
    with c2:
        st.info(
            "Use this score as a screening signal. A credit officer should still review the applicant context before a final decision."
        )


def home_page() -> None:
    st.markdown(
        '<h1 class="main-title">Zimbabwe Loan Default Prediction System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="subtitle">
        Helping credit teams make faster, fairer, and more explainable
        loan-risk decisions for Zimbabwe's lending market. The system estimates
        default probability from borrower, product, collateral, location, and
        repayment information, then keeps a human reviewer in the loop.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<a class="demo-link" href="{DEMO_URL}" target="_blank">Access the live demo</a>',
        unsafe_allow_html=True,
    )

    try:
        train = load_train_data()
        total_loans = f"{len(train):,}"
        default_rate = f"{train['Target'].mean() * 100:.1f}%"
    except Exception:
        total_loans = "Unavailable"
        default_rate = "Unavailable"

    cols = st.columns(4)
    with cols[0]:
        metric_card("Total Loans Analyzed", total_loans)
    with cols[1]:
        metric_card("Default Rate", default_rate)
    with cols[2]:
        metric_card("Model Accuracy", "61.3%")
    with cols[3]:
        metric_card("AUC Score", "0.6955")

    st.markdown('<h3 class="section-header">System Overview</h3>', unsafe_allow_html=True)
    st.write(
        """
        The application combines the saved CatBoost model with the same feature
        engineering used during training. Analysts can score one application,
        upload a portfolio file, review the probability of default, ask Brighty
        for a plain-English explanation, and inspect portfolio-level patterns
        from the training data.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">Score</div>
                <p>Estimate default probability for a single borrower or a CSV portfolio.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">Explain</div>
                <p>Turn model output into clear risk factors and human review steps.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="insight-card">
                <div class="insight-title">Review</div>
                <p>Compare results with portfolio patterns before making a final decision.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def predictor_page() -> None:
    st.markdown(
        '<h1 class="main-title">Loan Risk Predictor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="subtitle">Enter borrower and loan details to estimate default risk.</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="disclaimer">
        <strong>Human review required.</strong> This app is a screening tool, not an automatic approval or rejection decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading CatBoost model and feature schema..."):
            model, feature_names = load_model_and_features()
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        return

    with st.form("prediction_form"):
        st.markdown('<h3 class="section-header">Loan Details</h3>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            amount_usd = st.number_input("Amount USD", min_value=0.0, value=1000.0, step=50.0)
            annual_rate_pct = st.number_input("Annual Rate %", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
            term_months = st.number_input("Term Months", min_value=0, value=12, step=1)
            product_code = st.number_input("Product Code", min_value=0, value=1, step=1)
        with c2:
            monthly_income_usd = st.number_input("Monthly Income USD", min_value=0.0, value=450.0, step=25.0)
            num_dependents = st.number_input("Number of Dependents", min_value=0, value=2, step=1)
            months_at_employer = st.number_input("Months at Employer", min_value=0, value=36, step=1)
            existing_obligations = st.number_input("Existing Obligations", min_value=0.0, value=0.0, step=10.0)
        with c3:
            employment_sector = st.selectbox("Employment Sector", ["Agriculture", "Mining", "Manufacturing", "Construction", "Retail", "Transport", "Finance", "Education", "Health", "Government", "NGO", "Self Employed", "Informal", "Other"])
            collateral_type = st.selectbox("Collateral Type", ["Unknown", "Property", "Vehicle", "Equipment", "Savings", "Guarantor", "None", "Other"])
            province = st.selectbox("Province", ["Harare", "Bulawayo", "Manicaland", "Mashonaland Central", "Mashonaland East", "Mashonaland West", "Masvingo", "Matabeleland North", "Matabeleland South", "Midlands"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

        st.markdown('<h3 class="section-header">Customer and Repayment Details</h3>', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        with c4:
            client_gender = st.selectbox("Client Gender", ["Male", "Female"])
            loan_purpose = st.selectbox("Loan Purpose", ["Business", "Personal", "Agriculture", "Education", "Housing", "Medical", "Vehicle", "Equipment", "Other"])
            payment_frequency = st.selectbox("Payment Frequency", ["Monthly", "Weekly", "Bi-Weekly", "Quarterly"])
            disbursement_channel = st.selectbox("Disbursement Channel", ["Bank", "Mobile Money", "Cash", "Cheque"])
        with c5:
            date_approved = st.date_input("Date Approved", value=date(2026, 1, 15))
            date_disbursed = st.date_input("Date Disbursed", value=date(2026, 1, 20))
            first_payment_due = st.date_input("First Payment Due", value=date(2026, 2, 20))
        with c6:
            maturity_date = st.date_input("Maturity Date", value=date(2027, 1, 20))
            client_dob = st.date_input("Client DOB", value=date(1988, 1, 1))

        submitted = st.form_submit_button("Predict Default Risk")

    if submitted:
        raw_inputs = {
            "product_code": int(product_code),
            "amount_usd": float(amount_usd),
            "annual_rate_pct": float(annual_rate_pct),
            "term_months": int(term_months),
            "loan_purpose": loan_purpose,
            "marital_status": marital_status,
            "num_dependents": int(num_dependents),
            "employment_sector": employment_sector,
            "months_at_employer": int(months_at_employer),
            "monthly_income_usd": float(monthly_income_usd),
            "existing_obligations": float(existing_obligations),
            "disbursement_channel": disbursement_channel,
            "client_gender": client_gender,
            "collateral_type": collateral_type,
            "province": province,
            "payment_frequency": payment_frequency,
            "date_approved": date_approved,
            "date_disbursed": date_disbursed,
            "first_payment_due": first_payment_due,
            "maturity_date": maturity_date,
            "client_dob": client_dob,
        }

        try:
            with st.spinner("Engineering features and scoring loan application..."):
                model_input = build_features(raw_inputs, feature_names)
                probability = float(model.predict_proba(model_input)[:, 1][0])
                level, color, text_color, factors = risk_details(probability, raw_inputs)
                recommendations = review_recommendations(probability, raw_inputs, factors)

            st.session_state.latest_prediction_context = {
                "probability": probability,
                "probability_pct": f"{probability * 100:.1f}%",
                "level": level,
                "color": color,
                "text_color": text_color,
                "factors": factors,
                "recommendations": recommendations,
                "inputs": raw_inputs,
            }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "latest_prediction_context" in st.session_state:
        render_prediction_result(st.session_state.latest_prediction_context)


def score_dataframe(raw_df: pd.DataFrame, model, feature_names: list[str]) -> pd.DataFrame:
    model_input = build_feature_frame(raw_df, feature_names)
    probabilities = model.predict_proba(model_input)[:, 1]

    results = raw_df.copy()
    results["Target"] = probabilities
    results["Default Probability"] = (probabilities * 100).round(1).astype(str) + "%"
    results["Risk Level"] = [risk_label(float(prob))[0] for prob in probabilities]
    return results


def batch_upload_page() -> None:
    st.markdown('<h1 class="main-title">Batch Loan Scoring</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Upload a borrower CSV and score many applications in one run.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="disclaimer">
        <strong>Human review required.</strong> Batch scores should be treated as a triage queue for credit officers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading CatBoost model and feature schema..."):
            model, feature_names = load_model_and_features()
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        return

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.caption("Expected columns match `Test.csv`: borrower, loan, dates, collateral, channel, and province fields.")

    with st.expander("Required columns"):
        st.write(", ".join(RAW_INPUT_COLS))

    if uploaded_file is None:
        return

    try:
        batch = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    missing_cols = [col for col in RAW_INPUT_COLS if col not in batch.columns]
    if missing_cols:
        st.error("Missing required columns: " + ", ".join(missing_cols))
        return

    try:
        with st.spinner("Scoring uploaded applications..."):
            results = score_dataframe(batch, model, feature_names)
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
        return

    total = len(results)
    avg_probability = results["Target"].mean() if total else 0
    high_risk_count = int((results["Risk Level"] == "HIGH RISK").sum())
    medium_risk_count = int((results["Risk Level"] == "MEDIUM RISK").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Applications Scored", f"{total:,}")
    with c2:
        metric_card("Average Risk", f"{avg_probability * 100:.1f}%")
    with c3:
        metric_card("High Risk", f"{high_risk_count:,}")
    with c4:
        metric_card("Medium Risk", f"{medium_risk_count:,}")

    chart_col, queue_col = st.columns([1, 1])
    with chart_col:
        fig_hist = px.histogram(
            results,
            x="Target",
            nbins=30,
            title="Default Probability Distribution",
            color_discrete_sequence=[PRIMARY],
        )
        fig_hist.update_layout(xaxis_title="Default Probability", yaxis_title="Applications")
        st.plotly_chart(fig_hist, width="stretch")
    with queue_col:
        risk_counts = results["Risk Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Applications"]
        fig_risk = px.bar(
            risk_counts,
            x="Risk Level",
            y="Applications",
            title="Risk Queue",
            color="Risk Level",
            color_discrete_map={
                "LOW RISK": PRIMARY,
                "MEDIUM RISK": "#F59E0B",
                "HIGH RISK": "#C62828",
            },
        )
        st.plotly_chart(fig_risk, width="stretch")

    st.markdown('<h3 class="section-header">Scored Applications</h3>', unsafe_allow_html=True)
    display_cols = [
        "ID",
        "Target",
        "Default Probability",
        "Risk Level",
        "amount_usd",
        "monthly_income_usd",
        "existing_obligations",
        "term_months",
        "province",
        "collateral_type",
    ]
    display_cols = [col for col in display_cols if col in results.columns]
    st.dataframe(
        results.sort_values("Target", ascending=False)[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    full_csv = results.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download full scored CSV",
            data=full_csv,
            file_name="loan_risk_predictions_full.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        if "ID" in results.columns:
            submission_csv = results[["ID", "Target"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download ID + Target CSV",
                data=submission_csv,
                file_name="loan_risk_submission.csv",
                mime="text/csv",
                use_container_width=True,
            )


def dashboard_page() -> None:
    st.markdown('<h1 class="main-title">Portfolio Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Training-data insights for loan default behavior.</p>', unsafe_allow_html=True)

    try:
        with st.spinner("Loading training data..."):
            train = load_train_data()
    except Exception as exc:
        st.error(f"Could not load Train.csv: {exc}")
        return

    def default_rate_chart(col: str, title: str):
        data = (
            train.assign(**{col: train[col].fillna("Unknown").astype(str)})
            .groupby(col, as_index=False)["Target"]
            .mean()
        )
        data["Default Rate"] = data["Target"] * 100
        fig = px.bar(
            data.sort_values("Default Rate", ascending=False),
            x=col,
            y="Default Rate",
            title=title,
            color="Default Rate",
            color_continuous_scale=["#A5D6A7", SECONDARY, "#C62828"],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Default Rate (%)")
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(default_rate_chart("employment_sector", "Default Rate by Employment Sector"), width="stretch")
    with c2:
        st.plotly_chart(default_rate_chart("province", "Default Rate by Province"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(default_rate_chart("collateral_type", "Default Rate by Collateral Type"), width="stretch")
    with c4:
        fig_amount = px.histogram(
            train,
            x="amount_usd",
            nbins=45,
            title="Loan Amount Distribution",
            color_discrete_sequence=[PRIMARY],
        )
        fig_amount.update_layout(xaxis_title="Loan Amount USD", yaxis_title="Count")
        st.plotly_chart(fig_amount, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        gender_counts = train["client_gender"].fillna("Unknown").value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig_gender = px.pie(
            gender_counts,
            names="Gender",
            values="Count",
            title="Gender Distribution",
            color_discrete_sequence=[PRIMARY, SECONDARY, "#9CA3AF"],
        )
        st.plotly_chart(fig_gender, width="stretch")
    with c6:
        term_data = train.groupby("term_months", as_index=False)["Target"].mean()
        term_data["Default Rate"] = term_data["Target"] * 100
        fig_term = px.line(
            term_data,
            x="term_months",
            y="Default Rate",
            markers=True,
            title="Default Rate by Loan Term",
            color_discrete_sequence=[PRIMARY],
        )
        fig_term.update_layout(xaxis_title="Term Months", yaxis_title="Default Rate (%)")
        st.plotly_chart(fig_term, width="stretch")


def get_config_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    try:
        secret_value = st.secrets.get(name)
        if secret_value:
            value = str(secret_value)
    except Exception:
        pass
    return value or default


def local_assistant_reply(question: str, prediction_context: dict | None = None) -> str:
    q = question.lower()
    prefix = (
        "I am in built-in guide mode because no OpenAI API key is configured yet. "
    )

    if prediction_context and any(
        word in q for word in ["latest", "this prediction", "explain", "result", "review next"]
    ):
        context_text = format_prediction_context(prediction_context)
        return (
            prefix
            + "Here is the latest prediction in plain English:\n\n"
            + context_text
            + "\n\nA credit officer should use this as a review queue signal, not as an automatic decision."
        )

    if any(word in q for word in ["api", "key", "openai", "setup", "configure"]):
        return (
            prefix
            + "To enable full AI chat, add `OPENAI_API_KEY` to Streamlit secrets "
            + "or set it as an environment variable. You can also set "
            + "`OPENAI_MODEL`; otherwise the app uses `gpt-4o-mini`."
        )
    if any(word in q for word in ["default", "risk", "score", "probability", "result"]):
        return (
            prefix
            + "The predictor estimates the probability that a borrower may default. "
            + "Low risk is below 35%, medium risk is 35% to 65%, and high risk is "
            + "above 65%. Treat the score as a screening signal, then review income, "
            + "obligations, collateral, employment stability, and repayment terms."
        )
    if any(word in q for word in ["reduce", "improve", "lower", "better"]):
        return (
            prefix
            + "Ways to reduce risk usually include lowering the loan amount, extending "
            + "the term when appropriate, reducing existing obligations, adding stronger "
            + "collateral, or verifying stable income and employment history."
        )
    if any(word in q for word in ["dashboard", "chart", "portfolio", "data"]):
        return (
            prefix
            + "The dashboard summarizes training-data patterns by employment sector, "
            + "province, collateral type, amount, gender, and term. Use it to spot "
            + "portfolio trends, not to judge one borrower by a group average alone."
        )
    if any(word in q for word in ["model", "catboost", "accuracy", "auc"]):
        return (
            prefix
            + "The app loads a saved CatBoost model plus the feature list used during "
            + "training. The Home page shows the current reported accuracy and AUC. "
            + "Those metrics describe screening performance, not guaranteed outcomes."
        )
    if any(word in q for word in ["run", "start", "streamlit", "local"]):
        return (
            prefix
            + "To run the app locally, open PowerShell in the project folder, activate "
            + "the virtual environment, then run `streamlit run app.py`."
        )

    return (
        prefix
        + "I can answer common questions about the predictor, dashboard, risk levels, "
        + "and app setup. For richer free-form answers, configure `OPENAI_API_KEY`."
    )


def generate_assistant_reply(
    messages: list[dict[str, str]],
    prediction_context: dict | None = None,
) -> str:
    api_key = get_config_value("OPENAI_API_KEY")
    model = get_config_value("OPENAI_MODEL", OPENAI_MODEL_DEFAULT)

    if not api_key:
        return local_assistant_reply(messages[-1]["content"], prediction_context)

    try:
        from openai import OpenAI
    except ImportError:
        return (
            "The app found an API key, but the `openai` package is not installed. "
            "Run `python -m pip install -r requirements.txt`, then restart Streamlit."
        )

    try:
        client = OpenAI(api_key=api_key)
        instructions = ASSISTANT_SYSTEM_PROMPT
        if prediction_context:
            instructions += (
                "\n\nLatest prediction context available to explain:\n"
                + format_prediction_context(prediction_context)
            )
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=messages[-10:],
            store=False,
        )
        answer = response.output_text.strip()
        return answer or "I did not get a readable answer back. Please try again."
    except Exception as exc:
        return (
            "I could not reach the AI service right now. "
            f"Technical detail: {exc}"
        )


def assistant_page() -> None:
    api_key = get_config_value("OPENAI_API_KEY")
    model = get_config_value("OPENAI_MODEL", OPENAI_MODEL_DEFAULT)
    mode_label = "Brighty is ready" if api_key else "Guide mode"
    status_class = "status-ready" if api_key else "status-guide"
    status_note = (
        f"Ask natural questions about risk scores, borrower review, and dashboard patterns. Model: {model}."
        if api_key
        else "Add OPENAI_API_KEY in Streamlit secrets to unlock full AI chat. Until then, Brighty can still answer common app questions."
    )
    prediction_context = st.session_state.get("latest_prediction_context")

    st.markdown(
        """
        <div class="assistant-hero">
            <h1>Ask Brighty</h1>
            <p>
            A friendly assistant for explaining credit-risk scores, dashboard
            patterns, borrower factors, and next review steps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_col, side_col = st.columns([2.3, 1])

    with side_col:
        st.markdown(
            f"""
            <div class="assistant-panel">
                <div class="insight-title">Assistant Status</div>
                <div class="status-pill {status_class}">{mode_label}</div>
                <p class="tiny-muted">
                {status_note}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if prediction_context:
            st.markdown(
                f"""
                <div class="assistant-panel" style="margin-top:0.8rem;">
                    <div class="insight-title">Latest Prediction</div>
                    <p class="tiny-muted">
                    <strong>{prediction_context["level"].title()}</strong><br>
                    Default probability: <strong>{prediction_context["probability_pct"]}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("#### Suggested questions")
        quick_questions = []
        if prediction_context:
            quick_questions.append("Explain the latest prediction")
        quick_questions += [
            "Explain a high-risk score",
            "Suggest ways to lower risk",
            "List the next review checks",
            "How do I enable full AI chat?",
        ]
        for idx, item in enumerate(quick_questions):
            if st.button(item, key=f"quick_question_{idx}", use_container_width=True):
                st.session_state.assistant_pending_prompt = item

        if st.button("Start over", use_container_width=True):
            st.session_state.assistant_messages = [
                {"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE.strip()}
            ]
            st.rerun()

    with chat_col:
        if "assistant_messages" not in st.session_state:
            st.session_state.assistant_messages = [
                {"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE.strip()}
            ]

        for message in st.session_state.assistant_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about risk scores, borrower review, or the dashboard")
        pending_prompt = st.session_state.pop("assistant_pending_prompt", None)
        if pending_prompt and not prompt:
            prompt = pending_prompt

        if prompt:
            st.session_state.assistant_messages.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking through the question..."):
                    reply = generate_assistant_reply(
                        st.session_state.assistant_messages,
                        prediction_context,
                    )
                st.markdown(reply)

            st.session_state.assistant_messages.append(
                {"role": "assistant", "content": reply}
            )


def main() -> None:
    st.sidebar.title("Navigation")
    pages = ["Home", "Loan Risk Predictor", "Batch Upload", "Dashboard", "AI Assistant"]
    if st.session_state.get("page") not in pages:
        st.session_state.page = "Home"
    page = st.sidebar.radio(
        "Select Page",
        pages,
        key="page",
    )

    if page == "Home":
        home_page()
    elif page == "Loan Risk Predictor":
        predictor_page()
    elif page == "Batch Upload":
        batch_upload_page()
    elif page == "Dashboard":
        dashboard_page()
    else:
        assistant_page()


if __name__ == "__main__":
    main()
