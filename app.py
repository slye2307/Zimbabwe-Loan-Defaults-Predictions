from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import LabelEncoder


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
FEATURES_PATH = APP_DIR / "features.pkl"
TRAIN_PATH = APP_DIR / "Train.csv"

PRIMARY = "#2E7D32"
SECONDARY = "#FFC107"
TEXT = "#111827"
MUTED_TEXT = "#4B5563"
SURFACE = "#FFFFFF"
BORDER = "#D1D5DB"

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


def build_features(raw_inputs: dict, feature_names: list[str]) -> pd.DataFrame:
    encoders = load_label_encoders()
    row = normalize_inputs(raw_inputs.copy())
    df = pd.DataFrame([row])

    missing_cols = [
        "collateral_type",
        "monthly_income_usd",
        "months_at_employer",
        "num_dependents",
        "annual_rate_pct",
    ]
    for col in missing_cols:
        df[f"{col}_missing"] = 0

    for col in DATE_COLS:
        dt = pd.to_datetime(df.loc[0, col])
        df[f"{col}_year"] = dt.year
        df[f"{col}_month"] = dt.month
        df[f"{col}_day"] = dt.day

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
    df["log_income"] = np.log1p(df["monthly_income_usd"])
    df["log_amount"] = np.log1p(df["amount_usd"])
    df["log_obligations"] = np.log1p(df["existing_obligations"])
    df["credit_stress"] = df["existing_obligations"] + df["monthly_payment"] / (
        df["monthly_income_usd"] + 1
    )
    df["income_per_dependent"] = df["monthly_income_usd"] / (df["num_dependents"] + 1)
    df["job_stability"] = df["months_at_employer"] / (df["monthly_income_usd"] + 1)

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = encode_value(df.loc[0, col], encoders[col])

    df = df.drop(columns=[col for col in DATE_COLS if col in df.columns])
    df = df.drop(columns=[col for col in DROP_WEAK if col in df.columns])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]
    return df


def risk_details(probability: float, inputs: dict) -> tuple[str, str, str, list[str]]:
    if probability < 0.35:
        level, color, text_color = "LOW RISK", "#2E7D32", "#FFFFFF"
    elif probability < 0.65:
        level, color, text_color = "MEDIUM RISK", "#F59E0B", TEXT
    else:
        level, color, text_color = "HIGH RISK", "#C62828", "#FFFFFF"

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


def home_page() -> None:
    st.markdown(
        '<h1 class="main-title">Zimbabwe Loan Default Prediction System</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <p class="subtitle">
        A professional credit-risk screening tool for Zimbabwe's banking sector,
        built for the IndabaX Zimbabwe 2026 Hackathon. The system estimates loan
        default probability from borrower, product, collateral, location, and
        repayment information.
        </p>
        """,
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
        engineering used during training. Analysts can score a new loan, review
        the probability of default, and inspect portfolio-level patterns from
        the training data.
        """
    )


def predictor_page() -> None:
    st.markdown(
        '<h1 class="main-title">Loan Risk Predictor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="subtitle">Enter borrower and loan details to estimate default risk.</p>', unsafe_allow_html=True)

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
            st.info(
                "Use this score as a screening signal. A credit officer should still review the applicant context before a final decision."
            )

            st.markdown('<h3 class="section-header">Key Risk Factors</h3>', unsafe_allow_html=True)
            for factor in factors:
                st.write(f"- {factor}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


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


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Loan Risk Predictor", "Dashboard"],
    )

    if page == "Home":
        home_page()
    elif page == "Loan Risk Predictor":
        predictor_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()
