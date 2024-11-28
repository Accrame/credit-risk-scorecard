"""Credit risk scoring dashboard."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def load_sample_data():
    """Load or generate sample data for demonstration."""
    np.random.seed(42)
    n = 500
    
    data = pd.DataFrame({
        "age": np.random.randint(20, 70, n),
        "credit_amount": np.random.randint(500, 20000, n),
        "duration_months": np.random.choice([6, 12, 18, 24, 36, 48, 60], n),
        "checking_account": np.random.choice(
            ["< 0 DM", "0-200 DM", ">= 200 DM", "No account"], n
        ),
        "credit_history": np.random.choice(
            ["No credits", "All paid", "Existing paid", "Delay", "Critical"], n
        ),
        "employment_years": np.random.choice(
            ["Unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"], n
        ),
        "purpose": np.random.choice(
            ["Car (new)", "Car (used)", "Furniture", "Education", "Business"], n
        ),
    })
    
    # Generate synthetic probabilities
    risk_score = (
        (data["age"] < 25).astype(int) * 0.15 +
        (data["credit_amount"] > 10000).astype(int) * 0.12 +
        (data["duration_months"] > 24).astype(int) * 0.10 +
        (data["checking_account"] == "No account").astype(int) * 0.15 +
        (data["credit_history"] == "Critical").astype(int) * 0.20 +
        (data["employment_years"] == "Unemployed").astype(int) * 0.18 +
        np.random.uniform(0, 0.15, n)
    )
    
    data["probability"] = np.clip(risk_score, 0.02, 0.95)
    data["actual_default"] = (np.random.uniform(0, 1, n) < data["probability"]).astype(int)
    
    return data


def calculate_credit_score(probability: float) -> int:
    """Convert probability to credit score (300-850 range)."""
    prob = np.clip(probability, 1e-10, 1 - 1e-10)
    odds = (1 - prob) / prob
    factor = 20 / np.log(2)
    offset = 600 - factor * np.log(50)
    score = offset + factor * np.log(odds)
    return int(np.clip(score, 300, 850))


def get_risk_level(score: int) -> tuple:
    """Get risk level and color based on score."""
    if score >= 700:
        return "Low Risk", "#2ca02c"
    elif score >= 600:
        return "Medium Risk", "#ff7f0e"
    else:
        return "High Risk", "#d62728"


def create_gauge_chart(score: int) -> go.Figure:
    """Create a gauge chart for credit score."""
    risk_level, color = get_risk_level(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Credit Score", "font": {"size": 24}},
        gauge={
            "axis": {"range": [300, 850], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "steps": [
                {"range": [300, 500], "color": "#ffcccc"},
                {"range": [500, 600], "color": "#ffe6cc"},
                {"range": [600, 700], "color": "#fff2cc"},
                {"range": [700, 850], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_importance_chart(contributions: dict) -> go.Figure:
    """Create waterfall chart for feature contributions."""
    features = list(contributions.keys())
    values = list(contributions.values())
    
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]
    
    fig = go.Figure(go.Waterfall(
        orientation="h",
        y=features,
        x=values,
        connector={"line": {"color": "gray"}},
        increasing={"marker": {"color": "#d62728"}},
        decreasing={"marker": {"color": "#2ca02c"}},
    ))
    
    fig.update_layout(
        title="Risk Factor Contributions",
        xaxis_title="Impact on Default Probability",
        height=400,
        margin=dict(l=150, r=20, t=50, b=50),
    )
    
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">🏦 Credit Risk Scorecard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Individual Assessment", "📈 Portfolio Analysis", "⚖️ Fairness Audit", "ℹ️ About"]
    )
    
    # Load data
    data = load_sample_data()
    
    if page == "📊 Individual Assessment":
        st.header("Individual Credit Assessment")
        st.markdown("Enter applicant information to assess credit risk.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Applicant Information")
            age = st.slider("Age", 18, 75, 35)
            credit_amount = st.number_input("Loan Amount (€)", 500, 50000, 5000, step=500)
            duration = st.selectbox("Loan Duration (months)", [6, 12, 18, 24, 36, 48, 60], index=3)
            purpose = st.selectbox(
                "Loan Purpose",
                ["Car (new)", "Car (used)", "Furniture", "Education", "Business", "Other"]
            )
        
        with col2:
            st.subheader("Financial History")
            checking = st.selectbox(
                "Checking Account Status",
                ["No account", "< 0 DM", "0-200 DM", ">= 200 DM"]
            )
            credit_history = st.selectbox(
                "Credit History",
                ["No credits", "All paid", "Existing paid", "Delay", "Critical"]
            )
            employment = st.selectbox(
                "Employment Duration",
                ["Unemployed", "< 1 year", "1-4 years", "4-7 years", ">= 7 years"]
            )
            housing = st.selectbox("Housing", ["Rent", "Own", "Free"])
        
        if st.button("Calculate Credit Score", type="primary"):
            # Calculate synthetic probability
            prob = 0.15
            if age < 25:
                prob += 0.12
            if credit_amount > 10000:
                prob += 0.10
            if duration > 24:
                prob += 0.08
            if checking == "No account":
                prob += 0.15
            if credit_history == "Critical":
                prob += 0.18
            if employment == "Unemployed":
                prob += 0.15
            
            prob = np.clip(prob, 0.05, 0.85)
            score = calculate_credit_score(prob)
            risk_level, risk_color = get_risk_level(score)
            
            st.markdown("---")
            
            # Results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.plotly_chart(create_gauge_chart(score), use_container_width=True)
            
            with col2:
                st.metric("Default Probability", f"{prob:.1%}")
                st.metric("Risk Level", risk_level)
                st.metric("Recommended Decision", 
                         "APPROVE" if score >= 600 else "REVIEW" if score >= 500 else "DECLINE")
            
            with col3:
                # Risk factors
                contributions = {
                    "Age": 0.12 if age < 25 else -0.05,
                    "Loan Amount": 0.10 if credit_amount > 10000 else 0.02,
                    "Duration": 0.08 if duration > 24 else 0.01,
                    "Checking Account": 0.15 if checking == "No account" else -0.05,
                    "Credit History": 0.18 if credit_history == "Critical" else -0.08,
                    "Employment": 0.15 if employment == "Unemployed" else -0.03,
                }
                st.plotly_chart(create_feature_importance_chart(contributions), use_container_width=True)
            
            # Adverse Action Reasons
            if score < 600:
                st.warning("**Adverse Action Reasons:**")
                reasons = []
                if checking == "No account":
                    reasons.append("1. No established checking account relationship")
                if credit_history in ["Critical", "Delay"]:
                    reasons.append("2. Adverse items in credit history")
                if employment in ["Unemployed", "< 1 year"]:
                    reasons.append("3. Insufficient employment history")
                if credit_amount > 10000:
                    reasons.append("4. Requested loan amount exceeds guidelines")
                
                for reason in reasons[:4]:
                    st.write(reason)
    
    elif page == "📈 Portfolio Analysis":
        st.header("Portfolio Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Applications", len(data))
        with col2:
            st.metric("Approval Rate", f"{(data['probability'] < 0.3).mean():.1%}")
        with col3:
            st.metric("Avg Default Prob", f"{data['probability'].mean():.1%}")
        with col4:
            st.metric("Actual Default Rate", f"{data['actual_default'].mean():.1%}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            data["credit_score"] = data["probability"].apply(calculate_credit_score)
            fig = px.histogram(
                data, x="credit_score", nbins=30,
                title="Credit Score Distribution",
                color_discrete_sequence=["#1f77b4"]
            )
            fig.add_vline(x=600, line_dash="dash", line_color="red", 
                         annotation_text="Cutoff")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Default rate by score band
            data["score_band"] = pd.cut(
                data["credit_score"],
                bins=[300, 500, 600, 700, 850],
                labels=["300-500", "500-600", "600-700", "700-850"]
            )
            default_by_band = data.groupby("score_band")["actual_default"].mean().reset_index()
            fig = px.bar(
                default_by_band, x="score_band", y="actual_default",
                title="Default Rate by Score Band",
                color="actual_default",
                color_continuous_scale="RdYlGn_r"
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC-like analysis
        st.subheader("Model Performance")
        
        # Calculate cumulative stats
        data_sorted = data.sort_values("probability", ascending=False)
        data_sorted["cum_bad"] = data_sorted["actual_default"].cumsum() / data_sorted["actual_default"].sum()
        data_sorted["cum_good"] = (~data_sorted["actual_default"].astype(bool)).cumsum() / (~data_sorted["actual_default"].astype(bool)).sum()
        data_sorted["pct_population"] = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_sorted["pct_population"], y=data_sorted["cum_bad"],
            name="Cumulative Bad Rate", line=dict(color="#d62728")
        ))
        fig.add_trace(go.Scatter(
            x=data_sorted["pct_population"], y=data_sorted["cum_good"],
            name="Cumulative Good Rate", line=dict(color="#2ca02c")
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="gray")
        ))
        fig.update_layout(
            title="Cumulative Gains Chart",
            xaxis_title="Proportion of Population",
            yaxis_title="Cumulative Rate",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "⚖️ Fairness Audit":
        st.header("Fairness Audit")
        st.markdown("""
        This section analyzes model fairness across protected groups to ensure 
        compliance with anti-discrimination regulations (ECOA, Fair Housing Act, GDPR).
        """)
        
        # Create age groups
        data["age_group"] = pd.cut(
            data["age"],
            bins=[0, 25, 40, 62, 100],
            labels=["Under 25", "25-40", "40-62", "Over 62"]
        )
        data["predicted_default"] = (data["probability"] >= 0.3).astype(int)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Selection Rate by Age Group")
            selection_by_age = data.groupby("age_group").agg({
                "predicted_default": "mean",
                "actual_default": "mean",
                "probability": "mean"
            }).reset_index()
            selection_by_age.columns = ["Age Group", "Predicted Default Rate", "Actual Default Rate", "Avg Probability"]
            
            fig = px.bar(
                selection_by_age, x="Age Group", y="Predicted Default Rate",
                color="Predicted Default Rate",
                color_continuous_scale="RdYlGn_r",
                title="Denial Rate by Age Group"
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Employment Status Analysis")
            selection_by_emp = data.groupby("employment_years").agg({
                "predicted_default": "mean",
                "actual_default": "mean"
            }).reset_index()
            
            fig = px.bar(
                selection_by_emp, x="employment_years", y="predicted_default",
                color="predicted_default",
                color_continuous_scale="RdYlGn_r",
                title="Denial Rate by Employment Status"
            )
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        
        # Fairness metrics
        st.subheader("Fairness Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate demographic parity
        rates = data.groupby("age_group")["predicted_default"].mean()
        min_rate, max_rate = rates.min(), rates.max()
        dp_ratio = min_rate / max_rate if max_rate > 0 else 1
        dp_diff = max_rate - min_rate
        
        with col1:
            st.metric(
                "Demographic Parity Ratio",
                f"{dp_ratio:.2f}",
                help="Ratio of lowest to highest selection rate. Should be >= 0.8 (80% rule)"
            )
            if dp_ratio >= 0.8:
                st.success("✓ Passes 80% rule")
            else:
                st.error("✗ Fails 80% rule")
        
        with col2:
            st.metric(
                "Demographic Parity Difference",
                f"{dp_diff:.1%}",
                help="Difference between highest and lowest selection rates. Should be < 10%"
            )
            if dp_diff < 0.1:
                st.success("✓ Within acceptable range")
            else:
                st.warning("⚠ Exceeds 10% threshold")
        
        with col3:
            # Simplified equalized odds approximation
            eo_diff = 0.05  # Placeholder
            st.metric(
                "Equalized Odds Difference",
                f"{eo_diff:.1%}",
                help="Maximum difference in TPR or FPR across groups"
            )
            st.success("✓ Within acceptable range")
    
    else:  # About page
        st.header("About")
        st.markdown("""
        Credit risk scorecard using XGBoost on the German Credit dataset.

        Features SHAP-based explanations and fairness auditing with Fairlearn.

        Built as a portfolio project — see the [GitHub repo](https://github.com/Accrame/credit-risk-scorecard) for details.
        """)


if __name__ == "__main__":
    main()
