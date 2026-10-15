"""
=============================================================================
Application: app.py
Project: Global Public Health Trends Analysis and Disease-Risk Prediction System
Author: Senior Data Scientist, Public Health Analyst & ML Engineer
Description:
    Interactive, portfolio-ready Streamlit web application providing:
    - Global public health KPI monitoring & regional health equity benchmarks
    - Longitudinal multi-indicator time-series analytics
    - Comprehensive Country Health Profile Explorer with radar benchmark charts
    - Interactive global choropleth risk mapping
    - Live what-if policy simulator & disease-risk predictive engine
    - ML model performance benchmarking, ROC/PR diagnostics, and explainability
=============================================================================
"""

import os
import sys
import streamlit as st

# Automatically launch with streamlit CLI if executed via standard python
if not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())

import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Global Public Health Trends & Risk Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        padding: 0;
    }
    .main-header p {
        color: #e0eafc !important;
        font-size: 1.05rem;
        margin-top: 8px;
        margin-bottom: 0;
    }
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 5px solid #2a5298;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-high {
        color: #dc2626;
        font-weight: bold;
    }
    .risk-mod {
        color: #d97706;
        font-weight: bold;
    }
    .risk-low {
        color: #16a34a;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "engineered_health_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_health_risk_model.joblib")
BENCHMARK_PATH = os.path.join(BASE_DIR, "reports", "model_comparison_benchmark.csv")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "final_model_evaluation_metrics.json")
INSIGHTS_PATH = os.path.join(BASE_DIR, "reports", "eda_insights.json")
QUALITY_REPORT_PATH = os.path.join(BASE_DIR, "reports", "data_quality_report.json")
METADATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sources_metadata.json")


@st.cache_data
def load_health_dataset():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. Please run `python run_pipeline.py` first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_champion_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model artifact not found at {MODEL_PATH}. Please run `python run_pipeline.py` first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_data
def load_reports_and_metadata():
    benchmark_df = pd.read_csv(BENCHMARK_PATH) if os.path.exists(BENCHMARK_PATH) else pd.DataFrame()
    
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
    insights = []
    if os.path.exists(INSIGHTS_PATH):
        with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
            insights = json.load(f)
            
    quality = {}
    if os.path.exists(QUALITY_REPORT_PATH):
        with open(QUALITY_REPORT_PATH, "r", encoding="utf-8") as f:
            quality = json.load(f)
            
    meta = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
    return benchmark_df, metrics, insights, quality, meta


df = load_health_dataset()
model_pipeline = load_champion_model()
benchmark_df, final_metrics, insights_list, quality_report, metadata = load_reports_and_metadata()

# Sidebar Navigation & Filter Controls
st.sidebar.image("https://img.icons8.com/fluency/96/caduceus.png", width=64)
st.sidebar.title("Public Health System")
st.sidebar.markdown("**Global Epidemiology & ML Risk Surveillance**")
st.sidebar.markdown("---")

navigation = st.sidebar.radio(
    "Select System View:",
    [
        "📊 Global Overview & KPIs",
        "📈 Longitudinal Trends",
        "🔍 Country Health Explorer",
        "🗺️ Global Disease-Risk Map",
        "🔮 Live Risk Simulator & Predictor",
        "🤖 Model Evaluation & Benchmarks"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("⚡ **Built for Health Policy Analysts & Epidemiologists**")
st.sidebar.caption(f"📅 Data Temporal Coverage: 2000 - {int(df['year'].max())}")
st.sidebar.caption(f"🌐 Monitored Sovereign States: {df['country'].nunique()}")

# =============================================================================
# VIEW 1: GLOBAL OVERVIEW & KPIS
# =============================================================================
if navigation == "📊 Global Overview & KPIs":
    st.markdown("""
    <div class="main-header">
        <h1>Global Public Health Trends & Surveillance Hub</h1>
        <p>Comprehensive monitoring of epidemiological indicators, healthcare accessibility, immunization coverage, and environmental risk.</p>
    </div>
    """, unsafe_allow_html=True)
    
    latest_year = int(df["year"].max())
    df_latest = df[df["year"] == latest_year]
    
    # KPI Summary Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_life_exp = df_latest["life_expectancy"].mean()
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #3b82f6;">
            <div class="metric-label">Global Mean Life Expectancy</div>
            <div class="metric-value">{avg_life_exp:.1f} <span style="font-size: 1rem; color: #64748b;">Years</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        avg_u5_mort = df_latest["under_5_mortality_rate"].mean()
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ef4444;">
            <div class="metric-label">Under-5 Mortality Rate</div>
            <div class="metric-value">{avg_u5_mort:.1f} <span style="font-size: 1rem; color: #64748b;">per 1k</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        avg_dtp3 = df_latest["dtp3_vaccination_coverage"].mean()
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #10b981;">
            <div class="metric-label">DTP3 Immunization Coverage</div>
            <div class="metric-value">{avg_dtp3:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        high_risk_pct = (df_latest["high_health_risk"].mean()) * 100.0
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #f59e0b;">
            <div class="metric-label">Countries in High-Risk Tier</div>
            <div class="metric-value">{high_risk_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🌐 Regional Health Disparities & Distribution (2023)")
    c1, c2 = st.columns([3, 2])
    
    with c1:
        fig_bar = px.bar(
            df_latest.groupby("who_region")[["under_5_mortality_rate", "life_expectancy", "healthcare_access_score"]].mean().reset_index(),
            x="who_region",
            y="healthcare_access_score",
            color="who_region",
            title="Healthcare Access Capacity Index by WHO Region (0-100)",
            labels={"healthcare_access_score": "Access Capacity Index", "who_region": "WHO Region"}
        )
        fig_bar.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        tier_dist = df_latest["risk_category_tier"].value_counts().reset_index()
        tier_dist.columns = ["Risk Tier", "Country Count"]
        fig_pie = px.pie(
            tier_dist,
            names="Risk Tier",
            values="Country Count",
            color="Risk Tier",
            color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"},
            title="Global Sovereign States Risk Tier Allocation (2023)",
            hole=0.45
        )
        fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("### 💡 Core Public Health Findings & Empirical Insights")
    for ins in insights_list[:4]:
        with st.expander(f"📌 {ins['title']}", expanded=False):
            st.write(ins["finding"])

# =============================================================================
# VIEW 2: LONGITUDINAL TRENDS
# =============================================================================
elif navigation == "📈 Longitudinal Trends":
    st.markdown("""
    <div class="main-header">
        <h1>Longitudinal Multi-Indicator Time-Series Analysis</h1>
        <p>Explore multi-decade trajectories across health outcomes, immunization, clinical human resources, and environmental pollution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        selected_indicator = st.selectbox(
            "Select Primary Metric to Trend:",
            [
                "life_expectancy",
                "under_5_mortality_rate",
                "adult_mortality_rate",
                "dtp3_vaccination_coverage",
                "measles_vaccination_coverage",
                "tb_incidence_rate",
                "healthcare_access_score",
                "pm25_air_pollution",
                "health_expenditure_per_capita",
                "composite_vulnerability_score"
            ],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col2:
        selected_region = st.selectbox("Filter WHO Region:", ["All Regions"] + sorted(df["who_region"].unique().tolist()))
    with col3:
        compare_countries = st.multiselect(
            "Benchmark Specific Sovereign Countries:",
            sorted(df["country"].unique().tolist()),
            default=["United States", "Nigeria", "India", "Germany", "Brazil"]
        )
        
    filtered_df = df.copy()
    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["who_region"] == selected_region]
        
    # Regional aggregate trend
    fig_trend = px.line(
        filtered_df.groupby(["year", "who_region"])[selected_indicator].mean().reset_index(),
        x="year",
        y=selected_indicator,
        color="who_region",
        title=f"Regional Mean Trajectory: {selected_indicator.replace('_', ' ').title()} (2000-2023)",
        markers=True
    )
    fig_trend.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    if compare_countries:
        st.markdown(f"#### Country-Level Comparison: {selected_indicator.replace('_', ' ').title()}")
        c_df = df[df["country"].isin(compare_countries)]
        fig_c = px.line(
            c_df,
            x="year",
            y=selected_indicator,
            color="country",
            title=f"Comparative Country Trajectory: {selected_indicator.replace('_', ' ').title()}",
            markers=True
        )
        fig_c.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig_c, use_container_width=True)

# =============================================================================
# VIEW 3: COUNTRY HEALTH EXPLORER
# =============================================================================
elif navigation == "🔍 Country Health Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>Country Health Profile & Indicator Explorer</h1>
        <p>Granular epidemiological breakdown, regional benchmarks, and multidimensional radar analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_country = st.selectbox("Select Country:", sorted(df["country"].unique().tolist()), index=0)
    with c2:
        chosen_year = st.slider("Select Assessment Year:", int(df["year"].min()), int(df["year"].max()), int(df["year"].max()))
        
    c_data = df[(df["country"] == chosen_country) & (df["year"] == chosen_year)].iloc[0]
    region_data = df[(df["who_region"] == c_data["who_region"]) & (df["year"] == chosen_year)]
    
    # Country Header Stats
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Life Expectancy", f"{c_data['life_expectancy']:.1f} Yrs", delta=f"{c_data['life_expectancy'] - region_data['life_expectancy'].mean():.1f} vs Region")
    with k2:
        st.metric("Under-5 Mortality", f"{c_data['under_5_mortality_rate']:.1f} /1k", delta=f"{c_data['under_5_mortality_rate'] - region_data['under_5_mortality_rate'].mean():.1f} vs Region", delta_color="inverse")
    with k3:
        st.metric("Healthcare Capacity", f"{c_data['healthcare_access_score']:.1f} /100", delta=f"{c_data['healthcare_access_score'] - region_data['healthcare_access_score'].mean():.1f} vs Region")
    with k4:
        tier = c_data["risk_category_tier"]
        tier_class = "risk-high" if tier == "High Risk" else ("risk-mod" if tier == "Moderate Risk" else "risk-low")
        st.markdown(f"""
        <div class="kpi-card" style="padding: 10px 15px; margin: 0;">
            <div class="metric-label">Risk Category Tier</div>
            <div class="{tier_class}" style="font-size: 1.4rem;">{tier}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Radar Chart: Normalized Dimension Profile
    col_radar, col_timeline = st.columns([1, 1])
    with col_radar:
        categories = ['Immunization', 'Clinical Workforce', 'WASH Access', 'Clean Air (Inv)', 'Econ Spend']
        
        c_vals = [
            c_data['dtp3_vaccination_coverage'] / 100.0,
            min(1.0, c_data['physicians_per_10k'] / 35.0),
            c_data['clean_water_sanitation_pct'] / 100.0,
            max(0.0, 1.0 - (c_data['pm25_air_pollution'] / 80.0)),
            min(1.0, c_data['health_expenditure_per_capita'] / 3000.0)
        ]
        
        reg_vals = [
            region_data['dtp3_vaccination_coverage'].mean() / 100.0,
            min(1.0, region_data['physicians_per_10k'].mean() / 35.0),
            region_data['clean_water_sanitation_pct'].mean() / 100.0,
            max(0.0, 1.0 - (region_data['pm25_air_pollution'].mean() / 80.0)),
            min(1.0, region_data['health_expenditure_per_capita'].mean() / 3000.0)
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=c_vals, theta=categories, fill='toself', name=chosen_country, line_color="#2980b9"))
        fig_radar.add_trace(go.Scatterpolar(r=reg_vals, theta=categories, fill='toself', name=f"Region ({c_data['who_region']})", line_color="#95a5a6", opacity=0.5))
        
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=380, title="Multidimensional Health Capability Radar")
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_timeline:
        c_history = df[df["country"] == chosen_country].sort_values("year")
        fig_hist = px.line(
            c_history,
            x="year",
            y=["under_5_mortality_rate", "dtp3_vaccination_coverage", "healthcare_access_score"],
            title=f"Longitudinal Health Indicators: {chosen_country} (2000-2023)"
        )
        fig_hist.update_layout(height=380, hovermode="x unified")
        st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# VIEW 4: GLOBAL RISK MAP
# =============================================================================
elif navigation == "🗺️ Global Disease-Risk Map":
    st.markdown("""
    <div class="main-header">
        <h1>Interactive Global Public-Health Risk Choropleth Map</h1>
        <p>Global spatial visualization of public health risk tiers, mortality burdens, and immunization coverage.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_map1, col_map2 = st.columns([2, 1])
    with col_map1:
        map_metric = st.selectbox(
            "Select Map Indicator Layer:",
            [
                "risk_category_tier",
                "under_5_mortality_rate",
                "life_expectancy",
                "dtp3_vaccination_coverage",
                "healthcare_access_score",
                "pm25_air_pollution",
                "composite_vulnerability_score"
            ],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col_map2:
        map_year = st.slider("Map Year:", 2000, 2023, 2023)
        
    map_df = df[df["year"] == map_year]
    
    if map_metric == "risk_category_tier":
        fig_map = px.choropleth(
            map_df,
            locations="iso3",
            color="risk_category_tier",
            hover_name="country",
            hover_data=["life_expectancy", "under_5_mortality_rate", "dtp3_vaccination_coverage", "healthcare_access_score"],
            color_discrete_map={"Low Risk": "#2ecc71", "Moderate Risk": "#f39c12", "High Risk": "#e74c3c"},
            title=f"Global Sovereign States Public Health Risk Category Tier ({map_year})",
            projection="natural earth"
        )
    else:
        fig_map = px.choropleth(
            map_df,
            locations="iso3",
            color=map_metric,
            hover_name="country",
            hover_data=["life_expectancy", "under_5_mortality_rate", "healthcare_access_score"],
            color_continuous_scale="Viridis" if map_metric in ["life_expectancy", "dtp3_vaccination_coverage", "healthcare_access_score"] else "Reds",
            title=f"Global Spatial Distribution: {map_metric.replace('_', ' ').title()} ({map_year})",
            projection="natural earth"
        )
        
    fig_map.update_layout(height=580, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# =============================================================================
# VIEW 5: LIVE RISK SIMULATOR & PREDICTOR
# =============================================================================
elif navigation == "🔮 Live Risk Simulator & Predictor":
    st.markdown("""
    <div class="main-header">
        <h1>Live Public Health Risk Simulator & Policy Intervention Tool</h1>
        <p>Simulate policy interventions or input customized epidemiological scenarios to predict disease-risk classification in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Policy Simulator**: Adjust healthcare capacity, immunization coverage, and environmental parameters below to observe real-time predicted risk shifts.")
    
    # Preset loader
    p1, p2 = st.columns([2, 1])
    with p1:
        preset_country = st.selectbox("Load Baseline Country Profile:", ["Custom / Manual Inputs"] + sorted(df["country"].unique().tolist()))
        
    if preset_country != "Custom / Manual Inputs":
        c_base = df[(df["country"] == preset_country) & (df["year"] == 2023)].iloc[0]
        init_region = c_base["who_region"]
        init_income = c_base["income_group"]
        init_dtp3 = float(c_base["dtp3_vaccination_coverage"])
        init_measles = float(c_base["measles_vaccination_coverage"])
        init_tb = float(c_base["tb_incidence_rate"])
        init_physicians = float(c_base["physicians_per_10k"])
        init_nurses = float(c_base["nurses_per_10k"])
        init_beds = float(c_base["hospital_beds_per_10k"])
        init_exp_pct = float(c_base["health_expenditure_pct_gdp"])
        init_gdp = float(c_base["gdp_per_capita"])
        init_water = float(c_base["clean_water_sanitation_pct"])
        init_pm25 = float(c_base["pm25_air_pollution"])
        init_urban = float(c_base["urban_population_pct"])
        init_edu = float(c_base["education_index"])
    else:
        init_region = "African Region"
        init_income = "Lower middle income"
        init_dtp3 = 78.0
        init_measles = 75.0
        init_tb = 140.0
        init_physicians = 8.0
        init_nurses = 18.0
        init_beds = 15.0
        init_exp_pct = 4.5
        init_gdp = 3200.0
        init_water = 65.0
        init_pm25 = 48.0
        init_urban = 45.0
        init_edu = 0.55
        
    with st.form("risk_prediction_form"):
        st.subheader("1. Regional & Socioeconomic Context")
        f1, f2, f3 = st.columns(3)
        with f1:
            inp_region = st.selectbox("WHO Region:", df["who_region"].unique().tolist(), index=df["who_region"].unique().tolist().index(init_region))
        with f2:
            inp_income = st.selectbox("World Bank Income Group:", ["Low income", "Lower middle income", "Upper middle income", "High income"], index=["Low income", "Lower middle income", "Upper middle income", "High income"].index(init_income))
        with f3:
            inp_gdp = st.number_input("GDP Per Capita (USD):", 300.0, 120000.0, float(init_gdp), step=500.0)
            
        st.subheader("2. Healthcare System Capacity & Investment")
        f4, f5, f6, f7 = st.columns(4)
        with f4:
            inp_physicians = st.slider("Physicians per 10k:", 0.1, 80.0, float(init_physicians))
        with f5:
            inp_nurses = st.slider("Nurses per 10k:", 0.5, 180.0, float(init_nurses))
        with f6:
            inp_beds = st.slider("Hospital Beds per 10k:", 1.0, 140.0, float(init_beds))
        with f7:
            inp_exp_pct = st.slider("Health Expenditure (% GDP):", 1.0, 20.0, float(init_exp_pct))
            
        st.subheader("3. Immunization & Disease Burden")
        f8, f9, f10 = st.columns(3)
        with f8:
            inp_dtp3 = st.slider("DTP3 Immunization (%):", 10.0, 99.0, float(init_dtp3))
        with f9:
            inp_measles = st.slider("Measles Immunization (%):", 10.0, 99.0, float(init_measles))
        with f10:
            inp_tb = st.slider("Tuberculosis Incidence (per 100k):", 2.0, 800.0, float(init_tb))
            
        st.subheader("4. Environmental & Demographic Factors")
        f11, f12, f13, f14 = st.columns(4)
        with f11:
            inp_water = st.slider("Clean Water/Sanitation Access (%):", 10.0, 100.0, float(init_water))
        with f12:
            inp_pm25 = st.slider("PM2.5 Air Pollution (μg/m³):", 5.0, 150.0, float(init_pm25))
        with f13:
            inp_urban = st.slider("Urban Population (%):", 10.0, 100.0, float(init_urban))
        with f14:
            inp_edu = st.slider("Education / Schooling Index:", 0.10, 1.00, float(init_edu))
            
        submit_btn = st.form_submit_button("🚀 Compute Disease-Risk Classification", use_container_width=True)
        
    if submit_btn or preset_country != "Custom / Manual Inputs":
        # Calculate engineered composite features for live input
        p_capita_exp = inp_gdp * (inp_exp_pct / 100.0)
        
        # Healthcare access capacity score (0-100)
        p_norm = min(1.0, max(0.0, (inp_physicians - 0.2) / (80.0 - 0.2)))
        n_norm = min(1.0, max(0.0, (inp_nurses - 0.8) / (180.0 - 0.8)))
        b_norm = min(1.0, max(0.0, (inp_beds - 1.5) / (140.0 - 1.5)))
        w_norm = min(1.0, max(0.0, (inp_water - 15.0) / (99.9 - 15.0)))
        healthcare_access = round((0.30 * p_norm + 0.25 * n_norm + 0.20 * b_norm + 0.25 * w_norm) * 100.0, 2)
        
        # Pollution score
        pm_norm = min(1.0, max(0.0, (inp_pm25 - 5.0) / (120.0 - 5.0)))
        u_norm = min(1.0, max(0.0, (inp_urban - 10.0) / (99.0 - 10.0)))
        pollution_score = round((0.65 * pm_norm + 0.35 * u_norm) * 100.0, 2)
        
        # Normalized income spend
        group_meds = {"Low income": 35.0, "Lower middle income": 140.0, "Upper middle income": 650.0, "High income": 4100.0}
        norm_inc_exp = round(p_capita_exp / (group_meds.get(inp_income, 200.0) + 1.0), 3)
        
        # Interaction flag
        low_v_high_p = int(((inp_dtp3 + inp_measles) / 2.0 < 78.0) and (inp_pm25 > 40.0))
        
        # Build inference input dataframe
        sample_dict = {
            "dtp3_vaccination_coverage": [inp_dtp3],
            "measles_vaccination_coverage": [inp_measles],
            "dtp3_yoy_pct_change": [0.0],
            "tb_incidence_rate": [inp_tb],
            "tb_yoy_pct_change": [0.0],
            "physicians_per_10k": [inp_physicians],
            "nurses_per_10k": [inp_nurses],
            "hospital_beds_per_10k": [inp_beds],
            "health_expenditure_pct_gdp": [inp_exp_pct],
            "health_expenditure_per_capita": [p_capita_exp],
            "health_exp_normalized_income": [norm_inc_exp],
            "healthcare_access_score": [healthcare_access],
            "clean_water_sanitation_pct": [inp_water],
            "pm25_air_pollution": [inp_pm25],
            "pollution_health_risk_score": [pollution_score],
            "low_vacc_high_pollution_flag": [low_v_high_p],
            "gdp_per_capita": [inp_gdp],
            "urban_population_pct": [inp_urban],
            "education_index": [inp_edu],
            "who_region": [inp_region],
            "income_group": [inp_income]
        }
        
        sample_df = pd.DataFrame(sample_dict)
        
        pred_label = model_pipeline.predict(sample_df)[0]
        pred_proba = model_pipeline.predict_proba(sample_df)[0][1]
        
        st.markdown("### 📊 Live Model Risk Classification Outcome")
        r_col1, r_col2, r_col3 = st.columns([1.5, 1.5, 2])
        
        with r_col1:
            if pred_label == 1:
                st.error("🚨 Predicted Status: **HIGH PUBLIC HEALTH RISK**")
            else:
                st.success("✅ Predicted Status: **LOW / MODERATE RISK**")
                
        with r_col2:
            st.metric("High-Risk Probability", f"{pred_proba * 100.0:.1f}%")
            
        with r_col3:
            st.metric("Healthcare Capacity Score", f"{healthcare_access:.1f} / 100")
            
        # Probability Gauge Bar
        st.markdown(f"**High-Risk Probability Meter:**")
        st.progress(float(pred_proba))

# =============================================================================
# VIEW 6: MODEL EVALUATION & BENCHMARKS
# =============================================================================
elif navigation == "🤖 Model Evaluation & Benchmarks":
    st.markdown("""
    <div class="main-header">
        <h1>Machine Learning Evaluation & Benchmarks</h1>
        <p>Comprehensive validation diagnostics: 7-model cross-validation comparisons, confusion matrix, ROC-AUC, and feature importances.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model KPI summary
    if final_metrics:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Champion Model", final_metrics.get("champion_model", "Random Forest"))
        with m2:
            st.metric("Test Accuracy", f"{final_metrics.get('test_accuracy', 0.8551)*100:.2f}%")
        with m3:
            st.metric("Test F1-Score", f"{final_metrics.get('test_f1_score', 0.8186):.4f}")
        with m4:
            st.metric("Test Recall", f"{final_metrics.get('test_recall', 0.8852):.4f}")
        with m5:
            st.metric("Test ROC-AUC", f"{final_metrics.get('test_roc_auc', 0.9390):.4f}")
            
    st.markdown("### 📋 Multi-Model Cross-Validation Benchmark")
    if not benchmark_df.empty:
        st.dataframe(benchmark_df, use_container_width=True)
        
    st.markdown("### 📈 Champion Model Diagnostic Visualizations")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        cm_img = os.path.join(BASE_DIR, "reports", "figures", "eval_confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Champion Model Confusion Matrix (Test Split)", use_container_width=True)
    with col_v2:
        roc_img = os.path.join(BASE_DIR, "reports", "figures", "eval_roc_pr_curves.png")
        if os.path.exists(roc_img):
            st.image(roc_img, caption="ROC & Precision-Recall Diagnostic Curves", use_container_width=True)
            
    col_v3, col_v4 = st.columns(2)
    with col_v3:
        feat_img = os.path.join(BASE_DIR, "reports", "figures", "eval_feature_importance.png")
        if os.path.exists(feat_img):
            st.image(feat_img, caption="Top 15 Feature Importances (Gini Impurity)", use_container_width=True)
    with col_v4:
        cal_img = os.path.join(BASE_DIR, "reports", "figures", "eval_calibration_curve.png")
        if os.path.exists(cal_img):
            st.image(cal_img, caption="Probability Calibration Reliability Diagram", use_container_width=True)

st.markdown("---")
st.caption("🛡️ **Disclaimer**: Educational & public-health trend surveillance project. Not intended for clinical or individual medical diagnostic purposes.")
