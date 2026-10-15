"""
Analytics Views for Global Public Health Trends Analysis System.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

BASE_DIR = settings.BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "engineered_health_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_health_risk_model.joblib")
BENCHMARK_PATH = os.path.join(BASE_DIR, "reports", "model_comparison_benchmark.csv")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "final_model_evaluation_metrics.json")
INSIGHTS_PATH = os.path.join(BASE_DIR, "reports", "eda_insights.json")
METADATA_PATH = os.path.join(BASE_DIR, "data", "raw", "sources_metadata.json")

_DATA_CACHE = None
_MODEL_CACHE = None

def get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        if os.path.exists(DATA_PATH):
            _DATA_CACHE = pd.read_csv(DATA_PATH)
        else:
            raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}. Run run_pipeline.py first.")
    return _DATA_CACHE

def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if os.path.exists(MODEL_PATH):
            _MODEL_CACHE = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Run run_pipeline.py first.")
    return _MODEL_CACHE

def load_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def overview_view(request):
    """View for Global Overview & KPIs dashboard."""
    df = get_data()
    latest_year = int(df["year"].max())
    df_latest = df[df["year"] == latest_year]

    # Global KPIs
    avg_life_exp = round(df_latest["life_expectancy"].mean(), 1)
    avg_u5_mort = round(df_latest["under_5_mortality_rate"].mean(), 1)
    avg_dtp3 = round(df_latest["dtp3_vaccination_coverage"].mean(), 1)
    high_risk_pct = round(df_latest["high_health_risk"].mean() * 100.0, 1)

    # Figure 1: Regional Healthcare Access Score
    reg_summary = df_latest.groupby("who_region")[["healthcare_access_score", "under_5_mortality_rate"]].mean().reset_index()
    fig_bar = px.bar(
        reg_summary,
        x="who_region",
        y="healthcare_access_score",
        color="who_region",
        title="Healthcare Access Capacity Index by WHO Region (0-100)",
        labels={"healthcare_access_score": "Access Capacity Index", "who_region": "WHO Region"}
    )
    fig_bar.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    bar_json = pio.to_json(fig_bar)

    # Figure 2: Risk Tier Distribution Pie Chart
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
    fig_pie.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
    pie_json = pio.to_json(fig_pie)

    # Insights
    insights = load_json_file(INSIGHTS_PATH)

    context = {
        'active_page': 'overview',
        'latest_year': latest_year,
        'total_countries': df['country'].nunique(),
        'avg_life_exp': avg_life_exp,
        'avg_u5_mort': avg_u5_mort,
        'avg_dtp3': avg_dtp3,
        'high_risk_pct': high_risk_pct,
        'bar_json': bar_json,
        'pie_json': pie_json,
        'insights': insights[:4],
    }
    return render(request, 'analytics/overview.html', context)


def trends_view(request):
    """View for Longitudinal Multi-Indicator Trends."""
    df = get_data()
    selected_indicator = request.GET.get('indicator', 'life_expectancy')
    selected_region = request.GET.get('region', 'All Regions')
    compare_country_str = request.GET.get('countries', 'United States,Nigeria,India,Germany,Brazil')
    compare_countries = [c.strip() for c in compare_country_str.split(',') if c.strip()]

    all_regions = ["All Regions"] + sorted(df["who_region"].unique().tolist())
    all_countries = sorted(df["country"].unique().tolist())

    indicators = [
        ("life_expectancy", "Life Expectancy (Years)"),
        ("under_5_mortality_rate", "Under-5 Mortality Rate (per 1k)"),
        ("adult_mortality_rate", "Adult Mortality Rate (per 1k)"),
        ("dtp3_vaccination_coverage", "DTP3 Vaccination Coverage (%)"),
        ("measles_vaccination_coverage", "Measles Vaccination Coverage (%)"),
        ("tb_incidence_rate", "TB Incidence (per 100k)"),
        ("healthcare_access_score", "Healthcare Access Score (0-100)"),
        ("pm25_air_pollution", "PM2.5 Air Pollution (μg/m³)"),
        ("health_expenditure_per_capita", "Health Expenditure per Capita (USD)"),
        ("composite_vulnerability_score", "Composite Vulnerability Score (0-100)")
    ]

    filtered_df = df.copy()
    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["who_region"] == selected_region]

    # Regional Trend Chart
    reg_trend = filtered_df.groupby(["year", "who_region"])[selected_indicator].mean().reset_index()
    fig_trend = px.line(
        reg_trend,
        x="year",
        y=selected_indicator,
        color="who_region",
        title=f"Regional Mean Trajectory: {selected_indicator.replace('_', ' ').title()} (2000-2023)",
        markers=True
    )
    fig_trend.update_layout(height=420, hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)')
    trend_json = pio.to_json(fig_trend)

    # Country Comparison Chart
    c_df = df[df["country"].isin(compare_countries)]
    fig_c = px.line(
        c_df,
        x="year",
        y=selected_indicator,
        color="country",
        title=f"Country Benchmark Comparison: {selected_indicator.replace('_', ' ').title()}",
        markers=True
    )
    fig_c.update_layout(height=400, hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)')
    country_trend_json = pio.to_json(fig_c)

    context = {
        'active_page': 'trends',
        'indicators': indicators,
        'selected_indicator': selected_indicator,
        'all_regions': all_regions,
        'selected_region': selected_region,
        'all_countries': all_countries,
        'compare_countries': compare_countries,
        'trend_json': trend_json,
        'country_trend_json': country_trend_json,
    }
    return render(request, 'analytics/trends.html', context)


def country_explorer_view(request):
    """View for deep country health profiling."""
    df = get_data()
    all_countries = sorted(df["country"].unique().tolist())
    chosen_country = request.GET.get('country', all_countries[0])
    chosen_year = int(request.GET.get('year', int(df["year"].max())))

    c_match = df[(df["country"] == chosen_country) & (df["year"] == chosen_year)]
    if c_match.empty:
        c_data = df[df["country"] == chosen_country].sort_values("year").iloc[-1]
        chosen_year = int(c_data["year"])
    else:
        c_data = c_match.iloc[0]

    region_data = df[(df["who_region"] == c_data["who_region"]) & (df["year"] == chosen_year)]

    # Radar Chart
    categories = ['Immunization', 'Clinical Workforce', 'WASH Access', 'Clean Air (Inv)', 'Econ Spend']
    c_vals = [
        float(c_data['dtp3_vaccination_coverage']) / 100.0,
        min(1.0, float(c_data['physicians_per_10k']) / 35.0),
        float(c_data['clean_water_sanitation_pct']) / 100.0,
        max(0.0, 1.0 - (float(c_data['pm25_air_pollution']) / 80.0)),
        min(1.0, float(c_data['health_expenditure_per_capita']) / 3000.0)
    ]
    reg_vals = [
        float(region_data['dtp3_vaccination_coverage'].mean()) / 100.0,
        min(1.0, float(region_data['physicians_per_10k'].mean()) / 35.0),
        float(region_data['clean_water_sanitation_pct'].mean()) / 100.0,
        max(0.0, 1.0 - (float(region_data['pm25_air_pollution'].mean()) / 80.0)),
        min(1.0, float(region_data['health_expenditure_per_capita'].mean()) / 3000.0)
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=c_vals, theta=categories, fill='toself', name=chosen_country, line_color="#2980b9"))
    fig_radar.add_trace(go.Scatterpolar(r=reg_vals, theta=categories, fill='toself', name=f"Region ({c_data['who_region']})", line_color="#95a5a6", opacity=0.5))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=380, title="Multidimensional Health Capability Radar", paper_bgcolor='rgba(0,0,0,0)')
    radar_json = pio.to_json(fig_radar)

    # Timeline Chart
    c_history = df[df["country"] == chosen_country].sort_values("year")
    fig_hist = px.line(
        c_history,
        x="year",
        y=["under_5_mortality_rate", "dtp3_vaccination_coverage", "healthcare_access_score"],
        title=f"Historical Indicators: {chosen_country} (2000-2023)"
    )
    fig_hist.update_layout(height=380, hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)')
    history_json = pio.to_json(fig_hist)

    context = {
        'active_page': 'country_explorer',
        'all_countries': all_countries,
        'chosen_country': chosen_country,
        'chosen_year': chosen_year,
        'c_data': c_data,
        'delta_life': round(c_data['life_expectancy'] - region_data['life_expectancy'].mean(), 1),
        'delta_u5': round(c_data['under_5_mortality_rate'] - region_data['under_5_mortality_rate'].mean(), 1),
        'delta_access': round(c_data['healthcare_access_score'] - region_data['healthcare_access_score'].mean(), 1),
        'radar_json': radar_json,
        'history_json': history_json,
    }
    return render(request, 'analytics/country_explorer.html', context)


def risk_map_view(request):
    """View for Global Choropleth Risk Mapping."""
    df = get_data()
    map_year = int(request.GET.get('year', 2023))
    map_metric = request.GET.get('metric', 'risk_category_tier')

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

    fig_map.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)')
    map_json = pio.to_json(fig_map)

    metrics_list = [
        ("risk_category_tier", "Risk Category Tier (Low, Moderate, High)"),
        ("under_5_mortality_rate", "Under-5 Mortality Rate"),
        ("life_expectancy", "Life Expectancy (Years)"),
        ("dtp3_vaccination_coverage", "DTP3 Immunization Coverage (%)"),
        ("healthcare_access_score", "Healthcare Access Capacity Score"),
        ("pm25_air_pollution", "PM2.5 Air Pollution"),
        ("composite_vulnerability_score", "Composite Vulnerability Score")
    ]

    context = {
        'active_page': 'risk_map',
        'map_year': map_year,
        'map_metric': map_metric,
        'metrics_list': metrics_list,
        'map_json': map_json,
    }
    return render(request, 'analytics/risk_map.html', context)


def simulator_view(request):
    """View for Live ML Policy Simulator & Risk Prediction."""
    df = get_data()
    model = get_model()
    all_countries = ["Custom / Manual Inputs"] + sorted(df["country"].unique().tolist())

    preset_country = request.GET.get('preset', 'Custom / Manual Inputs')

    # Defaults
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

    # If POST or query param supplied
    if request.method == 'POST':
        inp_region = request.POST.get('who_region', init_region)
        inp_income = request.POST.get('income_group', init_income)
        inp_gdp = float(request.POST.get('gdp_per_capita', init_gdp))
        inp_physicians = float(request.POST.get('physicians_per_10k', init_physicians))
        inp_nurses = float(request.POST.get('nurses_per_10k', init_nurses))
        inp_beds = float(request.POST.get('hospital_beds_per_10k', init_beds))
        inp_exp_pct = float(request.POST.get('health_expenditure_pct_gdp', init_exp_pct))
        inp_dtp3 = float(request.POST.get('dtp3_vaccination_coverage', init_dtp3))
        inp_measles = float(request.POST.get('measles_vaccination_coverage', init_measles))
        inp_tb = float(request.POST.get('tb_incidence_rate', init_tb))
        inp_water = float(request.POST.get('clean_water_sanitation_pct', init_water))
        inp_pm25 = float(request.POST.get('pm25_air_pollution', init_pm25))
        inp_urban = float(request.POST.get('urban_population_pct', init_urban))
        inp_edu = float(request.POST.get('education_index', init_edu))
    else:
        inp_region = init_region
        inp_income = init_income
        inp_gdp = init_gdp
        inp_physicians = init_physicians
        inp_nurses = init_nurses
        inp_beds = init_beds
        inp_exp_pct = init_exp_pct
        inp_dtp3 = init_dtp3
        inp_measles = init_measles
        inp_tb = init_tb
        inp_water = init_water
        inp_pm25 = init_pm25
        inp_urban = init_urban
        inp_edu = init_edu

    # Derived feature computations
    p_capita_exp = inp_gdp * (inp_exp_pct / 100.0)
    p_norm = min(1.0, max(0.0, (inp_physicians - 0.2) / (80.0 - 0.2)))
    n_norm = min(1.0, max(0.0, (inp_nurses - 0.8) / (180.0 - 0.8)))
    b_norm = min(1.0, max(0.0, (inp_beds - 1.5) / (140.0 - 1.5)))
    w_norm = min(1.0, max(0.0, (inp_water - 15.0) / (99.9 - 15.0)))
    healthcare_access = round((0.30 * p_norm + 0.25 * n_norm + 0.20 * b_norm + 0.25 * w_norm) * 100.0, 2)

    pm_norm = min(1.0, max(0.0, (inp_pm25 - 5.0) / (120.0 - 5.0)))
    u_norm = min(1.0, max(0.0, (inp_urban - 10.0) / (99.0 - 10.0)))
    pollution_score = round((0.65 * pm_norm + 0.35 * u_norm) * 100.0, 2)

    group_meds = {"Low income": 35.0, "Lower middle income": 140.0, "Upper middle income": 650.0, "High income": 4100.0}
    norm_inc_exp = round(p_capita_exp / (group_meds.get(inp_income, 200.0) + 1.0), 3)
    low_v_high_p = int(((inp_dtp3 + inp_measles) / 2.0 < 78.0) and (inp_pm25 > 40.0))

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

    pred_label = int(model.predict(sample_df)[0])
    pred_proba = float(model.predict_proba(sample_df)[0][1])

    context = {
        'active_page': 'simulator',
        'all_countries': all_countries,
        'preset_country': preset_country,
        'who_regions': df["who_region"].unique().tolist(),
        'income_groups': ["Low income", "Lower middle income", "Upper middle income", "High income"],
        'inp_region': inp_region,
        'inp_income': inp_income,
        'inp_gdp': inp_gdp,
        'inp_physicians': inp_physicians,
        'inp_nurses': inp_nurses,
        'inp_beds': inp_beds,
        'inp_exp_pct': inp_exp_pct,
        'inp_dtp3': inp_dtp3,
        'inp_measles': inp_measles,
        'inp_tb': inp_tb,
        'inp_water': inp_water,
        'inp_pm25': inp_pm25,
        'inp_urban': inp_urban,
        'inp_edu': inp_edu,
        'pred_label': pred_label,
        'pred_proba': round(pred_proba * 100.0, 1),
        'pred_proba_raw': pred_proba,
        'healthcare_access': healthcare_access,
    }
    return render(request, 'analytics/simulator.html', context)


def model_evaluation_view(request):
    """View for ML Evaluation Diagnostics & Benchmarks."""
    benchmark_df = pd.read_csv(BENCHMARK_PATH) if os.path.exists(BENCHMARK_PATH) else pd.DataFrame()
    final_metrics = load_json_file(METRICS_PATH)

    clean_benchmark = []
    if not benchmark_df.empty:
        for r in benchmark_df.to_dict(orient='records'):
            cv_val = ""
            for k, v in r.items():
                if "CV Accuracy" in k:
                    cv_val = v
                    break
            clean_benchmark.append({
                'model': r.get('Model', ''),
                'cv_acc': cv_val,
                'val_acc': r.get('Val Accuracy', ''),
                'val_f1': r.get('Val F1-Score', ''),
                'val_auc': r.get('Val ROC-AUC', '')
            })

    context = {
        'active_page': 'model_evaluation',
        'final_metrics': final_metrics,
        'benchmark_rows': clean_benchmark,
    }
    return render(request, 'analytics/model_evaluation.html', context)


@csrf_exempt
def api_predict(request):
    """JSON REST API endpoint for real-time inference."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    try:
        data = json.loads(request.body)
        model = get_model()
        
        # Extract fields with safe defaults
        inp_dtp3 = float(data.get('dtp3_vaccination_coverage', 75.0))
        inp_measles = float(data.get('measles_vaccination_coverage', 72.0))
        inp_physicians = float(data.get('physicians_per_10k', 10.0))
        inp_nurses = float(data.get('nurses_per_10k', 25.0))
        inp_beds = float(data.get('hospital_beds_per_10k', 20.0))
        inp_water = float(data.get('clean_water_sanitation_pct', 70.0))
        inp_pm25 = float(data.get('pm25_air_pollution', 35.0))
        inp_urban = float(data.get('urban_population_pct', 50.0))
        inp_gdp = float(data.get('gdp_per_capita', 4000.0))
        inp_exp_pct = float(data.get('health_expenditure_pct_gdp', 5.0))
        inp_region = str(data.get('who_region', 'African Region'))
        inp_income = str(data.get('income_group', 'Lower middle income'))
        inp_tb = float(data.get('tb_incidence_rate', 100.0))
        inp_edu = float(data.get('education_index', 0.60))

        p_capita_exp = inp_gdp * (inp_exp_pct / 100.0)
        p_norm = min(1.0, max(0.0, (inp_physicians - 0.2) / (80.0 - 0.2)))
        n_norm = min(1.0, max(0.0, (inp_nurses - 0.8) / (180.0 - 0.8)))
        b_norm = min(1.0, max(0.0, (inp_beds - 1.5) / (140.0 - 1.5)))
        w_norm = min(1.0, max(0.0, (inp_water - 15.0) / (99.9 - 15.0)))
        healthcare_access = round((0.30 * p_norm + 0.25 * n_norm + 0.20 * b_norm + 0.25 * w_norm) * 100.0, 2)

        pm_norm = min(1.0, max(0.0, (inp_pm25 - 5.0) / (120.0 - 5.0)))
        u_norm = min(1.0, max(0.0, (inp_urban - 10.0) / (99.0 - 10.0)))
        pollution_score = round((0.65 * pm_norm + 0.35 * u_norm) * 100.0, 2)

        group_meds = {"Low income": 35.0, "Lower middle income": 140.0, "Upper middle income": 650.0, "High income": 4100.0}
        norm_inc_exp = round(p_capita_exp / (group_meds.get(inp_income, 200.0) + 1.0), 3)
        low_v_high_p = int(((inp_dtp3 + inp_measles) / 2.0 < 78.0) and (inp_pm25 > 40.0))

        sample_df = pd.DataFrame([{
            "dtp3_vaccination_coverage": inp_dtp3,
            "measles_vaccination_coverage": inp_measles,
            "dtp3_yoy_pct_change": 0.0,
            "tb_incidence_rate": inp_tb,
            "tb_yoy_pct_change": 0.0,
            "physicians_per_10k": inp_physicians,
            "nurses_per_10k": inp_nurses,
            "hospital_beds_per_10k": inp_beds,
            "health_expenditure_pct_gdp": inp_exp_pct,
            "health_expenditure_per_capita": p_capita_exp,
            "health_exp_normalized_income": norm_inc_exp,
            "healthcare_access_score": healthcare_access,
            "clean_water_sanitation_pct": inp_water,
            "pm25_air_pollution": inp_pm25,
            "pollution_health_risk_score": pollution_score,
            "low_vacc_high_pollution_flag": low_v_high_p,
            "gdp_per_capita": inp_gdp,
            "urban_population_pct": inp_urban,
            "education_index": inp_edu,
            "who_region": inp_region,
            "income_group": inp_income
        }])

        pred_label = int(model.predict(sample_df)[0])
        pred_proba = float(model.predict_proba(sample_df)[0][1])

        return JsonResponse({
            'status': 'success',
            'predicted_high_risk': pred_label == 1,
            'high_risk_probability': round(pred_proba, 4),
            'risk_tier': 'High Risk' if pred_label == 1 else 'Low / Moderate Risk',
            'healthcare_access_score': healthcare_access
        })
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)


def api_country_data(request, country_name):
    """Returns historical indicator data for a specific country."""
    df = get_data()
    c_df = df[df["country"].str.lower() == country_name.lower()].sort_values("year")
    if c_df.empty:
        return JsonResponse({'error': f'Country {country_name} not found'}, status=404)
    return JsonResponse({
        'country': country_name,
        'records': c_df[['year', 'life_expectancy', 'under_5_mortality_rate', 'dtp3_vaccination_coverage', 'healthcare_access_score', 'risk_category_tier']].to_dict(orient='records')
    })
