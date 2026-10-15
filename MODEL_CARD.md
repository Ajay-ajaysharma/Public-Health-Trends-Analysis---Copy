# 📋 Model Card: Global Public Health Disease-Risk Prediction System

## Model Details
- **Model Name**: Champion Random Forest Classifier for Public Health Risk Classification
- **Version**: 1.0.0
- **Model Type**: Supervised Ensemble Classifier (Random Forest with 150 Estimators, Max Depth 6, Balanced Class Weights)
- **Pipeline Architecture**: Scikit-Learn Pipeline (`StandardScaler` for numeric features + `OneHotEncoder` for categorical features + `RandomForestClassifier`)
- **Developer**: Senior Data Scientist & ML Engineer
- **License**: Creative Commons Open Data (CC-BY 4.0)
- **Release Date**: October 2026

---

## Intended Use
- **Primary Use**: Macro-level epidemiological risk surveillance, multi-year public health trend monitoring, and what-if policy intervention scenario simulation across sovereign nations.
- **Intended Users**: Public health policy analysts, epidemiologists, health economists, global development NGOs, and academic researchers.
- **Out-of-Scope Use Cases**: This model is **NOT** designed or approved for individual patient diagnosis, clinical bedside triage, or point-of-care medical decision-making.

---

## Data & Features
- **Training Temporal Window**: 2000 – 2023 (Longitudinal country-year records across 157 sovereign states).
- **Data Sources**:
  1. Our World in Data (OWID) - Demographics, Life Expectancy, Child & Adult Mortality.
  2. World Health Organization (WHO GHO) - Immunization coverage (DTP3, Measles), Clinical workforce density (physicians, nurses, hospital beds).
  3. World Bank Open Data - GDP per capita, Clean water & sanitation access, PM2.5 air pollution concentration.
- **Predictor Features (Upstream Structural Determinants)**:
  - *Clinical Human Resources & Bed Density*: `physicians_per_10k`, `nurses_per_10k`, `hospital_beds_per_10k`, `healthcare_access_score`
  - *Immunization Coverage & Dynamics*: `dtp3_vaccination_coverage`, `measles_vaccination_coverage`, `dtp3_yoy_pct_change`
  - *Communicable Disease Incidence*: `tb_incidence_rate`, `tb_yoy_pct_change`
  - *Environmental & Sanitation*: `clean_water_sanitation_pct`, `pm25_air_pollution`, `pollution_health_risk_score`, `low_vacc_high_pollution_flag`
  - *Economic Capacity*: `gdp_per_capita`, `health_expenditure_pct_gdp`, `health_expenditure_per_capita`, `health_exp_normalized_income`
  - *Demographics & Regional Structure*: `urban_population_pct`, `education_index`, `who_region`, `income_group`
- **Target Variable**: `high_health_risk` (Binary 0 / 1), identifying country-years where composite preventable mortality and communicable disease burden exceed critical epidemiological thresholds.

---

## Performance Benchmark & Evaluation Metrics

| Metric | Cross-Validation (5-Fold CV) | Validation Set | Untouched Test Partition |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **89.05% ± 1.1%** | **89.05%** | **85.51%** (Target Window: 80%–90%) |
| **Precision** | **81.40%** | **82.10%** | **76.13%** |
| **Recall** | **87.20%** | **88.40%** | **88.52%** |
| **F1-Score** | **84.18%** | **85.14%** | **81.86%** |
| **ROC-AUC** | **96.31%** | **96.31%** | **93.90%** |

---

## Ethical Considerations & Limitations
1. **Reporting Bias**: Low-income or politically unstable regions may have under-reported disease surveillance data; imputation relies on regional longitudinal medians.
2. **Aggregated Macro Scale**: National-level indicators conceal sub-national inequalities (e.g., rural vs. urban healthcare access disparities).
3. **No Clinical Generalizability**: Indicators represent population-level aggregate statistics, not individual biomarkers.
