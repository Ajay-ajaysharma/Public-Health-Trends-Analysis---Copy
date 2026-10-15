"""
=============================================================================
Module: data_cleaning.py
Project: Global Public Health Trends Analysis and Disease-Risk Prediction System
Author: Senior Data Scientist & Public Health Analyst
Description:
    Cleans, standardizes, validates, and imputes multi-source public health data.
    Generates a rigorous Data Quality Audit report covering missingness,
    outliers, schema validation, and geographic completeness.
=============================================================================
"""

import os
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "raw_global_public_health_indicators.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
CLEAN_DATA_PATH = os.path.join(PROCESSED_DIR, "clean_master_public_health_data.csv")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
DATA_QUALITY_REPORT = os.path.join(REPORTS_DIR, "data_quality_report.json")

# Physical/Domain bounds for public health metrics
INDICATOR_BOUNDS = {
    "life_expectancy": (30.0, 95.0),
    "under_5_mortality_rate": (0.0, 300.0),
    "adult_mortality_rate": (10.0, 800.0),
    "tb_incidence_rate": (0.0, 1200.0),
    "ncd_mortality_rate": (50.0, 1500.0),
    "dtp3_vaccination_coverage": (0.0, 100.0),
    "measles_vaccination_coverage": (0.0, 100.0),
    "physicians_per_10k": (0.0, 150.0),
    "nurses_per_10k": (0.0, 300.0),
    "hospital_beds_per_10k": (0.0, 200.0),
    "health_expenditure_pct_gdp": (0.5, 30.0),
    "clean_water_sanitation_pct": (0.0, 100.0),
    "pm25_air_pollution": (1.0, 250.0),
    "urban_population_pct": (0.0, 100.0),
    "education_index": (0.0, 1.0)
}


def clean_and_harmonize_data(raw_df=None):
    """
    Performs end-to-end data cleaning, standardization, domain validation,
    longitudinal missingness imputation, and generates a data quality audit.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if raw_df is None:
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data file not found at {RAW_DATA_PATH}. Run data_collection.py first.")
        logger.info(f"Loading raw dataset from {RAW_DATA_PATH}...")
        df = pd.read_csv(RAW_DATA_PATH)
    else:
        df = raw_df.copy()
        
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    # 1. Standardize string columns
    df["country"] = df["country"].astype(str).str.strip()
    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df["who_region"] = df["who_region"].astype(str).str.strip()
    df["income_group"] = df["income_group"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    
    # 2. Deduplication check on primary key (country, year)
    duplicate_mask = df.duplicated(subset=["country", "year"], keep="first")
    num_duplicates = int(duplicate_mask.sum())
    if num_duplicates > 0:
        logger.warning(f"Detected and removing {num_duplicates} duplicate country-year entries.")
        df = df[~duplicate_mask].copy()
    else:
        logger.info("Deduplication audit: 0 duplicate country-year rows found.")
        
    # 3. Sort chronologically by Country and Year for longitudinal integrity
    df = df.sort_values(by=["country", "year"]).reset_index(drop=True)
    
    # 4. Outlier detection and domain bounds enforcement
    outliers_rectified = 0
    for col, (min_val, max_val) in INDICATOR_BOUNDS.items():
        if col in df.columns:
            outlier_mask = (df[col] < min_val) | (df[col] > max_val)
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                outliers_rectified += outlier_count
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                
    # 5. Missing Data Imputation Strategy:
    #   Step A: Time-series linear interpolation within each country
    #   Step B: Forward & Backward fill per country for edge years
    #   Step C: Group median by (who_region, income_group, year) for any residual NaNs
    #   Step D: Global column median as final safety net
    
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["year", "population"]]
    missing_before = df[numeric_cols].isnull().sum().to_dict()
    
    # Step A & B: Within-country longitudinal imputation
    df[numeric_cols] = df.groupby("country")[numeric_cols].transform(
        lambda group: group.interpolate(method="linear", limit_direction="both")
    )
    
    # Step C: Regional + Income + Year median fallback
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df.groupby(["who_region", "income_group", "year"])[col].transform(
                lambda s: s.fillna(s.median())
            )
            
    # Step D: Final global column median fallback
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    missing_after = df[numeric_cols].isnull().sum().to_dict()
    
    # Recalculate derivative columns for consistency
    df["health_expenditure_per_capita"] = (df["gdp_per_capita"] * (df["health_expenditure_pct_gdp"] / 100.0)).round(2)
    
    # Save clean dataset
    df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info(f"Clean master dataset saved to {CLEAN_DATA_PATH} (Shape: {df.shape})")
    
    # Generate comprehensive Data Quality Report
    quality_report = {
        "dataset_name": "Harmonized Global Public Health Indicators",
        "initial_record_count": initial_rows,
        "clean_record_count": len(df),
        "total_countries": int(df["country"].nunique()),
        "total_regions": int(df["who_region"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "duplicates_removed": num_duplicates,
        "outliers_rectified": outliers_rectified,
        "missingness_summary_before": {k: int(v) for k, v in missing_before.items() if v > 0},
        "missingness_summary_after": {k: int(v) for k, v in missing_after.items()},
        "schema_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    with open(DATA_QUALITY_REPORT, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=4)
        
    logger.info(f"Data Quality Report successfully written to {DATA_QUALITY_REPORT}")
    return df, quality_report


if __name__ == "__main__":
    clean_and_harmonize_data()
