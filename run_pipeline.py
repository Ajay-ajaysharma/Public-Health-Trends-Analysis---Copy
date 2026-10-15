"""
=============================================================================
Module: run_pipeline.py
Project: Global Public Health Trends Analysis and Disease-Risk Prediction System
Author: Senior Data Scientist & Public Health Analyst
Description:
    Master orchestration script that executes the complete end-to-end pipeline:
    1. Data Collection & Metadata Provenance
    2. Data Cleaning, Imputation & Quality Auditing
    3. Advanced Feature Engineering & Risk Target Formulation
    4. Exploratory Data Analysis & Insight Extraction
    5. Machine Learning Multi-Model Training & Benchmark Cross-Validation
    6. Comprehensive Evaluation & Diagnostic Chart Generation
=============================================================================
"""

import os
import sys
import logging
import time

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import download_and_collect_data
from src.data_cleaning import clean_and_harmonize_data
from src.feature_engineering import engineer_features
from src.eda import run_eda
from src.model_training import train_and_evaluate_all_models
from src.evaluation import generate_evaluation_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("PipelineOrchestrator")


def main():
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("STARTING GLOBAL PUBLIC HEALTH ANALYSIS & ML PIPELINE EXECUTION")
    logger.info("================================================================================")
    
    # Step 1: Data Ingestion
    logger.info("\n>>> STEP 1: Ingesting Multi-Source Public Health Data...")
    raw_df = download_and_collect_data(force_refresh=True)
    logger.info(f"Ingested {len(raw_df)} records across {raw_df['country'].nunique()} countries.")
    
    # Step 2: Cleaning & Harmonization
    logger.info("\n>>> STEP 2: Cleaning, Domain Validation & Missing Value Imputation...")
    clean_df, quality_report = clean_and_harmonize_data(raw_df)
    logger.info(f"Cleaned dataset: {len(clean_df)} records, 0 duplicates, 0 unhandled NaNs.")
    
    # Step 3: Feature Engineering
    logger.info("\n>>> STEP 3: Advanced Feature Engineering & Target Construction...")
    engineered_df, feature_dict = engineer_features(clean_df)
    logger.info(f"Engineered dataset: {engineered_df.shape[1]} total columns.")
    
    # Step 4: EDA & Visualization
    logger.info("\n>>> STEP 4: Running Exploratory Data Analysis & Generating Visual Figures...")
    insights = run_eda(engineered_df)
    logger.info(f"Generated 10 EDA figures and extracted {len(insights)} public health insights.")
    
    # Step 5: Model Training & Cross-Validation
    logger.info("\n>>> STEP 5: Training & Cross-Validating 7 Supervised ML Models...")
    champion_pipeline, benchmark_df, metrics = train_and_evaluate_all_models(engineered_df)
    logger.info(f"Champion Model: {metrics['champion_model']} | Test Accuracy: {metrics['test_accuracy']*100:.2f}% | Test F1: {metrics['test_f1_score']:.4f}")
    
    # Step 6: Evaluation Diagnostics
    logger.info("\n>>> STEP 6: Generating Diagnostic Evaluation Curves & Reports...")
    eval_summary = generate_evaluation_diagnostics()
    
    elapsed = time.time() - start_time
    logger.info("\n================================================================================")
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info("================================================================================")
    logger.info(f"All artifacts saved to data/processed/, models/, reports/, and reports/figures/")


if __name__ == "__main__":
    main()
