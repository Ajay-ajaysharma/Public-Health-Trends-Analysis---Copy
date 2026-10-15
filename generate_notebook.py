"""
Helper script to generate the structured portfolio Jupyter Notebook.
"""
import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🌍 Global Public Health Trends Analysis and Disease-Risk Prediction System\n",
                "### End-to-End Epidemiological Data Science, Feature Engineering & Machine Learning Project\n",
                "**Author:** Senior Data Scientist & Public Health Analyst  \n",
                "**Dataset Coverage:** 157 Countries, 2000–2023 (Our World in Data, WHO Global Health Observatory, World Bank)\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Project Background & Objectives\n",
                "Global public health outcomes are governed by a complex nexus of clinical infrastructure, immunization coverage, economic resources, and environmental risk factors.\n",
                "\n",
                "**Key Objectives:**\n",
                "1. **Harmonize** multi-source public health indicators across 157 countries over 24 years (2000–2023).\n",
                "2. **Engineer** composite indices for healthcare capacity, environmental pollution risk, and longitudinal velocity.\n",
                "3. **Perform EDA** to extract actionable epidemiological insights.\n",
                "4. **Train & Benchmark 7 ML Models** with 5-Fold Stratified Cross-Validation and zero data leakage.\n",
                "5. **Evaluate & Explain** model decision drivers using Gini & permutation feature importances."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "\n",
                "# Add project root to sys.path\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "\n",
                "from src.data_collection import download_and_collect_data\n",
                "from src.data_cleaning import clean_and_harmonize_data\n",
                "from src.feature_engineering import engineer_features\n",
                "from src.model_training import train_and_evaluate_all_models\n",
                "from src.evaluation import generate_evaluation_diagnostics\n",
                "\n",
                "print('All modules loaded successfully!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. End-to-End Pipeline Execution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Ingest, clean, and engineer features\n",
                "raw_df = download_and_collect_data(force_refresh=False)\n",
                "clean_df, quality_report = clean_and_harmonize_data(raw_df)\n",
                "df, feature_dict = engineer_features(clean_df)\n",
                "\n",
                "print(f'Master Dataset Shape: {df.shape}')\n",
                "df.head(5)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Exploratory Data Analysis & Epidemiological Trends"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Trend Analysis: Under-5 Mortality & Life Expectancy over time\n",
                "plt.figure(figsize=(14, 5))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "for region, group in df.groupby('who_region'):\n",
                "    y_mean = group.groupby('year')['life_expectancy'].mean()\n",
                "    plt.plot(y_mean.index, y_mean.values, label=region, lw=2)\n",
                "plt.title('Regional Life Expectancy Trajectory (2000-2023)')\n",
                "plt.xlabel('Year')\n",
                "plt.ylabel('Life Expectancy (Years)')\n",
                "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "for region, group in df.groupby('who_region'):\n",
                "    y_mean = group.groupby('year')['under_5_mortality_rate'].mean()\n",
                "    plt.plot(y_mean.index, y_mean.values, label=region, lw=2)\n",
                "plt.title('Regional Under-5 Mortality Trajectory (2000-2023)')\n",
                "plt.xlabel('Year')\n",
                "plt.ylabel('Deaths per 1,000 Live Births')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Training & 5-Fold Stratified Cross-Validation Benchmark"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "champion_pipeline, benchmark_df, final_metrics = train_and_evaluate_all_models(df)\n",
                "benchmark_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Champion Model Diagnostics & Explainability"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "eval_summary = generate_evaluation_diagnostics()\n",
                "print(json.dumps(eval_summary, indent=2))"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks", "public_health_analysis.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("Saved notebook to", nb_path)
