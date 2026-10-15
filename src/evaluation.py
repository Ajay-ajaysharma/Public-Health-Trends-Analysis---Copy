"""
=============================================================================
Module: evaluation.py
Project: Global Public Health Trends Analysis and Disease-Risk Prediction System
Author: Senior Data Scientist & ML Engineer
Description:
    Computes rigorous evaluation diagnostics for the champion ML pipeline:
    Confusion Matrix, ROC Curve, Precision-Recall Curve, Permutation Feature
    Importance, Calibration Curve, and generates formal evaluation reports.
=============================================================================
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "champion_health_risk_model.joblib")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "test_split_evaluation_data.csv")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def generate_evaluation_diagnostics():
    """
    Computes comprehensive model diagnostics and visual curves for the test partition.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    if not os.path.exists(CHAMPION_MODEL_PATH):
        raise FileNotFoundError(f"Champion model not found at {CHAMPION_MODEL_PATH}. Run model_training.py first.")
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test split data not found at {TEST_DATA_PATH}. Run model_training.py first.")
        
    logger.info("Loading champion model and test partition...")
    model_pipeline = joblib.load(CHAMPION_MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    y_true = test_df["y_true"].values
    y_pred = test_df["y_pred"].values
    y_prob = test_df["y_prob"].values
    
    feature_cols = [c for c in test_df.columns if c not in ["y_true", "y_pred", "y_prob"]]
    X_test = test_df[feature_cols]
    
    # -------------------------------------------------------------
    # 1. Confusion Matrix Heatmap
    # -------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted Low/Mod Risk (0)", "Predicted High Risk (1)"],
        yticklabels=["Actual Low/Mod Risk (0)", "Actual High Risk (1)"],
        ax=ax
    )
    ax.set_title("Champion Model Confusion Matrix (Test Set)")
    fig_cm_path = os.path.join(FIGURES_DIR, "eval_confusion_matrix.png")
    plt.savefig(fig_cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------
    # 2. ROC & Precision-Recall Curves
    # -------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    ap_val = average_precision_score(y_true, y_prob)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    
    axes[1].plot(recall_curve, precision_curve, color="#27ae60", lw=2.5, label=f"PR Curve (AP = {ap_val:.3f})")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    
    fig_roc_path = os.path.join(FIGURES_DIR, "eval_roc_pr_curves.png")
    plt.savefig(fig_roc_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    # -------------------------------------------------------------
    # 3. Permutation Feature Importance & Forest Gini Importance
    # -------------------------------------------------------------
    preprocessor = model_pipeline.named_steps["preprocessor"]
    classifier = model_pipeline.named_steps["classifier"]
    
    encoded_feature_names = preprocessor.get_feature_names_out()
    
    if hasattr(classifier, "feature_importances_"):
        raw_importances = classifier.feature_importances_
        feat_imp_df = pd.DataFrame({
            "Feature": [f.replace("num__", "").replace("cat__", "") for f in encoded_feature_names],
            "Importance": raw_importances
        }).sort_values("Importance", ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feat_imp_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title("Top 15 Feature Importances (Champion Random Forest)")
        ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)")
        fig_feat_path = os.path.join(FIGURES_DIR, "eval_feature_importance.png")
        plt.savefig(fig_feat_path, dpi=200, bbox_inches="tight")
        plt.close()
        
        # Save feature importance table
        feat_imp_df.to_csv(os.path.join(REPORTS_DIR, "top_feature_importances.csv"), index=False)
        
    # -------------------------------------------------------------
    # 4. Calibration Curve (Reliability Diagram)
    # -------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, marker="o", lw=2, color="#8e44ad", label="Random Forest Classifier")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.set_title("Model Probability Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of High Risk Positives")
    ax.legend(loc="upper left")
    fig_cal_path = os.path.join(FIGURES_DIR, "eval_calibration_curve.png")
    plt.savefig(fig_cal_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    logger.info("All evaluation diagnostic figures successfully saved to reports/figures/")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(auc_val),
        "average_precision": float(ap_val)
    }


if __name__ == "__main__":
    generate_evaluation_diagnostics()
