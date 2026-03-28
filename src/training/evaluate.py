"""
Model evaluation utilities.

Why Precision-Recall AUC instead of ROC-AUC?
  With 3.5% fraud, a model that predicts 'not fraud' on everything achieves
  96.5% accuracy and ROC-AUC ~0.5. That looks fine on ROC-AUC but Precision-Recall
  AUC would correctly show it as ~0.035 (random baseline = class prior).

  Precision-Recall AUC is the right metric when:
    - Classes are highly imbalanced
    - False negatives (missed fraud) are expensive
    - You care about the performance at YOUR operating threshold, not all thresholds

Operating threshold concept:
  The model outputs a probability. The BUSINESS decides the threshold:
    < 0.3  → APPROVE  (low fraud risk)
    0.3-0.7 → REVIEW  (human review queue)
    > 0.7  → DECLINE  (high fraud risk, auto-block)

  The threshold determines your FP/FN trade-off. Lower threshold = catch more fraud
  but block more legitimate transactions (bad customer experience).
  This is a product decision, not a data science decision.
"""

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "",
) -> Dict[str, float]:
    """
    Compute full evaluation suite for a fraud model.

    Parameters
    ----------
    y_true : ground truth labels (0/1)
    y_prob : predicted fraud probabilities
    threshold : decision threshold for binary classification
    model_name : used in log output only

    Returns
    -------
    dict of metric_name → value (all floats, suitable for MLflow logging)
    """
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    # Precision @ 80% Recall: simulate a fraud team SLA
    # "We want to catch 80% of fraud. What precision can we achieve?"
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    recall_80_mask = recall_arr >= 0.80
    if recall_80_mask.any():
        prec_at_80_recall = precision_arr[recall_80_mask].max()
        threshold_at_80_recall = float(thresholds[recall_80_mask[:-1]].min())
    else:
        prec_at_80_recall = 0.0
        threshold_at_80_recall = 1.0

    metrics = {
        "pr_auc":              round(pr_auc, 4),
        "roc_auc":             round(roc_auc, 4),
        "f1":                  round(f1, 4),
        "precision_at_80_rec": round(prec_at_80_recall, 4),
        "threshold_at_80_rec": round(threshold_at_80_recall, 4),
    }

    name = f"[{model_name}] " if model_name else ""
    logger.info(f"{name}PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")
    logger.info(f"{name}Precision@80%Recall={prec_at_80_recall:.4f}")

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """
    Find the threshold that achieves a target recall while maximising precision.

    In a real fraud system, the business picks the recall target:
    'We want to catch X% of fraud — what threshold achieves that?'
    """
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
    valid = recall_arr[:-1] >= target_recall
    if valid.any():
        best_idx = np.argmax(precision_arr[:-1][valid])
        best_threshold = float(thresholds[valid][best_idx])
    else:
        best_threshold = 0.5
        logger.warning(f"Could not find threshold achieving {target_recall:.0%} recall; defaulting to 0.5")

    achieved_recall = recall_arr[:-1][thresholds <= best_threshold].max() if (thresholds <= best_threshold).any() else 0.0
    logger.info(f"Optimal threshold: {best_threshold:.3f} → recall {achieved_recall:.2%}")
    return best_threshold


def print_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))
