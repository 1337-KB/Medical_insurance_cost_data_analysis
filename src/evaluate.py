"""
evaluate.py
-----------
Evaluation utilities for the Medical Insurance Cost project.

Contains helpers for both regression and classification tasks, plus
the optimal-threshold.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    mean_squared_error,
    f1_score,
    classification_report,
)


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {"r2": round(r2, 4), "rmse": round(rmse, 2), "predictions": y_pred}


def print_regression_results(results: dict) -> None:
    print(f"R²   : {results['r2']}")
    print(f"RMSE : ${results['rmse']:,.2f}")


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(model, X_test, y_test, threshold: float = 0.5) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
        "probabilities": y_proba,
        "predictions": y_pred,
    }


def print_classification_results(results: dict) -> None:
    """Pretty-print classification evaluation results."""
    print(f"Accuracy       : {results['accuracy']}")
    print(f"Precision      : {results['precision']}")
    print(f"Recall         : {results['recall']}")
    print(f"F1 Score       : {results['f1']}")
    print(f"Avg Precision  : {results['avg_precision']}")


# ---------------------------------------------------------------------------
# Optimal threshold selection (Task 4)
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_test, y_proba) -> float:
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)

    # Avoid division by zero when both precision and recall are 0
    denom = precision_vals + recall_vals
    denom[denom == 0] = 1e-9

    f1_scores = 2 * (precision_vals * recall_vals) / denom
    best_idx = np.argmax(f1_scores)

    # precision_recall_curve returns one more precision/recall value than thresholds
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    return round(float(best_threshold), 2)


def print_threshold_report(y_test, y_pred_custom) -> None:
    """Print sklearn classification_report for predictions made at a custom threshold."""
    print(classification_report(y_test, y_pred_custom))
