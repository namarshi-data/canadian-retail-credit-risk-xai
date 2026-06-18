from __future__ import annotations

"""Evaluation utilities for credit-risk default prediction.

This module intentionally evaluates probabilities, fixed-threshold labels, ranking
quality, calibration, confusion-matrix counts, review burden, and simple business
cost. It does not choose a threshold by looking at the test set.
"""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_FALSE_NEGATIVE_COST = 5_000
DEFAULT_FALSE_POSITIVE_COST = 500


def predict_default_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class default probabilities for a fitted classifier."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1 / (1 + np.exp(-scores))
    raise TypeError("Model must expose predict_proba or decision_function.")


def classification_metrics_at_threshold(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.50,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> dict[str, float | int]:
    """Calculate ranking, classification, calibration, and cost metrics."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.asarray(y_score).astype(float)

    if y_true_arr.size == 0:
        raise ValueError("Cannot evaluate an empty target array.")
    if not np.isfinite(y_score_arr).all():
        raise ValueError("Predicted probabilities contain non-finite values.")

    y_pred = (y_score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    business_cost = fn * false_negative_cost + fp * false_positive_cost

    # ROC-AUC is undefined when only one class appears. Keep output robust.
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_score_arr))
    except ValueError:
        pr_auc = np.nan

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_true_arr, y_score_arr)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true_arr, y_pred)),
        "review_rate": float(y_pred.mean()),
        "business_cost": float(business_cost),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "default_count": int(y_true_arr.sum()),
        "non_default_count": int((1 - y_true_arr).sum()),
    }


def evaluate_fitted_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    threshold: float = 0.50,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
) -> dict[str, float | int | str]:
    """Evaluate a fitted model and return one row of metrics."""
    y_score = predict_default_probability(model, X)
    metrics = classification_metrics_at_threshold(
        y,
        y_score,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return {
        "model_name": model_name,
        "dataset": dataset_name,
        **metrics,
    }


def make_prediction_frame(
    model_name: str,
    model,
    features: pd.DataFrame,
    identity_frame: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Create an auditable prediction frame with IDs, target, split, and probabilities."""
    scores = predict_default_probability(model, features)
    output = identity_frame.copy()
    output["model_name"] = model_name
    output["predicted_default_probability"] = scores
    output[f"predicted_label_at_{str(threshold).replace('.', '_')}"] = (scores >= threshold).astype(int)
    return output


def model_selection_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Rank validation results using credit-risk oriented model-selection criteria."""
    if results.empty:
        return pd.DataFrame()

    sort_cols = ["pr_auc", "roc_auc", "balanced_accuracy", "mcc"]
    available_sort_cols = [col for col in sort_cols if col in results.columns]
    ranked = results.sort_values(available_sort_cols, ascending=[False] * len(available_sort_cols)).copy()
    ranked["selection_rank"] = range(1, len(ranked) + 1)

    preferred_cols = [
        "selection_rank",
        "model_name",
        "dataset",
        "pr_auc",
        "roc_auc",
        "brier_score",
        "recall",
        "precision",
        "specificity",
        "f1",
        "balanced_accuracy",
        "mcc",
        "review_rate",
        "business_cost",
        "false_negative",
        "false_positive",
        "true_positive",
        "true_negative",
    ]
    return ranked[[col for col in preferred_cols if col in ranked.columns]].reset_index(drop=True)


def confusion_matrix_frame(metrics_row: pd.Series | dict) -> pd.DataFrame:
    """Return a tidy confusion-matrix table from one metric row."""
    row = pd.Series(metrics_row)
    return pd.DataFrame(
        [
            {"actual": 0, "predicted": 0, "count": int(row["true_negative"]), "cell": "true_negative"},
            {"actual": 0, "predicted": 1, "count": int(row["false_positive"]), "cell": "false_positive"},
            {"actual": 1, "predicted": 0, "count": int(row["false_negative"]), "cell": "false_negative"},
            {"actual": 1, "predicted": 1, "count": int(row["true_positive"]), "cell": "true_positive"},
        ]
    )


def score_decile_lift_table(
    predictions: pd.DataFrame,
    probability_col: str = "predicted_default_probability",
    target_col: str = "defaulter",
    bins: int = 10,
) -> pd.DataFrame:
    """Create decile lift and capture table from predictions."""
    df = predictions[[probability_col, target_col]].copy().dropna()
    if df.empty:
        return pd.DataFrame()

    # Highest-risk decile should be rank 1.
    df["risk_decile"] = pd.qcut(
        df[probability_col].rank(method="first", ascending=False),
        q=bins,
        labels=range(1, bins + 1),
    ).astype(int)

    overall_default_rate = df[target_col].mean()
    total_defaults = df[target_col].sum()

    table = (
        df.groupby("risk_decile", observed=True)
        .agg(
            row_count=(target_col, "size"),
            default_count=(target_col, "sum"),
            avg_score=(probability_col, "mean"),
            min_score=(probability_col, "min"),
            max_score=(probability_col, "max"),
        )
        .reset_index()
        .sort_values("risk_decile")
    )
    table["default_rate"] = table["default_count"] / table["row_count"]
    table["lift_vs_average"] = table["default_rate"] / overall_default_rate if overall_default_rate else np.nan
    table["cumulative_default_capture"] = table["default_count"].cumsum() / total_defaults if total_defaults else np.nan
    table["cumulative_review_rate"] = table["row_count"].cumsum() / table["row_count"].sum()
    return table


def calibration_by_score_band(
    predictions: pd.DataFrame,
    probability_col: str = "predicted_default_probability",
    target_col: str = "defaulter",
    bins: int = 10,
) -> pd.DataFrame:
    """Create a simple calibration table by predicted-score bands."""
    df = predictions[[probability_col, target_col]].copy().dropna()
    if df.empty:
        return pd.DataFrame()

    df["score_band"] = pd.qcut(df[probability_col].rank(method="first"), q=bins, duplicates="drop")
    table = (
        df.groupby("score_band", observed=True)
        .agg(
            row_count=(target_col, "size"),
            observed_default_rate=(target_col, "mean"),
            average_predicted_probability=(probability_col, "mean"),
        )
        .reset_index()
    )
    table["calibration_gap"] = table["average_predicted_probability"] - table["observed_default_rate"]
    return table
