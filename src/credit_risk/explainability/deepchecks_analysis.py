from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def run_optional_deepchecks_model_evaluation(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_html_path: Path,
) -> pd.DataFrame:
    """Run Deepchecks model-evaluation suite when installed; otherwise return status."""
    rows: list[dict[str, Any]] = []
    try:
        from deepchecks.tabular import Dataset  # type: ignore
        from deepchecks.tabular.suites import model_evaluation  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return pd.DataFrame(
            [
                {
                    "check_name": "Deepchecks availability",
                    "status": "not_run",
                    "finding": f"Deepchecks is not installed or could not be imported: {exc}",
                    "recommended_action": "Install deepchecks to run automated model-diagnostics HTML reports; fallback diagnostics are still generated.",
                }
            ]
        )

    try:  # pragma: no cover - environment dependent
        train_df = X_train.copy()
        train_df["defaulter"] = y_train.values
        test_df = X_test.copy()
        test_df["defaulter"] = y_test.values
        cat_features = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
        train_ds = Dataset(train_df, label="defaulter", cat_features=cat_features)
        test_ds = Dataset(test_df, label="defaulter", cat_features=cat_features)
        suite = model_evaluation()
        result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=model)
        output_html_path.parent.mkdir(parents=True, exist_ok=True)
        result.save_as_html(str(output_html_path))
        rows.append(
            {
                "check_name": "Deepchecks model_evaluation suite",
                "status": "completed",
                "finding": f"Deepchecks report saved to {output_html_path}",
                "recommended_action": "Review failed/warned checks for weak segments, calibration issues, and overfitting symptoms.",
            }
        )
    except Exception as exc:
        rows.append(
            {
                "check_name": "Deepchecks model_evaluation suite",
                "status": "error",
                "finding": f"Deepchecks was available but failed during execution: {exc}",
                "recommended_action": "Review Deepchecks version/API compatibility; fallback diagnostics are still generated.",
            }
        )
    return pd.DataFrame(rows)


def fallback_model_weakness_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    scores: pd.Series,
    threshold: float,
    segment_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create dependency-light diagnostics when Deepchecks is unavailable."""
    segment_columns = list(segment_columns or [])
    pred = scores.ge(threshold).astype(int)
    df = X.copy()
    df["defaulter"] = y.astype(int)
    df["score"] = scores
    df["predicted_high_risk"] = pred
    df["false_positive"] = ((pred == 1) & (y == 0)).astype(int)
    df["false_negative"] = ((pred == 0) & (y == 1)).astype(int)

    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "diagnostic_area": "threshold_error_profile",
            "segment": "overall",
            "row_count": len(df),
            "observed_default_rate": float(df["defaulter"].mean()),
            "average_score": float(df["score"].mean()),
            "review_rate": float(df["predicted_high_risk"].mean()),
            "false_positive_rate_within_segment": float(df["false_positive"].mean()),
            "false_negative_rate_within_segment": float(df["false_negative"].mean()),
            "recommended_action": "Use threshold and segment analysis to manage false positives and false negatives.",
        }
    )

    for col in segment_columns:
        if col not in df.columns:
            continue
        grouped = df.groupby(df[col].fillna("Missing").astype(str), observed=False)
        for value, part in grouped:
            if len(part) < 100:
                continue
            rows.append(
                {
                    "diagnostic_area": f"segment_weakness::{col}",
                    "segment": str(value),
                    "row_count": int(len(part)),
                    "observed_default_rate": float(part["defaulter"].mean()),
                    "average_score": float(part["score"].mean()),
                    "review_rate": float(part["predicted_high_risk"].mean()),
                    "false_positive_rate_within_segment": float(part["false_positive"].mean()),
                    "false_negative_rate_within_segment": float(part["false_negative"].mean()),
                    "recommended_action": "Investigate segment-specific calibration, feature availability, and threshold behavior.",
                }
            )
    return pd.DataFrame(rows).sort_values(["diagnostic_area", "false_negative_rate_within_segment", "false_positive_rate_within_segment"], ascending=[True, False, False])


def model_enhancement_recommendations(metrics: dict[str, Any], weakness_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Turn diagnostics into practical model-improvement actions."""
    rows = [
        {
            "area": "False negatives",
            "finding": f"False negatives at threshold: {metrics.get('false_negative')}",
            "recommendation": "Review missed-default cohorts and consider segment-specific thresholds only if governance allows.",
            "expected_effect": "Reduce missed defaults, possibly at higher review rate.",
        },
        {
            "area": "False positives",
            "finding": f"False positives at threshold: {metrics.get('false_positive')}",
            "recommendation": "Review high-score non-default cohorts; improve calibration and add affordability/behavioural features if available.",
            "expected_effect": "Improve precision and reduce reviewer workload.",
        },
        {
            "area": "Calibration",
            "finding": f"Brier score: {metrics.get('brier_score'):.4f}" if isinstance(metrics.get("brier_score"), float) else "Brier score available in metrics.",
            "recommendation": "Consider calibration testing after champion selection; do not change ranking model solely for calibration.",
            "expected_effect": "More reliable risk probabilities and threshold interpretation.",
        },
        {
            "area": "Data and features",
            "finding": "Performance ceiling appears feature-limited, not purely algorithm-limited.",
            "recommendation": "Prioritize leakage-safe repayment trend, utilization, bureau, and affordability features where available.",
            "expected_effect": "Potentially higher PR-AUC and better separation between default/non-default accounts.",
        },
    ]
    if not weakness_diagnostics.empty:
        worst = weakness_diagnostics.sort_values("false_negative_rate_within_segment", ascending=False).head(1).iloc[0]
        rows.append(
            {
                "area": "Weak segment review",
                "finding": f"Highest FN-rate segment in fallback diagnostics: {worst.get('diagnostic_area')} / {worst.get('segment')}",
                "recommendation": "Inspect SHAP drivers and data quality for this segment before deploying segment-specific actions.",
                "expected_effect": "Improved governance understanding of model weaknesses.",
            }
        )
    return pd.DataFrame(rows)
