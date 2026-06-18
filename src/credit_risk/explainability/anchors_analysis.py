from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any

import numpy as np
import pandas as pd

from credit_risk.explainability.shap_analysis import humanize_feature_name, raw_feature_from_transformed_name


@dataclass
class AnchorCondition:
    """One human-readable condition used in an anchor-style rule."""

    raw_feature: str
    operator: str
    value: object

    def describe(self) -> str:
        if self.operator == "==":
            return f"{humanize_feature_name(self.raw_feature)} = {self.value}"
        if isinstance(self.value, (int, float, np.number)):
            return f"{humanize_feature_name(self.raw_feature)} {self.operator} {float(self.value):,.2f}"
        return f"{humanize_feature_name(self.raw_feature)} {self.operator} {self.value}"


def _condition_mask(X: pd.DataFrame, condition: AnchorCondition) -> pd.Series:
    if condition.raw_feature not in X.columns:
        return pd.Series(False, index=X.index)
    series = X[condition.raw_feature]
    if condition.operator == "==":
        if condition.value == "Missing":
            return series.isna()
        return series.fillna("Missing").astype(str).eq(str(condition.value))
    numeric = pd.to_numeric(series, errors="coerce")
    if condition.operator == ">=":
        return numeric.ge(float(condition.value))
    if condition.operator == "<":
        return numeric.lt(float(condition.value))
    raise ValueError(f"Unsupported operator: {condition.operator}")


def build_condition_for_feature(X_reference: pd.DataFrame, row: pd.Series, raw_feature: str) -> AnchorCondition | None:
    """Create a stable condition for a raw feature value."""
    if raw_feature not in X_reference.columns or raw_feature not in row.index:
        return None
    value = row[raw_feature]
    if pd.isna(value):
        return AnchorCondition(raw_feature, "==", "Missing")
    if pd.api.types.is_numeric_dtype(X_reference[raw_feature]):
        numeric_ref = pd.to_numeric(X_reference[raw_feature], errors="coerce")
        unique_values = pd.Series(numeric_ref.dropna().unique())
        if len(unique_values) <= 3:
            try:
                clean_value: Any = int(value) if float(value).is_integer() else float(value)
            except Exception:
                clean_value = value
            return AnchorCondition(raw_feature, "==", clean_value)
        median_value = float(numeric_ref.median())
        operator = ">=" if float(value) >= median_value else "<"
        return AnchorCondition(raw_feature, operator, median_value)
    return AnchorCondition(raw_feature, "==", value)


def evaluate_anchor_conditions(
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    scores_reference: pd.Series,
    conditions: list[AnchorCondition],
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate coverage and precision-style quality for an anchor-style rule."""
    if not conditions:
        return {
            "coverage": 0.0,
            "covered_rows": 0,
            "model_high_risk_precision": np.nan,
            "observed_default_rate": np.nan,
        }
    mask = pd.Series(True, index=X_reference.index)
    for condition in conditions:
        mask &= _condition_mask(X_reference, condition)
    covered_rows = int(mask.sum())
    if covered_rows == 0:
        return {
            "coverage": 0.0,
            "covered_rows": 0,
            "model_high_risk_precision": np.nan,
            "observed_default_rate": np.nan,
        }
    return {
        "coverage": float(mask.mean()),
        "covered_rows": covered_rows,
        "model_high_risk_precision": float((scores_reference.loc[mask] >= threshold).mean()),
        "observed_default_rate": float(y_reference.loc[mask].mean()),
    }


def build_anchor_like_rules(
    X_reference: pd.DataFrame,
    y_reference: pd.Series,
    scores_reference: pd.Series,
    X_explain: pd.DataFrame,
    shap_explain: pd.DataFrame,
    candidate_indices: Iterable[int],
    threshold: float,
    raw_feature_names: Sequence[str] | None = None,
    max_conditions: int = 3,
) -> pd.DataFrame:
    """Build anchor-style rules from top positive SHAP contributors.

    These are dependency-light, auditable anchor-style rules. If your project installs
    Alibi, these rows still provide a good business-friendly baseline explanation.
    """
    score_series = pd.Series(scores_reference, index=X_reference.index)
    rows: list[dict[str, Any]] = []
    for idx in candidate_indices:
        if idx not in X_explain.index or idx not in shap_explain.index or idx not in score_series.index:
            continue
        row = X_explain.loc[idx]
        positive_features = shap_explain.loc[idx].sort_values(ascending=False)
        conditions: list[AnchorCondition] = []
        used_raw_features: set[str] = set()
        for transformed_feature, shap_value in positive_features.items():
            if float(shap_value) <= 0:
                continue
            raw_feature = raw_feature_from_transformed_name(str(transformed_feature), raw_feature_names)
            if raw_feature in used_raw_features:
                continue
            condition = build_condition_for_feature(X_reference, row, raw_feature)
            if condition is None:
                continue
            conditions.append(condition)
            used_raw_features.add(raw_feature)
            if len(conditions) >= max_conditions:
                break
        metrics = evaluate_anchor_conditions(X_reference, y_reference, score_series, conditions, threshold)
        rows.append(
            {
                "row_index": idx,
                "predicted_default_probability": float(score_series.loc[idx]),
                "operating_threshold": float(threshold),
                "predicted_high_risk": int(score_series.loc[idx] >= threshold),
                "rule": " AND ".join(condition.describe() for condition in conditions),
                "condition_count": len(conditions),
                "explanation_type": "anchor_style_shap_rule",
                "interpretation": "When these conditions hold, the model often treats similar accounts as high risk.",
                "governance_note": "Use as a model explanation aid, not as a standalone decision rule or customer instruction.",
                **metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model_high_risk_precision", "coverage"], ascending=[False, False]).reset_index(drop=True)
