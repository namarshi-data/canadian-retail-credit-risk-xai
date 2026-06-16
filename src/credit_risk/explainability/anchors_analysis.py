from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk.explainability.shap_analysis import humanize_feature_name, raw_feature_from_transformed_name


@dataclass
class AnchorCondition:
    """One simple human-readable condition used in an anchor-like rule."""

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
    series = X[condition.raw_feature]
    if condition.operator == "==":
        return series.astype(str).eq(str(condition.value))
    numeric = pd.to_numeric(series, errors="coerce")
    if condition.operator == ">=":
        return numeric.ge(float(condition.value))
    if condition.operator == "<":
        return numeric.lt(float(condition.value))
    raise ValueError(f"Unsupported operator: {condition.operator}")


def build_condition_for_feature(X_reference: pd.DataFrame, row: pd.Series, raw_feature: str) -> AnchorCondition | None:
    """Create a stable rule condition for a raw feature."""
    if raw_feature not in X_reference.columns or raw_feature not in row.index:
        return None

    value = row[raw_feature]
    if pd.isna(value):
        return AnchorCondition(raw_feature, "==", "Missing")

    if pd.api.types.is_numeric_dtype(X_reference[raw_feature]):
        unique_values = pd.Series(X_reference[raw_feature].dropna().unique())
        if len(unique_values) <= 3:
            try:
                value = int(value) if float(value).is_integer() else value
            except Exception:
                pass
            return AnchorCondition(raw_feature, "==", value)
        median_value = float(pd.to_numeric(X_reference[raw_feature], errors="coerce").median())
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
    """Evaluate coverage and precision-style metrics for a set of conditions."""
    if not conditions:
        return {"coverage": 0.0, "covered_rows": 0, "model_high_risk_precision": np.nan, "observed_default_rate": np.nan}
    mask = pd.Series(True, index=X_reference.index)
    for condition in conditions:
        mask &= _condition_mask(X_reference, condition)

    covered_rows = int(mask.sum())
    if covered_rows == 0:
        return {"coverage": 0.0, "covered_rows": 0, "model_high_risk_precision": np.nan, "observed_default_rate": np.nan}

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
    max_conditions: int = 3,
) -> pd.DataFrame:
    """Build simple anchor-like rules from top positive SHAP contributors."""
    score_series = pd.Series(scores_reference, index=X_reference.index)
    rows = []
    for idx in candidate_indices:
        if idx not in X_explain.index or idx not in shap_explain.index:
            continue
        row = X_explain.loc[idx]
        positive_features = shap_explain.loc[idx].sort_values(ascending=False)
        conditions: list[AnchorCondition] = []
        used_raw_features = set()
        for transformed_feature, shap_value in positive_features.items():
            if shap_value <= 0:
                continue
            raw_feature = raw_feature_from_transformed_name(transformed_feature)
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
                "rule": " AND ".join(condition.describe() for condition in conditions),
                "condition_count": len(conditions),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["model_high_risk_precision", "coverage"], ascending=[False, False]).reset_index(drop=True)
