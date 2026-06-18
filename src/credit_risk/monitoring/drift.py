from __future__ import annotations

"""Population and feature drift utilities for model monitoring.

The functions are designed for Notebook 09 style monitoring. They compare a
reference population, usually training, with a comparison population, usually
validation/test/current-month, using PSI and missingness drift.
"""

from typing import Iterable

import numpy as np
import pandas as pd


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_proportions(counts: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    total = counts.sum()
    if total <= 0:
        return counts.astype(float) + epsilon
    return (counts / total).clip(lower=epsilon)


def psi_from_proportions(expected_pct: pd.Series, actual_pct: pd.Series) -> pd.Series:
    """Calculate per-bin PSI contributions from aligned proportions."""
    expected_pct, actual_pct = expected_pct.align(actual_pct, fill_value=1e-6)
    expected_pct = expected_pct.clip(lower=1e-6)
    actual_pct = actual_pct.clip(lower=1e-6)
    return (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)


def numeric_psi(
    expected: pd.Series,
    actual: pd.Series,
    *,
    buckets: int = 10,
    feature: str | None = None,
) -> tuple[float, pd.DataFrame]:
    """Calculate PSI for a numeric feature using reference quantile bins."""
    feature = feature or expected.name or "numeric_feature"
    expected_num = _as_numeric(expected).dropna()
    actual_num = _as_numeric(actual).dropna()

    if expected_num.empty or actual_num.empty or expected_num.nunique() < 2:
        detail = pd.DataFrame(
            [{"feature": feature, "bucket": "insufficient_numeric_variation", "expected_pct": np.nan, "actual_pct": np.nan, "psi_contribution": np.nan}]
        )
        return np.nan, detail

    try:
        _, bin_edges = pd.qcut(expected_num, q=min(buckets, expected_num.nunique()), retbins=True, duplicates="drop")
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    except Exception:
        bin_edges = np.linspace(expected_num.min(), expected_num.max(), buckets + 1)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    expected_bins = pd.cut(expected_num, bins=bin_edges, include_lowest=True).astype(str)
    actual_bins = pd.cut(actual_num, bins=bin_edges, include_lowest=True).astype(str)

    expected_pct = _safe_proportions(expected_bins.value_counts(dropna=False))
    actual_pct = _safe_proportions(actual_bins.value_counts(dropna=False))
    contribution = psi_from_proportions(expected_pct, actual_pct)

    detail = (
        pd.DataFrame(
            {
                "feature": feature,
                "bucket": contribution.index,
                "expected_pct": expected_pct.reindex(contribution.index).values,
                "actual_pct": actual_pct.reindex(contribution.index).values,
                "psi_contribution": contribution.values,
            }
        )
        .sort_values("bucket")
        .reset_index(drop=True)
    )
    return float(contribution.sum()), detail


def categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
    *,
    feature: str | None = None,
    max_categories: int = 50,
) -> tuple[float, pd.DataFrame]:
    """Calculate PSI for a categorical feature."""
    feature = feature or expected.name or "categorical_feature"

    expected_cat = expected.astype("object").where(expected.notna(), "Missing").astype(str)
    actual_cat = actual.astype("object").where(actual.notna(), "Missing").astype(str)

    top_categories = expected_cat.value_counts(dropna=False).head(max_categories).index
    expected_grouped = expected_cat.where(expected_cat.isin(top_categories), "Other")
    actual_grouped = actual_cat.where(actual_cat.isin(top_categories), "Other")

    expected_pct = _safe_proportions(expected_grouped.value_counts(dropna=False))
    actual_pct = _safe_proportions(actual_grouped.value_counts(dropna=False))
    contribution = psi_from_proportions(expected_pct, actual_pct)

    detail = pd.DataFrame(
        {
            "feature": feature,
            "bucket": contribution.index,
            "expected_pct": expected_pct.reindex(contribution.index).values,
            "actual_pct": actual_pct.reindex(contribution.index).values,
            "psi_contribution": contribution.values,
        }
    ).reset_index(drop=True)

    return float(contribution.sum()), detail


def psi_severity(psi_value: float) -> str:
    """Classify PSI using common monitoring thresholds."""
    if pd.isna(psi_value):
        return "not_available"
    if psi_value < 0.10:
        return "stable"
    if psi_value < 0.25:
        return "moderate_shift"
    return "material_shift"


def feature_psi(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    feature: str,
    *,
    buckets: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """Calculate PSI for one feature using automatic numeric/categorical handling."""
    if feature not in reference_df.columns or feature not in comparison_df.columns:
        summary = {
            "feature": feature,
            "feature_type": "missing",
            "psi": np.nan,
            "severity": "missing_feature",
            "reference_missing_pct": np.nan,
            "comparison_missing_pct": np.nan,
            "missing_pct_gap": np.nan,
        }
        return summary, pd.DataFrame()

    ref = reference_df[feature]
    cmp = comparison_df[feature]
    is_numeric = pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cmp)

    if is_numeric:
        psi, detail = numeric_psi(ref, cmp, buckets=buckets, feature=feature)
        feature_type = "numeric"
    else:
        psi, detail = categorical_psi(ref, cmp, feature=feature)
        feature_type = "categorical"

    ref_missing = float(ref.isna().mean() * 100)
    cmp_missing = float(cmp.isna().mean() * 100)

    summary = {
        "feature": feature,
        "feature_type": feature_type,
        "psi": psi,
        "severity": psi_severity(psi),
        "reference_missing_pct": ref_missing,
        "comparison_missing_pct": cmp_missing,
        "missing_pct_gap": cmp_missing - ref_missing,
    }
    return summary, detail


def build_drift_report(
    reference_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    features: Iterable[str],
    *,
    reference_label: str = "reference",
    comparison_label: str = "comparison",
    buckets: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature-level PSI summary and bin-level detail tables."""
    summaries = []
    details = []

    for feature in features:
        summary, detail = feature_psi(reference_df, comparison_df, feature, buckets=buckets)
        summary["reference_label"] = reference_label
        summary["comparison_label"] = comparison_label
        summaries.append(summary)
        if not detail.empty:
            detail["reference_label"] = reference_label
            detail["comparison_label"] = comparison_label
            details.append(detail)

    summary_df = pd.DataFrame(summaries).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary_df, detail_df


def build_split_drift_report(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    split_col: str = "split",
    reference_split: str = "train",
    comparison_splits: Iterable[str] = ("validation", "test"),
    buckets: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare validation/test distributions against training split."""
    if split_col not in df.columns:
        raise KeyError(f"Split column {split_col!r} not found.")

    reference = df[df[split_col] == reference_split]
    if reference.empty:
        raise ValueError(f"Reference split {reference_split!r} is empty.")

    all_summaries = []
    all_details = []
    for split in comparison_splits:
        comparison = df[df[split_col] == split]
        if comparison.empty:
            continue
        summary, detail = build_drift_report(
            reference,
            comparison,
            features,
            reference_label=reference_split,
            comparison_label=split,
            buckets=buckets,
        )
        all_summaries.append(summary)
        if not detail.empty:
            all_details.append(detail)

    return (
        pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame(),
        pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame(),
    )


def drift_action_register(drift_summary: pd.DataFrame) -> pd.DataFrame:
    """Create monitoring actions from a drift summary table."""
    if drift_summary.empty:
        return pd.DataFrame()

    rows = []
    for _, row in drift_summary.iterrows():
        if row["severity"] == "material_shift":
            action = "Escalate for data drift investigation, score calibration review, and possible retraining assessment."
        elif row["severity"] == "moderate_shift":
            action = "Monitor closely and review data source, portfolio mix, and feature distribution changes."
        elif abs(row.get("missing_pct_gap", 0)) >= 5:
            action = "Investigate missingness drift and upstream data-quality changes."
        else:
            action = "No immediate action; continue scheduled monitoring."

        rows.append(
            {
                "feature": row["feature"],
                "comparison_label": row.get("comparison_label"),
                "psi": row["psi"],
                "severity": row["severity"],
                "missing_pct_gap": row.get("missing_pct_gap"),
                "recommended_action": action,
            }
        )
    return pd.DataFrame(rows)
