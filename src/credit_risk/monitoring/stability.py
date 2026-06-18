from __future__ import annotations

"""Stability and monitoring limit utilities for model governance.

These functions support Notebook 09 by checking split stability, metric changes,
missingness stability, and monitoring risk-limit status.
"""

import re
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COL = "defaulter"
SPLIT_COL = "split"


def parse_numeric_limit(value) -> float:
    """Parse numbers from values such as '> 0.10', '29.23%', or '$5.8M'."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    multiplier = 1.0
    if "%" in text:
        multiplier = 0.01
    if text.upper().endswith("M"):
        multiplier = 1_000_000.0
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    return float(match.group()) * multiplier


def split_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    split_col: str = SPLIT_COL,
) -> pd.DataFrame:
    """Summarize row counts and default rates by split."""
    if split_col not in df.columns:
        raise KeyError(f"Split column {split_col!r} not found.")

    rows = []
    for split_name, part in df.groupby(split_col, dropna=False):
        row = {
            "split": split_name,
            "row_count": int(len(part)),
            "row_share_pct": round(float(len(part) / len(df) * 100), 4) if len(df) else np.nan,
        }
        if target_col in part.columns:
            y = pd.to_numeric(part[target_col], errors="coerce")
            row["default_count"] = int(y.eq(1).sum())
            row["default_rate_pct"] = round(float(y.mean() * 100), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def target_stability_by_split(
    split_summary: pd.DataFrame,
    baseline_split: str = "train",
) -> pd.DataFrame:
    """Compare default-rate stability by split against a baseline split."""
    if split_summary.empty or "default_rate_pct" not in split_summary.columns:
        return pd.DataFrame()

    baseline_rows = split_summary.loc[split_summary["split"].astype(str).eq(baseline_split)]
    if baseline_rows.empty:
        return split_summary.assign(default_rate_gap_vs_baseline=np.nan)

    baseline_rate = float(baseline_rows.iloc[0]["default_rate_pct"])
    out = split_summary.copy()
    out["baseline_split"] = baseline_split
    out["baseline_default_rate_pct"] = baseline_rate
    out["default_rate_gap_vs_baseline_pct_points"] = out["default_rate_pct"] - baseline_rate
    out["stability_status"] = np.where(
        out["default_rate_gap_vs_baseline_pct_points"].abs() <= 0.25,
        "stable",
        np.where(out["default_rate_gap_vs_baseline_pct_points"].abs() <= 1.00, "review", "escalate"),
    )
    return out


def missingness_stability_by_split(
    df: pd.DataFrame,
    features: Iterable[str],
    split_col: str = SPLIT_COL,
    baseline_split: str = "train",
) -> pd.DataFrame:
    """Compare feature missingness by split."""
    features = [f for f in features if f in df.columns]
    rows = []
    for split_name, part in df.groupby(split_col, dropna=False):
        for feature in features:
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "row_count": int(len(part)),
                    "missing_pct": round(float(part[feature].isna().mean() * 100), 4) if len(part) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    baseline = out.loc[out["split"].astype(str).eq(baseline_split), ["feature", "missing_pct"]].rename(
        columns={"missing_pct": "baseline_missing_pct"}
    )
    out = out.merge(baseline, on="feature", how="left")
    out["missing_pct_gap_vs_baseline"] = out["missing_pct"] - out["baseline_missing_pct"]
    out["stability_status"] = np.where(
        out["missing_pct_gap_vs_baseline"].abs() <= 2,
        "stable",
        np.where(out["missing_pct_gap_vs_baseline"].abs() <= 5, "review", "escalate"),
    )
    return out.sort_values(["stability_status", "missing_pct_gap_vs_baseline"], ascending=[True, False]).reset_index(drop=True)


def numeric_stability_by_split(
    df: pd.DataFrame,
    numeric_features: Iterable[str],
    split_col: str = SPLIT_COL,
    baseline_split: str = "train",
) -> pd.DataFrame:
    """Compare numeric medians and means by split against a baseline split."""
    numeric_features = [f for f in numeric_features if f in df.columns]
    rows = []
    for split_name, part in df.groupby(split_col, dropna=False):
        for feature in numeric_features:
            values = pd.to_numeric(part[feature], errors="coerce")
            rows.append(
                {
                    "split": split_name,
                    "feature": feature,
                    "non_null_count": int(values.notna().sum()),
                    "mean": float(values.mean()) if values.notna().any() else np.nan,
                    "median": float(values.median()) if values.notna().any() else np.nan,
                    "std": float(values.std()) if values.notna().sum() > 1 else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    baseline = out.loc[out["split"].astype(str).eq(baseline_split), ["feature", "mean", "median"]].rename(
        columns={"mean": "baseline_mean", "median": "baseline_median"}
    )
    out = out.merge(baseline, on="feature", how="left")
    out["mean_relative_change"] = np.where(out["baseline_mean"].abs() > 0, (out["mean"] - out["baseline_mean"]) / out["baseline_mean"], np.nan)
    out["median_relative_change"] = np.where(out["baseline_median"].abs() > 0, (out["median"] - out["baseline_median"]) / out["baseline_median"], np.nan)
    out["stability_status"] = np.where(
        out["median_relative_change"].abs() <= 0.05,
        "stable",
        np.where(out["median_relative_change"].abs() <= 0.10, "review", "escalate"),
    )
    return out.sort_values(["stability_status", "feature"]).reset_index(drop=True)


def model_metric_stability(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    metric_columns: Iterable[str] = ("roc_auc", "pr_auc", "recall", "precision", "review_rate", "business_cost"),
) -> pd.DataFrame:
    """Compare validation and test metrics for the selected operating model."""
    if validation_metrics.empty or test_metrics.empty:
        return pd.DataFrame()

    val = validation_metrics.iloc[0]
    test = test_metrics.iloc[0]
    rows = []
    for metric in metric_columns:
        if metric not in validation_metrics.columns or metric not in test_metrics.columns:
            continue
        val_value = float(val[metric])
        test_value = float(test[metric])
        abs_change = test_value - val_value
        rel_change = abs_change / val_value if val_value not in (0, 0.0) else np.nan
        rows.append(
            {
                "metric": metric,
                "validation_value": val_value,
                "test_value": test_value,
                "absolute_change": abs_change,
                "relative_change": rel_change,
                "stability_status": "stable" if pd.isna(rel_change) or abs(rel_change) <= 0.10 else "review",
            }
        )
    return pd.DataFrame(rows)


def evaluate_risk_limits(
    current_kpis: pd.DataFrame,
    risk_limit_register: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate current KPI values against warning/breach limits where possible.

    This is flexible because governance tables often store values as strings.
    It expects `risk_limit_register` columns:
    metric, baseline, warning_limit, breach_limit, monitoring_frequency, action
    """
    if risk_limit_register.empty:
        return pd.DataFrame()

    rows = []
    for _, row in risk_limit_register.iterrows():
        baseline = parse_numeric_limit(row.get("baseline"))
        warning = parse_numeric_limit(row.get("warning_limit"))
        breach = parse_numeric_limit(row.get("breach_limit"))

        status = "baseline_only"
        if pd.notna(breach) and pd.notna(baseline):
            # For the current project's limits, higher values indicate more risk.
            if baseline >= breach:
                status = "breach"
            elif pd.notna(warning) and baseline >= warning:
                status = "warning"
            else:
                status = "within_limit"

        rows.append(
            {
                "metric": row.get("metric"),
                "baseline": row.get("baseline"),
                "warning_limit": row.get("warning_limit"),
                "breach_limit": row.get("breach_limit"),
                "parsed_baseline": baseline,
                "parsed_warning_limit": warning,
                "parsed_breach_limit": breach,
                "status": status,
                "monitoring_frequency": row.get("monitoring_frequency"),
                "action": row.get("action"),
            }
        )
    return pd.DataFrame(rows)


def monitoring_readiness_gate(
    control_register: pd.DataFrame,
    risk_limit_register: pd.DataFrame,
    kpi_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact monitoring readiness checklist."""
    checks = [
        {
            "check": "control_register_available",
            "status": "pass" if not control_register.empty else "fail",
            "detail": f"{len(control_register)} controls",
        },
        {
            "check": "risk_limit_register_available",
            "status": "pass" if not risk_limit_register.empty else "fail",
            "detail": f"{len(risk_limit_register)} risk limits",
        },
        {
            "check": "kpi_snapshot_available",
            "status": "pass" if not kpi_snapshot.empty else "fail",
            "detail": f"{len(kpi_snapshot)} KPIs",
        },
        {
            "check": "manual_review_rate_limit_documented",
            "status": "pass"
            if not risk_limit_register.empty and risk_limit_register["metric"].astype(str).str.contains("review", case=False, na=False).any()
            else "review",
            "detail": "Review-rate capacity limit should be monitored.",
        },
    ]
    return pd.DataFrame(checks)
