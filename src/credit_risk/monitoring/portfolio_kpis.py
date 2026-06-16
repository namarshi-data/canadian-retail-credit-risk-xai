from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _safe_pct(numerator: float, denominator: float) -> float:
    """Return a percentage with zero-division protection."""
    if denominator in (0, 0.0) or pd.isna(denominator):
        return 0.0
    return float(numerator / denominator * 100)


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return only the requested columns that exist in the data frame."""
    return [col for col in columns if col in df.columns]


def portfolio_overview(df: pd.DataFrame, target: str = "defaulter") -> pd.DataFrame:
    """Build executive-level portfolio KPIs for the cleaned credit-risk dataset."""
    row_count = len(df)
    default_count = int(df[target].sum()) if target in df.columns else 0
    exposure = float(df["amount"].sum(skipna=True)) if "amount" in df.columns else np.nan
    default_exposure = (
        float(df.loc[df[target].eq(1), "amount"].sum(skipna=True))
        if target in df.columns and "amount" in df.columns
        else np.nan
    )

    metrics = [
        ("row_count", row_count),
        ("default_count", default_count),
        ("non_default_count", row_count - default_count),
        ("default_rate_pct", _safe_pct(default_count, row_count)),
        ("amount_missing_rate_pct", float(df["amount"].isna().mean() * 100) if "amount" in df.columns else np.nan),
        ("total_exposure", exposure),
        ("defaulted_exposure", default_exposure),
        ("defaulted_exposure_share_pct", _safe_pct(default_exposure, exposure) if pd.notna(exposure) else np.nan),
        ("median_loan_amount", float(df["amount"].median()) if "amount" in df.columns else np.nan),
        ("median_income", float(df["total_income_pa"].median()) if "total_income_pa" in df.columns else np.nan),
        ("average_interest_rate", float(df["interest_rate"].mean()) if "interest_rate" in df.columns else np.nan),
        ("median_loan_to_income_ratio", float(df["loan_to_income_ratio"].median()) if "loan_to_income_ratio" in df.columns else np.nan),
        ("core_data_quality_issue_rate_pct", float(df["has_core_data_quality_issue"].mean() * 100) if "has_core_data_quality_issue" in df.columns else np.nan),
        ("broad_data_quality_issue_rate_pct", float(df["has_broad_data_quality_issue"].mean() * 100) if "has_broad_data_quality_issue" in df.columns else np.nan),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def target_distribution(df: pd.DataFrame, target: str = "defaulter") -> pd.DataFrame:
    """Return default/non-default distribution."""
    out = (
        df[target]
        .value_counts(dropna=False)
        .rename_axis(target)
        .reset_index(name="row_count")
        .sort_values(target)
    )
    out["row_pct"] = out["row_count"] / len(df) * 100
    label_map = {0: "Non-default", 1: "Default"}
    out["target_label"] = out[target].map(label_map).fillna(out[target].astype(str))
    return out[[target, "target_label", "row_count", "row_pct"]]


def segment_profile(
    df: pd.DataFrame,
    segment_col: str,
    target: str = "defaulter",
    min_rows: int = 0,
    include_exposure: bool = True,
) -> pd.DataFrame:
    """Summarise default and exposure metrics by a categorical segment."""
    if segment_col not in df.columns:
        raise KeyError(f"{segment_col} not found in dataframe")

    grouped = df.groupby(segment_col, dropna=False)
    out = grouped.agg(
        row_count=(target, "size"),
        default_count=(target, "sum"),
        median_income=("total_income_pa", "median") if "total_income_pa" in df.columns else (target, "size"),
        median_interest_rate=("interest_rate", "median") if "interest_rate" in df.columns else (target, "size"),
    ).reset_index()

    out["default_rate_pct"] = out["default_count"] / out["row_count"] * 100
    out["portfolio_share_pct"] = out["row_count"] / len(df) * 100

    if include_exposure and "amount" in df.columns:
        exposure = grouped["amount"].sum(min_count=1).reset_index(name="exposure")
        missing = grouped["amount"].apply(lambda s: float(s.isna().mean() * 100)).reset_index(name="amount_missing_rate_pct")
        median_amount = grouped["amount"].median().reset_index(name="median_amount")
        out = out.merge(exposure, on=segment_col, how="left")
        out = out.merge(missing, on=segment_col, how="left")
        out = out.merge(median_amount, on=segment_col, how="left")
        total_exposure = df["amount"].sum(skipna=True)
        out["exposure_share_pct"] = np.where(total_exposure > 0, out["exposure"] / total_exposure * 100, np.nan)

    if "loan_to_income_ratio" in df.columns:
        lti = grouped["loan_to_income_ratio"].median().reset_index(name="median_loan_to_income_ratio")
        out = out.merge(lti, on=segment_col, how="left")

    out = out[out["row_count"] >= min_rows].copy()
    return out.sort_values(["default_rate_pct", "row_count"], ascending=[False, False]).reset_index(drop=True)


def quantile_profile(
    df: pd.DataFrame,
    value_col: str,
    target: str = "defaulter",
    q: int = 5,
) -> pd.DataFrame:
    """Create a default-rate profile across quantile buckets of a numeric variable."""
    if value_col not in df.columns:
        raise KeyError(f"{value_col} not found in dataframe")

    temp = df[[value_col, target]].copy()
    temp = temp[temp[value_col].notna()].copy()
    if temp.empty:
        return pd.DataFrame(columns=["bucket", "row_count", "default_count", "default_rate_pct"])

    unique_values = temp[value_col].nunique(dropna=True)

    # Discrete count-style variables such as delinquency count are more useful
    # as business buckets than as quantiles, especially when most values are zero.
    if value_col == "delinq_2yrs":
        temp["bucket"] = pd.cut(
            temp[value_col],
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=["0", "1", "2", "3+"],
            right=True,
        ).astype(str)
    elif value_col == "number_of_loans":
        temp["bucket"] = pd.cut(
            temp[value_col],
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=["0", "1", "2", "3+"],
            right=True,
        ).astype(str)
    elif unique_values <= q:
        temp["bucket"] = temp[value_col].astype(str)
    else:
        bucket_count = min(q, unique_values)
        temp["bucket"] = pd.qcut(temp[value_col], q=bucket_count, duplicates="drop")

        # Very skewed variables can collapse to one qcut bucket. Fall back to
        # actual values when the cardinality is manageable; otherwise use an
        # equal-width cut to preserve a risk-gradient view.
        bucket_categories = getattr(temp["bucket"], "cat", None)
        collapsed_bucket_count = len(bucket_categories.categories) if bucket_categories is not None else temp["bucket"].nunique()
        if collapsed_bucket_count <= 1:
            if unique_values <= 20:
                temp["bucket"] = temp[value_col].astype(str)
            else:
                temp["bucket"] = pd.cut(temp[value_col], bins=q, duplicates="drop")

        temp["bucket"] = temp["bucket"].astype(str)

    out = (
        temp.groupby("bucket", observed=False)
        .agg(
            row_count=(target, "size"),
            default_count=(target, "sum"),
            min_value=(value_col, "min"),
            max_value=(value_col, "max"),
            median_value=(value_col, "median"),
        )
        .reset_index()
    )
    out["default_rate_pct"] = out["default_count"] / out["row_count"] * 100
    return out.sort_values("min_value").reset_index(drop=True)


def flag_profile(
    df: pd.DataFrame,
    flag_cols: Iterable[str] | None = None,
    target: str = "defaulter",
) -> pd.DataFrame:
    """Summarise default rate by data-quality or governance flags."""
    if flag_cols is None:
        flag_cols = [
            col
            for col in df.columns
            if col.endswith("_flag")
            or col in ["has_core_data_quality_issue", "has_broad_data_quality_issue"]
        ]

    rows: list[dict[str, float | int | str]] = []
    for flag in _existing_columns(df, flag_cols):
        for flag_value, part in df.groupby(flag, dropna=False):
            rows.append(
                {
                    "flag": flag,
                    "flag_value": flag_value,
                    "row_count": int(len(part)),
                    "row_pct": float(len(part) / len(df) * 100),
                    "default_count": int(part[target].sum()),
                    "default_rate_pct": float(part[target].mean() * 100),
                }
            )
    return pd.DataFrame(rows).sort_values(["flag", "flag_value"]).reset_index(drop=True)


def save_portfolio_monitoring_tables(
    df: pd.DataFrame,
    output_dir: str | Path,
    target: str = "defaulter",
) -> dict[str, pd.DataFrame]:
    """Create and save the core monitoring tables used by Notebook 04."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pd.DataFrame] = {
        "portfolio_overview": portfolio_overview(df, target=target),
        "target_distribution": target_distribution(df, target=target),
        "data_quality_flag_profile": flag_profile(df, target=target),
    }

    for segment in _existing_columns(
        df,
        [
            "loan_category",
            "employment_type",
            "tier_of_employment",
            "work_experience",
            "home",
            "is_verified",
            "amount_missing_flag",
            "has_core_data_quality_issue",
        ],
    ):
        tables[f"segment_profile_{segment}"] = segment_profile(df, segment, target=target)

    for numeric_col in _existing_columns(
        df,
        ["amount", "total_income_pa", "interest_rate", "loan_to_income_ratio", "delinq_2yrs"],
    ):
        tables[f"quantile_profile_{numeric_col}"] = quantile_profile(df, numeric_col, target=target)

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    return tables
