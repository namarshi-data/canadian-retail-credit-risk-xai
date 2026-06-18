from __future__ import annotations

"""Portfolio KPI and EDA monitoring utilities.

These utilities support Notebook 04 and provide aggregate, GitHub-safe portfolio
monitoring outputs. They do not fit models, transform modelling features, or
produce borrower-level prediction data.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COL = "defaulter"


def safe_pct(numerator: float, denominator: float) -> float:
    """Return percentage with zero-division protection."""
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator / denominator * 100)


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return requested columns that exist in the DataFrame."""
    return [col for col in columns if col in df.columns]


def clean_for_grouping(series: pd.Series) -> pd.Series:
    """Use stable string labels for grouping outputs."""
    return series.astype("object").where(series.notna(), "Missing").astype(str)


def portfolio_overview(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Build executive-level portfolio KPIs for the cleaned dataset."""
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found.")

    y = pd.to_numeric(df[target], errors="coerce")
    row_count = int(len(df))
    default_count = int(y.eq(1).sum())
    non_default_count = int(y.eq(0).sum())

    amount = pd.to_numeric(df["amount"], errors="coerce") if "amount" in df.columns else pd.Series(dtype=float)
    total_exposure = float(amount.sum(skipna=True)) if not amount.empty else np.nan
    defaulted_exposure = float(amount.loc[y.eq(1)].sum(skipna=True)) if not amount.empty else np.nan

    metrics = [
        ("row_count", row_count),
        ("column_count", int(df.shape[1])),
        ("default_count", default_count),
        ("non_default_count", non_default_count),
        ("default_rate_pct", safe_pct(default_count, row_count)),
        ("amount_missing_rate_pct", float(amount.isna().mean() * 100) if not amount.empty else np.nan),
        ("total_exposure", total_exposure),
        ("defaulted_exposure", defaulted_exposure),
        ("defaulted_exposure_share_pct", safe_pct(defaulted_exposure, total_exposure)),
        ("average_amount", float(amount.mean()) if not amount.empty else np.nan),
        ("median_amount", float(amount.median()) if not amount.empty else np.nan),
        ("median_income", float(df["total_income_pa"].median()) if "total_income_pa" in df.columns else np.nan),
        ("average_interest_rate", float(df["interest_rate"].mean()) if "interest_rate" in df.columns else np.nan),
        ("median_loan_to_income_ratio", float(df["loan_to_income_ratio"].median()) if "loan_to_income_ratio" in df.columns else np.nan),
        ("core_data_quality_issue_rate_pct", float(df["has_core_data_quality_issue"].mean() * 100) if "has_core_data_quality_issue" in df.columns else np.nan),
        ("broad_data_quality_issue_rate_pct", float(df["has_broad_data_quality_issue"].mean() * 100) if "has_broad_data_quality_issue" in df.columns else np.nan),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def target_distribution(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Return target class counts and shares."""
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not found.")

    out = (
        df[target]
        .value_counts(dropna=False)
        .rename_axis(target)
        .reset_index(name="row_count")
        .sort_values(target)
    )
    out["row_pct"] = out["row_count"] / len(df) * 100
    out["target_label"] = out[target].map({0: "Non-default", 1: "Default"}).fillna(out[target].astype(str))
    return out[[target, "target_label", "row_count", "row_pct"]]


def column_role_summary(df: pd.DataFrame, target: str = TARGET_COL) -> pd.DataFrame:
    """Classify columns for monitoring and governance review."""
    rows = []
    for col in df.columns:
        lower = col.lower()
        if col == target:
            role = "target"
            note = "Outcome field; never use as a predictor."
        elif lower in {"user_id", "customer_id", "record_sequence"} or lower.endswith("_id"):
            role = "identifier"
            note = "Audit key only."
        elif any(token in lower for token in ["payment", "principal", "interest_received"]):
            role = "repayment_monitoring_only"
            note = "Potential timing leakage for predictive modelling."
        elif lower in {"gender", "married", "pincode", "social_profile", "has_social_profile"}:
            role = "sensitive_or_proxy_monitoring_only"
            note = "Use only for permitted governance/fairness review."
        elif lower.endswith("_flag") or lower.startswith("has_") or lower.endswith("_issue_count"):
            role = "data_quality_or_risk_flag"
            note = "Useful monitoring signal; interpret carefully."
        elif pd.api.types.is_numeric_dtype(df[col]):
            role = "numeric_monitoring_feature"
            note = "Safe for aggregate EDA review."
        else:
            role = "categorical_monitoring_feature"
            note = "Safe for aggregate EDA review."
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 4),
                "unique_values": int(df[col].nunique(dropna=True)),
                "monitoring_role": role,
                "governance_note": note,
            }
        )
    return pd.DataFrame(rows)


def segment_profile(
    df: pd.DataFrame,
    segment_col: str,
    target: str = TARGET_COL,
    min_rows: int = 100,
    include_exposure: bool = True,
) -> pd.DataFrame:
    """Summarize default, exposure, and concentration by a categorical segment."""
    if segment_col not in df.columns:
        raise KeyError(f"{segment_col!r} not found in DataFrame.")
    if target not in df.columns:
        raise KeyError(f"{target!r} not found in DataFrame.")

    temp_cols = [segment_col, target] + existing_columns(df, ["amount", "total_income_pa", "interest_rate", "loan_to_income_ratio"])
    temp = df[temp_cols].copy()
    temp[segment_col] = clean_for_grouping(temp[segment_col])
    temp[target] = pd.to_numeric(temp[target], errors="coerce")

    grouped = temp.groupby(segment_col, dropna=False)
    out = grouped[target].agg(row_count="size", default_count="sum", default_rate="mean").reset_index()
    out = out.rename(columns={segment_col: "segment_value"})
    out["segment_column"] = segment_col
    out["default_rate_pct"] = out["default_rate"] * 100
    out["portfolio_share_pct"] = out["row_count"] / len(df) * 100

    if include_exposure and "amount" in temp.columns:
        exposure = grouped["amount"].agg(exposure="sum", median_amount="median").reset_index().rename(columns={segment_col: "segment_value"})
        out = out.merge(exposure, on="segment_value", how="left")
        total_exposure = pd.to_numeric(temp["amount"], errors="coerce").sum(skipna=True)
        out["exposure_share_pct"] = np.where(total_exposure > 0, out["exposure"] / total_exposure * 100, np.nan)

    for optional_col, out_name in [
        ("total_income_pa", "median_income"),
        ("interest_rate", "median_interest_rate"),
        ("loan_to_income_ratio", "median_loan_to_income_ratio"),
    ]:
        if optional_col in temp.columns:
            add = grouped[optional_col].median().reset_index(name=out_name).rename(columns={segment_col: "segment_value"})
            out = out.merge(add, on="segment_value", how="left")

    out = out.loc[out["row_count"].ge(min_rows)].copy()
    ordered_cols = ["segment_column", "segment_value"] + [c for c in out.columns if c not in {"segment_column", "segment_value"}]
    return out[ordered_cols].sort_values(["default_rate_pct", "row_count"], ascending=[False, False]).reset_index(drop=True)


def build_segment_risk_all(
    df: pd.DataFrame,
    segment_columns: Iterable[str] | None = None,
    target: str = TARGET_COL,
    min_rows: int = 100,
) -> pd.DataFrame:
    """Build a combined segment-risk table for selected columns."""
    if segment_columns is None:
        segment_columns = [
            "loan_category",
            "employment_type",
            "tier_of_employment",
            "work_experience",
            "home",
            "married",
            "is_verified",
            "amount_missing_flag",
            "has_core_data_quality_issue",
            "has_broad_data_quality_issue",
        ]

    tables = []
    for col in existing_columns(df, segment_columns):
        table = segment_profile(df, col, target=target, min_rows=min_rows)
        if not table.empty:
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def high_risk_segments(
    segment_risk: pd.DataFrame,
    portfolio_default_rate_pct: float,
    min_rows: int = 250,
    lift_threshold: float = 1.25,
) -> pd.DataFrame:
    """Return segments with elevated default-rate lift."""
    if segment_risk.empty:
        return pd.DataFrame()
    out = segment_risk.copy()
    out["default_rate_lift"] = np.where(
        portfolio_default_rate_pct > 0,
        out["default_rate_pct"] / portfolio_default_rate_pct,
        np.nan,
    )
    return (
        out.loc[out["row_count"].ge(min_rows) & out["default_rate_lift"].ge(lift_threshold)]
        .sort_values(["default_rate_lift", "row_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def quantile_profile(
    df: pd.DataFrame,
    value_col: str,
    target: str = TARGET_COL,
    q: int = 10,
) -> pd.DataFrame:
    """Create default-rate profile across quantile buckets of a numeric variable."""
    if value_col not in df.columns:
        raise KeyError(f"{value_col!r} not found in DataFrame.")

    temp = df[[value_col, target]].copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp[target] = pd.to_numeric(temp[target], errors="coerce")
    temp = temp.dropna(subset=[value_col, target]).copy()
    if temp.empty or temp[value_col].nunique(dropna=True) < 2:
        return pd.DataFrame()

    if value_col in {"delinq_2yrs", "number_of_loans"}:
        temp["bucket"] = pd.cut(
            temp[value_col],
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=["0", "1", "2", "3+"],
            right=True,
        ).astype(str)
    elif temp[value_col].nunique(dropna=True) <= q:
        temp["bucket"] = temp[value_col].astype(str)
    else:
        temp["bucket"] = pd.qcut(temp[value_col], q=min(q, temp[value_col].nunique()), duplicates="drop").astype(str)

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
    out["feature"] = value_col
    out["default_rate_pct"] = out["default_count"] / out["row_count"] * 100
    return out[["feature", "bucket", "row_count", "default_count", "default_rate_pct", "min_value", "max_value", "median_value"]].sort_values("min_value").reset_index(drop=True)


def build_quantile_profiles(
    df: pd.DataFrame,
    numeric_columns: Iterable[str] | None = None,
    target: str = TARGET_COL,
) -> pd.DataFrame:
    """Build combined numeric quantile risk profiles."""
    if numeric_columns is None:
        numeric_columns = [
            "amount",
            "total_income_pa",
            "interest_rate",
            "loan_to_income_ratio",
            "tenure_years",
            "delinq_2yrs",
            "number_of_loans",
        ]
    tables = []
    for col in existing_columns(df, numeric_columns):
        table = quantile_profile(df, col, target=target)
        if not table.empty:
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def flag_profile(
    df: pd.DataFrame,
    flag_cols: Iterable[str] | None = None,
    target: str = TARGET_COL,
) -> pd.DataFrame:
    """Summarize default rates by data-quality and governance flags."""
    if flag_cols is None:
        flag_cols = [
            col for col in df.columns
            if col.endswith("_flag") or col.startswith("has_") or col.endswith("_issue_count")
        ]

    rows: list[dict] = []
    for flag in existing_columns(df, flag_cols):
        for flag_value, part in df.groupby(flag, dropna=False):
            rows.append(
                {
                    "flag": flag,
                    "flag_value": flag_value,
                    "row_count": int(len(part)),
                    "row_pct": float(len(part) / len(df) * 100),
                    "default_count": int(pd.to_numeric(part[target], errors="coerce").sum()),
                    "default_rate_pct": float(pd.to_numeric(part[target], errors="coerce").mean() * 100),
                    "monitoring_note": "Missingness/data-quality flags should not be interpreted as borrower behaviour alone.",
                }
            )
    return pd.DataFrame(rows).sort_values(["flag", "flag_value"]).reset_index(drop=True)


def hhi(series: pd.Series) -> float:
    """Herfindahl-Hirschman index for row concentration."""
    shares = series.value_counts(normalize=True, dropna=False)
    return float((shares ** 2).sum())


def concentration_metrics(
    df: pd.DataFrame,
    segment_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create concentration metrics for row count and exposure."""
    if segment_columns is None:
        segment_columns = ["loan_category", "employment_type", "tier_of_employment", "home", "is_verified"]

    rows = []
    for col in existing_columns(df, segment_columns):
        row = {
            "segment_column": col,
            "unique_segments": int(df[col].nunique(dropna=False)),
            "hhi_row_concentration": round(hhi(df[col]), 6),
        }
        if "amount" in df.columns:
            exposure = pd.to_numeric(df["amount"], errors="coerce")
            exposure_by_segment = exposure.groupby(clean_for_grouping(df[col])).sum(min_count=1)
            if exposure_by_segment.sum() > 0:
                shares = exposure_by_segment / exposure_by_segment.sum()
                row["hhi_exposure_concentration"] = round(float((shares ** 2).sum()), 6)
                row["top_exposure_segment"] = str(exposure_by_segment.sort_values(ascending=False).index[0])
                row["top_exposure_share_pct"] = round(float(shares.max() * 100), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hhi_row_concentration", ascending=False).reset_index(drop=True)


def save_portfolio_monitoring_tables(
    df: pd.DataFrame,
    output_dir: str | Path,
    target: str = TARGET_COL,
    figure_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Create and save core Notebook 04 aggregate monitoring tables.

    Returns a dictionary of table names to DataFrames. Output filenames use the
    `04_` prefix for notebook-stage traceability.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview = portfolio_overview(df, target=target)
    distribution = target_distribution(df, target=target)
    segment_risk = build_segment_risk_all(df, target=target)
    portfolio_default_rate = float(pd.to_numeric(df[target], errors="coerce").mean() * 100)
    high_risk = high_risk_segments(segment_risk, portfolio_default_rate)
    quantiles = build_quantile_profiles(df, target=target)
    flags = flag_profile(df, target=target)
    concentration = concentration_metrics(df)
    roles = column_role_summary(df, target=target)

    tables = {
        "04_portfolio_overview": overview,
        "04_target_distribution": distribution,
        "04_column_roles": roles,
        "04_segment_risk_all": segment_risk,
        "04_high_risk_segments": high_risk,
        "04_numeric_quantile_risk": quantiles,
        "04_flag_risk_profile": flags,
        "04_concentration_metrics": concentration,
    }

    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    if figure_dir is not None:
        save_portfolio_figures(tables, figure_dir)

    return tables


def save_portfolio_figures(tables: dict[str, pd.DataFrame], figure_dir: str | Path) -> list[Path]:
    """Save a small, GitHub-safe set of aggregate portfolio figures."""
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    dist = tables.get("04_target_distribution", pd.DataFrame())
    if not dist.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(dist["target_label"], dist["row_count"])
        ax.set_title("Portfolio Target Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Record count")
        for i, row in dist.reset_index(drop=True).iterrows():
            ax.text(i, row["row_count"], f"{row['row_pct']:.1f}%", ha="center", va="bottom")
        path = figure_dir / "portfolio_target_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    segment = tables.get("04_segment_risk_all", pd.DataFrame())
    for col, file_name in [
        ("loan_category", "default_rate_by_loan_category.png"),
        ("employment_type", "default_rate_by_employment_type.png"),
    ]:
        plot_df = segment.loc[segment.get("segment_column", pd.Series(dtype=str)).eq(col)].copy() if not segment.empty else pd.DataFrame()
        if not plot_df.empty:
            plot_df = plot_df.sort_values("default_rate_pct", ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(plot_df["segment_value"].astype(str), plot_df["default_rate_pct"])
            ax.invert_yaxis()
            ax.set_title(f"Default Rate by {col.replace('_', ' ').title()}")
            ax.set_xlabel("Default rate (%)")
            ax.set_ylabel(col)
            path = figure_dir / file_name
            fig.tight_layout()
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

    quantile = tables.get("04_numeric_quantile_risk", pd.DataFrame())
    if not quantile.empty and "interest_rate" in set(quantile["feature"]):
        plot_df = quantile.loc[quantile["feature"].eq("interest_rate")].copy()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(plot_df)), plot_df["default_rate_pct"])
        ax.set_title("Default Rate by Interest Rate Quantile")
        ax.set_xlabel("Interest-rate bucket")
        ax.set_ylabel("Default rate (%)")
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["bucket"].astype(str), rotation=45, ha="right")
        path = figure_dir / "default_rate_by_interest_rate_quantile.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved
