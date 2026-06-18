from __future__ import annotations

"""Run Notebook 03 data cleaning and preprocessing audit pipeline.

This script loads the safely merged interim dataset, applies centralized cleaning,
saves the cleaned analytical dataset, and exports audit tables used by later
feature-engineering, modelling, and governance notebooks.

It does not fit encoders, scalers, transformations, resampling methods, or models.
Those operations belong inside the modelling pipeline after train/validation/test
splitting.
"""

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from credit_risk.config import INTERIM_DIR, PROCESSED_DIR, TABLE_DIR, ensure_project_directories
except ImportError:  # pragma: no cover - portability fallback
    INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

    def ensure_project_directories() -> None:
        for path in [INTERIM_DIR, PROCESSED_DIR, TABLE_DIR]:
            path.mkdir(parents=True, exist_ok=True)

from credit_risk.data.cleaning import clean_credit_risk_dataset  # noqa: E402
from credit_risk.data.validation import build_logical_quality_checks, build_readiness_gate, validate_record_keys, validate_target  # noqa: E402

TARGET_COL = "defaulter"
INPUT_FILE = INTERIM_DIR / "credit_risk_merged_interim.csv"
CLEANED_CSV = PROCESSED_DIR / "credit_risk_cleaned.csv"
CLEANED_PARQUET = PROCESSED_DIR / "credit_risk_cleaned.parquet"


def present_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def save_table(df: pd.DataFrame, path: Path, *, float_format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)


def save_parquet_if_available(df: pd.DataFrame, path: Path) -> str:
    try:
        df.to_parquet(path, index=False)
        return "saved"
    except Exception as exc:  # engine unavailable or dtype issue
        return f"skipped_{type(exc).__name__}"


def make_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "row_count": len(df),
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(4),
            "unique_values": df.nunique(dropna=True).values,
        }
    )
    summary["complete_pct"] = (100 - summary["missing_pct"]).round(4)
    summary["missingness_severity"] = pd.cut(
        summary["missing_pct"],
        bins=[-0.01, 0, 5, 20, 50, 100],
        labels=["No missingness", "Low missingness", "Moderate missingness", "High missingness", "Severe missingness"],
    )
    return summary.sort_values(["missing_pct", "missing_count"], ascending=False).reset_index(drop=True)


def make_missingness_by_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    if target_col not in df.columns:
        return pd.DataFrame()
    feature_cols = [col for col in df.columns if col != target_col]
    class_counts = df[target_col].value_counts(dropna=False).reindex([0, 1], fill_value=0)
    counts = (
        df[feature_cols]
        .isna()
        .groupby(df[target_col])
        .sum()
        .T.reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: "missing_count_non_default", 1: "missing_count_default"})
        .reset_index()
        .rename(columns={"index": "column"})
    )
    counts.columns.name = None
    counts["non_default_row_count"] = int(class_counts.loc[0])
    counts["default_row_count"] = int(class_counts.loc[1])
    counts["missing_rate_non_default"] = np.where(
        counts["non_default_row_count"] > 0,
        counts["missing_count_non_default"] / counts["non_default_row_count"] * 100,
        0,
    )
    counts["missing_rate_default"] = np.where(
        counts["default_row_count"] > 0,
        counts["missing_count_default"] / counts["default_row_count"] * 100,
        0,
    )
    counts["absolute_gap"] = (counts["missing_rate_default"] - counts["missing_rate_non_default"]).abs()
    counts["total_missing_count"] = counts["missing_count_non_default"] + counts["missing_count_default"]
    rate_cols = ["missing_rate_non_default", "missing_rate_default", "absolute_gap"]
    counts[rate_cols] = counts[rate_cols].round(4)
    return counts.query("total_missing_count > 0").sort_values(["absolute_gap", "total_missing_count"], ascending=False).reset_index(drop=True)


def make_column_lineage(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> pd.DataFrame:
    raw_columns = set(raw_df.columns)
    cleaned_columns = set(cleaned_df.columns)
    return pd.concat(
        [
            pd.DataFrame({"column": sorted(raw_columns & cleaned_columns), "lineage_status": "kept_from_raw"}),
            pd.DataFrame({"column": sorted(cleaned_columns - raw_columns), "lineage_status": "added_during_cleaning"}),
            pd.DataFrame({"column": sorted(raw_columns - cleaned_columns), "lineage_status": "removed_during_cleaning"}),
        ],
        ignore_index=True,
    )


def make_quality_flag_dictionary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    flag_cols = [col for col in df.columns if col.endswith("_flag") or col.startswith("has_")]
    count_cols = [col for col in df.columns if col.endswith("_issue_count") or col.endswith("_missing_count")]
    dictionary = pd.DataFrame(
        {
            "field": flag_cols + count_cols,
            "field_type": ["binary_flag"] * len(flag_cols) + ["count_feature"] * len(count_cols),
        }
    )
    if not dictionary.empty:
        dictionary["business_meaning"] = dictionary["field"].str.replace("_", " ", regex=False)
        dictionary["recommended_use"] = np.where(
            dictionary["field"].str.contains("quality|missing|placeholder|non_positive|exceeds", case=False, na=False),
            "audit_monitoring_and_possible_model_feature_after_policy_review",
            "review_before_use",
        )
    summary_rows = []
    for col in flag_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary_rows.append({"flag": col, "flagged_count": int((df[col] == 1).sum()), "flagged_pct": round((df[col] == 1).mean() * 100, 4)})
    summary = pd.DataFrame(summary_rows).sort_values("flagged_pct", ascending=False) if summary_rows else pd.DataFrame()
    return dictionary, summary


def identify_mathematical_numeric_columns(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    review = []
    for col in numeric_cols:
        s = df[col]
        non_null = s.dropna()
        n = len(non_null)
        unique_count = non_null.nunique()
        unique_ratio = unique_count / n if n > 0 else 0
        is_integer_like = pd.api.types.is_integer_dtype(s) or (n > 0 and np.all(np.isclose(non_null, np.round(non_null))))
        unique_values = set(non_null.unique()) if unique_count <= 20 else set()
        is_binary = unique_values.issubset({0, 1})
        is_likely_id = unique_ratio >= 0.95 and is_integer_like
        is_low_cardinality_integer = unique_count <= 15 and is_integer_like
        include_in_math_profile = not (is_binary or is_likely_id or is_low_cardinality_integer or col == TARGET_COL)
        if is_binary:
            inferred_type = "binary_flag"
        elif col == TARGET_COL:
            inferred_type = "target"
        elif is_likely_id:
            inferred_type = "likely_identifier"
        elif is_low_cardinality_integer:
            inferred_type = "low_cardinality_numeric_or_ordinal"
        else:
            inferred_type = "mathematical_numeric"
        review.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 4),
                "unique_values": int(unique_count),
                "unique_ratio": round(unique_ratio, 6),
                "inferred_type": inferred_type,
                "include_in_math_profile": include_in_math_profile,
            }
        )
    review_df = pd.DataFrame(review)
    math_numeric_cols = review_df.loc[review_df["include_in_math_profile"], "column"].tolist() if not review_df.empty else []
    return math_numeric_cols, review_df


def make_numeric_profile(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    profile = df[numeric_cols].replace([np.inf, -np.inf], np.nan).agg(["count", "mean", "median", "std", "min", "max", "skew"]).T.reset_index().rename(columns={"index": "column"})
    profile["missing_count"] = df[numeric_cols].isna().sum().values
    profile["missing_pct"] = df[numeric_cols].isna().mean().values * 100
    return profile.round(4)


def make_high_correlation_pairs(df: pd.DataFrame, numeric_cols: list[str], threshold: float = 0.80) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation", "abs_correlation"])
    corr = df[numeric_cols].replace([np.inf, -np.inf], np.nan).corr(numeric_only=True)
    upper_mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(upper_mask).stack().reset_index().rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return pairs.query("abs_correlation >= @threshold").sort_values("abs_correlation", ascending=False).reset_index(drop=True).round(6)


def make_skewness_report(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        skew = s.skew() if len(s) > 2 else np.nan
        min_value = s.min() if len(s) else np.nan
        rows.append(
            {
                "column": col,
                "skew": round(skew, 6) if pd.notna(skew) else np.nan,
                "min": min_value,
                "max": s.max() if len(s) else np.nan,
                "log_transform_candidate": bool(pd.notna(skew) and abs(skew) > 2 and min_value >= 0),
                "recommended_action": "Consider train-only log1p or robust transform in feature engineering" if pd.notna(skew) and abs(skew) > 2 and min_value >= 0 else "No cleaning-stage transformation",
            }
        )
    return pd.DataFrame(rows).sort_values("skew", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def make_outlier_review(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "column": col,
                "non_null_count": int(len(s)),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "min": s.min(),
                "p01": s.quantile(0.01),
                "q1": q1,
                "q3": q3,
                "p99": s.quantile(0.99),
                "max": s.max(),
                "iqr": iqr,
                "iqr_lower_bound": lower,
                "iqr_upper_bound": upper,
                "iqr_outlier_count": outlier_count,
                "iqr_outlier_pct": round(outlier_count / len(s) * 100, 4),
                "recommended_action": "Review and document; do not cap/remove blindly in cleaning",
            }
        )
    return pd.DataFrame(rows).sort_values("iqr_outlier_pct", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def make_categorical_cardinality_summary(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    rows = []
    for col in categorical_cols:
        s = df[col]
        value_counts = s.value_counts(dropna=True, normalize=True)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "row_count": len(df),
                "non_null_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 4),
                "unique_values": int(s.nunique(dropna=True)),
                "top_value": value_counts.index[0] if not value_counts.empty else np.nan,
                "top_value_share_pct": round(value_counts.iloc[0] * 100, 4) if not value_counts.empty else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["cardinality_level"] = pd.cut(
        summary["unique_values"],
        bins=[-1, 1, 10, 50, 500, np.inf],
        labels=["single_value", "low_cardinality", "medium_cardinality", "high_cardinality", "very_high_cardinality"],
    )
    return summary.sort_values(["unique_values", "missing_pct"], ascending=False).reset_index(drop=True)


def make_encoding_plan(df: pd.DataFrame) -> pd.DataFrame:
    ordinal_candidates = ["tier_of_employment", "work_experience"]
    nominal_candidates = ["married", "home", "loan_category", "employment_type", "is_verified"]
    high_cardinality_or_governance_exclude = ["industry", "role", "pincode", "gender", "social_profile"]
    rows = []
    for col in present_columns(df, nominal_candidates):
        rows.append({"feature": col, "feature_type": "nominal_categorical", "recommended_encoding": "one_hot_encode_after_split", "cleaning_stage_action": "fill_missing_as_unknown_if_needed", "notes": "Fit encoder inside modelling pipeline only."})
    for col in present_columns(df, ordinal_candidates):
        rows.append({"feature": col, "feature_type": "ordinal_categorical", "recommended_encoding": "ordinal_encode_after_split", "cleaning_stage_action": "standardize_unknown_or_placeholder_values", "notes": "Use documented order. Do not infer order from target."})
    for col in present_columns(df, high_cardinality_or_governance_exclude):
        rows.append({"feature": col, "feature_type": "high_cardinality_or_governance_sensitive", "recommended_encoding": "exclude_from_baseline_or_group_before_encoding", "cleaning_stage_action": "retain_for_audit_monitoring", "notes": "Do not direct one-hot encode in baseline model."})
    return pd.DataFrame(rows)


def make_run_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, parquet_status: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": ["raw_row_count", "cleaned_row_count", "raw_column_count", "cleaned_column_count", "raw_missing_values", "cleaned_missing_values", "target_present", "record_sequence_present", "parquet_status"],
            "value": [raw_df.shape[0], cleaned_df.shape[0], raw_df.shape[1], cleaned_df.shape[1], int(raw_df.isna().sum().sum()), int(cleaned_df.isna().sum().sum()), TARGET_COL in cleaned_df.columns, "record_sequence" in cleaned_df.columns, parquet_status],
        }
    )


def make_baseline_feature_list(model_feature_policy: pd.DataFrame) -> pd.DataFrame:
    if model_feature_policy is None or model_feature_policy.empty:
        return pd.DataFrame()
    policy = model_feature_policy.copy()
    feature_col = "feature" if "feature" in policy.columns else "column" if "column" in policy.columns else None
    policy_col = "recommended_use" if "recommended_use" in policy.columns else None
    if feature_col is None:
        return pd.DataFrame()
    if policy_col is None:
        return policy[[feature_col]].rename(columns={feature_col: "feature"})
    use = policy[policy_col].astype(str)
    baseline_mask = use.str.contains("candidate|include|model", case=False, na=False) & ~use.str.contains("exclude|audit|monitor|leakage|target|identifier", case=False, na=False)
    return policy.loc[baseline_mask, [feature_col, policy_col]].rename(columns={feature_col: "feature", policy_col: "baseline_model_policy"}).drop_duplicates().reset_index(drop=True)


def main() -> None:
    ensure_project_directories()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Interim dataset not found at {INPUT_FILE}. Run scripts/run_data_pipeline.py first.")

    raw_df = pd.read_csv(INPUT_FILE, low_memory=False)
    cleaning_result = clean_credit_risk_dataset(raw_df)
    cleaned_df = cleaning_result.cleaned

    cleaned_df.to_csv(CLEANED_CSV, index=False)
    parquet_status = save_parquet_if_available(cleaned_df, CLEANED_PARQUET)

    run_summary = make_run_summary(raw_df, cleaned_df, parquet_status)
    record_key_checks = validate_record_keys(cleaned_df)
    target_checks = pd.DataFrame([validate_target(cleaned_df, target=TARGET_COL)])
    logical_quality_checks = build_logical_quality_checks(cleaned_df)
    readiness_gate = build_readiness_gate(
        {
            "record_key_checks": record_key_checks,
            "target_checks": target_checks,
            "logical_quality_checks": logical_quality_checks,
        }
    )

    column_lineage = make_column_lineage(raw_df, cleaned_df)
    flag_dictionary, quality_flag_summary = make_quality_flag_dictionary(cleaned_df)
    missingness_summary = make_missingness_summary(cleaned_df)
    missingness_by_target = make_missingness_by_target(cleaned_df)
    categorical_cardinality = make_categorical_cardinality_summary(cleaned_df)
    math_numeric_cols, numeric_column_review = identify_mathematical_numeric_columns(cleaned_df)
    numeric_profile = make_numeric_profile(cleaned_df, math_numeric_cols)
    high_corr_pairs = make_high_correlation_pairs(cleaned_df, math_numeric_cols)
    skewness_report = make_skewness_report(cleaned_df, math_numeric_cols)
    outlier_review = make_outlier_review(cleaned_df, math_numeric_cols)
    encoding_plan = make_encoding_plan(cleaned_df)
    baseline_feature_list = make_baseline_feature_list(cleaning_result.model_feature_policy)

    outputs = {
        "03_cleaning_policy.csv": cleaning_result.cleaning_policy,
        "03_cleaning_audit_summary.csv": cleaning_result.audit_summary,
        "03_cleaning_flag_summary.csv": cleaning_result.flag_summary,
        "03_model_feature_policy.csv": cleaning_result.model_feature_policy,
        "03_cleaning_pipeline_run_summary.csv": run_summary,
        "03_cleaning_record_key_checks.csv": record_key_checks,
        "03_cleaning_target_checks.csv": target_checks,
        "03_cleaning_logical_quality_checks.csv": logical_quality_checks,
        "03_cleaning_readiness_gate.csv": readiness_gate,
        "03_column_lineage.csv": column_lineage,
        "03_quality_flag_dictionary.csv": flag_dictionary,
        "03_quality_flag_summary_from_cleaned_data.csv": quality_flag_summary,
        "03_post_clean_missingness.csv": missingness_summary,
        "03_post_clean_missingness_by_target.csv": missingness_by_target,
        "03_post_clean_categorical_cardinality.csv": categorical_cardinality,
        "03_post_clean_numeric_column_review.csv": numeric_column_review,
        "03_post_clean_numeric_profile.csv": numeric_profile,
        "03_post_clean_high_correlation_pairs.csv": high_corr_pairs,
        "03_post_clean_skewness_review.csv": skewness_report,
        "03_post_clean_outlier_review.csv": outlier_review,
        "03_encoding_and_transformation_plan.csv": encoding_plan,
        "03_baseline_model_feature_list.csv": baseline_feature_list,
    }
    for filename, table in outputs.items():
        save_table(table, TABLE_DIR / filename, float_format="%.6f")

    print("Cleaning pipeline completed successfully.")
    print(f"Input shape: {raw_df.shape}")
    print(f"Output shape: {cleaned_df.shape}")
    print(f"Processed CSV: {CLEANED_CSV}")
    print(f"Parquet status: {parquet_status}")

    print("\nRun summary:")
    print(run_summary.to_string(index=False))
    print("\nReadiness gate:")
    print(readiness_gate.to_string(index=False))

    if not cleaning_result.flag_summary.empty:
        print("\nTop cleaning flags:")
        print(cleaning_result.flag_summary.head(12).to_string(index=False))

    if not outlier_review.empty:
        print("\nTop numeric outlier fields to review, not automatically treat:")
        print(outlier_review.head(12).to_string(index=False))

    print("\nReminder: fit encoders, scalers, transformations, and resampling methods inside the modelling pipeline after train/validation/test split.")


if __name__ == "__main__":
    main()
