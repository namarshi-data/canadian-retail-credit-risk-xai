from __future__ import annotations

"""Feature engineering utilities for the Canadian retail credit-risk project.

Design principles
-----------------
- Create deterministic, business-explainable features only.
- Do not fit encoders, imputers, scalers, target encoders, resamplers, or models here.
- Build an unencoded modelling dataset with a leakage-reviewed feature policy.
- Run supervised screening on the training split only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import json

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover
    train_test_split = None

from credit_risk.features.encoding import DEFAULT_ORDINAL_CATEGORIES, make_encoding_plan_table
from credit_risk.features.leakage import (
    GOVERNANCE_MONITORING_FIELDS,
    HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS,
    ID_COLS,
    REPAYMENT_MONITORING_FIELDS,
    SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS,
    SPLIT_COL,
    TARGET_COL,
    TIMING_REVIEW_FIELDS,
    build_feature_usage_policy,
    feature_family_for_column,
    make_leakage_review_table,
    present_columns,
    select_baseline_features,
    validate_modeling_dataset_against_policy,
)
from credit_risk.features.selection import (
    make_feature_family_summary,
    make_multicollinearity_review,
    make_rare_category_review,
    make_train_only_univariate_screening,
)

BUSINESS_BIN_FEATURES = [
    "income_band",
    "loan_amount_band",
    "interest_rate_band",
    "tenure_band",
    "loan_to_income_band",
]


@dataclass
class FeatureEngineeringArtifacts:
    """Container for Notebook 05 generated artifacts."""

    engineered_df: pd.DataFrame
    modeling_df: pd.DataFrame
    feature_policy: pd.DataFrame
    leakage_review: pd.DataFrame
    feature_catalog: pd.DataFrame
    feature_lineage: pd.DataFrame
    preprocessing_plan: pd.DataFrame
    ordinal_mapping_plan: pd.DataFrame
    split_distribution: pd.DataFrame
    missingness_by_split: pd.DataFrame
    rare_category_review: pd.DataFrame
    train_only_univariate_screening: pd.DataFrame
    multicollinearity_review: pd.DataFrame
    feature_family_summary: pd.DataFrame
    qa_checks: pd.DataFrame
    output_manifest: pd.DataFrame


def save_table(df: pd.DataFrame, path: Path, *, float_format: str | None = None) -> None:
    """Save a CSV table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)


def save_json(obj: Mapping | list, path: Path) -> None:
    """Save a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    ratio = np.where((denominator > 0) & numerator.notna(), numerator / denominator, np.nan)
    return pd.Series(ratio, index=numerator.index)


def _fixed_band(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=True)
    return out.astype("object").where(values.notna(), "Missing")


def add_credit_risk_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic row-level credit-risk features.

    No target-derived, split-derived, encoded, scaled, or fitted transformations
    are created here.
    """
    out = df.copy()
    lineage_rows: list[dict] = []

    def add_lineage(feature: str, source_columns: list[str], transformation: str, rationale: str) -> None:
        lineage_rows.append(
            {
                "engineered_feature": feature,
                "source_columns": ", ".join(source_columns),
                "transformation_type": transformation,
                "business_rationale": rationale,
            }
        )

    if {"amount", "total_income_pa"}.issubset(out.columns):
        if "loan_to_income_ratio" not in out.columns:
            out["loan_to_income_ratio"] = _safe_ratio(out["amount"], out["total_income_pa"])
            transformation_type = "safe_ratio"
        else:
            transformation_type = "existing_cleaning_feature"
        add_lineage(
            "loan_to_income_ratio",
            ["amount", "total_income_pa"],
            transformation_type,
            "Measures borrower leverage relative to annual income.",
        )

        out["income_to_loan_buffer"] = pd.to_numeric(out["total_income_pa"], errors="coerce") - pd.to_numeric(out["amount"], errors="coerce")
        add_lineage(
            "income_to_loan_buffer",
            ["total_income_pa", "amount"],
            "difference",
            "Captures income buffer after loan amount.",
        )

        out["loan_to_income_missing_flag"] = (
            out["amount"].isna()
            | out["total_income_pa"].isna()
            | (pd.to_numeric(out["total_income_pa"], errors="coerce") <= 0)
        ).astype(int)
        add_lineage(
            "loan_to_income_missing_flag",
            ["amount", "total_income_pa"],
            "binary_flag",
            "Identifies unavailable affordability ratio caused by missing or invalid inputs.",
        )

    if "amount" in out.columns:
        out["loan_amount_band"] = _fixed_band(
            out["amount"],
            bins=[-np.inf, 10_000, 25_000, 50_000, 100_000, np.inf],
            labels=["<=10K", "10K-25K", "25K-50K", "50K-100K", ">100K"],
        )
        add_lineage(
            "loan_amount_band",
            ["amount"],
            "fixed_business_band",
            "Supports loan-size segmentation without target-learned cut points.",
        )

    if "total_income_pa" in out.columns:
        out["income_band"] = _fixed_band(
            out["total_income_pa"],
            bins=[-np.inf, 50_000, 75_000, 100_000, 150_000, np.inf],
            labels=["<=50K", "50K-75K", "75K-100K", "100K-150K", ">150K"],
        )
        add_lineage(
            "income_band",
            ["total_income_pa"],
            "fixed_business_band",
            "Enables income-risk segmentation using stable business thresholds.",
        )

    if "interest_rate" in out.columns:
        out["interest_rate_band"] = _fixed_band(
            out["interest_rate"],
            bins=[-np.inf, 8, 12, 16, 20, np.inf],
            labels=["<=8%", "8%-12%", "12%-16%", "16%-20%", ">20%"],
        )
        out["high_interest_flag"] = (pd.to_numeric(out["interest_rate"], errors="coerce") >= 16).astype(int)
        add_lineage("interest_rate_band", ["interest_rate"], "fixed_business_band", "Groups pricing levels for risk review.")
        add_lineage("high_interest_flag", ["interest_rate"], "binary_flag", "Flags relatively high loan pricing.")

    if "tenure_years" in out.columns:
        out["tenure_band"] = _fixed_band(
            out["tenure_years"],
            bins=[-np.inf, 1, 3, 5, np.inf],
            labels=["<=1Y", "1Y-3Y", "3Y-5Y", ">5Y"],
        )
        out["long_tenure_flag"] = (pd.to_numeric(out["tenure_years"], errors="coerce") > 5).astype(int)
        add_lineage("tenure_band", ["tenure_years"], "fixed_business_band", "Supports term-risk segmentation.")
        add_lineage("long_tenure_flag", ["tenure_years"], "binary_flag", "Identifies longer-duration loans.")

    if "loan_to_income_ratio" in out.columns:
        out["loan_to_income_band"] = _fixed_band(
            out["loan_to_income_ratio"],
            bins=[-np.inf, 0.10, 0.25, 0.50, 1.00, np.inf],
            labels=["<=10%", "10%-25%", "25%-50%", "50%-100%", ">100%"],
        )
        out["high_loan_to_income_flag"] = (pd.to_numeric(out["loan_to_income_ratio"], errors="coerce") > 0.50).astype(int)
        add_lineage("loan_to_income_band", ["loan_to_income_ratio"], "fixed_business_band", "Groups affordability burden.")
        add_lineage("high_loan_to_income_flag", ["loan_to_income_ratio"], "binary_flag", "Flags elevated loan burden.")

    missing_flag_cols = [col for col in out.columns if col.endswith("_missing_flag") or col.endswith("_missing_raw_flag")]
    if missing_flag_cols:
        out["borrower_missingness_flag_count"] = (
            out[missing_flag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).astype(int)
        )
        out["has_any_missingness_flag"] = (out["borrower_missingness_flag_count"] > 0).astype(int)
        add_lineage(
            "borrower_missingness_flag_count",
            missing_flag_cols,
            "row_level_sum",
            "Summarizes source-data incompleteness.",
        )
        add_lineage(
            "has_any_missingness_flag",
            ["borrower_missingness_flag_count"],
            "binary_flag",
            "Flags any missingness-related issue.",
        )

    return out, pd.DataFrame(lineage_rows)


def create_stratified_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> pd.Series:
    """Create 70/15/15 train/validation/test split with target stratification."""
    if train_test_split is None:
        raise ImportError("scikit-learn is required for stratified splitting.")
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found.")

    eligible = df[df[target_col].notna()].copy()
    train_idx, temp_idx = train_test_split(
        eligible.index,
        test_size=0.30,
        stratify=eligible[target_col],
        random_state=random_state,
    )
    temp_targets = eligible.loc[temp_idx, target_col]
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_targets,
        random_state=random_state,
    )

    split = pd.Series(index=df.index, data="unused", dtype="object")
    split.loc[train_idx] = "train"
    split.loc[valid_idx] = "validation"
    split.loc[test_idx] = "test"
    return split


def build_modeling_dataset(
    engineered_df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """Create leakage-reviewed, unencoded modelling dataset."""
    feature_policy = build_feature_usage_policy(engineered_df)
    baseline_features = select_baseline_features(feature_policy)
    baseline_features = [
        col
        for col in baseline_features
        if col in engineered_df.columns and col not in ID_COLS + [TARGET_COL, SPLIT_COL]
    ]

    modeling_cols = present_columns(engineered_df, ID_COLS) + [TARGET_COL] + baseline_features
    modeling_cols = list(dict.fromkeys(modeling_cols))
    modeling_df = engineered_df[modeling_cols].copy()
    modeling_df[SPLIT_COL] = create_stratified_splits(modeling_df, random_state=random_state)

    feature_cols = [c for c in modeling_df.columns if c not in ID_COLS + [TARGET_COL, SPLIT_COL]]
    numeric_features = modeling_df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    ordinal_features = present_columns(modeling_df, DEFAULT_ORDINAL_CATEGORIES.keys())

    return modeling_df, feature_policy, numeric_features, categorical_features, ordinal_features


def make_split_distribution(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    split_col: str = SPLIT_COL,
) -> pd.DataFrame:
    rows = []
    for split_name in ["train", "validation", "test"]:
        part = df[df[split_col] == split_name]
        rows.append(
            {
                "split": split_name,
                "row_count": int(len(part)),
                "row_share_pct": round(len(part) / len(df) * 100, 4) if len(df) else np.nan,
                "default_count": int(part[target_col].sum()) if target_col in part else np.nan,
                "default_rate_pct": round(part[target_col].mean() * 100, 4) if target_col in part and len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_feature_catalog(modeling_df: pd.DataFrame, feature_policy: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in modeling_df.columns if c not in ID_COLS + [TARGET_COL, SPLIT_COL]]
    rows = []
    for col in feature_cols:
        s = modeling_df[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "feature_family": feature_family_for_column(col),
                "non_null_count": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 4),
                "unique_values": int(s.nunique(dropna=True)),
                "example_values": ", ".join(map(str, s.dropna().astype(str).unique()[:5])),
            }
        )
    catalog = pd.DataFrame(rows)
    policy_cols = ["feature", "decision", "baseline_model_policy", "reason"]
    return (
        catalog.merge(feature_policy[policy_cols], on="feature", how="left")
        .sort_values(["feature_family", "feature"])
        .reset_index(drop=True)
    )


def make_missingness_by_split(modeling_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in modeling_df.columns if c not in ID_COLS + [TARGET_COL, SPLIT_COL]]
    rows = []
    for split_name in ["train", "validation", "test"]:
        part = modeling_df[modeling_df[SPLIT_COL] == split_name]
        for col in feature_cols:
            rows.append(
                {
                    "split": split_name,
                    "feature": col,
                    "row_count": int(len(part)),
                    "missing_count": int(part[col].isna().sum()),
                    "missing_pct": round(part[col].isna().mean() * 100, 4) if len(part) else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["missing_pct", "feature"], ascending=[False, True]).reset_index(drop=True)


def make_ordinal_mapping_plan(modeling_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, ordered_values in DEFAULT_ORDINAL_CATEGORIES.items():
        if feature not in modeling_df.columns:
            continue
        present = set(modeling_df[feature].dropna().astype(str).unique())
        rows.append(
            {
                "feature": feature,
                "recommended_encoder": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)",
                "ordered_values": " | ".join(ordered_values),
                "observed_unmapped_values": " | ".join(sorted(present - set(ordered_values))),
                "fit_stage": "fit_on_training_split_only",
            }
        )
    return pd.DataFrame(rows)


def build_preprocessing_groups(
    modeling_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    ordinal_features: list[str],
    cleaned_df: pd.DataFrame,
) -> dict:
    """Create the JSON column groups consumed by Notebook 06."""
    nominal_features = [col for col in categorical_features if col not in ordinal_features]
    binary_features = [
        col for col in numeric_features
        if set(pd.to_numeric(modeling_df[col], errors="coerce").dropna().unique()).issubset({0, 1})
    ]
    continuous_numeric_features = [col for col in numeric_features if col not in binary_features]

    return {
        "target": TARGET_COL,
        "id_columns": present_columns(modeling_df, ID_COLS),
        "split_column": SPLIT_COL,
        "continuous_numeric_features": continuous_numeric_features,
        "binary_features": binary_features,
        "nominal_categorical_features": nominal_features,
        "ordinal_categorical_features": ordinal_features,
        "excluded_monitoring_only_fields": present_columns(cleaned_df, REPAYMENT_MONITORING_FIELDS),
        "excluded_sensitive_or_proxy_fields": present_columns(cleaned_df, SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS),
        "excluded_high_cardinality_or_encrypted_fields": present_columns(cleaned_df, HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS),
        "timing_review_fields": present_columns(cleaned_df, TIMING_REVIEW_FIELDS),
        "fit_policy": "fit imputers, encoders, scalers, resamplers, and models on training split only",
    }


def make_qa_checks(raw_df: pd.DataFrame, engineered_df: pd.DataFrame, modeling_df: pd.DataFrame, feature_policy: pd.DataFrame) -> pd.DataFrame:
    checks = [
        {
            "check": "row_count_preserved_after_feature_engineering",
            "status": "pass" if len(raw_df) == len(engineered_df) == len(modeling_df) else "fail",
            "value": f"raw={len(raw_df)}, engineered={len(engineered_df)}, modeling={len(modeling_df)}",
        }
    ]
    policy_checks = validate_modeling_dataset_against_policy(modeling_df, feature_policy).to_dict("records")
    checks.extend(policy_checks)
    return pd.DataFrame(checks)


def make_output_manifest(paths: Mapping[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"artifact": name, "path": str(path), "exists": path.exists()} for name, path in paths.items()]
    )


def build_feature_engineering_artifacts(df: pd.DataFrame, random_state: int = 42) -> FeatureEngineeringArtifacts:
    engineered_df, feature_lineage = add_credit_risk_features(df)
    modeling_df, feature_policy, numeric_features, categorical_features, ordinal_features = build_modeling_dataset(engineered_df, random_state=random_state)

    groups = build_preprocessing_groups(modeling_df, numeric_features, categorical_features, ordinal_features, df)

    feature_catalog = make_feature_catalog(modeling_df, feature_policy)
    preprocessing_plan = make_encoding_plan_table(groups)
    ordinal_mapping_plan = make_ordinal_mapping_plan(modeling_df)
    split_distribution = make_split_distribution(modeling_df)
    missingness_by_split = make_missingness_by_split(modeling_df)
    rare_category_review = make_rare_category_review(modeling_df, categorical_features)
    train_only_univariate_screening = make_train_only_univariate_screening(modeling_df, numeric_features, categorical_features)
    multicollinearity_review = make_multicollinearity_review(modeling_df, numeric_features)
    feature_family_summary = make_feature_family_summary(feature_catalog)
    leakage_review = make_leakage_review_table(engineered_df)
    qa_checks = make_qa_checks(df, engineered_df, modeling_df, feature_policy)

    return FeatureEngineeringArtifacts(
        engineered_df=engineered_df,
        modeling_df=modeling_df,
        feature_policy=feature_policy,
        leakage_review=leakage_review,
        feature_catalog=feature_catalog,
        feature_lineage=feature_lineage,
        preprocessing_plan=preprocessing_plan,
        ordinal_mapping_plan=ordinal_mapping_plan,
        split_distribution=split_distribution,
        missingness_by_split=missingness_by_split,
        rare_category_review=rare_category_review,
        train_only_univariate_screening=train_only_univariate_screening,
        multicollinearity_review=multicollinearity_review,
        feature_family_summary=feature_family_summary,
        qa_checks=qa_checks,
        output_manifest=pd.DataFrame(),
    )


def save_feature_engineering_outputs(
    cleaned_df: pd.DataFrame,
    processed_dir: Path,
    table_dir: Path,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Build, save, and return all Notebook 05 artifacts."""
    processed_dir = Path(processed_dir)
    table_dir = Path(table_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_feature_engineering_artifacts(cleaned_df, random_state=random_state)

    feature_cols = [c for c in artifacts.modeling_df.columns if c not in ID_COLS + [TARGET_COL, SPLIT_COL]]
    numeric_features = artifacts.modeling_df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    ordinal_features = present_columns(artifacts.modeling_df, DEFAULT_ORDINAL_CATEGORIES.keys())
    groups = build_preprocessing_groups(
        artifacts.modeling_df,
        numeric_features,
        categorical_features,
        ordinal_features,
        cleaned_df,
    )

    output_paths = {
        "engineered_dataset": processed_dir / "credit_risk_engineered_full_audit_dataset.csv",
        "modeling_dataset": processed_dir / "credit_risk_modeling_dataset.csv",
        "feature_policy": table_dir / "05_feature_leakage_and_usage_policy.csv",
        "leakage_review": table_dir / "05_feature_leakage_review.csv",
        "feature_catalog": table_dir / "05_modeling_feature_catalog.csv",
        "feature_lineage": table_dir / "05_feature_lineage.csv",
        "preprocessing_plan": table_dir / "05_preprocessing_pipeline_design.csv",
        "ordinal_mapping_plan": table_dir / "05_ordinal_mapping_plan.csv",
        "split_distribution": table_dir / "05_modeling_split_distribution.csv",
        "missingness_by_split": table_dir / "05_modeling_feature_missingness_by_split.csv",
        "rare_category_review": table_dir / "05_rare_category_review.csv",
        "train_only_univariate_screening": table_dir / "05_train_only_univariate_feature_screening.csv",
        "multicollinearity_review": table_dir / "05_multicollinearity_review.csv",
        "feature_family_summary": table_dir / "05_feature_family_summary.csv",
        "qa_checks": table_dir / "05_feature_engineering_qa_checks.csv",
        "numeric_feature_list": table_dir / "05_numeric_feature_list.csv",
        "categorical_feature_list": table_dir / "05_categorical_feature_list.csv",
        "preprocessing_column_groups_json": table_dir / "05_preprocessing_column_groups.json",
    }

    save_table(artifacts.engineered_df, output_paths["engineered_dataset"])
    save_table(artifacts.modeling_df, output_paths["modeling_dataset"])
    save_table(artifacts.feature_policy, output_paths["feature_policy"])
    save_table(artifacts.leakage_review, output_paths["leakage_review"])
    save_table(artifacts.feature_catalog, output_paths["feature_catalog"])
    save_table(artifacts.feature_lineage, output_paths["feature_lineage"])
    save_table(artifacts.preprocessing_plan, output_paths["preprocessing_plan"])
    save_table(artifacts.ordinal_mapping_plan, output_paths["ordinal_mapping_plan"])
    save_table(artifacts.split_distribution, output_paths["split_distribution"])
    save_table(artifacts.missingness_by_split, output_paths["missingness_by_split"])
    save_table(artifacts.rare_category_review, output_paths["rare_category_review"])
    save_table(artifacts.train_only_univariate_screening, output_paths["train_only_univariate_screening"])
    save_table(artifacts.multicollinearity_review, output_paths["multicollinearity_review"])
    save_table(artifacts.feature_family_summary, output_paths["feature_family_summary"])
    save_table(artifacts.qa_checks, output_paths["qa_checks"])
    save_table(pd.DataFrame({"feature": numeric_features}), output_paths["numeric_feature_list"])
    save_table(pd.DataFrame({"feature": categorical_features}), output_paths["categorical_feature_list"])
    save_json(groups, output_paths["preprocessing_column_groups_json"])

    manifest = make_output_manifest(output_paths)
    manifest_path = table_dir / "05_feature_engineering_output_manifest.csv"
    save_table(manifest, manifest_path)
    artifacts.output_manifest = manifest

    return {
        "engineered_df": artifacts.engineered_df,
        "modeling_df": artifacts.modeling_df,
        "feature_policy": artifacts.feature_policy,
        "leakage_review": artifacts.leakage_review,
        "feature_catalog": artifacts.feature_catalog,
        "feature_lineage": artifacts.feature_lineage,
        "preprocessing_plan": artifacts.preprocessing_plan,
        "ordinal_mapping_plan": artifacts.ordinal_mapping_plan,
        "split_distribution": artifacts.split_distribution,
        "missingness_by_split": artifacts.missingness_by_split,
        "rare_category_review": artifacts.rare_category_review,
        "train_only_univariate_screening": artifacts.train_only_univariate_screening,
        "multicollinearity_review": artifacts.multicollinearity_review,
        "feature_family_summary": artifacts.feature_family_summary,
        "qa_checks": artifacts.qa_checks,
        "output_manifest": manifest,
    }
