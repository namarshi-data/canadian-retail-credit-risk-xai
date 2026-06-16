from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "defaulter"
IDENTIFIER_COLUMNS = ["user_id", "record_sequence"]

LEAKAGE_SENSITIVE_COLUMNS = [
    "total_payment",
    "received_principal",
    "interest_received",
    "payment_to_amount_ratio",
    "principal_to_amount_ratio",
    "interest_to_amount_ratio",
    "principal_exceeds_amount_flag",
]

FAIRNESS_PROXY_COLUMNS = [
    "gender",
    "married",
    "pincode",
    "social_profile",
]

HIGH_CARDINALITY_OR_LOW_INTERPRETABILITY_COLUMNS = [
    "industry",
    "role",
]

BASE_SAFE_NUMERIC_COLUMNS = [
    "amount",
    "interest_rate",
    "tenure_years",
    "total_income_pa",
    "dependents",
    "delinq_2yrs",
    "number_of_loans",
    "loan_to_income_ratio",
    "amount_missing_flag",
    "employment_type_missing_flag",
    "tier_of_employment_missing_flag",
    "industry_missing_flag",
    "work_experience_missing_flag",
    "married_missing_flag",
    "social_profile_missing_flag",
    "is_verified_missing_flag",
    "amount_non_positive_flag",
    "industry_placeholder_zero_flag",
    "work_experience_placeholder_zero_flag",
    "core_data_quality_issue_count",
    "has_core_data_quality_issue",
    "broad_data_quality_issue_count",
    "has_broad_data_quality_issue",
]

BASE_SAFE_CATEGORICAL_COLUMNS = [
    "loan_category",
    "employment_type",
    "tier_of_employment",
    "work_experience",
    "home",
    "is_verified",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "amount_log1p",
    "total_income_pa_log1p",
    "loan_to_income_ratio_log1p",
    "interest_rate_x_lti",
]

ENGINEERED_BINARY_COLUMNS = [
    "has_prior_delinquency_flag",
    "has_existing_loans_flag",
    "multiple_loans_flag",
    "high_interest_flag",
    "high_loan_to_income_flag",
    "very_high_loan_to_income_flag",
    "low_income_flag",
    "high_amount_flag",
]

ENGINEERED_CATEGORICAL_COLUMNS = [
    "interest_rate_band",
    "loan_to_income_band",
    "income_band",
    "amount_band",
    "dependents_band",
    "tenure_band",
]


def _existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def safe_log1p(series: pd.Series) -> pd.Series:
    """Return log1p after clipping negative values to zero."""
    return np.log1p(pd.to_numeric(series, errors="coerce").clip(lower=0))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-aware borrower and loan features for modelling.

    This function only uses variables that are available at or near underwriting / portfolio
    review time. Repayment-realization fields are intentionally not used.
    """
    output = df.copy()

    if "amount" in output.columns:
        output["amount_log1p"] = safe_log1p(output["amount"])
        output["high_amount_flag"] = (output["amount"] >= output["amount"].quantile(0.75)).astype("int64")
        output["amount_band"] = pd.cut(
            output["amount"],
            bins=[-np.inf, 25000, 50000, 100000, 250000, np.inf],
            labels=["<=25K", "25K-50K", "50K-100K", "100K-250K", ">250K"],
        ).astype("object").fillna("Missing")

    if "total_income_pa" in output.columns:
        output["total_income_pa_log1p"] = safe_log1p(output["total_income_pa"])
        output["low_income_flag"] = (output["total_income_pa"] <= output["total_income_pa"].quantile(0.25)).astype("int64")
        output["income_band"] = pd.cut(
            output["total_income_pa"],
            bins=[-np.inf, 40000, 60000, 90000, 120000, np.inf],
            labels=["<=40K", "40K-60K", "60K-90K", "90K-120K", ">120K"],
        ).astype("object").fillna("Missing")

    if "loan_to_income_ratio" in output.columns:
        output["loan_to_income_ratio_log1p"] = safe_log1p(output["loan_to_income_ratio"])
        output["high_loan_to_income_flag"] = (output["loan_to_income_ratio"] >= 2.0).astype("int64")
        output["very_high_loan_to_income_flag"] = (output["loan_to_income_ratio"] >= 4.0).astype("int64")
        output["loan_to_income_band"] = pd.cut(
            output["loan_to_income_ratio"],
            bins=[-np.inf, 0.5, 1.0, 2.0, 4.0, np.inf],
            labels=["<=0.5", "0.5-1.0", "1.0-2.0", "2.0-4.0", ">4.0"],
        ).astype("object").fillna("Missing")

    if {"interest_rate", "loan_to_income_ratio"}.issubset(output.columns):
        output["interest_rate_x_lti"] = output["interest_rate"] * output["loan_to_income_ratio"]

    if "interest_rate" in output.columns:
        output["high_interest_flag"] = (output["interest_rate"] >= 16).astype("int64")
        output["interest_rate_band"] = pd.cut(
            output["interest_rate"],
            bins=[-np.inf, 8, 12, 16, np.inf],
            labels=["<=8%", "8%-12%", "12%-16%", ">16%"],
        ).astype("object").fillna("Missing")

    if "delinq_2yrs" in output.columns:
        output["has_prior_delinquency_flag"] = (output["delinq_2yrs"] > 0).astype("int64")

    if "number_of_loans" in output.columns:
        output["has_existing_loans_flag"] = (output["number_of_loans"] > 0).astype("int64")
        output["multiple_loans_flag"] = (output["number_of_loans"] > 1).astype("int64")

    if "dependents" in output.columns:
        output["dependents_band"] = pd.cut(
            output["dependents"],
            bins=[-np.inf, 0, 2, np.inf],
            labels=["0", "1-2", "3+"],
        ).astype("object").fillna("Missing")

    if "tenure_years" in output.columns:
        output["tenure_band"] = pd.cut(
            output["tenure_years"],
            bins=[-np.inf, 3, 5, np.inf],
            labels=["<=3 years", "4-5 years", "6+ years"],
        ).astype("object").fillna("Missing")

    for col in ENGINEERED_CATEGORICAL_COLUMNS:
        if col in output.columns:
            output[col] = output[col].astype("object").fillna("Missing")

    return output


def build_feature_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-governance feature policy table."""
    rows: list[dict[str, str]] = []

    for col in df.columns:
        if col == TARGET_COLUMN:
            decision = "target"
            rationale = "Outcome variable; never used as predictor."
        elif col in IDENTIFIER_COLUMNS:
            decision = "exclude_from_model"
            rationale = "Identifier or record key; useful for audit but not predictive modelling."
        elif col in LEAKAGE_SENSITIVE_COLUMNS:
            decision = "exclude_from_baseline_model"
            rationale = "Repayment/outcome-period information; useful for monitoring, not baseline prediction."
        elif col in FAIRNESS_PROXY_COLUMNS:
            decision = "exclude_from_baseline_model"
            rationale = "Sensitive or proxy-sensitive field; retain for fairness audit and governance review."
        elif col in HIGH_CARDINALITY_OR_LOW_INTERPRETABILITY_COLUMNS:
            decision = "exclude_from_baseline_model"
            rationale = "High-cardinality or encrypted field; low interpretability for baseline model."
        else:
            decision = "candidate_baseline_feature"
            rationale = "Potentially available pre-outcome feature after leakage and proxy review."

        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "decision": decision,
                "rationale": rationale,
            }
        )

    policy = pd.DataFrame(rows)
    decision_order = {
        "target": 0,
        "exclude_from_model": 1,
        "exclude_from_baseline_model": 2,
        "candidate_baseline_feature": 3,
    }
    return policy.assign(decision_order=policy["decision"].map(decision_order)).sort_values(
        ["decision_order", "feature"]
    ).drop(columns="decision_order")


def get_baseline_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature lists for the baseline model."""
    numeric = _existing(BASE_SAFE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS + ENGINEERED_BINARY_COLUMNS, df)
    categorical = _existing(BASE_SAFE_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS, df)
    return numeric, categorical


def build_modeling_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Create model-ready raw feature table and governance policy.

    The returned dataset is intentionally not one-hot encoded. Encoding and imputation
    should be performed inside a scikit-learn Pipeline in the modelling notebook to avoid
    train/test contamination.
    """
    engineered = add_engineered_features(df)
    policy = build_feature_policy(engineered)
    numeric_features, categorical_features = get_baseline_feature_columns(engineered)

    keep_cols = IDENTIFIER_COLUMNS + [TARGET_COLUMN] + numeric_features + categorical_features
    keep_cols = _existing(keep_cols, engineered)
    modeling_df = engineered[keep_cols].copy()

    return modeling_df, policy, numeric_features, categorical_features


def create_stratified_splits(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.Series:
    """Create train/validation/test labels with stratified default rates."""
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must equal 1.0")

    y = df[target_col]
    idx = df.index.to_numpy()

    train_idx, temp_idx = train_test_split(
        idx,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )

    temp_y = y.loc[temp_idx]
    relative_validation_size = validation_size / (validation_size + test_size)
    validation_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_validation_size,
        random_state=random_state,
        stratify=temp_y,
    )

    splits = pd.Series(index=df.index, data="", dtype="object", name="split")
    splits.loc[train_idx] = "train"
    splits.loc[validation_idx] = "validation"
    splits.loc[test_idx] = "test"
    return splits


def split_distribution(df: pd.DataFrame, split_col: str = "split", target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """Summarize split counts and target rates."""
    summary = (
        df.groupby(split_col, dropna=False)
        .agg(
            row_count=(target_col, "size"),
            default_count=(target_col, "sum"),
            default_rate=(target_col, "mean"),
        )
        .reset_index()
    )
    summary["row_pct"] = summary["row_count"] / len(df) * 100
    summary["default_rate_pct"] = summary["default_rate"] * 100
    return summary[[split_col, "row_count", "row_pct", "default_count", "default_rate_pct"]]


def feature_catalog(
    modeling_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Create feature catalog for model governance documentation."""
    rows = []
    for col in numeric_features + categorical_features:
        if col not in modeling_df.columns:
            continue
        role = "numeric" if col in numeric_features else "categorical"
        rows.append(
            {
                "feature": col,
                "feature_type": role,
                "dtype": str(modeling_df[col].dtype),
                "missing_count": int(modeling_df[col].isna().sum()),
                "missing_pct": float(modeling_df[col].isna().mean() * 100),
                "unique_values": int(modeling_df[col].nunique(dropna=True)),
                "included_in_baseline_model": True,
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_type", "feature"]).reset_index(drop=True)


def feature_missingness_by_split(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    split_col: str = "split",
) -> pd.DataFrame:
    """Calculate missingness by train/validation/test split."""
    rows = []
    for split_name, split_df in modeling_df.groupby(split_col):
        for col in feature_cols:
            rows.append(
                {
                    "split": split_name,
                    "feature": col,
                    "missing_count": int(split_df[col].isna().sum()),
                    "missing_pct": float(split_df[col].isna().mean() * 100),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "split"]).reset_index(drop=True)


def save_feature_engineering_outputs(
    cleaned_df: pd.DataFrame,
    processed_dir,
    table_dir,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate and save model-ready feature-engineering artifacts."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    modeling_df, policy, numeric_features, categorical_features = build_modeling_dataset(cleaned_df)
    modeling_df["split"] = create_stratified_splits(modeling_df, random_state=random_state)

    catalog = feature_catalog(modeling_df, numeric_features, categorical_features)
    split_summary = split_distribution(modeling_df)
    missingness = feature_missingness_by_split(modeling_df, numeric_features + categorical_features)

    modeling_df.to_csv(processed_dir / "credit_risk_modeling_dataset.csv", index=False)
    policy.to_csv(table_dir / "feature_leakage_and_usage_policy.csv", index=False)
    catalog.to_csv(table_dir / "modeling_feature_catalog.csv", index=False)
    split_summary.to_csv(table_dir / "modeling_split_distribution.csv", index=False)
    missingness.to_csv(table_dir / "modeling_feature_missingness_by_split.csv", index=False)

    return {
        "modeling_dataset_preview": modeling_df.head(10),
        "feature_policy": policy,
        "feature_catalog": catalog,
        "split_distribution": split_summary,
        "feature_missingness_by_split": missingness,
    }
