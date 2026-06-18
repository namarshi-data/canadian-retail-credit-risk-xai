from __future__ import annotations

"""Cleaning utilities for the Canadian retail credit-risk project.

Cleaning decisions are intentionally conservative: rows are preserved, missingness
is flagged, and modelling transformations are deferred to the modelling pipeline
after train/validation/test splitting.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

TARGET_COLUMN = "defaulter"

TEXT_COLUMNS = [
    "loan_category",
    "employment_type",
    "tier_of_employment",
    "industry",
    "role",
    "work_experience",
    "gender",
    "married",
    "home",
    "pincode",
    "social_profile",
    "is_verified",
]

CATEGORICAL_FILL_COLUMNS = [
    "employment_type",
    "tier_of_employment",
    "industry",
    "work_experience",
    "married",
    "social_profile",
    "is_verified",
]

MISSING_STRINGS = {"", "nan", "null", "na", "n/a", "<na>", "none", "unknown?"}

NUMERIC_COLUMNS = [
    "amount",
    "interest_rate",
    "tenure_years",
    "total_income_pa",
    "dependents",
    "delinq_2yrs",
    "total_payment",
    "received_principal",
    "interest_received",
    "number_of_loans",
    TARGET_COLUMN,
]


@dataclass(frozen=True)
class CleaningResult:
    """Container returned by the cleaning pipeline."""

    cleaned: pd.DataFrame
    audit_summary: pd.DataFrame
    flag_summary: pd.DataFrame
    model_feature_policy: pd.DataFrame
    cleaning_policy: pd.DataFrame


def _present_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _normalise_string_series(series: pd.Series) -> pd.Series:
    """Trim strings and convert common textual missing tokens to pandas NA."""
    out = series.astype("string").str.strip()
    out = out.mask(out.str.lower().isin(MISSING_STRINGS), pd.NA)
    return out


def _standardise_yes_no(series: pd.Series) -> pd.Series:
    """Standardize yes/no style fields."""
    out = _normalise_string_series(series).str.lower().replace(
        {"y": "yes", "n": "no", "true": "yes", "false": "no", "1": "yes", "0": "no"}
    )
    out = out.str.title()
    return out.where(out.isin(["Yes", "No"]), other=pd.NA)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe vectorized ratio returning NaN when denominator is not positive."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return pd.Series(np.where(denominator > 0, numerator / denominator, np.nan), index=numerator.index)


def add_missing_indicators(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add binary missing indicators for selected columns."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype(int)
    return out


def coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str] = NUMERIC_COLUMNS) -> pd.DataFrame:
    """Coerce expected numeric fields to numeric dtype when available."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_cleaning_policy() -> pd.DataFrame:
    """Document cleaning-stage decisions."""
    rows = [
        ("row_deletion", "preserve_rows", "Avoid deleting borrower records in cleaning; flag issues instead."),
        ("text_missing_tokens", "convert_to_missing", "Convert blank/nan/null/n/a style tokens to pandas NA."),
        ("amount_non_positive", "set_to_missing_and_flag", "Non-positive loan amount is not analytically valid."),
        ("industry_zero_placeholder", "set_to_missing_and_flag", "Source placeholder zero is not a valid industry label."),
        ("work_experience_zero_placeholder", "set_to_missing_and_flag", "Source placeholder zero is not a valid experience label."),
        ("categorical_missing", "flag_then_fill_unknown", "Create missing flags before filling selected categoricals as Unknown."),
        ("repayment_ratios", "monitoring_only", "Create ratios for audit/EDA; exclude from baseline modelling if leakage-prone."),
        ("encoding_scaling_resampling", "defer_to_model_pipeline", "Fit encoders, scalers, winsorization, and resampling after train split only."),
    ]
    return pd.DataFrame(rows, columns=["cleaning_area", "decision", "rationale"])


def build_model_feature_policy(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Document feature-use policy for modelling and governance."""
    base_rows = [
        ("user_id", "exclude_from_model", "Identifier; useful for joins and audit only."),
        ("record_sequence", "exclude_from_model", "Technical merge key; no business meaning."),
        (TARGET_COLUMN, "target_only", "Target variable; never used as predictor."),
        ("total_payment", "exclude_from_baseline_model", "May include post-origination repayment behaviour and create target leakage."),
        ("received_principal", "exclude_from_baseline_model", "May be observed after lending decision or outcome window."),
        ("interest_received", "exclude_from_baseline_model", "May be post-outcome repayment information."),
        ("payment_to_amount_ratio", "exclude_from_baseline_model", "Derived from repayment information; monitoring-only unless timing is proven safe."),
        ("principal_to_amount_ratio", "exclude_from_baseline_model", "Derived from received principal; may reveal repayment outcome information."),
        ("interest_to_amount_ratio", "exclude_from_baseline_model", "Derived from interest received; may reveal repayment/outcome timing."),
        ("gender", "fairness_audit_only", "Sensitive/proxy field; do not use for baseline model training."),
        ("married", "fairness_audit_or_exclude", "Household-status proxy; include only with clear governance justification."),
        ("pincode", "portfolio_monitoring_or_exclude", "Masked geographic field; may encode socioeconomic proxy risk."),
        ("social_profile", "exclude_or_governance_review", "Unclear business meaning and potential behavioural/social proxy."),
        ("industry", "governance_review_before_model", "High-cardinality/masked values; needs grouping before modelling."),
        ("role", "governance_review_before_model", "High-cardinality/masked values; needs grouping or exclusion."),
        ("amount_missing_raw_flag", "candidate_feature_with_governance_note", "Data-quality signal; monitor for drift and avoid overinterpreting as borrower behaviour."),
        ("core_data_quality_issue_count", "candidate_feature_with_governance_note", "Operational data-quality signal; useful but requires monitoring."),
    ]
    policy = pd.DataFrame(base_rows, columns=["column", "recommended_use", "reason"])
    if columns is not None:
        known = set(policy["column"])
        extra = [col for col in columns if col not in known]
        extra_rows = pd.DataFrame(
            {
                "column": extra,
                "recommended_use": "candidate_after_leakage_and_quality_review",
                "reason": "No automatic exclusion at cleaning stage.",
            }
        )
        policy = pd.concat([policy, extra_rows], ignore_index=True)
    return policy.drop_duplicates("column").reset_index(drop=True)


def clean_credit_risk_dataset(df: pd.DataFrame) -> CleaningResult:
    """Clean the merged credit-risk dataset while preserving auditability."""
    original = df.copy()
    cleaned = df.copy()

    # Standardize dtypes first.
    cleaned = coerce_numeric_columns(cleaned)

    # Standardize text fields and textual missing values.
    for col in _present_columns(cleaned, TEXT_COLUMNS):
        cleaned[col] = _normalise_string_series(cleaned[col])

    # Amount-quality signals before modifying field.
    if "amount" in cleaned.columns:
        cleaned["amount_missing_raw_flag"] = cleaned["amount"].isna().astype(int)
        cleaned["amount_non_positive_flag"] = cleaned["amount"].le(0).fillna(False).astype(int)
        cleaned.loc[cleaned["amount"].le(0), "amount"] = np.nan

    # Treat source placeholder zeros as missing business information.
    for col in ["industry", "work_experience"]:
        if col in cleaned.columns:
            placeholder_flag = cleaned[col].astype("string").eq("0").fillna(False).astype(int)
            cleaned[f"{col}_placeholder_zero_flag"] = placeholder_flag
            cleaned.loc[placeholder_flag.eq(1), col] = pd.NA

    # Standardize common categorical values.
    if "employment_type" in cleaned.columns:
        cleaned["employment_type"] = cleaned["employment_type"].replace(
            {
                "Self - Employeed": "Self-Employed",
                "Self-Employeed": "Self-Employed",
                "Self Employed": "Self-Employed",
                "Self-employed": "Self-Employed",
            }
        )

    if "home" in cleaned.columns:
        cleaned["home"] = cleaned["home"].astype("string").str.lower().replace(
            {
                "mortgage": "Mortgage",
                "rent": "Rent",
                "own": "Own",
                "other": "Other/None",
                "none": "Other/None",
            }
        )

    if "gender" in cleaned.columns:
        cleaned["gender"] = cleaned["gender"].astype("string").str.title()

    for col in ["married", "social_profile"]:
        if col in cleaned.columns:
            cleaned[col] = _standardise_yes_no(cleaned[col])

    if "is_verified" in cleaned.columns:
        cleaned["is_verified"] = cleaned["is_verified"].replace(
            {
                "Verified": "Verified",
                "Source Verified": "Source Verified",
                "Not Verified": "Not Verified",
                "source verified": "Source Verified",
                "verified": "Verified",
                "not verified": "Not Verified",
            }
        )

    # Create missingness indicators after converting placeholders to missing.
    missing_indicator_cols = [
        "amount",
        "employment_type",
        "tier_of_employment",
        "industry",
        "work_experience",
        "married",
        "social_profile",
        "is_verified",
        "total_income_pa",
    ]
    cleaned = add_missing_indicators(cleaned, missing_indicator_cols)

    # Fill selected categoricals after flags are created.
    for col in _present_columns(cleaned, CATEGORICAL_FILL_COLUMNS):
        cleaned[col] = cleaned[col].fillna("Unknown")

    # Audit/EDA ratios. These are monitoring-only unless timing is justified.
    if {"amount", "total_income_pa"}.issubset(cleaned.columns):
        cleaned["loan_to_income_ratio"] = _safe_rate(cleaned["amount"], cleaned["total_income_pa"])
    if {"total_payment", "amount"}.issubset(cleaned.columns):
        cleaned["payment_to_amount_ratio"] = _safe_rate(cleaned["total_payment"], cleaned["amount"])
    if {"received_principal", "amount"}.issubset(cleaned.columns):
        cleaned["principal_to_amount_ratio"] = _safe_rate(cleaned["received_principal"], cleaned["amount"])
        cleaned["principal_exceeds_amount_flag"] = (
            cleaned["received_principal"].gt(cleaned["amount"]) & cleaned["amount"].notna()
        ).astype(int)
    if {"interest_received", "amount"}.issubset(cleaned.columns):
        cleaned["interest_to_amount_ratio"] = _safe_rate(cleaned["interest_received"], cleaned["amount"])

    # Ensure quality flags exist before issue-count rollups.
    for col in ["amount_missing_raw_flag", "amount_non_positive_flag", "principal_exceeds_amount_flag"]:
        if col not in cleaned.columns:
            cleaned[col] = 0
    for col in ["industry_placeholder_zero_flag", "work_experience_placeholder_zero_flag"]:
        if col not in cleaned.columns:
            cleaned[col] = 0

    core_data_quality_flag_cols = [
        "amount_missing_raw_flag",
        "amount_non_positive_flag",
        "principal_exceeds_amount_flag",
    ]
    broad_data_quality_flag_cols = core_data_quality_flag_cols + [
        "industry_placeholder_zero_flag",
        "work_experience_placeholder_zero_flag",
    ]

    cleaned["core_data_quality_issue_count"] = cleaned[core_data_quality_flag_cols].sum(axis=1)
    cleaned["has_core_data_quality_issue"] = cleaned["core_data_quality_issue_count"].gt(0).astype(int)
    cleaned["broad_data_quality_issue_count"] = cleaned[broad_data_quality_flag_cols].sum(axis=1)
    cleaned["has_broad_data_quality_issue"] = cleaned["broad_data_quality_issue_count"].gt(0).astype(int)

    record_key_duplicate_count = (
        int(cleaned[["user_id", "record_sequence"]].duplicated().sum())
        if {"user_id", "record_sequence"}.issubset(cleaned.columns)
        else np.nan
    )
    target_default_rate = (
        float(pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").mean()) if TARGET_COLUMN in cleaned.columns else np.nan
    )

    audit_rows = [
        ("rows_before", int(original.shape[0])),
        ("rows_after", int(cleaned.shape[0])),
        ("columns_before", int(original.shape[1])),
        ("columns_after", int(cleaned.shape[1])),
        ("rows_preserved", bool(original.shape[0] == cleaned.shape[0])),
        ("target_default_rate_after", target_default_rate),
        ("full_duplicate_rows_after", int(cleaned.duplicated().sum())),
        ("record_key_duplicate_count_after", record_key_duplicate_count),
        ("remaining_missing_values_after", int(cleaned.isna().sum().sum())),
    ]
    audit_summary = pd.DataFrame(audit_rows, columns=["metric", "value"])

    flag_cols = [
        c for c in cleaned.columns if c.endswith("_flag") or c in ["has_core_data_quality_issue", "has_broad_data_quality_issue"]
    ]
    flag_rows = []
    for col in flag_cols:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            flag_rows.append(
                {
                    "flag": col,
                    "flagged_row_count": int((cleaned[col] == 1).sum()),
                    "flagged_row_pct": round(float((cleaned[col] == 1).mean() * 100), 4),
                }
            )
    flag_summary = pd.DataFrame(flag_rows).sort_values("flagged_row_count", ascending=False).reset_index(drop=True)

    return CleaningResult(
        cleaned=cleaned,
        audit_summary=audit_summary,
        flag_summary=flag_summary,
        model_feature_policy=build_model_feature_policy(cleaned.columns),
        cleaning_policy=build_cleaning_policy(),
    )
