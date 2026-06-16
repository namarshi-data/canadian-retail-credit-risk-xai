from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

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

MODEL_EXCLUDE_COLUMNS_BASE = [
    "user_id",
    "record_sequence",
    "defaulter",
    "total_payment",
    "received_principal",
    "interest_received",
    "gender",
    "married",
    "pincode",
    "social_profile",
]

MISSING_STRINGS = {"", "nan", "null", "na", "n/a", "<na>"}


@dataclass(frozen=True)
class CleaningResult:
    """Container returned by the cleaning pipeline."""

    cleaned: pd.DataFrame
    audit_summary: pd.DataFrame
    flag_summary: pd.DataFrame
    model_feature_policy: pd.DataFrame


def _normalise_string_series(series: pd.Series) -> pd.Series:
    """Trim strings and convert common textual missing tokens to pandas NA."""
    out = series.astype("string").str.strip()
    out = out.mask(out.str.lower().isin(MISSING_STRINGS), pd.NA)
    return out


def _standardise_yes_no(series: pd.Series) -> pd.Series:
    out = _normalise_string_series(series).str.title()
    return out.where(out.isin(["Yes", "No"]), other=pd.NA)


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator / denominator, np.nan)


def add_missing_indicators(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add binary missing indicators for selected columns."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_missing_flag"] = df[col].isna().astype(int)
    return df


def build_model_feature_policy() -> pd.DataFrame:
    """Document how potentially problematic columns should be handled later."""
    rows = [
        {
            "column": "user_id",
            "recommended_use": "exclude_from_model",
            "reason": "Identifier; useful for joins and audit only.",
        },
        {
            "column": "record_sequence",
            "recommended_use": "exclude_from_model",
            "reason": "Technical merge key; no business meaning.",
        },
        {
            "column": "defaulter",
            "recommended_use": "target_only",
            "reason": "Target variable; never used as predictor.",
        },
        {
            "column": "total_payment",
            "recommended_use": "exclude_from_baseline_model",
            "reason": "May include post-origination repayment behaviour and can create target leakage.",
        },
        {
            "column": "received_principal",
            "recommended_use": "exclude_from_baseline_model",
            "reason": "May be observed after the lending decision or outcome window.",
        },
        {
            "column": "interest_received",
            "recommended_use": "exclude_from_baseline_model",
            "reason": "May be post-outcome repayment information.",
        },
        {
            "column": "gender",
            "recommended_use": "fairness_audit_only",
            "reason": "Sensitive/proxy field; use for bias diagnostics, not model training.",
        },
        {
            "column": "married",
            "recommended_use": "fairness_audit_or_exclude",
            "reason": "Household status proxy; include only with clear governance justification.",
        },
        {
            "column": "pincode",
            "recommended_use": "portfolio_monitoring_or_exclude",
            "reason": "Masked geographic field; can encode socioeconomic proxy risk.",
        },
        {
            "column": "social_profile",
            "recommended_use": "exclude_or_governance_review",
            "reason": "Unclear business meaning and potential behavioural/social proxy.",
        },
        {
            "column": "industry",
            "recommended_use": "governance_review_before_model",
            "reason": "Mostly placeholder or high-cardinality masked values; needs grouping before modelling.",
        },
        {
            "column": "role",
            "recommended_use": "governance_review_before_model",
            "reason": "High-cardinality masked values; needs grouping or exclusion.",
        },
    ]
    return pd.DataFrame(rows)


def clean_credit_risk_dataset(df: pd.DataFrame) -> CleaningResult:
    """Clean the merged credit risk dataset while preserving auditability.

    The function intentionally avoids aggressive row deletion. In credit-risk work,
    missingness and data-quality defects often carry business signal and should be
    flagged before modelling rather than silently removed.
    """
    original = df.copy()
    cleaned = df.copy()

    # Standardise text fields and textual missing values.
    for col in TEXT_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = _normalise_string_series(cleaned[col])

    # Preserve raw amount quality signals before modifying the field.
    cleaned["amount_missing_raw_flag"] = cleaned["amount"].isna().astype(int)
    cleaned["amount_non_positive_flag"] = cleaned["amount"].le(0).fillna(False).astype(int)
    cleaned.loc[cleaned["amount"] <= 0, "amount"] = np.nan

    # Treat source placeholder zeros as missing business information.
    for col in ["industry", "work_experience"]:
        if col in cleaned.columns:
            placeholder_flag = cleaned[col].astype("string").eq("0").fillna(False).astype(int)
            cleaned[f"{col}_placeholder_zero_flag"] = placeholder_flag
            cleaned.loc[placeholder_flag.eq(1), col] = pd.NA

    # Standardise common categorical values.
    if "employment_type" in cleaned.columns:
        cleaned["employment_type"] = cleaned["employment_type"].replace(
            {
                "Self - Employeed": "Self-Employed",
                "Self-Employeed": "Self-Employed",
                "Self Employed": "Self-Employed",
            }
        )

    if "home" in cleaned.columns:
        cleaned["home"] = cleaned["home"].str.lower().replace(
            {
                "mortgage": "Mortgage",
                "rent": "Rent",
                "own": "Own",
                "other": "Other/None",
                "none": "Other/None",
            }
        )

    if "gender" in cleaned.columns:
        cleaned["gender"] = cleaned["gender"].str.title()

    for col in ["married", "social_profile"]:
        if col in cleaned.columns:
            cleaned[col] = _standardise_yes_no(cleaned[col])

    if "is_verified" in cleaned.columns:
        cleaned["is_verified"] = cleaned["is_verified"].replace(
            {
                "Verified": "Verified",
                "Source Verified": "Source Verified",
                "Not Verified": "Not Verified",
            }
        )

    # Create missingness indicators after converting placeholders to NA.
    cleaned = add_missing_indicators(
        cleaned,
        [
            "amount",
            "employment_type",
            "tier_of_employment",
            "industry",
            "work_experience",
            "married",
            "social_profile",
            "is_verified",
        ],
    )

    # Fill selected categorical values with Unknown after flags are created.
    for col in CATEGORICAL_FILL_COLUMNS:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna("Unknown")

    # Derive simple audit/EDA fields. These are not final modelling features by default.
    cleaned["loan_to_income_ratio"] = _safe_rate(cleaned["amount"], cleaned["total_income_pa"])
    cleaned["payment_to_amount_ratio"] = _safe_rate(cleaned["total_payment"], cleaned["amount"])
    cleaned["principal_to_amount_ratio"] = _safe_rate(cleaned["received_principal"], cleaned["amount"])
    cleaned["interest_to_amount_ratio"] = _safe_rate(cleaned["interest_received"], cleaned["amount"])

    cleaned["principal_exceeds_amount_flag"] = (
        (cleaned["received_principal"] > cleaned["amount"]) & cleaned["amount"].notna()
    ).astype(int)

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

    audit_rows = [
        {
            "metric": "rows_before",
            "value": int(original.shape[0]),
        },
        {
            "metric": "rows_after",
            "value": int(cleaned.shape[0]),
        },
        {
            "metric": "columns_before",
            "value": int(original.shape[1]),
        },
        {
            "metric": "columns_after",
            "value": int(cleaned.shape[1]),
        },
        {
            "metric": "target_default_rate_after",
            "value": float(cleaned["defaulter"].mean()),
        },
        {
            "metric": "full_duplicate_rows_after",
            "value": int(cleaned.duplicated().sum()),
        },
        {
            "metric": "record_key_duplicate_count_after",
            "value": int(cleaned.duplicated(["user_id", "record_sequence"]).sum()),
        },
    ]
    audit_summary = pd.DataFrame(audit_rows)

    flag_cols = [
        c
        for c in cleaned.columns
        if c.endswith("_flag")
        or c in ["has_core_data_quality_issue", "has_broad_data_quality_issue"]
    ]
    flag_summary = pd.DataFrame(
        {
            "flag": flag_cols,
            "flagged_row_count": [int(cleaned[c].sum()) for c in flag_cols],
            "flagged_row_pct": [round(float(cleaned[c].mean() * 100), 4) for c in flag_cols],
        }
    ).sort_values("flagged_row_count", ascending=False)

    return CleaningResult(
        cleaned=cleaned,
        audit_summary=audit_summary,
        flag_summary=flag_summary,
        model_feature_policy=build_model_feature_policy(),
    )
