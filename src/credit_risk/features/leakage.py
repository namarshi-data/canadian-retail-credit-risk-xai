from __future__ import annotations

"""Feature leakage and feature-use policy utilities.

The goal is to keep Notebook 05 conservative and auditable. This module marks
identifiers, target fields, repayment-derived fields, sensitive/proxy-sensitive
fields, high-cardinality encrypted fields, and timing-review fields before the
model-training stage.
"""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

TARGET_COL = "defaulter"
ID_COLS = ["user_id", "record_sequence"]
SPLIT_COL = "split"

REPAYMENT_MONITORING_FIELDS = [
    "total_payment",
    "received_principal",
    "interest_received",
    "payment_to_amount_ratio",
    "principal_to_amount_ratio",
    "interest_to_amount_ratio",
    "principal_exceeds_amount_flag",
]

SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS = [
    "gender",
    "pincode",
    "social_profile",
    "married",
]

GOVERNANCE_MONITORING_FIELDS = [
    "dependents",
    "home",
    "is_verified",
]

HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS = [
    "industry",
    "role",
]

TIMING_REVIEW_FIELDS = [
    "delinq_2yrs",
    "number_of_loans",
]

INCLUDE_BASELINE_POLICIES = {"include_in_baseline", "include_with_monitoring"}


@dataclass(frozen=True)
class LeakagePolicy:
    """Named feature policy constants used across Notebook 05 and 06."""

    target: str = TARGET_COL
    id_columns: tuple[str, ...] = tuple(ID_COLS)
    split_column: str = SPLIT_COL
    repayment_monitoring_fields: tuple[str, ...] = tuple(REPAYMENT_MONITORING_FIELDS)
    sensitive_or_proxy_fields: tuple[str, ...] = tuple(SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS)
    high_cardinality_or_encrypted_fields: tuple[str, ...] = tuple(HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS)
    timing_review_fields: tuple[str, ...] = tuple(TIMING_REVIEW_FIELDS)


def present_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return the requested columns that exist in df."""
    return [col for col in columns if col in df.columns]


def feature_family_for_column(column: str) -> str:
    """Classify a feature into a business/governance family."""
    if column in ID_COLS:
        return "identifier"
    if column == TARGET_COL:
        return "target"
    if column == SPLIT_COL:
        return "split_metadata"
    if column in REPAYMENT_MONITORING_FIELDS:
        return "repayment_monitoring"
    if column in SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS:
        return "sensitive_or_high_risk_proxy"
    if column in GOVERNANCE_MONITORING_FIELDS:
        return "proxy_governance_review"
    if column in HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS:
        return "high_cardinality_or_encrypted"
    if column in TIMING_REVIEW_FIELDS:
        return "timing_review"
    if column.endswith("_flag") or column.endswith("_count"):
        return "data_quality_or_binary_flag"
    if column.endswith("_band"):
        return "business_band_feature"
    if column in {"amount", "interest_rate", "tenure_years", "loan_category"}:
        return "loan_terms"
    if column in {"total_income_pa", "loan_to_income_ratio", "income_to_loan_buffer"}:
        return "affordability"
    if column in {"employment_type", "tier_of_employment", "work_experience"}:
        return "employment_profile"
    return "baseline_candidate"


def policy_decision_for_column(column: str) -> tuple[str, str, str]:
    """Return decision, baseline policy, and reason for a feature."""
    if column == TARGET_COL:
        return "target_only", "exclude_from_features", "Target variable; never used as a predictor."

    if column in ID_COLS:
        return "audit_key_only", "exclude_from_model", "Identifier/audit grain; useful for traceability only."

    if column == SPLIT_COL:
        return "split_assignment", "exclude_from_model", "Split label is metadata, not a feature."

    if column in REPAYMENT_MONITORING_FIELDS:
        return (
            "monitoring_only_leakage_risk",
            "exclude_from_baseline",
            "Repayment-derived/post-origination information may leak target timing.",
        )

    if column in SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS:
        return (
            "audit_governance_only",
            "exclude_from_baseline",
            "Sensitive or high-risk proxy field retained only for governance review.",
        )

    if column in HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS:
        return (
            "audit_or_future_grouping_only",
            "exclude_from_baseline",
            "High-cardinality/encrypted field is not explainable enough for the conservative baseline.",
        )

    if column in TIMING_REVIEW_FIELDS:
        return (
            "candidate_requires_timing_confirmation",
            "review_before_modelling",
            "Use only if confirmed known at prediction time.",
        )

    if column in GOVERNANCE_MONITORING_FIELDS:
        return (
            "include_with_governance_monitoring",
            "include_with_monitoring",
            "Usable with documented monitoring because it may proxy operational or socioeconomic factors.",
        )

    return (
        "include_conservative_baseline",
        "include_in_baseline",
        "Deterministic row-level feature not marked as leakage or high-risk proxy.",
    )


def build_feature_usage_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Create a feature-use policy table for every available column."""
    rows: list[dict] = []
    for col in df.columns:
        decision, baseline_policy, reason = policy_decision_for_column(col)
        rows.append(
            {
                "feature": col,
                "feature_family": feature_family_for_column(col),
                "decision": decision,
                "baseline_model_policy": baseline_policy,
                "reason": reason,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 4),
                "unique_values": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def select_baseline_features(
    policy: pd.DataFrame,
    *,
    include_policies: set[str] | None = None,
) -> list[str]:
    """Select conservative baseline features from a feature policy table."""
    include_policies = include_policies or INCLUDE_BASELINE_POLICIES
    if policy.empty:
        return []
    required = {"feature", "baseline_model_policy"}
    missing = required - set(policy.columns)
    if missing:
        raise KeyError(f"Feature policy missing required column(s): {sorted(missing)}")
    return policy.loc[policy["baseline_model_policy"].isin(include_policies), "feature"].tolist()


def make_leakage_review_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize available leakage/proxy-risk fields before modelling."""
    rows = []
    review_groups = {
        "repayment_monitoring_only": REPAYMENT_MONITORING_FIELDS,
        "sensitive_or_high_risk_proxy": SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS,
        "high_cardinality_or_encrypted": HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS,
        "timing_review_required": TIMING_REVIEW_FIELDS,
    }
    for group, cols in review_groups.items():
        for col in cols:
            rows.append(
                {
                    "feature": col,
                    "review_group": group,
                    "present_in_dataset": col in df.columns,
                    "recommended_model_use": (
                        "exclude_from_baseline"
                        if group != "timing_review_required"
                        else "review_before_modelling"
                    ),
                    "reason": policy_decision_for_column(col)[2],
                }
            )
    return pd.DataFrame(rows)


def validate_modeling_dataset_against_policy(
    modeling_df: pd.DataFrame,
    policy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create QA checks for leakage-sensitive modelling datasets."""
    blocked_fields = set(REPAYMENT_MONITORING_FIELDS + SENSITIVE_OR_HIGH_RISK_PROXY_FIELDS + HIGH_CARDINALITY_OR_ENCRYPTED_FIELDS)
    present_blocked = sorted(blocked_fields.intersection(modeling_df.columns))

    if policy is not None and not policy.empty and {"feature", "baseline_model_policy"}.issubset(policy.columns):
        policy_blocked = policy.loc[
            policy["baseline_model_policy"].astype(str).str.contains("exclude", case=False, na=False),
            "feature",
        ].tolist()
        present_policy_blocked = sorted(set(policy_blocked).intersection(modeling_df.columns))
    else:
        present_policy_blocked = []

    checks = [
        {
            "check": "target_present",
            "status": "pass" if TARGET_COL in modeling_df.columns else "fail",
            "value": TARGET_COL in modeling_df.columns,
        },
        {
            "check": "split_column_present",
            "status": "pass" if SPLIT_COL in modeling_df.columns else "fail",
            "value": SPLIT_COL in modeling_df.columns,
        },
        {
            "check": "record_key_duplicate_count",
            "status": "pass"
            if set(ID_COLS).issubset(modeling_df.columns) and modeling_df.duplicated(ID_COLS).sum() == 0
            else "review",
            "value": int(modeling_df.duplicated(ID_COLS).sum()) if set(ID_COLS).issubset(modeling_df.columns) else "missing_id_columns",
        },
        {
            "check": "no_known_blocked_fields_in_modeling_dataset",
            "status": "pass" if not present_blocked else "fail",
            "value": ", ".join(present_blocked),
        },
        {
            "check": "no_policy_excluded_fields_in_modeling_dataset",
            "status": "pass" if not present_policy_blocked else "fail",
            "value": ", ".join(present_policy_blocked),
        },
    ]
    return pd.DataFrame(checks)
