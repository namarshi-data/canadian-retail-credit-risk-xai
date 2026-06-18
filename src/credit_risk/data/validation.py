from __future__ import annotations

"""Validation utilities for ingestion, schema review, and cleaning."""

from typing import Mapping

import numpy as np
import pandas as pd

EXPECTED_COLUMNS: dict[str, list[str]] = {
    "loan_information": ["user_id", "loan_category", "amount", "interest_rate", "tenure_years"],
    "employment": [
        "user_id",
        "employment_type",
        "tier_of_employment",
        "industry",
        "role",
        "work_experience",
        "total_income_pa",
    ],
    "personal_information": [
        "user_id",
        "gender",
        "married",
        "dependents",
        "home",
        "pincode",
        "social_profile",
        "is_verified",
    ],
    "other_information": [
        "user_id",
        "delinq_2yrs",
        "total_payment",
        "received_principal",
        "interest_received",
        "number_of_loans",
        "defaulter",
    ],
}

TARGET_COLUMN = "defaulter"
KEY_COLUMNS = ["user_id", "record_sequence"]


def validate_expected_columns(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare standardized source sheets against expected schema."""
    rows: list[dict[str, object]] = []
    for sheet_name, expected_cols in EXPECTED_COLUMNS.items():
        actual_cols = set(sheets[sheet_name].columns) if sheet_name in sheets else set()
        for col in expected_cols:
            rows.append(
                {
                    "dataset": sheet_name,
                    "expected_column": col,
                    "present": col in actual_cols,
                }
            )
        extra_cols = sorted(actual_cols - set(expected_cols))
        for col in extra_cols:
            rows.append(
                {
                    "dataset": sheet_name,
                    "expected_column": col,
                    "present": True,
                    "note": "extra_column_not_in_expected_schema",
                }
            )
    return pd.DataFrame(rows)


def validate_target(df: pd.DataFrame, target: str = TARGET_COLUMN) -> dict[str, object]:
    """Return target checks for binary default modelling."""
    if target not in df.columns:
        return {"target_present": False}

    target_series = pd.to_numeric(df[target], errors="coerce")
    non_missing_values = sorted(target_series.dropna().unique().tolist())
    allowed = {0, 1}
    unexpected = sorted(set(non_missing_values) - allowed)

    return {
        "target_present": True,
        "row_count": int(len(df)),
        "target_missing_count": int(target_series.isna().sum()),
        "target_values": non_missing_values,
        "unexpected_target_values": unexpected,
        "binary_target_valid": len(unexpected) == 0 and len(non_missing_values) > 0,
        "default_count": int((target_series == 1).sum()),
        "non_default_count": int((target_series == 0).sum()),
        "default_rate": float(target_series.mean()) if target_series.notna().any() else np.nan,
    }


def validate_record_keys(df: pd.DataFrame, key_columns: list[str] | None = None) -> pd.DataFrame:
    """Validate record keys after ingestion or cleaning."""
    key_columns = key_columns or KEY_COLUMNS
    rows = []
    for col in key_columns:
        rows.append(
            {
                "check": f"{col}_present",
                "passed": col in df.columns,
                "value": col in df.columns,
            }
        )
        if col in df.columns:
            rows.append(
                {
                    "check": f"{col}_missing_count",
                    "passed": int(df[col].isna().sum()) == 0,
                    "value": int(df[col].isna().sum()),
                }
            )

    keys_present = all(col in df.columns for col in key_columns)
    duplicate_count = int(df[key_columns].duplicated().sum()) if keys_present else None
    rows.append(
        {
            "check": "record_key_duplicate_count",
            "passed": duplicate_count == 0 if duplicate_count is not None else False,
            "value": duplicate_count,
        }
    )
    return pd.DataFrame(rows)


def validate_merge_integrity(
    merged: pd.DataFrame,
    expected_row_count: int,
    key_columns: list[str] | None = None,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Validate safe merge output before downstream cleaning."""
    key_columns = key_columns or KEY_COLUMNS
    key_checks = validate_record_keys(merged, key_columns=key_columns)
    target_check = validate_target(merged, target=target)

    checks = [
        {
            "check": "row_count_preserved",
            "passed": int(len(merged)) == int(expected_row_count),
            "value": int(len(merged)),
            "expected": int(expected_row_count),
        },
        {
            "check": "target_present",
            "passed": bool(target_check.get("target_present", False)),
            "value": target_check.get("target_present", False),
            "expected": True,
        },
        {
            "check": "binary_target_valid",
            "passed": bool(target_check.get("binary_target_valid", False)),
            "value": target_check.get("target_values"),
            "expected": "0/1 only",
        },
    ]
    return pd.concat([pd.DataFrame(checks), key_checks], ignore_index=True)


def build_logical_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run business logical checks used by data-quality and cleaning notebooks."""
    checks: list[dict[str, object]] = []

    def add_check(name: str, condition: pd.Series | bool, severity: str, note: str) -> None:
        if isinstance(condition, pd.Series):
            fail_count = int(condition.fillna(False).sum())
        else:
            fail_count = int(bool(condition))
        checks.append(
            {
                "check": name,
                "severity": severity,
                "failed_count": fail_count,
                "passed": fail_count == 0,
                "note": note,
            }
        )

    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        add_check("amount_non_positive", amount.le(0), "high", "Loan amount should be positive when present.")

    if "interest_rate" in df.columns:
        rate = pd.to_numeric(df["interest_rate"], errors="coerce")
        add_check("interest_rate_outside_0_100", rate.lt(0) | rate.gt(100), "high", "Interest rate should be a percentage between 0 and 100.")

    if "tenure_years" in df.columns:
        tenure = pd.to_numeric(df["tenure_years"], errors="coerce")
        add_check("tenure_years_non_positive", tenure.le(0), "medium", "Loan tenure should be positive.")

    if {"total_income_pa", "amount"}.issubset(df.columns):
        income = pd.to_numeric(df["total_income_pa"], errors="coerce")
        amount = pd.to_numeric(df["amount"], errors="coerce")
        add_check("income_non_positive", income.le(0), "high", "Annual income should be positive when present.")
        add_check("extreme_loan_to_income_gt_20", (amount / income.replace(0, np.nan)).gt(20), "medium", "Extreme affordability ratio for review.")

    if {"received_principal", "amount"}.issubset(df.columns):
        principal = pd.to_numeric(df["received_principal"], errors="coerce")
        amount = pd.to_numeric(df["amount"], errors="coerce")
        add_check("principal_exceeds_amount", principal.gt(amount) & amount.notna(), "monitoring", "May be valid due to data timing; monitor and document.")

    if TARGET_COLUMN in df.columns:
        target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        add_check("target_missing", target.isna(), "critical", "Target must be present for modelling.")
        add_check("target_not_binary", ~target.dropna().isin([0, 1]), "critical", "Target must be binary 0/1.")

    return pd.DataFrame(checks)


def build_leakage_review_table(columns: list[str] | pd.Index) -> pd.DataFrame:
    """Create a leakage/fairness review table for available columns."""
    rows = []
    for col in columns:
        lower = str(col).lower()
        if lower == TARGET_COLUMN:
            decision = "target_only"
            reason = "Outcome variable; never use as predictor."
        elif lower in {"user_id", "record_sequence"} or lower.endswith("_id"):
            decision = "exclude_identifier"
            reason = "Identifier/key only."
        elif any(token in lower for token in ["payment", "principal", "interest_received"]):
            decision = "monitoring_only_possible_leakage"
            reason = "Repayment-derived field may reveal post-origination behaviour."
        elif lower in {"gender", "married", "pincode", "social_profile", "has_social_profile"}:
            decision = "audit_only_sensitive_or_proxy"
            reason = "Sensitive/proxy-sensitive or hard-to-defend feature."
        else:
            decision = "candidate_after_quality_review"
            reason = "No automatic exclusion at validation stage."
        rows.append({"column": col, "governance_decision": decision, "reason": reason})
    return pd.DataFrame(rows)


def build_readiness_gate(check_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a compact pass/fail readiness gate from validation outputs."""
    rows = []
    for name, table in check_tables.items():
        if table is None or table.empty:
            rows.append({"check_group": name, "passed": False, "failed_count": None, "note": "table_missing_or_empty"})
            continue
        if "passed" in table.columns:
            failed = int((~table["passed"].astype(bool)).sum())
            rows.append({"check_group": name, "passed": failed == 0, "failed_count": failed, "note": "derived_from_passed_column"})
        elif "present" in table.columns:
            failed = int((~table["present"].astype(bool)).sum())
            rows.append({"check_group": name, "passed": failed == 0, "failed_count": failed, "note": "derived_from_present_column"})
        else:
            rows.append({"check_group": name, "passed": True, "failed_count": 0, "note": "table_created"})
    return pd.DataFrame(rows)
