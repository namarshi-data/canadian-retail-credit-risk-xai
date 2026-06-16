from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = {
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


def validate_expected_columns(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare standardized source sheets against expected schema."""
    rows = []
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
    return pd.DataFrame(rows)


def validate_target(df: pd.DataFrame, target: str = "defaulter") -> dict[str, object]:
    """Return basic target checks for binary default modelling."""
    if target not in df.columns:
        return {"target_present": False}
    return {
        "target_present": True,
        "row_count": len(df),
        "target_missing_count": int(df[target].isna().sum()),
        "target_values": sorted(df[target].dropna().unique().tolist()),
        "default_rate": float(df[target].mean()) if pd.api.types.is_numeric_dtype(df[target]) else None,
    }
