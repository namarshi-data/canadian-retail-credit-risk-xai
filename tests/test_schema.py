from __future__ import annotations

"""Unit tests for ingestion, schema, and validation helpers.

These tests focus on the project risks that matter before modelling:

- raw column-name standardization
- duplicate user_id handling
- safe one-to-one merge using user_id + record_sequence
- schema summary output
- expected-column validation
- binary target validation
"""

import pandas as pd
import pytest

from credit_risk.data.ingestion import (
    add_record_sequence,
    merge_credit_risk_sheets,
    normalize_id_columns,
    standardize_column_name,
    standardize_columns,
    standardize_workbook_sheets,
)
from credit_risk.data.schema import duplicate_id_summary, summarize_dataframe, summarize_workbook
from credit_risk.data.validation import validate_expected_columns, validate_target


def _sample_standardized_sheets() -> dict[str, pd.DataFrame]:
    """Create minimal standardized source sheets with duplicate user_id values.

    This mimics the real project issue: user_id is not unique, so merging on
    user_id alone would inflate rows. The safe merge should use:
    [user_id, record_sequence].
    """
    return {
        "loan_information": pd.DataFrame(
            {
                "user_id": [101, 101, 202],
                "loan_category": ["Credit Card", "Consolidation", "Auto"],
                "amount": [10_000, 15_000, 20_000],
                "interest_rate": [12.0, 14.5, 10.0],
                "tenure_years": [3, 4, 5],
            }
        ),
        "employment": pd.DataFrame(
            {
                "user_id": [101, 101, 202],
                "employment_type": ["Salaried", "Self-Employed", "Salaried"],
                "tier_of_employment": ["A", "B", "C"],
                "industry": ["Tech", "Retail", "Finance"],
                "role": ["Analyst", "Owner", "Manager"],
                "work_experience": ["1-2", "3-5", "5-10"],
                "total_income_pa": [80_000, 95_000, 110_000],
            }
        ),
        "personal_information": pd.DataFrame(
            {
                "user_id": [101, 101, 202],
                "gender": ["Female", "Female", "Male"],
                "married": ["No", "Yes", "Yes"],
                "dependents": [0, 1, 2],
                "home": ["Rent", "Mortgage", "Own"],
                "pincode": ["A1A", "B2B", "C3C"],
                "social_profile": ["Yes", "No", "Yes"],
                "is_verified": ["Verified", "Source Verified", "Not Verified"],
            }
        ),
        "other_information": pd.DataFrame(
            {
                "user_id": [101, 101, 202],
                "delinq_2yrs": [0, 1, 0],
                "total_payment": [1_000, 2_000, 3_000],
                "received_principal": [900, 1_800, 2_700],
                "interest_received": [100, 200, 300],
                "number_of_loans": [1, 2, 1],
                "defaulter": [0, 1, 0],
            }
        ),
    }


def test_standardize_known_raw_columns() -> None:
    """Known source typos should map to clean project column names."""
    assert standardize_column_name("Employmet type") == "employment_type"
    assert standardize_column_name("Total Payement ") == "total_payment"
    assert standardize_column_name("Tenure(years)") == "tenure_years"
    assert standardize_column_name("Total Income(PA)") == "total_income_pa"


def test_standardize_column_name_general_snake_case() -> None:
    """Generic raw names should become stable snake_case names."""
    assert standardize_column_name("Loan Category") == "loan_category"
    assert standardize_column_name("Interest Rate") == "interest_rate"
    assert standardize_column_name(" User ID ") == "user_id"


def test_standardize_columns_returns_copy_with_expected_names() -> None:
    raw = pd.DataFrame(
        {
            "User ID": [1],
            "Loan Category": ["Credit Card"],
            "Total Income(PA)": [75_000],
        }
    )

    out = standardize_columns(raw)

    assert list(out.columns) == ["user_id", "loan_category", "total_income_pa"]
    assert list(raw.columns) == ["User ID", "Loan Category", "Total Income(PA)"]


def test_standardize_workbook_sheets_and_normalize_id_columns() -> None:
    sheets = {
        "Loan Information": pd.DataFrame({"User ID": [1], "Loan Category": ["A"]}),
        "Employment": pd.DataFrame({"Userid": [1], "Employmet type": ["Salaried"]}),
    }

    standardized = standardize_workbook_sheets(sheets)
    normalized = normalize_id_columns(standardized)

    assert "loan_information" in standardized
    assert "employment" in standardized
    assert "user_id" in normalized["employment"].columns


def test_add_record_sequence_handles_duplicate_user_ids() -> None:
    df = pd.DataFrame({"user_id": [101, 101, 202, 101], "value": [1, 2, 3, 4]})

    out = add_record_sequence(df)

    assert out["record_sequence"].tolist() == [1, 2, 1, 3]
    assert out[["user_id", "record_sequence"]].duplicated().sum() == 0


def test_merge_credit_risk_sheets_preserves_observation_grain() -> None:
    sheets = _sample_standardized_sheets()

    merged = merge_credit_risk_sheets(sheets)

    assert merged.shape[0] == 3
    assert {"user_id", "record_sequence", "loan_category", "employment_type", "defaulter"}.issubset(
        merged.columns
    )
    assert merged[["user_id", "record_sequence"]].duplicated().sum() == 0
    assert merged["defaulter"].tolist() == [0, 1, 0]


def test_merge_credit_risk_sheets_requires_all_expected_sheets() -> None:
    sheets = _sample_standardized_sheets()
    sheets.pop("other_information")

    with pytest.raises(KeyError, match="Missing expected.*sheet"):
        merge_credit_risk_sheets(sheets)


def test_summarize_dataframe_contains_quality_columns() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "x", "y"]})

    summary = summarize_dataframe(df, "sample")

    assert {"dataset", "column", "dtype", "missing_count", "missing_pct", "unique_values"}.issubset(
        summary.columns
    )
    assert summary.loc[summary["column"].eq("a"), "missing_count"].iloc[0] == 1


def test_summarize_workbook_combines_sheet_summaries() -> None:
    sheets = {
        "sheet_a": pd.DataFrame({"id": [1, 2]}),
        "sheet_b": pd.DataFrame({"id": [1], "value": [10]}),
    }

    summary = summarize_workbook(sheets)

    assert set(summary["dataset"]) == {"sheet_a", "sheet_b"}
    assert "column" in summary.columns


def test_duplicate_id_summary_detects_duplicate_ids() -> None:
    sheets = {"loan_information": pd.DataFrame({"user_id": [1, 1, 2]})}

    summary = duplicate_id_summary(sheets)

    assert bool(summary.loc[0, "id_column_present"]) is True
    assert summary.loc[0, "row_count"] == 3
    assert summary.loc[0, "unique_id_count"] == 2
    assert summary.loc[0, "duplicate_id_count"] == 1


def test_validate_expected_columns_reports_presence() -> None:
    sheets = _sample_standardized_sheets()

    result = validate_expected_columns(sheets)

    assert {"dataset", "expected_column", "present"}.issubset(result.columns)
    assert result["present"].all()


def test_validate_target_numeric_binary() -> None:
    df = pd.DataFrame({"defaulter": [0, 1, 0, 1]})

    result = validate_target(df)

    assert result["target_present"] is True
    assert result["row_count"] == 4
    assert result["target_missing_count"] == 0
    assert result["target_values"] == [0, 1]
    assert result["default_rate"] == 0.5


def test_validate_target_missing_column() -> None:
    result = validate_target(pd.DataFrame({"x": [1, 2, 3]}))

    assert result == {"target_present": False}
