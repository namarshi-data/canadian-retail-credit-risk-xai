from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_SHEET_NAMES = [
    "loan_information",
    "Employment",
    "Personal_information",
    "Other_information",
]


def standardize_column_name(column: str) -> str:
    """Convert raw source column names to snake_case names used in the project."""
    replacements = {
        "Employmet type": "employment_type",
        "Total Payement ": "total_payment",
        "Total Income(PA)": "total_income_pa",
        "Tenure(years)": "tenure_years",
    }
    if column in replacements:
        return replacements[column]

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def read_source_workbook(file_path: str | Path) -> dict[str, pd.DataFrame]:
    """Read all expected source sheets from the raw Excel workbook."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw workbook not found at {file_path}. "
            "Place Credit_Risk_Dataset.xlsx in data/raw/."
        )
    return pd.read_excel(file_path, sheet_name=RAW_SHEET_NAMES, engine="openpyxl")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized column names."""
    df = df.copy()
    df.columns = [standardize_column_name(col) for col in df.columns]
    return df


def standardize_workbook_sheets(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Standardize sheet and column names used in the project."""
    return {standardize_column_name(name): standardize_columns(df) for name, df in sheets.items()}


def normalize_id_columns(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Rename likely ID columns to user_id after column standardization."""
    normalized: dict[str, pd.DataFrame] = {}
    id_candidates = ["user_id", "userid", "customer_id", "borrower_id"]

    for name, df in sheets.items():
        df = df.copy()
        for candidate in id_candidates:
            if candidate in df.columns and candidate != "user_id":
                df = df.rename(columns={candidate: "user_id"})
                break
        normalized[name] = df

    return normalized


def add_record_sequence(
    df: pd.DataFrame,
    id_column: str = "user_id",
    sequence_column: str = "record_sequence",
) -> pd.DataFrame:
    """Add a deterministic sequence number within each user_id.

    The source workbook contains repeated user_id values. Merging all sheets on
    user_id alone creates a many-to-many Cartesian expansion. The sequence column
    preserves the source observation grain and allows a one-to-one merge on
    [user_id, record_sequence].
    """
    if id_column not in df.columns:
        raise KeyError(f"{id_column} not found in DataFrame columns.")

    df = df.copy()
    df[sequence_column] = df.groupby(id_column, sort=False).cumcount() + 1
    return df


def merge_credit_risk_sheets(
    sheets: dict[str, pd.DataFrame],
    id_column: str = "user_id",
    sequence_column: str = "record_sequence",
) -> pd.DataFrame:
    """Merge source sheets safely at observation-level grain.

    Important:
    - user_id is not unique in the source workbook.
    - A merge on user_id alone inflates row count and changes target prevalence.
    - This function adds record_sequence to each sheet and merges on
      [user_id, record_sequence] with one-to-one validation.
    """
    required_sheets = ["loan_information", "employment", "personal_information", "other_information"]
    missing_sheets = [name for name in required_sheets if name not in sheets]
    if missing_sheets:
        raise KeyError(f"Missing expected sheet(s): {missing_sheets}")

    sequenced = {
        name: add_record_sequence(df, id_column=id_column, sequence_column=sequence_column)
        for name, df in sheets.items()
    }

    merge_keys = [id_column, sequence_column]

    merged = sequenced["loan_information"].merge(
        sequenced["employment"],
        on=merge_keys,
        how="left",
        validate="1:1",
    )
    merged = merged.merge(
        sequenced["personal_information"],
        on=merge_keys,
        how="left",
        validate="1:1",
    )
    merged = merged.merge(
        sequenced["other_information"],
        on=merge_keys,
        how="left",
        validate="1:1",
    )

    return merged


def merge_credit_risk_sheets_many_to_many_for_audit(
    sheets: dict[str, pd.DataFrame],
    id_column: str = "user_id",
) -> pd.DataFrame:
    """Legacy many-to-many merge retained only for audit comparison.

    Do not use this output for modelling when user_id is duplicated.
    """
    loan = sheets["loan_information"]
    employment = sheets["employment"]
    personal = sheets["personal_information"]
    other = sheets["other_information"]

    merged = loan.merge(employment, on=id_column, how="left", validate="m:m")
    merged = merged.merge(personal, on=id_column, how="left", validate="m:m")
    merged = merged.merge(other, on=id_column, how="left", validate="m:m")
    return merged
