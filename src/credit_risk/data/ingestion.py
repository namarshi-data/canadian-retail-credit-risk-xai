from __future__ import annotations

"""Data-ingestion utilities for the Canadian retail credit-risk project.

This module owns the raw Excel ingestion and observation-grain merge logic used
by Notebook 01 and ``scripts/run_data_pipeline.py``.

Important design decision
-------------------------
The raw workbook can contain repeated ``user_id`` values in each sheet. Merging
on ``user_id`` alone can create a many-to-many Cartesian expansion and distort
portfolio metrics, target rate, and model performance. The approved merge grain
is therefore ``user_id`` + ``record_sequence``.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence

import pandas as pd

RAW_SHEET_NAMES: list[str] = [
    "loan_information",
    "Employment",
    "Personal_information",
    "Other_information",
]

STANDARD_SHEET_NAME_MAP: dict[str, str] = {
    "loan_information": "loan_information",
    "loan information": "loan_information",
    "loan": "loan_information",
    "employment": "employment",
    "personal_information": "personal_information",
    "personal information": "personal_information",
    "personal": "personal_information",
    "other_information": "other_information",
    "other information": "other_information",
    "other": "other_information",
}

COLUMN_REPLACEMENTS: dict[str, str] = {
    "User_id": "user_id",
    "User id": "user_id",
    "User ID": "user_id",
    "Employmet type": "employment_type",
    "Employment type": "employment_type",
    "Total Payement ": "total_payment",
    "Total Payment": "total_payment",
    "Total Income(PA)": "total_income_pa",
    "Tenure(years)": "tenure_years",
    "Loan Category": "loan_category",
    "Interest Rate": "interest_rate",
    "Received Principal": "received_principal",
    "Interest Received": "interest_received",
    "Number of loans": "number_of_loans",
    "Social Profile": "social_profile",
    "Is_verified": "is_verified",
}

ID_CANDIDATES: tuple[str, ...] = (
    "user_id",
    "userid",
    "user id",
    "customer_id",
    "customer id",
    "borrower_id",
    "borrower id",
)

REQUIRED_STANDARD_SHEETS: tuple[str, ...] = (
    "loan_information",
    "employment",
    "personal_information",
    "other_information",
)


@dataclass(frozen=True)
class IngestionResult:
    """Container returned by the data-ingestion pipeline."""

    raw_sheets: dict[str, pd.DataFrame]
    standardized_sheets: dict[str, pd.DataFrame]
    merged: pd.DataFrame
    raw_sheet_overview: pd.DataFrame
    standardized_sheet_overview: pd.DataFrame
    merge_audit: pd.DataFrame


def standardize_column_name(column: object) -> str:
    """Convert raw source column names to stable snake_case names."""
    if column in COLUMN_REPLACEMENTS:
        return COLUMN_REPLACEMENTS[str(column)]

    cleaned = str(column).strip().lower()
    cleaned = cleaned.replace("%", "pct")
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def standardize_sheet_name(sheet_name: object) -> str:
    """Normalize workbook sheet names to project-standard keys."""
    raw = str(sheet_name).strip()
    lowered = raw.lower().replace("_", " ")
    return STANDARD_SHEET_NAME_MAP.get(raw, STANDARD_SHEET_NAME_MAP.get(lowered, standardize_column_name(raw)))


def read_source_workbook(file_path: str | Path, sheet_names: Sequence[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read expected source sheets from the raw Excel workbook.

    Parameters
    ----------
    file_path:
        Path to the raw workbook.
    sheet_names:
        Optional list of raw sheet names. Defaults to ``RAW_SHEET_NAMES``.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw workbook not found at {file_path}. Place Credit_Risk_Dataset.xlsx in data/raw/."
        )

    try:
        return pd.read_excel(file_path, sheet_name=list(sheet_names or RAW_SHEET_NAMES), engine="openpyxl")
    except ValueError as exc:
        available = pd.ExcelFile(file_path, engine="openpyxl").sheet_names
        raise ValueError(
            "Could not read the expected workbook sheets. "
            f"Expected: {list(sheet_names or RAW_SHEET_NAMES)}. Available: {available}."
        ) from exc


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with standardized column names."""
    out = df.copy()
    out.columns = [standardize_column_name(col) for col in out.columns]
    return out


def standardize_workbook_sheets(sheets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Standardize both sheet names and column names used in the project."""
    standardized: dict[str, pd.DataFrame] = {}
    for raw_name, df in sheets.items():
        standardized[standardize_sheet_name(raw_name)] = standardize_columns(df)
    return standardized


def normalize_id_columns(sheets: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Rename likely ID columns to ``user_id`` after column standardization."""
    normalized: dict[str, pd.DataFrame] = {}
    standardized_id_candidates = [standardize_column_name(col) for col in ID_CANDIDATES]

    for name, df in sheets.items():
        out = df.copy()
        for candidate in standardized_id_candidates:
            if candidate in out.columns and candidate != "user_id":
                out = out.rename(columns={candidate: "user_id"})
                break
        normalized[name] = out
    return normalized


def source_sheet_overview(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize raw or standardized workbook sheets."""
    return pd.DataFrame(
        [
            {
                "sheet_name": sheet_name,
                "row_count": int(sheet_df.shape[0]),
                "column_count": int(sheet_df.shape[1]),
                "columns": ", ".join(map(str, sheet_df.columns)),
            }
            for sheet_name, sheet_df in sheets.items()
        ]
    )


def validate_required_sheets(sheets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Check that all required standardized sheets are available."""
    rows = []
    for sheet_name in REQUIRED_STANDARD_SHEETS:
        rows.append({"sheet_name": sheet_name, "present": sheet_name in sheets})
    return pd.DataFrame(rows)


def add_record_sequence(
    df: pd.DataFrame,
    id_column: str = "user_id",
    sequence_column: str = "record_sequence",
) -> pd.DataFrame:
    """Add a deterministic observation sequence number within each borrower ID.

    This preserves the source observation grain and supports a one-to-one merge on
    ``[user_id, record_sequence]``.
    """
    if id_column not in df.columns:
        raise KeyError(f"{id_column!r} not found in DataFrame columns.")

    out = df.copy()
    out[sequence_column] = out.groupby(id_column, sort=False).cumcount() + 1
    return out


def build_sheet_grain_summary(
    sheets: Mapping[str, pd.DataFrame],
    id_column: str = "user_id",
) -> pd.DataFrame:
    """Summarize row count, duplicate ID count, and maximum observations per ID."""
    rows: list[dict[str, object]] = []
    for name, df in sheets.items():
        if id_column not in df.columns:
            rows.append(
                {
                    "sheet_name": name,
                    "id_column_present": False,
                    "row_count": len(df),
                    "unique_id_count": None,
                    "duplicate_id_count": None,
                    "max_records_per_id": None,
                    "merge_grain_decision": "cannot_validate_missing_id",
                }
            )
            continue

        counts = df[id_column].value_counts(dropna=False)
        duplicate_id_count = int(df[id_column].duplicated().sum())
        rows.append(
            {
                "sheet_name": name,
                "id_column_present": True,
                "row_count": int(len(df)),
                "unique_id_count": int(df[id_column].nunique(dropna=True)),
                "duplicate_id_count": duplicate_id_count,
                "max_records_per_id": int(counts.max()) if not counts.empty else 0,
                "merge_grain_decision": (
                    "use_user_id_plus_record_sequence" if duplicate_id_count > 0 else "user_id_unique_in_sheet"
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_credit_risk_sheets(
    sheets: Mapping[str, pd.DataFrame],
    id_column: str = "user_id",
    sequence_column: str = "record_sequence",
) -> pd.DataFrame:
    """Merge source sheets safely at observation-level grain.

    Important:
    - ``user_id`` is not guaranteed unique in the source workbook.
    - A merge on ``user_id`` alone can inflate row count and change target prevalence.
    - This function adds ``record_sequence`` to each sheet and merges on
      ``[user_id, record_sequence]`` with one-to-one validation.
    """
    missing_sheets = [name for name in REQUIRED_STANDARD_SHEETS if name not in sheets]
    if missing_sheets:
        raise KeyError(f"Missing expected standardized sheet(s): {missing_sheets}")

    sequenced = {
        name: add_record_sequence(df, id_column=id_column, sequence_column=sequence_column)
        for name, df in sheets.items()
    }

    merge_keys = [id_column, sequence_column]
    merged = sequenced["loan_information"].merge(
        sequenced["employment"], on=merge_keys, how="left", validate="1:1"
    )
    merged = merged.merge(
        sequenced["personal_information"], on=merge_keys, how="left", validate="1:1"
    )
    merged = merged.merge(
        sequenced["other_information"], on=merge_keys, how="left", validate="1:1"
    )
    return merged


def merge_credit_risk_sheets_many_to_many_for_audit(
    sheets: Mapping[str, pd.DataFrame],
    id_column: str = "user_id",
) -> pd.DataFrame:
    """Legacy many-to-many merge retained only for audit comparison."""
    required = REQUIRED_STANDARD_SHEETS
    missing_sheets = [name for name in required if name not in sheets]
    if missing_sheets:
        raise KeyError(f"Missing expected standardized sheet(s): {missing_sheets}")

    merged = sheets["loan_information"].merge(sheets["employment"], on=id_column, how="left", validate="m:m")
    merged = merged.merge(sheets["personal_information"], on=id_column, how="left", validate="m:m")
    merged = merged.merge(sheets["other_information"], on=id_column, how="left", validate="m:m")
    return merged


def build_merge_audit(
    sheets: Mapping[str, pd.DataFrame],
    merged: pd.DataFrame,
    id_column: str = "user_id",
    sequence_column: str = "record_sequence",
    target_column: str = "defaulter",
) -> pd.DataFrame:
    """Build merge validation and grain-audit summary."""
    first_sheet_name = REQUIRED_STANDARD_SHEETS[0]
    expected_row_count = len(sheets[first_sheet_name]) if first_sheet_name in sheets else None
    key_cols = [id_column, sequence_column]
    key_present = all(col in merged.columns for col in key_cols)
    duplicate_key_count = int(merged[key_cols].duplicated().sum()) if key_present else None

    target_present = target_column in merged.columns
    default_rate = None
    if target_present:
        target = pd.to_numeric(merged[target_column], errors="coerce")
        default_rate = float(target.mean())

    rows = [
        ("expected_row_count_from_loan_information", expected_row_count),
        ("merged_row_count", int(len(merged))),
        ("merged_column_count", int(merged.shape[1])),
        ("row_count_preserved", bool(expected_row_count == len(merged)) if expected_row_count is not None else False),
        ("record_sequence_present", sequence_column in merged.columns),
        ("record_key_duplicate_count", duplicate_key_count),
        ("target_present", target_present),
        ("default_rate", default_rate),
    ]
    return pd.DataFrame(rows, columns=["check", "result"])


def run_ingestion_workflow(raw_path: str | Path) -> IngestionResult:
    """Execute the core ingestion workflow and return all key objects."""
    raw_sheets = read_source_workbook(raw_path)
    raw_overview = source_sheet_overview(raw_sheets)
    standardized_sheets = normalize_id_columns(standardize_workbook_sheets(raw_sheets))
    standardized_overview = source_sheet_overview(standardized_sheets)
    merged = merge_credit_risk_sheets(standardized_sheets)
    merge_audit = build_merge_audit(standardized_sheets, merged)
    return IngestionResult(
        raw_sheets=raw_sheets,
        standardized_sheets=standardized_sheets,
        merged=merged,
        raw_sheet_overview=raw_overview,
        standardized_sheet_overview=standardized_overview,
        merge_audit=merge_audit,
    )
